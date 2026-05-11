"""
Convergence Diagnostic for PyBaMM DFN + Degradation Models

Tests stress protocols across degradation configurations to characterize
CasADi/SUNDIALS solver failure modes. Run BEFORE implementing robustness
fixes to establish a baseline, then again AFTER fixes to compare.

Usage:
    # Baseline (uses original Level-0 solver config):
    python convergence_diagnostic.py

    # After fix (uses PyBaMMSimulator's fallback hierarchy):
    python convergence_diagnostic.py --after-fix

Output:
    Printed summary table + CSV at multi_objective/results/diagnostic_results.csv
"""

import sys
import os
import warnings
import logging
import re
import time
import csv
import concurrent.futures
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup — capture ALL warnings, do NOT suppress anything
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.captureWarnings(True)
logger = logging.getLogger("diagnostic")

import pybamm
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Test matrix
# ──────────────────────────────────────────────────────────────────────────────

# (label, c_rates, step_durations_min)
STRESS_PROTOCOLS: List[Tuple[str, List[float], List[float]]] = [
    # --- High C-rate single steps ---
    ("4C_30min",         [4.0],                [30.0]),
    ("5C_30min",         [5.0],                [30.0]),
    ("6C_30min",         [6.0],                [30.0]),
    # --- Abrupt step transitions (10 min each) ---
    ("rest-5C-rest",     [0.0, 5.0, 0.0],      [10.0, 10.0, 10.0]),
    ("5C-rest-5C",       [5.0, 0.0, 5.0],      [10.0, 10.0, 10.0]),
    ("0C-to-4C",         [0.0, 4.0],           [5.0, 25.0]),
    # --- Very short step durations ---
    ("3C_1min",          [3.0],                [1.0]),
    ("3C_2min",          [3.0],                [2.0]),
    ("3C-3C_1min_each",  [3.0, 3.0],           [1.0, 1.0]),
    # --- Rest-interspersed (these should work) ---
    ("3C-rest-3C",       [3.0, 0.0, 3.0],      [10.0, 10.0, 10.0]),
    ("3C-rest-rest-3C",  [3.0, 0.0, 0.0, 3.0], [7.5, 7.5, 7.5, 7.5]),
    # --- Low C reference (should always succeed) ---
    ("2C_30min",         [2.0],                [30.0]),
]

DEGRADATION_COMBOS: List[Tuple[str, List[str], str]] = [
    # (label, degradation_modes, sei_model)
    ("plating_only",    ["plating"],         "ec reaction limited"),
    ("SEI_ec",          ["SEI"],             "ec reaction limited"),
    ("SEI_solvent",     ["SEI"],             "solvent-diffusion limited"),
    ("plating_SEI_ec",  ["plating", "SEI"],  "ec reaction limited"),
    ("plating_SEI_sd",  ["plating", "SEI"],  "solvent-diffusion limited"),
]

TIMEOUT_SECONDS = 120  # Per-sim timeout; enough for fallback hierarchy to try L0→L1

# ──────────────────────────────────────────────────────────────────────────────
# Solver configurations
# ──────────────────────────────────────────────────────────────────────────────

# Level 0 — original config from pybamm_simulator.py (used for baseline)
SOLVER_CONFIG_LEVEL0 = dict(
    mode="fast with events",
    dt_max=1,
    rtol=1e-3,
    atol=1e-6,
    extra_options_setup={"max_num_steps": 20000},
)

# ──────────────────────────────────────────────────────────────────────────────
# Model / experiment builders (mirrors pybamm_simulator.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

V_MIN = 3.0
V_MAX = 4.2
SOC_START = 0.1


def build_model_options(degradation_modes: List[str], sei_model: str) -> Dict:
    options: Dict = {}
    if "plating" in degradation_modes:
        options["lithium plating"] = "reversible"
    if "SEI" in degradation_modes:
        options["SEI"] = sei_model
        options["SEI film resistance"] = "distributed"
        options["SEI porosity change"] = "true"
    return options


def build_experiment_original(c_rates: List[float], step_durations: List[float]) -> pybamm.Experiment:
    """Exact replica of the CURRENT (pre-fix) _build_experiment()."""
    v_start = V_MIN + (V_MAX - V_MIN) * SOC_START  # 3.12 V
    discharge_step = f"Discharge at 1C until {v_start:.2f}V"
    hold_step = f"Hold at {v_start:.2f}V for 10 minutes"  # buggy form
    charge_steps = []
    for c_rate, duration in zip(c_rates, step_durations):
        if c_rate == 0.0:
            charge_steps.append(f"Rest for {duration:.1f} minutes")
        else:
            charge_steps.append(f"Charge at {c_rate}C for {duration:.1f} minutes")
    return pybamm.Experiment([discharge_step, hold_step] + charge_steps)


# ──────────────────────────────────────────────────────────────────────────────
# Failure classification
# ──────────────────────────────────────────────────────────────────────────────

_FAILURE_PATTERNS = [
    ("NEWTON_FAIL",      r"corrector convergence failed"),
    ("LINESEARCH_FAIL",  r"linesearch algorithm"),
    ("STIFF_STEP_LIMIT", r"maximum number of steps"),
    ("EVENT_FAIL",       r"event detection"),
    ("INFEASIBLE",       r"infeasible"),
    ("VOLTAGE_EVENT",    r"maximum voltage"),
]


def classify_failure(error_msg: str, warning_strs: List[str]) -> str:
    combined = " ".join([error_msg or ""] + warning_strs).lower()
    for label, pattern in _FAILURE_PATTERNS:
        if re.search(pattern, combined):
            return label
    if error_msg:
        return "OTHER_ERROR"
    return "SUCCESS"


# ──────────────────────────────────────────────────────────────────────────────
# Single simulation (no warning suppression)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_baseline(
    protocol_label: str,
    c_rates: List[float],
    step_durations: List[float],
    deg_label: str,
    degradation_modes: List[str],
    sei_model: str,
    solver_config: Dict,
) -> Dict:
    """Run one simulation with the original solver config (no suppression)."""
    result: Dict = {
        "protocol":      protocol_label,
        "c_rates":       str(c_rates),
        "step_durations": str(step_durations),
        "degradation":   deg_label,
        "sei_model":     sei_model,
        "success":       False,
        "failure_type":  "UNKNOWN",
        "error_msg":     "",
        "warning_count": 0,
        "warning_sample": "",
        "elapsed_s":     0.0,
    }

    t_start = time.time()
    try:
        model_options = build_model_options(degradation_modes, sei_model)
        experiment = build_experiment_original(c_rates, step_durations)
        model = pybamm.lithium_ion.DFN(options=model_options)
        param_values = pybamm.ParameterValues("OKane2022")
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param_values)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solution = sim.solve(solver=pybamm.CasadiSolver(**solver_config))

        warning_strs = [str(w.message) for w in caught]
        result["warning_count"] = len(warning_strs)
        result["warning_sample"] = "; ".join(warning_strs[:3])

        # Classify: if warnings contain failure text, mark as WARNING_<type>
        ft = classify_failure("", warning_strs)
        if ft != "SUCCESS":
            result["failure_type"] = f"WARN_{ft}"
            result["success"] = True  # solution was returned despite warning
        else:
            result["failure_type"] = "SUCCESS"
            result["success"] = True

    except Exception as exc:
        error_msg = str(exc)
        result["error_msg"] = error_msg[:500]
        result["failure_type"] = classify_failure(error_msg, [])
        result["success"] = False

    result["elapsed_s"] = round(time.time() - t_start, 2)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# After-fix runner — uses PyBaMMSimulator with fallback hierarchy
# ──────────────────────────────────────────────────────────────────────────────

def run_single_afterfix(
    protocol_label: str,
    c_rates: List[float],
    step_durations: List[float],
    deg_label: str,
    degradation_modes: List[str],
    sei_model: str,
) -> Dict:
    """Run via the fixed PyBaMMSimulator (fallback hierarchy active)."""
    # Import here so baseline run doesn't depend on the updated simulator
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from multi_objective.utils.pybamm_simulator import PyBaMMSimulator

    result: Dict = {
        "protocol":      protocol_label,
        "c_rates":       str(c_rates),
        "step_durations": str(step_durations),
        "degradation":   deg_label,
        "sei_model":     sei_model,
        "success":       False,
        "failure_type":  "UNKNOWN",
        "error_msg":     "",
        "warning_count": 0,
        "warning_sample": "",
        "elapsed_s":     0.0,
        "solver_level":  -1,
        "approximate":   False,
    }

    t_start = time.time()
    try:
        sim_obj = PyBaMMSimulator(
            degradation_modes=degradation_modes,
            sei_model=sei_model,
        )
        run_result = sim_obj.run_protocol(c_rates, step_durations)

        result["success"]      = run_result["success"]
        result["error_msg"]    = run_result.get("error", "") or ""
        result["solver_level"] = run_result.get("solver_level", -1)
        result["approximate"]  = run_result.get("approximate", False)
        result["failure_type"] = "SUCCESS" if run_result["success"] else classify_failure(result["error_msg"], [])

    except Exception as exc:
        result["error_msg"]   = str(exc)[:500]
        result["failure_type"] = "OTHER_ERROR"

    result["elapsed_s"] = round(time.time() - t_start, 2)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Timeout wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run_with_timeout(fn, args=(), kwargs=None, timeout=TIMEOUT_SECONDS) -> Dict:
    if kwargs is None:
        kwargs = {}
    # NOTE: Do NOT use `with ThreadPoolExecutor` — the context manager calls
    # shutdown(wait=True) on exit, which blocks indefinitely when the C++
    # solver thread is stuck.  We call shutdown(wait=False) explicitly so the
    # stuck thread is abandoned and the diagnostic loop keeps moving.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)  # abandon stuck C++ thread, do not block
        return {
            "protocol":       kwargs.get("protocol_label", args[0] if args else "unknown"),
            "c_rates":        str(kwargs.get("c_rates", "")),
            "step_durations": str(kwargs.get("step_durations", "")),
            "degradation":    kwargs.get("deg_label", ""),
            "sei_model":      kwargs.get("sei_model", ""),
            "success":        False,
            "failure_type":   "TIMEOUT",
            "error_msg":      f"Exceeded {timeout}s timeout",
            "warning_count":  0,
            "warning_sample": "",
            "elapsed_s":      float(timeout),
            "solver_level":   -1,
            "approximate":    False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: List[Dict], label: str = "") -> None:
    total = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 72}")
    print(f"SUMMARY TABLE  {label}")
    print(f"{'=' * 72}")
    print(f"Total simulations: {total}   Succeeded: {successes}   Failed: {total - successes}")

    # Failure-type distribution
    by_type = Counter(r["failure_type"] for r in all_results)
    print(f"\nFailure-type distribution:")
    for ft, count in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        bar = "█" * max(1, int(pct / 3))
        marker = "✓" if ft == "SUCCESS" else "✗"
        print(f"  {marker} {ft:28s}: {count:3d}  ({pct:5.1f}%)  {bar}")

    # Per degradation combo
    print(f"\nFailure rate by degradation combo:")
    for deg_label, _, _ in DEGRADATION_COMBOS:
        sub = [r for r in all_results if r["degradation"] == deg_label]
        failed = [r for r in sub if not r["success"]]
        pct = 100.0 * len(failed) / max(len(sub), 1)
        bar = "█" * max(1, int(pct / 5))
        print(f"  {deg_label:28s}: {len(failed):2d}/{len(sub)}  ({pct:5.1f}%)  {bar}")

    # Per protocol
    print(f"\nFailure rate by protocol:")
    for proto_label, _, _ in STRESS_PROTOCOLS:
        sub = [r for r in all_results if r["protocol"] == proto_label]
        failed = [r for r in sub if not r["success"]]
        pct = 100.0 * len(failed) / max(len(sub), 1)
        bar = "█" * max(1, int(pct / 5))
        print(f"  {proto_label:28s}: {len(failed):2d}/{len(sub)}  ({pct:5.1f}%)  {bar}")


def print_before_after(before: List[Dict], after: List[Dict]) -> None:
    """Print a side-by-side comparison of before/after failure rates."""
    print(f"\n{'=' * 80}")
    print("BEFORE vs AFTER FIX COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Protocol':28s}  {'Degradation':20s}  {'Before':15s}  {'After':15s}  {'Change'}")
    print("-" * 90)

    def find(results, protocol, deg):
        for r in results:
            if r["protocol"] == protocol and r["degradation"] == deg:
                return r
        return None

    recovered = 0
    still_failed = 0
    for proto_label, _, _ in STRESS_PROTOCOLS:
        for deg_label, _, _ in DEGRADATION_COMBOS:
            b = find(before, proto_label, deg_label)
            a = find(after, proto_label, deg_label)
            if b is None or a is None:
                continue
            b_ok = "OK" if b["success"] else b["failure_type"]
            a_ok = "OK" if a["success"] else a["failure_type"]
            change = ""
            if not b["success"] and a["success"]:
                change = "✓ FIXED"
                recovered += 1
            elif not b["success"] and not a["success"]:
                change = "✗ STILL FAILING"
                still_failed += 1
            elif b["success"] and not a["success"]:
                change = "⚠ REGRESSION"
            print(f"  {proto_label:26s}  {deg_label:20s}  {b_ok:15s}  {a_ok:15s}  {change}")

    print(f"\nRecovered: {recovered}   Still failing: {still_failed}")


def save_csv(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "protocol", "c_rates", "step_durations", "degradation", "sei_model",
        "success", "failure_type", "error_msg", "warning_count",
        "warning_sample", "elapsed_s", "solver_level", "approximate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostic(after_fix: bool = False) -> List[Dict]:
    total = len(STRESS_PROTOCOLS) * len(DEGRADATION_COMBOS)
    mode = "AFTER-FIX" if after_fix else "BASELINE (before fix)"
    print(f"\n{'=' * 72}")
    print(f"PyBaMM Convergence Diagnostic — {mode}")
    print(f"{'=' * 72}")
    print(f"Protocols: {len(STRESS_PROTOCOLS)}   Degradation combos: {len(DEGRADATION_COMBOS)}")
    print(f"Total simulations: {total}   Timeout per sim: {TIMEOUT_SECONDS}s")
    print()

    all_results: List[Dict] = []
    idx = 0

    for proto_label, c_rates, step_durations in STRESS_PROTOCOLS:
        for deg_label, degradation_modes, sei_model in DEGRADATION_COMBOS:
            idx += 1
            print(
                f"[{idx:3d}/{total}] {proto_label:22s} × {deg_label:22s} ...",
                end="",
                flush=True,
            )

            if after_fix:
                result = run_with_timeout(
                    run_single_afterfix,
                    kwargs=dict(
                        protocol_label=proto_label,
                        c_rates=c_rates,
                        step_durations=step_durations,
                        deg_label=deg_label,
                        degradation_modes=degradation_modes,
                        sei_model=sei_model,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
            else:
                result = run_with_timeout(
                    run_single_baseline,
                    kwargs=dict(
                        protocol_label=proto_label,
                        c_rates=c_rates,
                        step_durations=step_durations,
                        deg_label=deg_label,
                        degradation_modes=degradation_modes,
                        sei_model=sei_model,
                        solver_config=SOLVER_CONFIG_LEVEL0,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )

            status = result["failure_type"]
            elapsed = result["elapsed_s"]
            lvl_info = f" [L{result.get('solver_level', 0)}]" if after_fix and result.get("solver_level", -1) >= 0 else ""
            approx = " [APPROX]" if result.get("approximate") else ""
            print(f" {status:25s}  {elapsed:6.1f}s{lvl_info}{approx}")

            all_results.append(result)

    return all_results


def main() -> None:
    after_fix = "--after-fix" in sys.argv

    results = run_diagnostic(after_fix=after_fix)

    suffix = "after_fix" if after_fix else "baseline"
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "results", f"diagnostic_results_{suffix}.csv"
    )
    save_csv(results, csv_path)
    print_summary(results, label=f"({'AFTER FIX' if after_fix else 'BASELINE'})")

    # If both files exist, show comparison
    baseline_csv = os.path.join(
        os.path.dirname(__file__), "..", "results", "diagnostic_results_baseline.csv"
    )
    afterfix_csv = os.path.join(
        os.path.dirname(__file__), "..", "results", "diagnostic_results_after_fix.csv"
    )
    if after_fix and os.path.exists(baseline_csv):
        import csv as _csv
        with open(baseline_csv) as f:
            before = list(_csv.DictReader(f))
        # Convert string booleans back
        for r in before:
            r["success"] = r["success"].lower() == "true"
        print_before_after(before, results)

    print("\nDone.")


if __name__ == "__main__":
    main()
