# Convergence Robustness Report: PyBaMM DFN + Degradation Solver Fixes

**Date:** 2026-05-06  
**Engineer:** DJ Walker  
**Scope:** `multi_objective/utils/pybamm_simulator.py` — solver and experiment construction only.  
Physical model (OKane2022 parameter set) is unchanged.

---

## 1. Baseline Failure Characterisation

A stress-protocol diagnostic was run across 60 combinations of 12 protocols × 5 degradation
configurations before any changes were made.

### Test matrix

| Axis | Values |
|------|--------|
| Protocols | 4C/30min, 5C/30min, 6C/30min, rest–5C–rest, 5C–rest–5C, 0C→4C, 3C/1min, 3C/2min, 3C–3C/1min, 3C–rest–3C, 3C–rest–rest–3C, 2C/30min |
| Degradation | plating_only, SEI_ec, SEI_solvent, plating+SEI_ec, plating+SEI_sd |

### Baseline results (original code, Level 0 solver only)

| Outcome | Count | % |
|---------|-------|---|
| SUCCESS | 59 | 98.3 % |
| TIMEOUT | 1  | 1.7 %  |

The single timeout was `0C→4C × plating+SEI_ec`.  All other configurations — including 6C single-step,
5C–rest–5C, and 1-minute steps — returned a valid solution within 45 seconds.

### Important observations

* **Voltage-event warnings are not failures.** High C-rate protocols (4C, 5C, 6C) trigger the
  "Maximum voltage [V] was triggered" PyBaMM warning and return a *partial* solution.  The original
  code suppressed this warning silently; the new code logs it.

* **Corrector-convergence warnings are not failures.** The SUNDIALS IDAS solver emits "corrector
  convergence failed" messages for many normal protocols but still converges and returns a valid
  solution.  Suppressing these with `warnings.filterwarnings("ignore")` was hiding legitimate signal.

---

## 2. Root Causes (ranked by impact)

### RC-1 — Silent failure masking (lines 185-186, highest priority)

`pybamm_simulator.py` applied `warnings.filterwarnings("ignore", ...)` for two critical warning
classes inside `run_protocol()`:

```python
warnings.filterwarnings("ignore", message=".*corrector convergence.*")
warnings.filterwarnings("ignore", message=".*linesearch algorithm.*")
```

These are exactly the warning strings that indicate the IDAS solver is struggling.  Suppressing them
means every convergence warning is invisible to operators and to the optimizer.  A trial that returns
a partial solution looks identical to a trial that ran cleanly.

**Fix:** Removed these suppressions.  All solver warnings now flow through the Python `logging`
module (`logging.captureWarnings(True)` at module level) and are emitted at `WARNING` level.

---

### RC-2 — No fallback when Level 0 fails (highest operational impact)

The original code used a single fixed solver configuration.  When that configuration failed, the
optimizer received `NaN` metrics with no recovery attempt.  Level 0 is fast but is known to produce
corrector-convergence failures for stiff protocols.

**Fix:** Implemented a three-level fallback hierarchy (see Section 4.1).

---

### RC-3 — IDAS corrector-loop hang: C++ solver does not honour `max_num_steps` when stuck

When the IDAS Newton corrector collapses to h ≈ 10⁻²⁰ s, the SUNDIALS step counter (`max_num_steps`)
never increments because each "step" immediately fails before converging.  IDAS spins in an internal
halving loop (up to `max_conv_fails=10` failures per step) before returning `IDA_CONV_FAIL`.
PyBaMM then halves `dt_max` and retries up to `max_step_decrease_count=5` times (default).

For the one failing case (`0C→4C × plating+SEI_ec`), this cycle took **10–15 minutes** of wall-clock
time before PyBaMM raised `"Maximum number of decreased steps"`.  This made the fallback hierarchy
useless within any reasonable trial timeout: Level 0 could not exhaust its retries fast enough.

**Fix:** Each solver level is run in a `ThreadPoolExecutor` worker with a hard wall-clock timeout
(`_LEVEL_WALL_TIMEOUTS = [25, 50, 60]` seconds).  When the timeout fires, the C++ thread is abandoned
(it cannot be forcibly killed in CPython without process isolation) and the next level begins.
Cumulative worst-case time for one trial: 25 + 50 + 60 + 60 (Level 3) = 195 seconds.

---

### RC-4 — "Hold at V" pre-conditioning: retained as-is

An initial hypothesis proposed replacing `"Hold at {v}V for 10 minutes"` with `"Rest for 10 minutes"`
to avoid an extra algebraic equation (voltage control) in the DAE system.

**Empirical test disproved this.** With `"Rest"`, the `4C_30min × plating+SEI_ec` combination
(which succeeded in 8 seconds under the original code) became a timeout.  The voltage-hold step
provides a consistent algebraic initial condition for the combined degradation model.  The constraint
anchors all algebraic variables (solid/electrolyte potentials, SEI film) to a coherent steady state
before the charge steps begin.  This is beneficial even though it adds one algebraic equation.

The `"Hold at V"` step is **retained unchanged**.

---

### RC-5 — No protocol pre-screening

The optimizer could generate protocols with C-rates > 6C or step durations < 1 min without any
early warning.

**Fix:** Added `validate_protocol()` method.

---

## 3. Code Changes

### 3.1 Solver fallback hierarchy

**File:** `multi_objective/utils/pybamm_simulator.py`

```python
_SOLVER_LEVELS = [
    # Level 0: original fast config
    dict(mode="fast with events", dt_max=1, rtol=1e-3, atol=1e-6,
         extra_options_setup={"max_num_steps": 20000}),
    # Level 1: safe mode, tighter tolerances
    dict(mode="safe", dt_max=10, rtol=1e-4, atol=1e-8,
         extra_options_setup={"max_num_steps": 40000}),
    # Level 2: most robust
    dict(mode="safe", dt_max=None, rtol=1e-6, atol=1e-10,
         extra_options_setup={"max_num_steps": 100000}),
]
_LEVEL_WALL_TIMEOUTS = [25, 50, 60]  # hard wall-clock timeout per level (seconds)
```

Each level is attempted in a `ThreadPoolExecutor` worker.  On timeout or exception the worker thread
is abandoned and the next level begins.  A fresh `pybamm.Simulation` object is created for each
level to avoid shared-state race conditions between active and abandoned threads.

Level 3 (last resort) rebuilds the model without `"SEI porosity change"` (slightly different physics)
and marks the result as `approximate=True`.  This option is only attempted when `"SEI"` is in
`degradation_modes`.

### 3.2 `_solver_stats` tracking

Each `PyBaMMSimulator` instance now exposes:
```python
self._solver_stats = {"level_0": 0, "level_1": 0, "level_2": 0, "level_3": 0, "failed": 0}
```
Incremented per successful trial.  Use this to monitor optimizer health over many trials.

### 3.3 Logging instead of warning suppression

Removed `warnings.filterwarnings("ignore", ...)` calls.  Module now uses:
```python
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
```
All SUNDIALS corrector-convergence and linesearch warnings are routed through the Python logging
system.  Callers can configure log level and handlers; nothing is silently discarded.

### 3.4 `validate_protocol()` method

New public method:
```python
sim.validate_protocol(c_rates, step_durations=None)
# → {'valid': bool, 'warnings': list[str], 'risk_score': float}
```

Risk scoring:
| Condition | Risk added |
|-----------|-----------|
| max C-rate > 6C | +0.9 |
| max C-rate > 4C | +0.4 |
| max C-rate 3–4C | +0.15 |
| min step duration < 1 min | +0.4 |
| min step duration 1–2 min | +0.2 |
| C-rate transition ≥ 5C | +0.35 per step |
| C-rate transition 3–5C | +0.10 per step |
| Solvent-diffusion SEI model | +0.2 (baseline elevated risk) |

`valid` = False when `risk_score ≥ 0.9`.

### 3.5 `run_and_extract()` — optional pre-screening

```python
sim.run_and_extract(c_rates, step_durations=None, verbose=False, validate_first=False)
```

When `validate_first=True`, a risk score > 0.5 emits a `logger.warning` before solving.
Default is `False` for backward compatibility.

---

## 4. Diagnostic Results (Before vs After)

### 4.1 Baseline (original code)

| Degradation combo | Failed / Total | Failure type |
|-------------------|---------------|--------------|
| plating_only | 0 / 12 | — |
| SEI_ec | 0 / 12 | — |
| SEI_solvent | 0 / 12 | — |
| plating+SEI_ec | 1 / 12 | TIMEOUT |
| plating+SEI_sd | 0 / 12 | — |
| **Total** | **1 / 60** | **1.7 %** |

### 4.2 After fix (empirical — full 60-sim diagnostic re-run)

| Degradation combo | Failed / Total | Notes |
|-------------------|---------------|-------|
| plating_only | 0 / 12 | — |
| SEI_ec | 0 / 12 | — |
| SEI_solvent | 0 / 12 | — |
| plating+SEI_ec | 0 / 12 | `0C→4C` recovered at Level 1 (62 s total) |
| plating+SEI_sd | 0 / 12 | — |
| **Total** | **0 / 60** | **0 % failure rate** |

**Solver level breakdown:**

| Level | Trials | Description |
|-------|--------|-------------|
| 0 (fast with events) | 59 / 60 | Original fast configuration; handles all common protocols |
| 1 (safe, rtol=1e-4) | 1 / 60 | `0C→4C × plating+SEI_ec`; Level 0 timed out at 25 s, Level 1 solved in ~37 s more |
| 2 (most robust) | 0 | Not needed in this test matrix |
| 3 (simplified model) | 0 | Not needed; no approximate results produced |

The previously-failing `0C→4C × plating+SEI_ec` case was resolved exactly (not approximately) at
Level 1 in 62 s wall-clock total.

---

## 5. Remaining Failure Modes

### 5.1 Zombie C++ threads

When a level is abandoned due to wall-clock timeout, the underlying IDAS computation continues in a
background thread until IDAS exhausts its internal retry logic (10–15 minutes in the worst case).
During this time, the process has additional CPU load.  In an optimization loop with many simultaneous
threads (e.g., qNEHVI with batch size 3), a pathological protocol can leave multiple zombie threads.

**Mitigation:** The optimizer should limit concurrent trials; the existing notebook uses a single
`ThreadPoolExecutor` with 120 s timeout per batch, which bounds the accumulation.

**Root fix (not implemented):** Spawn each simulation in a subprocess (using `multiprocessing` or a
process pool) so it can be forcibly terminated.  This is a significant refactor and was deemed out
of scope for this session.

### 5.2 `plating+SEI_ec` with abrupt high-C transitions is fundamentally stiff

The `0C→4C × plating+SEI_ec` case exhibits a Newton corrector collapse that begins at t ≈ 4247 s
(107 seconds into the 4C charge step).  This is caused by the combined Butler–Volmer nonlinearity
(exponential kinetics from both plating and SEI models) reacting to an abrupt step change in
applied current.  The stiffness is physical: abrupt current steps create discontinuities in
electrode overpotentials that propagate through the coupled SEI reaction and porosity feedback loop.

Workarounds:
- Avoid 0C→high-C abrupt transitions in the optimizer parameter space
- Add a ramp step (e.g., 1 s linear interpolation) between abrupt transitions
- Consider switching to the `IDAKLUSolver` (sparse direct linear algebra, generally more robust
  for DFN-scale problems)

---

## 6. Recommended Next Steps

1. **Switch to IDAKLUSolver for production runs** (see PyBaMM docs).  IDAKLU uses KLU sparse direct
   solver instead of dense LAPACK, which is better conditioned for the DFN DAE system at scale.

2. **Add multiprocessing isolation** so solver timeouts actually kill C++ computation.  Use
   `concurrent.futures.ProcessPoolExecutor` with a custom initialiser that pre-builds the model.

3. **Integrate `validate_protocol()` with the Ax experiment** as a constraint function so the
   optimizer avoids high-risk regions of parameter space automatically.

4. **Monitor `_solver_stats`** across optimiser trials.  If `level_1 + level_2 + level_3 > 5 %`
   of trials, the parameter space may include structurally difficult regions and the model should
   be re-evaluated.

5. **Consider smooth transitions** between charge steps.  A 30-second ramp (linearly increasing
   current) between large C-rate steps dramatically reduces the overpotential discontinuity and
   removes the primary source of corrector-convergence warnings in the stiff combined models.
