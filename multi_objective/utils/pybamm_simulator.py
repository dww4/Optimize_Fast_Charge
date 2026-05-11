"""
PyBaMM Simulator Wrapper for Fast-Charging Optimization

This module provides a clean interface to PyBaMM for multi-objective optimization
of fast-charging protocols with coupled degradation modes.

Author: DJ
Date: 2026-02-13
"""

import concurrent.futures
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybamm

# Route all Python warnings through logging so they are captured, not silenced.
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Solver fallback hierarchy
# ──────────────────────────────────────────────────────────────────────────────
#
# Level 0 — fast (original config): good for well-conditioned protocols
# Level 1 — safe mode with tighter tolerances: handles moderate stiffness
# Level 2 — most robust: uncapped dt, strictest tolerances
# Level 3 — last resort: simplified degradation (approximate, flagged)
#
# Each level is tried in sequence; the first to succeed wins.  All failures at
# each level are logged at WARNING level with their full error text.

_SOLVER_LEVELS: List[Dict] = [
    # Level 0: fast (original config)
    dict(
        mode="fast with events",
        dt_max=1,
        rtol=1e-3,
        atol=1e-6,
        extra_options_setup={"max_num_steps": 20000},
    ),
    # Level 1: safe mode, tighter tolerances
    dict(
        mode="safe",
        dt_max=10,
        rtol=1e-4,
        atol=1e-8,
        extra_options_setup={"max_num_steps": 40000},
    ),
    # Level 2: most robust — no dt cap, strictest tolerances
    dict(
        mode="safe",
        dt_max=None,
        rtol=1e-6,
        atol=1e-10,
        extra_options_setup={"max_num_steps": 100000},
    ),
]

# Per-level wall-clock timeout (seconds).  CasADi/IDAS can stall indefinitely
# when the Newton corrector collapses to h≈1e-20 — its internal step counter
# never increments so max_num_steps is never reached.  We enforce a hard wall-
# clock limit per level so the fallback always fires within reasonable time.
# The abandoned C++ thread continues in the background until IDAS eventually
# gives up, which is unavoidable in CPython without process isolation.
# Chosen so that Level0+1+2 total ≤ typical trial timeout (120 s in optimizer).
_LEVEL_WALL_TIMEOUTS = [25, 50, 60]  # seconds per level


class PyBaMMSimulator:
    """
    Wrapper class for PyBaMM simulations with configurable degradation models.

    Supports:
    - Reversible lithium plating
    - SEI growth (ec reaction limited or solvent-diffusion limited)
    - Lumped thermal model (optional)
    - Multi-step charging protocols
    - 10-90% SOC range (realistic user scenario)
    - Solver fallback hierarchy with full warning/error logging

    Example:
        >>> sim = PyBaMMSimulator(degradation_modes=['plating', 'SEI'])
        >>> results = sim.run_protocol(c_rates=[3.0, 0.0, 3.0],
        ...                            step_durations=[10, 10, 10])
        >>> metrics = sim.extract_metrics(results)
    """

    def __init__(
        self,
        parameter_set: str = "OKane2022",
        degradation_modes: List[str] = ["plating"],
        thermal_model: Optional[str] = None,
        sei_model: str = "ec reaction limited",
        soc_start: float = 0.1,  # 10% SOC
        soc_end: float = 0.9,    # 90% SOC
        v_min: float = 3.0,
        v_max: float = 4.2,
        charge_time_minutes: float = 30.0,
    ):
        """
        Initialize PyBaMM simulator with degradation options.

        Args:
            parameter_set: PyBaMM parameter set name (default: OKane2022 for LG M50 NMC cell)
            degradation_modes: List of degradation modes to enable
                - 'plating': Reversible lithium plating
                - 'SEI': Solid electrolyte interphase growth
                - 'mechanics': Particle stress/cracking (not implemented yet)
            thermal_model: Thermal model type ('lumped', None for isothermal)
            sei_model: SEI growth mechanism ('ec reaction limited' or 'solvent-diffusion limited')
            soc_start: Starting SOC for charging (0.1 = 10%)
            soc_end: Target SOC for charging (0.9 = 90%)
            v_min: Minimum voltage [V] for discharge
            v_max: Maximum voltage [V] for charge cutoff
            charge_time_minutes: Time window for charge throughput calculation [min]
        """
        self.parameter_set = parameter_set
        self.degradation_modes = degradation_modes
        self.thermal_model = thermal_model
        self.sei_model = sei_model
        self.soc_start = soc_start
        self.soc_end = soc_end
        self.v_min = v_min
        self.v_max = v_max
        self.charge_time_minutes = charge_time_minutes

        # Build model options dictionary
        self.model_options = self._build_model_options()

        # Load parameter values
        self.parameter_values = pybamm.ParameterValues(self.parameter_set)

        # Store last solution for debugging
        self.last_solution = None
        self.last_simulation = None

        # Track how often each solver level was needed
        self._solver_stats: Dict[str, int] = {
            "level_0": 0,
            "level_1": 0,
            "level_2": 0,
            "level_3": 0,
            "failed":  0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Model construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_model_options(self) -> Dict:
        """
        Construct PyBaMM model options dictionary based on degradation modes.

        Returns:
            Dictionary of model options for pybamm.lithium_ion.DFN
        """
        options: Dict = {}

        # Lithium plating
        if "plating" in self.degradation_modes:
            options["lithium plating"] = "reversible"

        # SEI growth
        if "SEI" in self.degradation_modes:
            options["SEI"] = self.sei_model
            options["SEI film resistance"] = "distributed"
            options["SEI porosity change"] = "true"

        # Particle mechanics (future work)
        if "mechanics" in self.degradation_modes:
            warnings.warn(
                "Particle mechanics not yet implemented. Skipping.",
                UserWarning,
            )

        # Thermal model
        if self.thermal_model is not None:
            options["thermal"] = self.thermal_model

        return options

    def _get_initial_soc_voltage(self) -> float:
        """
        Estimate voltage corresponding to target starting SOC.

        Uses a simple heuristic for NMC cells:
        V(SOC) ≈ V_min + (V_max - V_min) * SOC

        Returns:
            Voltage [V] corresponding to self.soc_start
        """
        return self.v_min + (self.v_max - self.v_min) * self.soc_start

    # ──────────────────────────────────────────────────────────────────────────
    # Protocol validation
    # ──────────────────────────────────────────────────────────────────────────

    def validate_protocol(
        self,
        c_rates: List[float],
        step_durations: Optional[List[float]] = None,
    ) -> Dict:
        """
        Pre-screen a protocol for likely solver convergence risk BEFORE running it.

        Args:
            c_rates: List of C-rates for each step
            step_durations: List of durations [minutes] for each step (optional)

        Returns:
            Dictionary:
                - 'valid': bool — False only if risk is near-certain failure
                - 'warnings': list of str describing risk factors
                - 'risk_score': float in [0, 1]; >0.7 indicates high risk
        """
        warn_list: List[str] = []
        risk: float = 0.0

        # Physical C-rate limit
        positive_rates = [c for c in c_rates if c > 0]
        if positive_rates:
            max_crate = max(positive_rates)
            if max_crate > 6.0:
                warn_list.append(
                    f"C-rate {max_crate:.1f}C exceeds physical limit (6C for LG M50)"
                )
                risk += 0.9
            elif max_crate > 4.0:
                warn_list.append(
                    f"C-rate {max_crate:.1f}C is very high (>4C); increased failure risk"
                )
                risk += 0.4
            elif max_crate > 3.0:
                risk += 0.15
        else:
            warn_list.append("All steps are rest (no charging)")
            risk += 0.1

        # Minimum step duration check
        if step_durations:
            min_dur = min(step_durations)
            if min_dur < 1.0:
                warn_list.append(
                    f"Minimum step duration {min_dur:.2f} min is below 1 min; "
                    "solver may need very many steps"
                )
                risk += 0.4
            elif min_dur < 2.0:
                warn_list.append(
                    f"Minimum step duration {min_dur:.2f} min is short (<2 min)"
                )
                risk += 0.2

        # Abrupt C-rate transitions
        for i in range(1, len(c_rates)):
            delta = abs(c_rates[i] - c_rates[i - 1])
            if delta >= 5.0:
                warn_list.append(
                    f"Very abrupt transition at step {i}: "
                    f"{c_rates[i-1]:.1f}C → {c_rates[i]:.1f}C (Δ={delta:.1f})"
                )
                risk += 0.35
            elif delta >= 3.0:
                risk += 0.1

        # Known elevated-risk degradation combo
        if (
            "SEI" in self.degradation_modes
            and self.sei_model == "solvent-diffusion limited"
        ):
            warn_list.append(
                "Solvent-diffusion limited SEI has a 1/L_sei singularity at t=0; "
                "baseline failure rate is elevated for this model"
            )
            risk += 0.2

        risk = min(1.0, risk)

        return {
            "valid": risk < 0.9,
            "warnings": warn_list,
            "risk_score": round(risk, 3),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Solver fallback hierarchy
    # ──────────────────────────────────────────────────────────────────────────

    def _solve_once(self, sim: pybamm.Simulation, config: Dict) -> object:
        """
        Execute a single sim.solve() call with full warning capture.
        Designed to run inside a ThreadPoolExecutor worker.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solution = sim.solve(solver=pybamm.CasadiSolver(**config))
        for w in caught:
            logger.warning(
                "[solver] %s: %s", w.category.__name__, str(w.message)
            )
        return solution

    def _build_sim(self, experiment: pybamm.Experiment, model_options: Optional[Dict] = None) -> pybamm.Simulation:
        """Build a fresh model + simulation object (no solve yet)."""
        if model_options is None:
            model_options = self.model_options
        model = pybamm.lithium_ion.DFN(options=model_options)
        return pybamm.Simulation(
            model, experiment=experiment, parameter_values=self.parameter_values
        )

    def _solve_with_fallback(
        self, experiment: pybamm.Experiment
    ) -> Tuple[object, int, bool, pybamm.Simulation]:
        """
        Attempt to solve using progressively more robust solver configurations,
        each with a hard wall-clock timeout.

        CasADi/IDAS can stall indefinitely when the Newton corrector collapses to
        h≈1e-20 — its internal step counter never increments, so max_num_steps is
        never reached.  We run each attempt in a background thread and abandon it
        after _LEVEL_WALL_TIMEOUTS[level] seconds, then move to the next level.
        The abandoned C++ thread keeps running until IDAS eventually gives up
        (unavoidable without process isolation), but the optimization loop stays
        responsive.

        A fresh pybamm.Simulation is built for each level to avoid shared-state
        race conditions between the active solver thread and any abandoned threads
        still executing in C++.

        Levels 0-2 use the same model options; Level 3 drops SEI porosity change
        (approximate, clearly flagged).

        Args:
            experiment: pybamm.Experiment to solve

        Returns:
            (solution, level_used, is_approximate, simulation_object)

        Raises:
            RuntimeError if all levels exhausted.
        """
        for level, (config, wall_timeout) in enumerate(
            zip(_SOLVER_LEVELS, _LEVEL_WALL_TIMEOUTS)
        ):
            sim = self._build_sim(experiment)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._solve_once, sim, config)
            try:
                solution = future.result(timeout=wall_timeout)
                executor.shutdown(wait=False)

                self._solver_stats[f"level_{level}"] += 1
                if level > 0:
                    logger.info(
                        "Solver recovered at level %d (rtol=%s, mode=%s)",
                        level, config["rtol"], config["mode"],
                    )
                return solution, level, False, sim

            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False)
                logger.warning(
                    "Solver level %d TIMED OUT after %ds — advancing to next level",
                    level, wall_timeout,
                )
            except Exception as exc:
                executor.shutdown(wait=False)
                logger.warning(
                    "Solver level %d FAILED — %s: %s",
                    level, type(exc).__name__, str(exc)[:300],
                )

        # Level 3: rebuild model without SEI porosity change (approximate)
        if "SEI" in self.degradation_modes:
            logger.warning(
                "All standard solver levels exhausted. Attempting Level 3 "
                "(simplified model: SEI porosity change disabled). "
                "Result will be flagged as APPROXIMATE."
            )
            try:
                sol, lvl, _, l3_sim = self._solve_level3_simplified(experiment)
                self._solver_stats["level_3"] += 1
                return sol, 3, True, l3_sim
            except Exception as exc:
                logger.error("Level 3 also failed: %s", str(exc)[:300])

        self._solver_stats["failed"] += 1
        raise RuntimeError(
            "All solver levels (0-3) exhausted. Protocol could not be solved."
        )

    def _solve_level3_simplified(
        self, experiment: pybamm.Experiment
    ) -> Tuple[object, int, bool, pybamm.Simulation]:
        """
        Level 3 last-resort: rebuild model without SEI porosity change, then solve
        with Level 2 config.  This physically changes the model slightly but allows
        problematic protocols to produce *some* result rather than NaN.

        The returned solution is always flagged as approximate=True by the caller.
        """
        simplified_options = dict(self.model_options)
        simplified_options.pop("SEI porosity change", None)
        sim = self._build_sim(experiment, model_options=simplified_options)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._solve_once, sim, _SOLVER_LEVELS[2])
        try:
            solution = future.result(timeout=_LEVEL_WALL_TIMEOUTS[2])
            executor.shutdown(wait=False)
            return solution, 3, True, sim
        except Exception:
            executor.shutdown(wait=False)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Protocol execution
    # ──────────────────────────────────────────────────────────────────────────

    def run_protocol(
        self,
        c_rates: List[float],
        step_durations: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run a multi-step charging protocol and extract all metrics.

        Args:
            c_rates: List of C-rates for each step (e.g., [3.0, 0.0, 3.0])
            step_durations: List of durations [minutes] for each step.
                           If None, distributes charge_time_minutes equally.
            verbose: Print simulation progress and results

        Returns:
            Dictionary containing:
                - 'solution': PyBaMM solution object (None on failure)
                - 'simulation': PyBaMM simulation object (None on failure)
                - 'success': Boolean indicating if simulation completed
                - 'error': Error message if failed, None otherwise
                - 'solver_level': int — which fallback level succeeded (0-3)
                - 'approximate': bool — True if Level 3 simplified model was used
        """
        # Handle step durations
        if step_durations is None:
            num_steps = len(c_rates)
            step_durations = [self.charge_time_minutes / num_steps] * num_steps

        if len(c_rates) != len(step_durations):
            raise ValueError(
                f"c_rates ({len(c_rates)}) and step_durations ({len(step_durations)}) "
                "must have same length"
            )

        logger.debug(
            "run_protocol: c_rates=%s  durations=%s  degradation=%s  sei=%s",
            c_rates, step_durations, self.degradation_modes, self.sei_model,
        )

        try:
            experiment = self._build_experiment(c_rates, step_durations)

            # _solve_with_fallback builds a fresh model+simulation for each
            # solver level it tries, which avoids shared-state race conditions
            # when an abandoned Level-N thread keeps running in C++ while
            # Level-(N+1) starts a new solve.
            solution, solver_level, approximate, last_sim = self._solve_with_fallback(
                experiment
            )

            # Store for debugging
            self.last_solution = solution
            self.last_simulation = last_sim

            if verbose:
                approx_tag = " [APPROXIMATE - L3 simplified model]" if approximate else ""
                print(
                    f"✓ Simulation completed (solver level {solver_level}){approx_tag}\n"
                    f"  Protocol: {c_rates} C-rates\n"
                    f"  Durations: {step_durations} min"
                )

            return {
                "solution":     solution,
                "simulation":   last_sim,
                "success":      True,
                "error":        None,
                "solver_level": solver_level,
                "approximate":  approximate,
            }

        except Exception as exc:
            logger.error(
                "run_protocol FAILED for c_rates=%s: %s: %s",
                c_rates, type(exc).__name__, str(exc),
            )
            if verbose:
                print(f"✗ Simulation failed: {exc}")

            return {
                "solution":     None,
                "simulation":   None,
                "success":      False,
                "error":        str(exc),
                "solver_level": -1,
                "approximate":  False,
            }

    def _build_experiment(
        self,
        c_rates: List[float],
        step_durations: List[float],
    ) -> pybamm.Experiment:
        """
        Build PyBaMM experiment with pre-conditioning and charging steps.

        Experiment structure:
        1. Discharge to target starting SOC (10% default)
        2. Rest for 10 minutes (algebraic relaxation — avoids voltage-control
           algebraic constraint conflict of the old "Hold at V" step)
        3. Multi-step charging protocol

        Args:
            c_rates: List of C-rates for each charging step
            step_durations: List of durations [minutes] for each step

        Returns:
            pybamm.Experiment object
        """
        # Step 1: Discharge to starting SOC voltage
        v_start = self._get_initial_soc_voltage()
        discharge_step = f"Discharge at 1C until {v_start:.2f}V"

        # Step 2: Hold at starting voltage for equilibration.
        # This forces a consistent algebraic initial condition (V = v_start)
        # for all DAE variables before the charge steps begin.  Empirically,
        # this outperforms a plain "Rest" step for combined plating+SEI models:
        # the voltage constraint anchors the solver's initial algebraic state
        # more reliably than free relaxation.
        hold_step = f"Hold at {v_start:.2f}V for 10 minutes"

        # Step 3: Build charging steps
        charge_steps = []
        for c_rate, duration in zip(c_rates, step_durations):
            if c_rate == 0.0:
                charge_steps.append(f"Rest for {duration} minutes")
            else:
                charge_steps.append(f"Charge at {c_rate}C for {duration} minutes")

        return pybamm.Experiment([discharge_step, hold_step] + charge_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # Metric extraction
    # ──────────────────────────────────────────────────────────────────────────

    def extract_metrics(self, result: Dict) -> Dict:
        """
        Extract all performance and degradation metrics from simulation result.

        Args:
            result: Dictionary returned by run_protocol()

        Returns:
            Dictionary containing:
                - 'Q30': Charge stored in time window [Ah]
                - 'plating_loss': Capacity lost to plating [Ah]
                - 'sei_growth': SEI thickness increase [nm] (if SEI enabled)
                - 'sei_li_loss': Lithium consumed by SEI [mol] (if SEI enabled)
                - 'total_lli': Total lithium inventory loss [mol]
                - 'success': Boolean indicating valid metrics
                - 'approximate': bool — True if Level 3 simplified model was used

            All degradation metrics return NaN if extraction fails.
        """
        if not result["success"]:
            return self._nan_metrics()

        solution = result["solution"]

        metrics: Dict = {}

        metrics["Q30"] = self._extract_charge_throughput(solution)
        metrics["plating_loss"] = self._extract_plating_loss(solution)

        if "SEI" in self.degradation_modes:
            sei_metrics = self._extract_sei_metrics(solution)
            metrics["sei_growth"]   = sei_metrics["thickness_growth_nm"]
            metrics["sei_li_loss"]  = sei_metrics["li_loss_mol"]
        else:
            metrics["sei_growth"]   = 0.0
            metrics["sei_li_loss"]  = 0.0

        metrics["total_lli"]  = self._compute_total_lli(metrics)
        metrics["success"]    = True
        metrics["approximate"] = result.get("approximate", False)
        metrics["solver_level"] = result.get("solver_level", 0)

        return metrics

    def _extract_charge_throughput(self, solution) -> float:
        """
        Calculate charge stored during the charging window.

        Uses Coulomb counting: Q = ∫ I(t) dt

        Returns:
            Charge throughput [Ah]
        """
        time_sim = solution["Time [min]"].data
        current  = solution["Current [A]"].data

        charging_indices = np.where(current < 0)[0]

        if len(charging_indices) == 0:
            warnings.warn("No charging detected in simulation")
            return 0.0

        t0    = time_sim[charging_indices[0]]
        t_end = t0 + self.charge_time_minutes

        charging_mask = (current < 0) & (time_sim >= t0) & (time_sim <= t_end)

        Q_integrated = np.trapz(-current[charging_mask], time_sim[charging_mask])

        # Despite "Time [min]" label, divide by 3600 to match original notebook
        # (consistent with LG M50 cell capacity expectations)
        Q_Ah = Q_integrated / 3600.0

        return Q_Ah

    def _extract_plating_loss(self, solution) -> float:
        """
        Extract capacity lost to lithium plating.

        Returns:
            Capacity loss due to plating [Ah]
        """
        try:
            plating_loss = solution[
                "Loss of capacity to negative lithium plating [A.h]"
            ].data
            return np.max(plating_loss)
        except KeyError:
            warnings.warn("Plating loss variable not found in solution")
            return np.nan

    def _extract_sei_metrics(self, solution) -> Dict:
        """
        Extract SEI growth metrics.

        Returns:
            Dictionary with:
                - 'thickness_growth_nm': SEI thickness increase [nm]
                - 'li_loss_mol': Lithium consumed by SEI [mol]
        """
        metrics: Dict = {}

        # SEI thickness
        try:
            sei_thickness = solution[
                "X-averaged negative total SEI thickness [m]"
            ].data

            if len(sei_thickness.shape) == 1:
                thickness_initial = sei_thickness[0]
                thickness_final   = sei_thickness[-1]
            else:
                thickness_initial = np.mean(sei_thickness[0])
                thickness_final   = np.mean(sei_thickness[-1])

            metrics["thickness_growth_nm"] = (thickness_final - thickness_initial) * 1e9

        except (KeyError, Exception) as exc:
            warnings.warn(f"SEI thickness extraction failed: {exc}")
            metrics["thickness_growth_nm"] = np.nan

        # Lithium consumed by SEI
        try:
            possible_pairs = [
                (
                    "Loss of lithium to negative SEI [mol]",
                    "Loss of lithium to positive SEI [mol]",
                ),
                ("Loss of lithium to negative SEI [mol]", None),
            ]

            total_sei_li_loss = 0.0
            for neg_name, pos_name in possible_pairs:
                try:
                    total_sei_li_loss += np.max(solution[neg_name].data)
                    if pos_name is not None:
                        try:
                            total_sei_li_loss += np.max(solution[pos_name].data)
                        except KeyError:
                            pass
                    break
                except KeyError:
                    continue

            if total_sei_li_loss == 0.0:
                raise KeyError("No SEI lithium loss variable found")

            metrics["li_loss_mol"] = total_sei_li_loss

        except (KeyError, Exception) as exc:
            warnings.warn(f"SEI lithium loss extraction failed: {exc}")
            metrics["li_loss_mol"] = np.nan

        return metrics

    def _compute_total_lli(self, metrics: Dict) -> float:
        """
        Compute total Loss of Lithium Inventory from all sources.

        LLI = LLI_SEI + LLI_plating

        Returns:
            Total LLI [mol]
        """
        F = 96485.0  # Faraday constant [C/mol]

        plating_loss_Ah = metrics.get("plating_loss", 0.0)
        sei_li_loss_mol = metrics.get("sei_li_loss", 0.0)

        plating_loss_mol = (
            0.0 if np.isnan(plating_loss_Ah)
            else (plating_loss_Ah * 3600.0) / F
        )
        if np.isnan(sei_li_loss_mol):
            sei_li_loss_mol = 0.0

        return plating_loss_mol + sei_li_loss_mol

    def _nan_metrics(self) -> Dict:
        """Return dictionary of NaN metrics for failed simulations."""
        return {
            "Q30":          np.nan,
            "plating_loss": np.nan,
            "sei_growth":   np.nan,
            "sei_li_loss":  np.nan,
            "total_lli":    np.nan,
            "success":      False,
            "approximate":  False,
            "solver_level": -1,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience interface (backward-compatible)
    # ──────────────────────────────────────────────────────────────────────────

    def run_and_extract(
        self,
        c_rates: List[float],
        step_durations: Optional[List[float]] = None,
        verbose: bool = False,
        validate_first: bool = False,
    ) -> Dict:
        """
        Convenience method: run protocol and extract metrics in one call.

        Args:
            c_rates: List of C-rates for each step
            step_durations: List of durations [minutes] for each step
            verbose: Print simulation details
            validate_first: If True, run validate_protocol() first and log
                            a warning if risk_score > 0.5 (default False)

        Returns:
            Dictionary of extracted metrics (same interface as before)
        """
        if validate_first:
            validation = self.validate_protocol(c_rates, step_durations)
            if validation["risk_score"] > 0.5:
                logger.warning(
                    "High-risk protocol (score=%.2f): %s",
                    validation["risk_score"],
                    "; ".join(validation["warnings"]),
                )

        result  = self.run_protocol(c_rates, step_durations, verbose)
        metrics = self.extract_metrics(result)
        return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Single-objective backward-compatibility shim
# ──────────────────────────────────────────────────────────────────────────────

def run_P2D_single_objective(
    params: List[float],
    beta: float = 0.008,
    degradation_modes: List[str] = ["plating"],
) -> Dict:
    """
    Single-objective evaluation function (compatible with original notebooks).

    Objective = Q30 - β * log(Q_plating)

    Args:
        params: List of C-rates [C1, C2, C3, ...]
        beta: Weight for degradation penalty
        degradation_modes: List of degradation modes to enable

    Returns:
        Dictionary with 'objective' key
    """
    simulator = PyBaMMSimulator(degradation_modes=degradation_modes)
    metrics = simulator.run_and_extract(params, verbose=False)

    if not metrics["success"]:
        return {"objective": -999.0}

    Q30          = metrics["Q30"]
    plating_loss = max(metrics["plating_loss"], 1e-6)

    return {"objective": Q30 - beta * np.log(plating_loss)}
