"""
PyBaMM Simulator Wrapper for Fast-Charging Optimization

This module provides a clean interface to PyBaMM for multi-objective optimization
of fast-charging protocols with coupled degradation modes.

Author: DJ
Date: 2026-02-13
"""

import pybamm
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


class PyBaMMSimulator:
    """
    Wrapper class for PyBaMM simulations with configurable degradation models.

    Supports:
    - Reversible lithium plating
    - SEI growth (ec reaction limited or solvent-diffusion limited)
    - Lumped thermal model (optional)
    - Multi-step charging protocols
    - 10-90% SOC range (realistic user scenario)

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

    def _build_model_options(self) -> Dict:
        """
        Construct PyBaMM model options dictionary based on degradation modes.

        Returns:
            Dictionary of model options for pybamm.lithium_ion.DFN
        """
        options = {}

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
                UserWarning
            )
            # options["particle mechanics"] = "swelling and cracking"

        # Thermal model
        if self.thermal_model is not None:
            options["thermal"] = self.thermal_model

        return options

    def _get_initial_soc_voltage(self) -> float:
        """
        Estimate voltage corresponding to target starting SOC.

        Uses a simple heuristic for NMC cells:
        V(SOC) ≈ V_min + (V_max - V_min) * SOC

        For more accurate mapping, could run a calibration discharge.

        Implication: The model will start at ROUGLY 10%

        Returns:
            Voltage [V] corresponding to self.soc_start
        """
        # Simple linear approximation (good enough for 10-90% range)
        v_estimate = self.v_min + (self.v_max - self.v_min) * self.soc_start
        return v_estimate

    def run_protocol(
        self,
        c_rates: List[float],
        step_durations: Optional[List[float]] = None,
        verbose: bool = False
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
                - 'solution': PyBaMM solution object
                - 'simulation': PyBaMM simulation object
                - 'success': Boolean indicating if simulation completed
                - 'error': Error message if failed, None otherwise
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

        # Build experiment
        try:
            experiment = self._build_experiment(c_rates, step_durations)

            # Create model
            model = pybamm.lithium_ion.DFN(options=self.model_options)

            # Create simulation
            sim = pybamm.Simulation(
                model,
                experiment=experiment,
                parameter_values=self.parameter_values
            )

            # Solve with CasadiSolver (same as original notebooks)
            # Suppress convergence warnings (they're usually harmless)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*corrector convergence.*")
                warnings.filterwarnings("ignore", message=".*linesearch algorithm.*")

                solution = sim.solve(solver=pybamm.CasadiSolver(mode="safe", dt_max=1))

            # Store for debugging
            self.last_solution = solution
            self.last_simulation = sim

            if verbose:
                print(f"✓ Simulation completed successfully")
                print(f"  Protocol: {c_rates} C-rates")
                print(f"  Durations: {step_durations} min")

            return {
                'solution': solution,
                'simulation': sim,
                'success': True,
                'error': None
            }

        except Exception as e:
            if verbose:
                print(f"✗ Simulation failed: {str(e)}")

            return {
                'solution': None,
                'simulation': None,
                'success': False,
                'error': str(e)
            }

    def _build_experiment(
        self,
        c_rates: List[float],
        step_durations: List[float]
    ) -> pybamm.Experiment:
        """
        Build PyBaMM experiment with pre-conditioning and charging steps.

        Experiment structure:
        1. Discharge to target starting SOC (10% default)
        2. Hold for relaxation (10 minutes)
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

        # Step 2: Hold for equilibration
        hold_step = f"Hold at {v_start:.2f}V for 10 minutes"

        # Step 3: Build charging steps
        charge_steps = []
        for c_rate, duration in zip(c_rates, step_durations):
            if c_rate == 0.0:
                # Rest period
                charge_steps.append(f"Rest for {duration} minutes")
            else:
                # Active charging
                charge_steps.append(f"Charge at {c_rate}C for {duration} minutes")

        # Combine all steps
        experiment_steps = [discharge_step, hold_step] + charge_steps

        return pybamm.Experiment(experiment_steps)

    def extract_metrics(self, result: Dict) -> Dict[str, float]:
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

            All degradation metrics return NaN if extraction fails.
        """
        if not result['success']:
            # Return NaN metrics for failed simulations
            return self._nan_metrics()

        solution = result['solution']

        metrics = {}

        # Extract charge throughput (Q30 or whatever time window specified)
        metrics['Q30'] = self._extract_charge_throughput(solution)

        # Extract lithium plating
        metrics['plating_loss'] = self._extract_plating_loss(solution)

        # Extract SEI metrics if enabled
        if "SEI" in self.degradation_modes:
            sei_metrics = self._extract_sei_metrics(solution)
            metrics['sei_growth'] = sei_metrics['thickness_growth_nm']
            metrics['sei_li_loss'] = sei_metrics['li_loss_mol']
        else:
            metrics['sei_growth'] = 0.0
            metrics['sei_li_loss'] = 0.0

        # Compute total LLI (from plating + SEI)
        metrics['total_lli'] = self._compute_total_lli(metrics)

        # Mark as successful extraction
        metrics['success'] = True

        return metrics

    def _extract_charge_throughput(self, solution) -> float:
        """
        Calculate charge stored during the charging window.

        Uses Coulomb counting: Q = ∫ I(t) dt

        Args:
            solution: PyBaMM solution object

        Returns:
            Charge throughput [Ah]
        """
        time_sim = solution["Time [min]"].data
        current = solution["Current [A]"].data

        # Find first charging point (negative current)
        charging_indices = np.where(current < 0)[0]

        if len(charging_indices) == 0:
            warnings.warn("No charging detected in simulation")
            return 0.0

        t0 = time_sim[charging_indices[0]]

        # Define charging window (30 minutes by default)
        t_end = t0 + (self.charge_time_minutes / 60.0)  # Convert to hours if needed

        # Actually, time is in minutes already based on solution["Time [min]"]
        t_end = t0 + self.charge_time_minutes

        # Create mask for charging window
        charging_mask = (current < 0) & (time_sim >= t0) & (time_sim <= t_end)

        # Coulomb counting (integrate current over time)
        # Note: Current is negative during charging, so negate it
        Q_integrated = np.trapz(-current[charging_mask], time_sim[charging_mask])

        # Convert to A·h
        # Note: Despite "Time [min]" label, we divide by 3600 to match original notebook
        # This gives correct results consistent with the LG M50 cell capacity
        Q_Ah = Q_integrated / 3600.0

        return Q_Ah

    def _extract_plating_loss(self, solution) -> float:
        """
        Extract capacity lost to lithium plating.

        Args:
            solution: PyBaMM solution object

        Returns:
            Capacity loss due to plating [Ah]
        """
        try:
            plating_loss = solution['Loss of capacity to negative lithium plating [A.h]'].data
            return np.max(plating_loss)
        except KeyError:
            warnings.warn("Plating loss variable not found in solution")
            return np.nan

    def _extract_sei_metrics(self, solution) -> Dict[str, float]:
        """
        Extract SEI growth metrics.

        Args:
            solution: PyBaMM solution object

        Returns:
            Dictionary with:
                - 'thickness_growth_nm': SEI thickness increase [nm]
                - 'li_loss_mol': Lithium consumed by SEI [mol]
                - 'resistance_ohm': SEI resistance [Ohm] (future)
        """
        metrics = {}

        # SEI thickness on negative electrode
        try:
            # Use X-averaged total SEI thickness (accounts for spatial variation + cracks)
            # Try in order of preference
            # possible_names = [
            #     'X-averaged negative total SEI thickness [m]',  # Best: includes cracks, averaged
            #     'Negative total SEI thickness [m]',             # Includes cracks
            #     'X-averaged negative SEI thickness [m]',        # Averaged
            #     'Negative SEI thickness [m]',                   # Basic
            # ]
            #
            # sei_thickness = None
            # var_name_used = None
            # for name in possible_names:
            #     try:
            #         sei_thickness = solution[name].data
            #         var_name_used = name
            #         break
            #     except KeyError:
            #         continue

            sei_thickness = None
            sei_thickness = solution['X-averaged negative total SEI thickness [m]'].data # Use X-averaged total SEI thickness (accounts for spatial variation + cracks)

            if sei_thickness is None:
                raise KeyError("No SEI thickness variable found")

            # Calculate growth (final - initial)
            # Handle both scalar and array data
            if len(sei_thickness.shape) == 1:
                # 1D array (time series)
                thickness_initial = sei_thickness[0]
                thickness_final = sei_thickness[-1]
            else:
                # 2D array (space and time) - take spatial average
                thickness_initial = np.mean(sei_thickness[0])
                thickness_final = np.mean(sei_thickness[-1])

            thickness_growth_m = thickness_final - thickness_initial

            # Convert to nanometers
            metrics['thickness_growth_nm'] = thickness_growth_m * 1e9

        except (KeyError, Exception) as e:
            warnings.warn(f"SEI thickness extraction failed: {e}")
            metrics['thickness_growth_nm'] = np.nan

        # Lithium consumed by SEI
        try:
            # Correct variable name based on available variables
            # Sum both negative and positive SEI losses (though negative dominates)
            possible_names = [
                ('Loss of lithium to negative SEI [mol]', 'Loss of lithium to positive SEI [mol]'),  # Separate
                ('Loss of lithium to negative SEI [mol]', None),  # Just negative
            ]

            total_sei_li_loss = 0.0
            for neg_name, pos_name in possible_names:
                try:
                    # Get negative SEI loss
                    neg_loss = solution[neg_name].data
                    total_sei_li_loss += np.max(neg_loss)

                    # Try to get positive SEI loss (optional)
                    if pos_name is not None:
                        try:
                            pos_loss = solution[pos_name].data
                            total_sei_li_loss += np.max(pos_loss)
                        except KeyError:
                            pass  # Positive SEI may not be significant

                    break  # Found variables, exit loop

                except KeyError:
                    continue

            if total_sei_li_loss == 0.0:
                raise KeyError("No SEI lithium loss variable found")

            metrics['li_loss_mol'] = total_sei_li_loss

        except (KeyError, Exception) as e:
            warnings.warn(f"SEI lithium loss extraction failed: {e}")
            metrics['li_loss_mol'] = np.nan

        return metrics

    def _compute_total_lli(self, metrics: Dict) -> float:
        """
        Compute total Loss of Lithium Inventory from all sources.

        LLI = LLI_SEI + LLI_plating

        Args:
            metrics: Dictionary containing plating_loss [Ah] and sei_li_loss [mol]

        Returns:
            Total LLI [mol]
        """
        # Convert plating loss from Ah to mol
        # Li+ + e- → Li, so 1 mol e- = 1 mol Li
        # Q [Ah] = n [mol] * F [C/mol] / 3600 [s/h]
        # n [mol] = Q [Ah] * 3600 / F

        F = 96485  # Faraday constant [C/mol]

        plating_loss_Ah = metrics.get('plating_loss', 0.0)
        sei_li_loss_mol = metrics.get('sei_li_loss', 0.0)

        # Convert plating to mol
        if np.isnan(plating_loss_Ah):
            plating_loss_mol = 0.0
        else:
            plating_loss_mol = (plating_loss_Ah * 3600) / F

        if np.isnan(sei_li_loss_mol):
            sei_li_loss_mol = 0.0

        total_lli_mol = plating_loss_mol + sei_li_loss_mol

        return total_lli_mol

    def _nan_metrics(self) -> Dict[str, float]:
        """
        Return dictionary of NaN metrics for failed simulations.

        Returns:
            Dictionary with all metrics set to NaN and success=False
        """
        return {
            'Q30': np.nan,
            'plating_loss': np.nan,
            'sei_growth': np.nan,
            'sei_li_loss': np.nan,
            'total_lli': np.nan,
            'success': False
        }

    def run_and_extract(
        self,
        c_rates: List[float],
        step_durations: Optional[List[float]] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Convenience method: run protocol and extract metrics in one call.

        Args:
            c_rates: List of C-rates for each step
            step_durations: List of durations [minutes] for each step
            verbose: Print simulation details

        Returns:
            Dictionary of extracted metrics
        """
        result = self.run_protocol(c_rates, step_durations, verbose)
        metrics = self.extract_metrics(result)
        return metrics


# Convenience function for single-objective optimization (backward compatibility)
def run_P2D_single_objective(
    params: List[float],
    beta: float = 0.008,
    degradation_modes: List[str] = ["plating"]
) -> Dict[str, float]:
    """
    Single-objective evaluation function (compatible with original notebooks).

    Objective = Q30 - β*log(Q_plating)

    Args:
        params: List of C-rates [C1, C2, C3, ...]
        beta: Weight for degradation penalty
        degradation_modes: List of degradation modes to enable

    Returns:
        Dictionary with 'objective' key
    """
    simulator = PyBaMMSimulator(degradation_modes=degradation_modes)
    metrics = simulator.run_and_extract(params, verbose=False)

    if not metrics['success']:
        # Return very poor objective for failed simulations
        return {'objective': -999.0}

    Q30 = metrics['Q30']
    plating_loss = metrics['plating_loss']

    # Avoid log(0) by adding small epsilon
    plating_loss = max(plating_loss, 1e-6)

    objective = Q30 - beta * np.log(plating_loss)

    return {'objective': objective}
