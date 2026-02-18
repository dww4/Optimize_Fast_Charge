"""
Baseline Fast-Charging Protocol Definitions

This module provides standard baseline charging protocols for comparison
against Bayesian optimization results.

Baselines included:
1. CCCV (Constant Current - Constant Voltage) - Industry standard
2. Linear Taper - Tesla-style decreasing current
3. BO Optimal - Best from single-objective optimization

Author: DJ
Date: 2026-02-17
"""

import numpy as np
from typing import List, Tuple, Dict


def get_cccv_protocol(
    charge_rate_C: float = 2.0,
    time_minutes: float = 30.0,
    v_max: float = 4.2
) -> Dict[str, any]:
    """
    Generate standard CCCV (Constant Current - Constant Voltage) protocol.

    This is the industry standard fast-charging approach:
    - Phase 1: Charge at constant current until v_max
    - Phase 2: Hold at v_max until current drops to cutoff (or time expires)

    For comparison purposes, we approximate this as a multi-step protocol
    since the BO-optimized protocols use discrete steps.

    Args:
        charge_rate_C: Constant current rate [C] (2.0 = 2C fast charge)
        time_minutes: Total charging time [min]
        v_max: Maximum voltage cutoff [V]

    Returns:
        Dictionary with:
            - 'c_rates': List of C-rates for each step
            - 'step_durations': List of durations [min]
            - 'name': Protocol name
            - 'description': Brief description

    Example:
        >>> protocol = get_cccv_protocol(charge_rate_C=2.0)
        >>> # Use with simulator:
        >>> sim.run_protocol(c_rates=protocol['c_rates'],
        ...                  step_durations=protocol['step_durations'])
    """
    # For true CCCV, we'd have variable time until voltage limit
    # For comparison, we approximate as constant current for full time
    # (PyBaMM will stop if voltage limit reached)

    # Single constant-current step
    c_rates = [charge_rate_C]
    step_durations = [time_minutes]

    return {
        'c_rates': c_rates,
        'step_durations': step_durations,
        'name': f'CCCV_{charge_rate_C}C',
        'description': f'Constant current at {charge_rate_C}C until {v_max}V, then constant voltage',
        'type': 'CCCV',
        'parameters': {
            'charge_rate_C': charge_rate_C,
            'v_max': v_max
        }
    }


def get_linear_taper_protocol(
    i_start_C: float = 3.0,
    i_end_C: float = 1.0,
    num_steps: int = 6,
    time_minutes: float = 30.0
) -> Dict[str, any]:
    """
    Generate linear current taper protocol (Tesla/industry heuristic).

    Current decreases linearly over time:
    I(t) = I_start - (I_start - I_end) * (t / t_total)

    Rationale: High current at low SOC (fast Li+ intercalation),
               lower current at high SOC (avoid plating/overpotential)

    Args:
        i_start_C: Starting C-rate (default: 3.0C for fast initial charge)
        i_end_C: Ending C-rate (default: 1.0C for gentle finish)
        num_steps: Number of discrete steps (higher = smoother taper)
        time_minutes: Total charging time [min]

    Returns:
        Dictionary with protocol definition

    Example:
        >>> protocol = get_linear_taper_protocol(i_start_C=3.0, i_end_C=1.0, num_steps=6)
        >>> # Results in: [3.0, 2.6, 2.2, 1.8, 1.4, 1.0]C over 30 minutes
    """
    # Create linearly decreasing C-rates
    c_rates = np.linspace(i_start_C, i_end_C, num_steps).tolist()

    # Equal duration per step
    step_duration = time_minutes / num_steps
    step_durations = [step_duration] * num_steps

    return {
        'c_rates': c_rates,
        'step_durations': step_durations,
        'name': f'LinearTaper_{i_start_C}C-{i_end_C}C',
        'description': f'Linear current taper from {i_start_C}C to {i_end_C}C over {num_steps} steps',
        'type': 'LinearTaper',
        'parameters': {
            'i_start_C': i_start_C,
            'i_end_C': i_end_C,
            'num_steps': num_steps
        }
    }


def get_bo_optimal_3step_aggressive() -> Dict[str, any]:
    """
    Get the best 3-step protocol from single-objective BO (aggressive, β=0.008).

    This is the "charge-rest-charge" protocol that emerged from your
    original Bayesian optimization:
    - Step 1: 3C for 10 minutes (fast initial charge)
    - Step 2: 0C for 10 minutes (rest/relaxation)
    - Step 3: 3C for 10 minutes (fast final charge)

    Performance (from original notebook):
    - Q30: 0.125 Ah
    - Plating loss: 1.267 Ah
    - Objective: 0.123

    Returns:
        Dictionary with protocol definition
    """
    return {
        'c_rates': [3.0, 0.0, 3.0],
        'step_durations': [10.0, 10.0, 10.0],
        'name': 'BO_3step_aggressive',
        'description': 'Best 3-step protocol from single-objective BO (β=0.008): Charge-Rest-Charge',
        'type': 'BO_Optimal',
        'parameters': {
            'beta': 0.008,
            'strategy': 'charge-rest-charge'
        }
    }


def get_bo_optimal_3step_conservative() -> Dict[str, any]:
    """
    Get the best 3-step protocol from single-objective BO (conservative, β=0.015).

    This protocol prioritizes lower degradation over high capacity:
    - Step 1: 0.91C for 10 minutes
    - Step 2: 0C for 10 minutes (rest)
    - Step 3: 0.54C for 10 minutes

    Performance (from original notebook):
    - Q30: 0.030 Ah
    - Plating loss: very low
    - Objective: 0.142

    Returns:
        Dictionary with protocol definition
    """
    return {
        'c_rates': [0.91, 0.0, 0.54],
        'step_durations': [10.0, 10.0, 10.0],
        'name': 'BO_3step_conservative',
        'description': 'Best 3-step protocol from single-objective BO (β=0.015): Low degradation focus',
        'type': 'BO_Optimal',
        'parameters': {
            'beta': 0.015,
            'strategy': 'low-degradation'
        }
    }


def get_bo_optimal_5step_long_rest() -> Dict[str, any]:
    """
    Get the best 5-step protocol from single-objective BO (long rest strategy).

    This "long rest" protocol had the best objective in the 5-step optimization:
    - Steps 1: 3C for 6 minutes
    - Steps 2-4: 0C for 18 minutes (long rest)
    - Step 5: 2.52C for 6 minutes

    Performance (from original notebook):
    - Q30: 0.115 Ah
    - Plating loss: 0.048 Ah
    - Objective: 0.139

    Returns:
        Dictionary with protocol definition
    """
    return {
        'c_rates': [3.0, 0.0, 0.0, 0.0, 2.52],
        'step_durations': [6.0, 6.0, 6.0, 6.0, 6.0],
        'name': 'BO_5step_long_rest',
        'description': 'Best 5-step protocol from single-objective BO: Long rest strategy',
        'type': 'BO_Optimal',
        'parameters': {
            'beta': 0.008,
            'strategy': 'long-rest'
        }
    }


def get_all_baselines(time_minutes: float = 30.0) -> List[Dict[str, any]]:
    """
    Get all baseline protocols for comparison.

    Args:
        time_minutes: Total charging time [min] (default: 30)

    Returns:
        List of protocol dictionaries

    Example:
        >>> baselines = get_all_baselines()
        >>> for protocol in baselines:
        ...     print(f"{protocol['name']}: {protocol['description']}")
    """
    baselines = [
        # Industry standard
        get_cccv_protocol(charge_rate_C=2.0, time_minutes=time_minutes),

        # Heuristic tapers
        get_linear_taper_protocol(i_start_C=3.0, i_end_C=1.0, num_steps=6, time_minutes=time_minutes),
        get_linear_taper_protocol(i_start_C=3.0, i_end_C=0.5, num_steps=6, time_minutes=time_minutes),

        # BO-optimized from previous work
        get_bo_optimal_3step_aggressive(),
        get_bo_optimal_3step_conservative(),
    ]

    return baselines


def protocol_to_string(protocol: Dict[str, any]) -> str:
    """
    Convert protocol to human-readable string.

    Args:
        protocol: Protocol dictionary

    Returns:
        Formatted string representation

    Example:
        >>> protocol = get_cccv_protocol(2.0)
        >>> print(protocol_to_string(protocol))
        CCCV_2C: [2.0]C for [30.0]min
    """
    c_rates = protocol['c_rates']
    durations = protocol['step_durations']

    # Format C-rates and durations
    c_str = '[' + ', '.join(f'{c:.2f}' for c in c_rates) + ']'
    d_str = '[' + ', '.join(f'{d:.1f}' for d in durations) + ']'

    return f"{protocol['name']}: {c_str}C for {d_str}min"


def validate_protocol(protocol: Dict[str, any]) -> Tuple[bool, str]:
    """
    Validate that a protocol dictionary has the correct structure.

    Args:
        protocol: Protocol dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> protocol = get_cccv_protocol()
        >>> is_valid, msg = validate_protocol(protocol)
        >>> assert is_valid
    """
    required_keys = ['c_rates', 'step_durations', 'name', 'description', 'type']

    # Check required keys
    for key in required_keys:
        if key not in protocol:
            return False, f"Missing required key: {key}"

    # Check c_rates and step_durations are lists
    if not isinstance(protocol['c_rates'], list):
        return False, "c_rates must be a list"
    if not isinstance(protocol['step_durations'], list):
        return False, "step_durations must be a list"

    # Check equal length
    if len(protocol['c_rates']) != len(protocol['step_durations']):
        return False, f"c_rates ({len(protocol['c_rates'])}) and step_durations ({len(protocol['step_durations'])}) must have same length"

    # Check for empty lists
    if len(protocol['c_rates']) == 0:
        return False, "c_rates cannot be empty"

    # Check for negative values
    if any(c < 0 for c in protocol['c_rates']):
        return False, "c_rates cannot be negative"
    if any(d <= 0 for d in protocol['step_durations']):
        return False, "step_durations must be positive"

    return True, "Valid protocol"


# Define standard baseline set for quick access
STANDARD_BASELINES = {
    'cccv_2c': get_cccv_protocol(2.0),
    'taper_3to1': get_linear_taper_protocol(3.0, 1.0, 6),
    'taper_3to0p5': get_linear_taper_protocol(3.0, 0.5, 6),
    'bo_aggressive': get_bo_optimal_3step_aggressive(),
    'bo_conservative': get_bo_optimal_3step_conservative(),
}
