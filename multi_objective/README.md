# Multi-Objective Fast-Charging Optimization

## Overview

This directory contains the implementation of multi-objective Bayesian optimization for fast-charging protocols with coupled degradation modes. The optimization seeks to find Pareto-optimal trade-offs between:

1. **Charge throughput** (maximize)
2. **Lithium plating** (minimize)
3. **SEI growth** (minimize)
4. **Loss of Lithium Inventory** (derived metric, minimize)

## Key Features

- **Multi-objective BO**: Uses Ax Platform with qNEHVI acquisition function
- **Coupled degradation modes**: PyBaMM DFN model with reversible plating + SEI growth
- **Pareto front visualization**: Interactive 3D plots with baseline comparisons
- **Realistic SOC range**: 10-90% charging (typical user scenario)

## Project Configuration

- **Cell**: LG M50 (NMC-graphite, OKane2022 parameter set)
- **Model**: DFN (P2D) with reversible plating + ec reaction limited SEI
- **Thermal**: Isothermal (no thermal model)
- **Time horizon**: 30 minutes charging
- **SOC range**: 10-90%
- **Optimization budget**: 150 evaluations (50 iterations × 3 trials)

## Notebooks (Execution Order)

1. **1_baseline_protocols.ipynb** - Evaluate standard charging protocols
2. **2_degradation_models.ipynb** - Test degradation metric extraction
3. **3_multi_obj_optimization.ipynb** - Run multi-objective Bayesian optimization
4. **4_pareto_visualization.ipynb** - Analyze results and create Pareto front plots

## Utilities

- `pybamm_simulator.py` - Wrapper class for PyBaMM simulations
- `degradation_metrics.py` - Extraction functions for all degradation modes
- `baseline_protocols.py` - Standard protocol definitions (CCCV, taper, etc.)
- `visualization.py` - Plotting utilities for Pareto fronts

## Results

- `results/pareto_fronts/` - Saved Pareto front visualizations
- `results/optimal_protocols/` - Best charging protocols (CSV/JSON)
- `results/trials_database/` - Ax experiment data for reproducibility

## Installation

```bash
cd /Users/dejua/Optimize_Fast_Charge
pip install -r requirements.txt
```

## Quick Start

```python
# Run notebooks in order:
# 1. baseline_protocols.ipynb
# 2. degradation_models.ipynb
# 3. multi_obj_optimization.ipynb
# 4. pareto_visualization.ipynb
```

## Author

DJ (dejuante1503@gmail.com)
Battery Modeling Engineer
