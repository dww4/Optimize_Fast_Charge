
# Bayesian Optimization of Fast Charging

############################### Description ###############################

Uses a physics-based DFN/P2D cell model with degradation mechanisms to optimize multi-step CC-CV fast charging protocols with Bayesian Optimization (via the Ax Platform). The optimization seeks to maximize charge stored in 30 minutes while minimizing aging due to degradation.

**Single-Objective Versions (3step, 3step_thermals, 5step):**
- Single objective function: Q₃₀ - β*log(Q_plating)
	- Q₃₀ = Charge Throughput in 30 minutes
	- Q_plating = Capacity Lost to Plating
- Outputs: Best performing multi-step protocol (C1,C2,C3,...)

**Multi-Objective Version (multi_objective):**
- Multiple competing objectives: maximize charge throughput (Q₃₀), minimize lithium plating (LLI), minimize SEI growth (impedance growth)
- Coupled degradation modes: Reversible lithium plating + SEI formation
- Outputs: Pareto-optimal protocols, 3D Pareto front visualizations, baseline comparisons
- See `multi_objective/README.md` for details

############################### Versions ###############################

- **3step**: Optimizes a 3-step CC-CV fast charge protocol (no thermals/temperature variation)
- **3step_thermals**: Optimizes a 3-step CC-CV fast charge protocol with a lumped thermal model (time-varying temperature)
- **5step**: Optimizes a 5-step CC-CV fast charge protocol (no thermals/temperature variation)
- **multi_objective**: Multi-objective Bayesian optimization with coupled degradation modes (plating + SEI) and Pareto front visualization (no thermals/temperature variation)

############################### Dependencies ###############################

0. PyBaMM Installation (pip install pybamm)  

1. Ax Platform Installation (pip install ax-platform)  

2. Version Directory (ex: 3step, multi_objective, etc.)

############################### Executing Program ###############################

1. Run Simulation:
	a. Open: `BO-P2D-FastCharge-3step.ipynb` in JupyterLab or Jupyter Notebook

	b. Execute all cells:
		i. Defines model and objective
		ii. Runs Bayesian Optimization
		iii. Outputs:
			- Optimization results

############################### Help ###############################

*If you encounter any issues with Ax or PyBaMM installs, check their respective documentation.*

*Ignore flags that are outputted - the model is working*

*Ask DJ (dejuante1503@gmail.com)*
