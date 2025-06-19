
# Bayesian Optimization of Fast Charging

############################### Description ###############################

Uses a DFN (P2D) cell model with lithium plating to optimize a multi-step CC-CV fast charging protocol via Bayesian Optimization (Ax Platform). The optimization seeks to maximize charge stored in 30 minutes while minimizing aging due to lithium plating.  

Outputs:
- Best performing multi-step protocol (C1,C2,C3,...)
- Optimization history and performance plots

############################### Versions ###############################

- 3step: Optimizes a 3-step CC-CV fast charge protocol (no thermals/temperature variation)
- 3step_thermals: Optimizes a 3-step CC-CV fast charge protocol with a lumped thermal model (time-varying temperature)

############################### Dependencies ###############################

0. PyBaMM Installation (pip install pybamm)  

1. Ax Platform Installation (pip install ax-platform)  

2. BO-P2D-FastCharge-3step.ipynb  

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

Ask DJ (dejuante1503@gmail.com)  
