# Multi-Information Source Bayesian Optimization over Combinatorial Structures

This repository contains the code for the experiments reported in the following paper:

Sabbatella, A., Ponti, A., Candelieri, A., & Archetti, F. **Bayesian Optimization using simulation based multiple information sources over combinatorial structures.**

## Python dependencies
Use the `requirements.txt` file as reference.  
You can automatically install all the dependencies using the following command. 
````bash
pip install -r requirements.txt
````

## How to use the code
There are two main entrypoints to run the experiments with the `AGP` algorithm:
- `run_agp_bqp.py`: run the experiments using the AGP model and acquisition function on the Binary Quadratic Programming test problem.
- `run_agp_osp.py`: run the experiments using the AGP model and acquisition function on the Optimal Sensors Placement problem.

There are two main entrypoints to run the experiments with the `MES` and `GIBBON` algorithm:
- `run_mes_bqp.py`: run the experiments using the MES or GIBBON model and acquisition function on the Binary Quadratic Programming test problem.
- `run_mes_osp.py`: run the experiments using the MES or GIBBON model and acquisition function on the Optimal Sensors Placement problem.

In all the scripts, it is possible to modify the algorithm configurations and parameters related to the problems.
