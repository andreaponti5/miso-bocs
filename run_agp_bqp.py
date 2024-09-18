import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorStandardDeviation
from botorch.exceptions import ModelFittingError, InputDataWarning, OptimizationWarning
from botorch_community.acquisition.augmented_multisource import AugmentedUpperConfidenceBound
from botorch_community.models.gp_regression_multisource import get_random_x_for_agp
from gpytorch import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning
from pymoo.config import Config

from bo.agp import SingleTaskAugmentedGP
from ga.acq_wrapper import AcqWrapper, opt_acqf_ga
from test_functions import AugmentedBQP

warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)
Config.warnings['not_compiled'] = False

# Problem config
n_vars = 10 + 1
n_qs = [50, 5]
alpha = 100
reg = 1

# Algo config
n_trials = 10
eval_budget = 120
n_init = 20

res_path = f".results/bqp/lc{alpha}_lambda{reg}_s_{n_qs[0]}_{n_qs[1]}"
os.makedirs(res_path, exist_ok=True)

np.random.seed(3)
torch.manual_seed(3)
problem = AugmentedBQP(n_vars=n_vars, n_qs=n_qs, alpha=alpha, reg=reg)

for tr in range(n_trials):
    print(f"\nTRIAL {tr + 1}/{n_trials}...")
    cost, it = 0, 0
    model_times, acq_times = [], []

    np.random.seed(tr)
    torch.manual_seed(tr)
    # problem = AugmentedBQP(n_vars=n_vars, n_qs=n_qs, alpha=alpha, reg=reg)
    train_x = get_random_x_for_agp(n=n_init, bounds=problem.bounds, q=1, seed=tr)
    train_x = torch.round(train_x, decimals=0)
    train_obj = problem(train_x)

    true_indeces = torch.where(train_x[:, -1] == 1)[0]
    print(
        f"Init Design:"
        f"\t Best seen = {train_obj[true_indeces].max():.4f};"
    )

    while cost < eval_budget:
        # Model init and fitting
        start = time.perf_counter()
        gp = SingleTaskAugmentedGP(train_x, train_obj, m=1)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            optimizer = torch.optim.Adam([{"params": gp.parameters()}], lr=0.1)
            for _ in range(100):
                optimizer.zero_grad()
                output = gp(gp.train_inputs[0])
                loss = -mll(output, gp.train_targets)
                loss.backward()
                optimizer.step()
        model_times.append(time.perf_counter() - start)

        # Acquisition init and optimization
        start = time.perf_counter()
        # TODO: Implement 3.3 of "Sparsifying to optimize..."
        acq = AugmentedUpperConfidenceBound(
            gp,
            beta=3,
            best_f=train_obj[torch.where(train_x[:, -1] == 1)].max(),
            cost={0: 0.5, 1: 1},
        )
        acqf_wrapper = AcqWrapper(acq, problem.bounds, problem.dim)
        new_x = opt_acqf_ga(acqf_wrapper)
        if gp.n_true_points < gp.max_n_cheap_points:
            new_x[:, -1] = 1
        new_x = torch.round(new_x, decimals=0)
        # Optimize the posterior standard deviation if `new_x` is already in the dataset
        if any([torch.equal(new_x[0], x) for x in train_x]):
            acq = PosteriorStandardDeviation(gp.models[-1])
            acqf_wrapper = AcqWrapper(acq, problem.bounds[..., :-1], problem.dim - 1)
            new_x = opt_acqf_ga(acqf_wrapper)
            new_x = torch.round(new_x, decimals=0)
            new_x = torch.cat([new_x, torch.tensor([[1]])], dim=1)
        acq_times.append(time.perf_counter() - start)

        # Dataset update
        train_x = torch.cat([train_x, new_x])
        new_obj = problem(new_x)
        train_obj = torch.cat([train_obj, new_obj])
        cost = sum(train_x[:, -1] == 0) * (n_qs[-1] / n_qs[0]) + sum(train_x[:, -1] == 1)
        it += 1

        true_indeces = torch.where(train_x[:, -1] == 1)[0]
        print(
            f"Iter {it};"
            f"\tCost: {cost};"
            f"\t Best seen = {train_obj[true_indeces].max():.4f};"
            f"\t Source: {train_x[-1, -1].tolist():.2f};"
            f"\t Time: {sum(model_times) + sum(acq_times):.2f}"
        )

    # Save results
    res = pd.DataFrame(train_x.tolist(), columns=[f"x_{i}" for i in range(problem.dim)])
    res["obj"] = train_obj.flatten().tolist()
    res["model_times"] = [0] * n_init + model_times
    res["acq_times"] = [0] * n_init + acq_times
    res["it"] = [0] * n_init + list(range(1, it + 1))
    res.to_csv(f"{res_path}/AGP_BQP_T{tr}.csv", index=False)
