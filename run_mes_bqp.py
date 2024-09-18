import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    InverseCostWeightedUtility,
    qMultiFidelityMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy, )
from botorch.exceptions import InputDataWarning, ModelFittingError, OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize
from botorch_community.models.gp_regression_multisource import get_random_x_for_agp
from gpytorch import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning
from pymoo.config import Config

from bo.cost_model import FixedCostModel
from ga.acq_wrapper import AcqWrapper, opt_acqf_ga
from problems.test_functions import AugmentedBQP

warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)
Config.warnings['not_compiled'] = False
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Problem config
n_vars = 10 + 1
n_qs = [50, 5]
alpha = 100
reg = 1

# Algo config
algo = "MES"  # MES or GIBBON
n_trials = 10
eval_budget = 120
n_init = 20

true_point_ratio = 0.2

res_path = f".results/bqp/lc{alpha}_lambda{reg}_s_{n_qs[0]}_{n_qs[1]}"
os.makedirs(res_path, exist_ok=True)

torch.manual_seed(0)
cost_model = FixedCostModel(cost=n_qs[1] / n_qs[0])
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

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

    bounds = problem.bounds
    candidate_set = torch.rand(
        1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype
    )
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
    candidate_set = torch.round(candidate_set, decimals=0)

    true_indeces = torch.where(train_x[:, -1] == 1)[0]
    print(
        f"Init Design:"
        f"\t Best seen = {train_obj[true_indeces].max():.4f};"
    )

    while cost < eval_budget:
        # Model init and fitting
        start = time.perf_counter()
        model = SingleTaskMultiFidelityGP(
            train_x,
            train_obj,
            outcome_transform=Standardize(m=1),
            data_fidelity=problem.dim - 1,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # Fit the model
        try:
            fit_gpytorch_mll(mll)  # , optimizer_kwargs={'options': {'maxls': 500}})
        except ModelFittingError:
            optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
            for _ in range(100):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_obj)
                loss.backward()
                optimizer.step()
        model_times.append(time.perf_counter() - start)

        # Acquisition init and optimization
        start = time.perf_counter()
        if algo == "MES":
            qMES = qMultiFidelityMaxValueEntropy(
                model, candidate_set, cost_aware_utility=cost_aware_utility
            )
        else:
            qMES = qMultiFidelityLowerBoundMaxValueEntropy(
                model, candidate_set, cost_aware_utility=cost_aware_utility
            )
        acqf_wrapper = AcqWrapper(qMES, problem.bounds, problem.dim)
        new_x = opt_acqf_ga(acqf_wrapper)
        if len(train_x[torch.where(train_x[..., -1] == 1)]) / len(train_x) <= true_point_ratio:
            new_x[:, -1] = 1.0
        new_x = torch.round(new_x, decimals=0)
        acq_times.append(time.perf_counter() - start)

        # Dataset update
        train_x = torch.cat([train_x, new_x])
        new_obj = problem(new_x)
        train_obj = torch.cat([train_obj, new_obj])
        cost = float(sum(train_x[:, -1] == 0) * (n_qs[-1] / n_qs[0]) + sum(train_x[:, -1] == 1))
        it += 1

        true_indeces = torch.where(train_x[:, -1] == 1)[0]
        print(
            f"Iter {it};"
            f"\tCost: {cost:.2f};"
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
    res.to_csv(f"{res_path}/{algo}_BQP_T{tr}.csv", index=False)
