import json
import time
import warnings

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.exceptions import InputDataWarning, ModelFittingError, OptimizationWarning
from botorch_community.acquisition.augmented_multisource import AugmentedUpperConfidenceBound
from botorch_community.models.gp_regression_multisource import get_random_x_for_agp
from gpytorch import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning

from bo.agp import SingleTaskAugmentedGP
from ga.acq_wrapper import AcqWrapper, opt_acqf_ga
from problems.osp import AugmentedOSP, verify_sp_budget, get_feasible_indeces

warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

network = "Apulian_5"
BUDGET = 15
COST_BUDGET = 150
N_TRIAL = 5
look_for_checkpoint = False

sensors = None
impact_matrix = pd.read_csv(f"data/{network}_det_times.csv",
                            dtype={'sensor': 'str', 'scenario': 'str', 'impact': 'float'})
if network == "Neptun" or network == "Apulian_5":
    sensors = json.load(open(f".data/{network}_sensors.json", "r"))
    impact_matrix = impact_matrix[impact_matrix["sensor"].isin(sensors)]
fidelities = [0.5, 1.0]
n_sources = len(fidelities)
problem = AugmentedOSP(impact_matrix=impact_matrix, budget=BUDGET,
                       fidelities=fidelities, seed=0).to(**tkwargs)
acqf_constraint = [problem.get_inequality_constraint(**tkwargs)]

N_INIT = (len(impact_matrix["sensor"].unique()) + 1) // 5
agp_m = 1

for tr in range(0, N_TRIAL):
    print(f"\nTRIAL {tr + 1}/{N_TRIAL}...")
    torch.manual_seed(tr)

    if look_for_checkpoint:
        checkpoint = pd.read_csv(f".result/AGP{agp_m}_{network}_OSP{problem.dim}_B{BUDGET}_T{tr}.csv")
        train_x = torch.tensor(checkpoint[[f"x_{i}" for i in range(problem.dim)]].to_numpy().tolist(), **tkwargs)
        train_x[torch.where(train_x[:, -1] == 0.5)[0], -1] = 0
        train_x[torch.where(train_x[:, -1] == 1)[0], -1] = 1
        train_obj = torch.tensor(checkpoint["obj"].tolist(), **tkwargs).unsqueeze(-1)
        model_times = checkpoint["model_times"].tolist()[N_INIT:]
        acq_times = checkpoint["acq_times"].tolist()[N_INIT:]
        start_iter = checkpoint["it"].max()
        cost = torch.sum((train_x[:, -1] + 1) / 2)
    else:
        train_x = get_random_x_for_agp(n=N_INIT, bounds=problem.bounds, q=1, seed=tr)
        train_x = torch.round(train_x, decimals=0)
        train_x = verify_sp_budget(train_x, problem.budget)
        train_obj = problem(train_x)
        model_times, acq_times = [], []
        start_iter = 0
        cost = torch.sum((train_x[:, -1] + 1) / 2)

    feasible_indices = get_feasible_indeces(train_x, problem.budget)[0]
    true_indeces = torch.where(train_x[:, -1] == len(fidelities) - 1)[0]
    true_feasible_indices = list(set(feasible_indices.tolist()) and set(true_indeces.tolist()))
    print(
        f"Init Design:"
        f"\t Best seen = {train_obj[true_feasible_indices].max():.4f};"
    )
    it = start_iter
    # for it in range(start_iter, N_ITER):
    while cost < COST_BUDGET:
        # Model init and fitting
        start = time.time()
        gp = SingleTaskAugmentedGP(
            train_x,
            train_obj,
            m=agp_m,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # Fit the model
        try:
            fit_gpytorch_mll(mll)  # , optimizer_kwargs={'options': {'maxls': 500}})
        except ModelFittingError:
            optimizer = torch.optim.Adam([{"params": gp.parameters()}], lr=0.1)
            for _ in range(100):
                optimizer.zero_grad()
                output = gp(gp.train_inputs[0])
                loss = -mll(output, gp.train_targets)
                loss.backward()
                optimizer.step()

        model_times.append(time.time() - start)

        # Acquisition init and optimization
        start = time.time()
        acq = AugmentedUpperConfidenceBound(
            gp,
            beta=3,
            maximize=True,
            best_f=train_obj[torch.where(train_x[:, -1] == n_sources - 1)].max(),
            cost={i: fid for i, fid in enumerate(fidelities)},
        )
        acqf_wrapper = AcqWrapper(acq, problem.bounds, problem.dim, problem.budget)
        new_x = opt_acqf_ga(acqf_wrapper)
        # candidates, value = optimize_acqf(
        #     acq_function=acq,
        #     bounds=problem.bounds,
        #     q=1,
        #     num_restarts=5,
        #     raw_samples=1024,
        #     inequality_constraints=acqf_constraint
        # )
        # new_x = candidates.detach()
        if gp.n_true_points < gp.max_n_cheap_points:
            new_x[:, -1] = n_sources - 1
        new_x = torch.round(new_x, decimals=0)
        acq_times.append(time.time() - start)

        # Dataset update
        train_x = torch.cat([train_x, new_x])
        new_obj = problem(new_x)
        train_obj = torch.cat([train_obj, new_obj])
        cost = torch.sum((train_x[:, -1] + 1) / 2)
        it += 1

        feasible_indices = get_feasible_indeces(train_x, problem.budget)[0]
        true_indeces = torch.where(train_x[:, -1] == len(fidelities) - 1)[0]
        true_feasible_indices = list(set(feasible_indices.tolist()) and set(true_indeces.tolist()))
        print(
            f"Iter {it};"
            f"\tCost: {cost};"
            f"\t Best seen = {train_obj[true_feasible_indices].max():.4f};"
            f"\t Source: {train_x[-1, -1].tolist():.2f};"
            f"\t Time: {sum(model_times) + sum(acq_times):.2f}"
        )

    # Convert source to fidelities
    mapping_fid = dict(zip(range(len(fidelities)), fidelities))
    train_fid = [mapping_fid[int(source)] for source in train_x[:, -1].tolist()]

    # Save results
    res = pd.DataFrame(
        train_x[:, :-1].tolist(), columns=[f"x_{i}" for i in range(problem.dim - 1)]
    )
    res[f"x_{problem.dim - 1}"] = train_fid
    res["obj"] = train_obj.flatten().tolist()
    res["model_times"] = [0] * N_INIT + model_times
    res["acq_times"] = [0] * N_INIT + acq_times
    res["it"] = [0] * N_INIT + list(range(1, it + 1))
    res.to_csv(f".result/osp/AGP{agp_m}_{network}_OSP{problem.dim}_B{BUDGET}_T{tr}.csv", index=False)
