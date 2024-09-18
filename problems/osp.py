from typing import List, Union, Optional

import numpy as np
import pandas as pd
import torch
from botorch.test_functions import SyntheticTestFunction
from botorch.utils import draw_sobol_samples
from pymoo.core.problem import ElementwiseProblem
from torch import Tensor


class AugmentedOSP(SyntheticTestFunction):

    def __init__(
            self,
            impact_matrix: pd.DataFrame,
            budget: int,
            fidelities: Union[List[float], Tensor],
            sensors: Optional[List[str]] = None,
            negate: bool = False,
            seed: Optional[int] = None
    ):
        self.budget = budget
        self.scenarios = impact_matrix["scenario"].unique()
        if sensors is None:
            self.sensors = impact_matrix["sensor"].unique()
        else:
            self.sensors = np.array(sensors)

        # Source 1 has the highest fidelity
        self.fidelities = sorted(fidelities)

        # Set the impact matrix for each source
        impact_matrix = impact_matrix.drop_duplicates(subset=["sensor", "scenario"])
        impact_matrix = impact_matrix.pivot(index="scenario", columns="sensor", values="impact").fillna(10e4)
        self.impact_matrix = [impact_matrix.sample(frac=fid, random_state=seed) for fid in self.fidelities]

        # The dimensions are the number of sensors, plus one for the source
        self.dim = len(self.sensors) + 1
        self._bounds = [(0, 1)] * self.dim
        self._bounds[-1] = (0, len(fidelities) - 1)
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X = torch.round(X, decimals=0)
        objs = [self._evaluate(x[:-1], int(x[-1])) for x in X]
        return torch.tensor(objs).unsqueeze(-1)

    def get_equality_constraint(self, **tkwargs):
        indices = torch.arange(0, self.dim - 1, **tkwargs).to(dtype=torch.int)
        coefficients = torch.ones(indices.shape, **tkwargs)
        rhf = self.budget
        return indices, coefficients, rhf

    def get_inequality_constraint(self, **tkwargs):
        indices = torch.arange(0, self.dim - 1, **tkwargs).to(dtype=torch.int)
        coefficients = -torch.ones(indices.shape, **tkwargs)
        rhf = -self.budget
        return indices, coefficients, rhf

    def _evaluate(self, x: Tensor, s: int) -> float:
        if x.sum() == 0:
            return -9e4
        sensors = self.sensors[torch.where(x)]
        sensors = [sensors] if not isinstance(sensors, Union[list, np.ndarray]) else sensors
        impact_distr = self.impact_matrix[s][sensors].min(axis=1)
        # CVaR
        impact_distr = sorted(impact_distr.tolist())
        var_index = int(len(impact_distr) * (1 - 0.95))
        return -np.mean(impact_distr[var_index + 1:])


def verify_sp_budget(X: Tensor, budget: int):
    for x in X:
        if torch.sum(x[:-1]) > budget:
            one_idxs = torch.where(x[:-1])[0]
            n_turn_off = torch.sum(x[:-1]).item() - budget
            turn_off_idxs = torch.randperm(len(one_idxs))[:int(n_turn_off)]
            x[one_idxs[turn_off_idxs]] = 0
    return X


def get_feasible_indeces(X: Tensor, budget: int):
    return torch.where(torch.sum(X[:, :-1], dim=1) <= budget)
