from typing import List

import numpy as np
import torch

from botorch.test_functions import SyntheticTestFunction
from torch import Tensor
from torch.distributions import Beta


class AugmentedBQP(SyntheticTestFunction):

    def __init__(
            self,
            n_vars: int,
            n_qs: List[int],
            alpha: float,
            reg: float,
            negate: bool = False,

    ):
        self.dim = n_vars
        super().__init__(negate=negate, bounds=[(0, 1)] * n_vars)
        self.model = lambda x, Q: (x.matmul(Q) * x).sum(dim=1)
        self.penalty = lambda x: reg * torch.sum(x, dim=1)
        self.Qs = torch.tensor([get_quad_mat(n_vars - 1, alpha) for _ in range(n_qs[0])], dtype=torch.float64)
        self.Qs_reduced = self.Qs[:n_qs[1]]

    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        zero_idxs = torch.where(X[..., -1] == 0)[0]
        one_idxs = torch.where(X[..., -1] == 1)[0]

        penalty_zero = self.penalty(X[zero_idxs, :-1])
        res[zero_idxs] = torch.stack([self.model(X[zero_idxs, :-1], Q) + penalty_zero for Q in self.Qs_reduced]).mean(dim=0)
        penalty_one = self.penalty(X[one_idxs, :-1])
        res[one_idxs] = torch.stack([self.model(X[one_idxs, :-1], Q) + penalty_one for Q in self.Qs]).mean(dim=0)
        return res.reshape(-1, 1)


class AugmentedContamination(SyntheticTestFunction):

    def __init__(
            self,
            n_vars: int,
            n_gens: List[int],
            reg: float,
            prop_limit: float = 0.1,
            negate: bool = False,
    ):
        self.dim = n_vars
        super().__init__(negate=negate, bounds=[(0, 1)] * n_vars)
        # Number of independent simulations
        self.n_gens = n_gens
        # Regularization parameter
        self.reg = reg
        # Upper fraction limit of contam food at each stage
        self.prop_limit = prop_limit
        # Beta distributions for Z (init), \Lambda (contam), \Gamma (restore)
        self.init_beta = Beta(torch.tensor(1, dtype=torch.float64),
                              torch.tensor(30, dtype=torch.float64))
        self.contam_beta = Beta(torch.tensor(1, dtype=torch.float64),
                                torch.tensor(17 / 3, dtype=torch.float64))
        self.restore_beta = Beta(torch.tensor(1, dtype=torch.float64),
                                 torch.tensor(3 / 7, dtype=torch.float64))
        # Cost of applying a prevention
        self.cost = torch.ones(n_vars - 1, dtype=torch.float64)

    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        for idx, x in enumerate(X):
            source = int(x[-1])
            x_var = x[:-1]
            init_z = self.init_beta.sample(torch.Size([self.n_gens[source]]))
            contam_lambda = self.contam_beta.sample(torch.Size([self.dim - 1, self.n_gens[source]]))
            restore_gamma = self.restore_beta.sample(torch.Size([self.dim - 1, self.n_gens[source]]))

            # Compute fraction of contamination at each stage
            contam_frac = torch.zeros(torch.Size([self.dim - 1, self.n_gens[source]]))
            contam_frac[0] = contam_lambda[0] * (1 - x_var[0]) * (1 - init_z) + (1 - restore_gamma[0] * x_var[0]) * init_z
            for i in range(1, self.dim - 1):
                contam_frac[i] = (contam_lambda[i] * (1 - x_var[i]) * (1 - contam_frac[i - 1]) +
                                  (1 - restore_gamma[i] * x_var[i]) * contam_frac[i - 1])
            # Compute cost of prevention
            cost = torch.sum(self.cost * x_var)
            # Check constraints with 95% confidence interval
            constr = contam_frac <= self.prop_limit
            constr = constr.sum(dim=1) / self.n_gens[source] - 0.95

            res[idx] = cost - constr.sum() + torch.norm(x_var, p=1) * self.reg
        return -res.reshape(-1, 1)


def get_quad_mat(n_vars, alpha):
    # evaluate decay function
    i = np.linspace(1, n_vars, n_vars)
    j = np.linspace(1, n_vars, n_vars)

    K = lambda s, t: np.exp(-1 * (s - t) ** 2 / alpha)
    decay = K(i[:, None], j[None, :])

    # Generate random quadratic model and apply exponential decay to Q
    # TODO: Try with ~N(0, 0.1) -> replace `randn` with `normal`
    Q = np.random.randn(n_vars, n_vars)
    # Q = np.random.normal(loc=0.0, scale=0.1, size=(n_vars, n_vars))
    Qa = Q * decay

    return Qa.tolist()
