import torch
from botorch.models.deterministic import DeterministicModel
from torch import Tensor


class FixedCostModel(DeterministicModel):
    def __init__(
            self,
            cost
    ) -> None:
        super().__init__()
        self._num_outputs = 1
        self.cost = cost

    def forward(self, X: Tensor) -> Tensor:
        lin_cost = torch.sum(X[..., -1] == 0) * self.cost + torch.sum(X[..., -1] == 1)
        return lin_cost.unsqueeze(-1).unsqueeze(-1)
