import numpy as np

from pymoo.core.mutation import Mutation


class BinFlip(Mutation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.where(np.random.random((X.shape[0], X.shape[1])) < prob_var)
        Xp[flip[0], flip[1]] = 1 - Xp[flip[0], flip[1]]
        return Xp
