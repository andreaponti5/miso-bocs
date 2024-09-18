import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling, IntegerRandomSampling

from ga.bin_crossover import BinCrossover
from ga.bin_flip import BinFlip


class AcqWrapper(ElementwiseProblem):

    def __init__(self, acquisition_function, bound, n_vars):
        bound = bound.numpy()
        xl = bound[0]
        xu = bound[1]
        super().__init__(n_var=n_vars, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=xl, xu=xu, vtype=int)
        self.acquisition_function = acquisition_function

    def _evaluate(self, x, out, *args, **kwargs):
        x_torch = torch.tensor(x, dtype=torch.float64)
        out["F"] = -self.acquisition_function(x_torch.unsqueeze(0)).item()


def opt_acqf_ga(problem, crossover=PointCrossover(2)):
    algo = GA(pop_size=10,
              sampling=IntegerRandomSampling(),
              crossover=crossover,
              mutation=BinFlip())
    algo.setup(problem, termination=('n_gen', 50), seed=1, verbose=False)

    while algo.has_next():
        pop = algo.ask()
        algo.evaluator.eval(problem, pop)
        algo.tell(infills=pop)

    res = algo.result()
    return torch.tensor(res.X).int().double().unsqueeze(0)
