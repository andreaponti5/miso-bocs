import numpy as np

from pymoo.core.crossover import Crossover


class BinCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):

        # get the X of parents and count the matings
        n_parents, n_matings, n_var = X.shape

        offspring = np.full((n_parents, n_matings, n_var), 0)
        for index_mating in range(n_matings):
            # Get the two parents
            father = X[0, index_mating]
            mother = X[1, index_mating]
            # Get the indeces of True elements
            father_idx = np.where(father)[0]
            mother_idx = np.where(mother)[0]
            # Sample without replacement from this indeces
            father_samples = np.random.choice(father_idx, father_idx.shape[0], replace=False)
            mother_samples = np.random.choice(mother_idx, mother_idx.shape[0], replace=False)

            index_offspring = 0

            for index_var in range(max(father_samples.shape[0], mother_samples.shape[0])):
                if index_var < father_samples.shape[0]:
                    offspring[index_offspring, index_mating, father_samples[index_var]] = father[father_samples[index_var]]
                if index_var < mother_samples.shape[0]:
                    offspring[index_offspring, index_mating, mother_samples[index_var]] = mother[mother_samples[index_var]]
                # Each iteration change child
                index_offspring = 1 if index_offspring == 0 else 0

        return offspring
