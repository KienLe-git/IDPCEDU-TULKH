from copy import deepcopy
from re import I
from typing import Deque, Tuple, Type
from matplotlib import pyplot as plt
import numpy as np

from ..tasks.task import AbstractTask
from ..EA import Individual, Population, SubPopulation

class AbstractMutation():
    def __init__(self, *arg, **kwargs):
        self.pm = None
        pass
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        if self.pm is None:
            self.pm = 1/self.dim_uss
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    def update(self, *arg, **kwargs) -> None:
        pass
    def compute_accept_prob(self, new_fcost, min_fcost): 
        pass
class NoMutation(AbstractMutation):
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            newInd =  self.IndClass(genes= np.copy(ind.genes), parent=ind)
            newInd.skill_factor = ind.skill_factor
            newInd.fcost = ind.fcost
            return newInd
        else:
            return ind
    
class IDPCEDU_Mutation(AbstractMutation):
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        self.S_tasks = [np.amax(t.count_paths, axis= 1)  for t in tasks]
        

    def __call__(self, ind: Individual, return_newInd: bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            new_genes:np.ndarray = np.copy(ind.genes)

            i, j = np.random.randint(0, self.dim_uss, 2)
            new_genes[0, i], new_genes[0, j] = new_genes[0, j], new_genes[0, i]
            i = np.random.randint(0, len(self.S_tasks[ind.skill_factor]))
            new_genes[1, i] = np.random.randint(0, self.S_tasks[ind.skill_factor][i])
            newInd = self.IndClass(genes= new_genes)
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            i, j = np.random.randint(0, self.dim_uss, 2)
            ind.genes[0, i], ind.genes[0, j] = ind.genes[0, j], ind.genes[0, i]
            i = np.random.randint(0, len(self.S_tasks[ind.skill_factor]))
            ind.genes[1, i] = np.random.randint(0, self.S_tasks[ind.skill_factor][i])

            return ind

