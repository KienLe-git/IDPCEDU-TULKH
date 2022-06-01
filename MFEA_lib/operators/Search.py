from typing import Tuple, Type
import numpy as np
import scipy.stats

from ..tasks.task import AbstractTask
from ..EA import Individual, Population

class AbstractSearch():
    def __init__(self) -> None:
        pass
    def __call__(self, *args, **kwargs) -> Individual:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: list[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    
    def update(self, *args, **kwargs) -> None:
        pass

