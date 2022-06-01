from typing import Deque, Tuple, Type
import numpy as np
from ..tasks.task import AbstractTask
from ..EA import Individual, Population
from .Search import *
import numba as nb

class AbstractCrossover():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
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

class NoCrossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        oa = self.IndClass(None, self.dim_uss)
        ob = self.IndClass(None, self.dim_uss)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob
        


@nb.njit
def pmx_func(p1, p2, t1, t2,  dim_uss):
    oa = np.empty_like(p1)
    ob = np.empty_like(p1)
    
    mid = np.copy(p2[t1:t2])
    mid_b = np.copy(p1[t1 : t2])
    
    added = np.zeros_like(p1)
    added_b = np.zeros_like(p2)
    
    added[mid] = 1
    added_b[mid_b] = 1
    
    redundant_idx = []
    redundant_idx_b = []
    
    
    for i in range(t1):
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
            
    for i in range(t2, dim_uss):
        
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
    
    redundant = np.empty(len(redundant_idx))
    redundant_b = np.empty(len(redundant_idx_b))
    
    cnt = 0
    cnt_b = 0
    
    for i in range(t1, t2):
        if added[p1[i]] == 0:
            redundant[cnt] = p1[i]
            cnt+=1
        if added_b[p2[i]] == 0:
            redundant_b[cnt_b] = p2[i]
            cnt_b+=1
    
    redundant_idx = np.array(redundant_idx)
    redundant_idx_b = np.array(redundant_idx_b)
    
    oa[redundant_idx] = redundant
    ob[redundant_idx_b] = redundant_b
    
    oa[t1:t2] = mid
    ob[t1:t2] = mid_b
    return oa, ob
    

class PMX_Crossover(AbstractCrossover):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1 = np.random.randint(0, self.dim_uss + 1)
        t2 = np.random.randint(t1, self.dim_uss + 1)
        genes_oa, genes_ob = pmx_func(pa.genes, pb.genes, t1, t2, self.dim_uss)
        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

        

class TPX_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1

        genes_oa = np.copy(pa.genes)
        genes_ob = np.copy(pb.genes)

        genes_oa[t1:t2], genes_ob[t1:t2] = genes_ob[t1:t2], genes_oa[t1:t2]

        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob


class IDPCEDU_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        genes_oa, genes_ob = np.empty_like(pa), np.empty_like(pb)

        #PMX
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1
        genes_oa[0], genes_ob[0] = pmx_func(pa.genes[0], pb.genes[0], t1, t2, self.dim_uss)

        #TPX
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1
        genes_oa[1], genes_ob[1] = pa.genes[1], pb.genes[1]
        genes_oa[1, t1:t2], genes_ob[1, t1:t2] = genes_ob[1, t1:t2], genes_oa[1, t1:t2]

        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob
