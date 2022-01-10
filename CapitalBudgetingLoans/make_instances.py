from CapitalBudgetingLoans.Environment.Env import ProjectsInstance

from joblib import Parallel, delayed
import numpy as np
import pickle

N = 10

thread_count = 8
num_instances = 16
xi_dim = 4

env_list = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N, xi_dim=xi_dim, inst_num=i)
                                                 for i in np.arange(num_instances))

with open(f"Data/Instances/env_list_cb_N{N}_{num_instances}.pickle", "wb") as handle:
    pickle.dump(env_list, handle)