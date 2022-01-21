from CapitalBudgetingLoans.Environment.Env import ProjectsInstance

from joblib import Parallel, delayed
import numpy as np
import pickle

test_data = False
N = 10
num_instances = 5
xi_dim = 4

thread_count = 8

env_list = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N, xi_dim=xi_dim, inst_num=i)
                                                 for i in np.arange(num_instances))
if test_data:
    with open(f"Data/Instances/env_list_cb_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)
else:
    # save train instances
    with open(f"Data/Instances/train_env_cb_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)