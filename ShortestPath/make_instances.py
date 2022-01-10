from ShortestPath.Environment.Env import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle

N = 40

thread_count = 8
num_instances = 16
gamma_perc = 0.3
first_stage_ratio = 0.1

env_list = Parallel(n_jobs=max(8, thread_count))(delayed(Graph)(N=N,
                                                                gamma_perc=gamma_perc,
                                                                first_stage_ratio=first_stage_ratio,
                                                                max_degree=5,
                                                                throw_away_perc=0.3,
                                                                inst_num=i)
                                                 for i in np.arange(num_instances))

with open(f"Instances/env_list_sp_N{N}_g{int(gamma_perc * 100)}_"
          f"fs{int(first_stage_ratio * 100)}_{num_instances}.pickle", "wb") as handle:
    pickle.dump(env_list, handle)
