from ProblemFunctions.Env import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle
import sys


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])

    env_list = Parallel(n_jobs=-1)(delayed(Graph)(N=N,
                                                  gamma_perc=0.3,
                                                  first_stage_ratio=0.1,
                                                  max_degree=5,
                                                  throw_away_perc=0.3,
                                                  inst_num=i)
                                   for i in np.arange(num_instances))

    # save per instance
    for i, env in enumerate(env_list):
        with open(f"ShortestPathCluster/Data/Instances/inst_results/sp_env_N{N}_{i}.pickle", "wb") as handle:
            pickle.dump(env, handle)

    # save all
    with open(f"ShortestPathCluster/Data/Instances/sp_env_list_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)
