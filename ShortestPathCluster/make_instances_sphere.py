from ProblemFunctions.EnvSphere import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle
import copy
import sys


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])
    throw_away_percent = int(sys.argv[3])
    degree = int(sys.argv[4])
    env_list_init = Parallel(n_jobs=-1)(delayed(Graph)(N=N,
                                                       gamma=None,
                                                       throw_away_perc=float(throw_away_percent)/100,
                                                       inst_num=i)
                                        for i in np.arange(num_instances))

    # save per instance
    for gamma in [1, 2, 3, 4]:
        env_list_change = copy.deepcopy(env_list_init)
        for i, env in enumerate(env_list_change):
            env.set_gamma(gamma)
            with open(f"ShortestPathCluster/Data/Instances/inst_results/sp_env_sphere_N{N}_d{degree}_tap{throw_away_percent}_g{gamma}_{i}.pickle", "wb") as handle:
                pickle.dump(env, handle)

        # save all
        with open(f"ShortestPathCluster/Data/Instances/sp_env_sphere_list_N{N}_d{degree}_tap{throw_away_percent}_g{gamma}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(env_list_change, handle)
