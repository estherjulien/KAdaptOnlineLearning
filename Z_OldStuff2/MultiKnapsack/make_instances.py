import copy

from ProblemFunctions.Env import KnapsackEnv
from joblib import Parallel, delayed
import numpy as np
import pickle
import sys


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])

    env_list = Parallel(n_jobs=-1)(delayed(KnapsackEnv)(N=N, inst_num=i) for i in
                                   np.arange(num_instances))

    for gamma_perc in [5, 15, 25, 50]:
        env_list_gamma = copy.deepcopy(env_list)
        for env in env_list_gamma:
            env.set_gamma(gamma_perc=float(gamma_perc)/100)

        # save per instance
        for i, env in enumerate(env_list_gamma):
            with open(f"Knapsack/Data/Instances/inst_results/ks_test_env_N{N}_g{gamma_perc}_{i}.pickle", "wb") as handle:
                pickle.dump(env, handle)
        # save all
        with open(f"Knapsack/Data/Instances/ks_test_env_list_N{N}_g{gamma_perc}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(env_list_gamma, handle)

