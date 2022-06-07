from joblib import Parallel, delayed
from Method.Random import algorithm
import numpy as np
import pickle


N = 100
num_instances = 8

for K in [2, 3, 4]:
    for gamma_perc in [25]:
        print(f"\nK = {K}   GAMMA PERCENT = {gamma_perc}\n")
        # load environment
        with open(f"Knapsack/Data/Instances/ks_test_env_list_N{N}_g{gamma_perc}_{num_instances}.pickle", "rb") as handle:
            env_list = pickle.load(handle)

        problem_type = f"ks_random_N{N}_g{gamma_perc}_K{K}"
        Parallel(n_jobs=8)(delayed(algorithm)(K, env_list[i], problem_type=problem_type) for i in np.arange(num_instances))
