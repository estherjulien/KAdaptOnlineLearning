from Method.Random import algorithm as algorithm_r
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
gamma_perc = 0.3
first_stage_ratio = 0.1

obtain_data = True

# todo: for N = [30, 40, 50] and K = [2, 3, 4]
for N in [30, 40, 50]:
    for K in [2, 3, 4]:
        # LOAD TEST INSTANCES
        try:
            with open(
                    f"Data/Instances/env_list_sp_N{N}_g{int(gamma_perc * 100)}_fs{int(first_stage_ratio * 100)}_{num_instances}.pickle", "rb") as handle:
                env_list = pickle.load(handle)
        except FileNotFoundError:
            print(f"Instances not found, N = {N}")
            continue

        time_limit = 0.5 * 60 * 60
        print()
        print(f"START RANDOM K = {K}, N = {N}\n")

        problem_type = f"sp_random_new_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
        algorithm_r(K, env_list[0], problem_type=problem_type, time_limit=time_limit)
        Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
                                                           time_limit=time_limit) for i in np.arange(num_instances))
