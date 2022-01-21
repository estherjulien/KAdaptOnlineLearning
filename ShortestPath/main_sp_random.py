from Method.Random import algorithm as algorithm_r
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
gamma_perc = 0.3
first_stage_ratio = 0.1

for N in [40]:
    for K in [3, 4]:
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
        Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
                                                           time_limit=time_limit) for i in np.arange(num_instances))

        # combine all results
        results = dict()
        for i in np.arange(num_instances):
            with open(f"Data/Results/Decisions/inst_results/final_results_{problem_type}_inst{i}.pickle", "rb")\
                    as handle:
                results[i] = pickle.load(handle)
        with open(f"Data/Results/Decisions/FINAL_results_{problem_type}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(results, handle)