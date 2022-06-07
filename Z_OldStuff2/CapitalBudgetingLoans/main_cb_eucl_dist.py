from Z_OldStuff2.Method.EuclDist.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
train_on = None
K_train = 3
N_train = 10
obtain_data = True
xi_dim = 4

# SUCCESS PREDICTION
att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]

for N in [10]:
    for K in [3, 4]:
        # LOAD TEST INSTANCES
        try:
            with open(f"Data/Instances/env_list_cb_N{N}_{num_instances}.pickle", "rb") as handle:
                env_list = pickle.load(handle)
        except FileNotFoundError:
            print(f"Instances not found, N = {N}")
            continue

        # TEST
        time_limit = 0.5 * 60 * 60
        print()
        print(f"START STRATEGY K = {K}, N = {N}")

        train_combinations = [10, 30, 50]

        for comb in train_combinations:
            problem_type = f"cb_eucl_dist_ML[K{K_train}_N{N_train}_rf{comb}]_T[K{K}_N{N}]"
            success_model_name = f"Data/ResultsEuclDist/RFModels/rf_model_cb_eucl_dist_K{K_train}_N{N_train}_rf{comb}.joblib"
            Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series,
                                                               success_model_name=success_model_name,
                                                               problem_type=problem_type,
                                                               time_limit=time_limit)
                                          for i in np.arange(num_instances))

            # combine all results
            results = dict()
            for i in np.arange(num_instances):
                with open(f"Data/ResultsEuclDist/Decisions/inst_results/final_results_{problem_type}_inst{i}.pickle", "rb") \
                        as handle:
                    results[i] = pickle.load(handle)
            with open(f"Data/ResultsEuclDist/Decisions/FINAL_results_{problem_type}_{num_instances}.pickle", "wb") as handle:
                pickle.dump(results, handle)