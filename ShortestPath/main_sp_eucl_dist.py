from Method.EuclDist.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import pickle

time_limit = 0.5 * 60 * 60

thread_count = 8
num_instances = 16
gamma_perc = 0.3
first_stage_ratio = 0.1
depth = 1
width = 50
train_on = [25, 50, 75, 100]
K_train = 2
N_train = 30
obtain_data = True

# EUCLIDEAN DISTANCE
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

for N in [30, 40]:
    for K in [3, 4, 5]:
        # LOAD TEST INSTANCES
        try:
            with open(
                    f"Data/Instances/env_list_sp_N{N}_g{int(gamma_perc * 100)}_fs{int(first_stage_ratio * 100)}_{num_instances}.pickle", "rb") as handle:
                env_list = pickle.load(handle)
        except FileNotFoundError:
            print(f"Instances not found, N = {N}")
            continue

        # TEST
        print()
        print(f"START STRATEGY K = {K}, N = {N}")

        for data_num in train_on:
            problem_type = f"sp_suc_pred_strategy_ML[K{K_train}_N{N_train}_rf{data_num}]_T[K{K}_N{N}]"
            success_model_name = f"ResultsSucPred/RFModels/rf_model_sp_suc_pred_data_K{K_train}_N{N_train}_rf{data_num}.joblib"
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