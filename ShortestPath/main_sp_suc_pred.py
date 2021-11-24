from ShortestPath.Environment.Env import Graph
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_r
from SuccessPrediction.OnlineLearning import algorithm as algorithm_o
from SuccessPrediction.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
gamma_perc = 0.3
first_stage_ratio = 0.1
depth = 1

# ONLINE LEARNING
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

max_depth = 6
n_back_track = 3

for N in [30, 40, 50]:
    for K in [4]:
        time_limit = 10 * 60
        for rf_num in np.arange(3):
            print(f"START ONLINE LEARNING K = {K}, N = {N}\n")
            env = Graph(N=N, gamma_perc=gamma_perc, first_stage_ratio=first_stage_ratio, max_degree=5, throw_away_perc=0.3, inst_num=0)
            problem_type = f"sp_suc_pred_K{K}_N{N}_rf{rf_num}"
            algorithm_o(K, env, att_series, time_limit=time_limit, max_depth=max_depth, n_back_track=n_back_track, problem_type=problem_type, width=100, depth=2)

        try:
            with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "rb") as handle:
                env_list = pickle.load(handle)
        except FileNotFoundError:
            env_list = Parallel(n_jobs=max(8, thread_count))(delayed(Graph)(N=N, gamma_perc=gamma_perc,
                                                                            first_stage_ratio=first_stage_ratio,
                                                                            max_degree=5,
                                                                            throw_away_perc=0.3,
                                                                            inst_num=i) for i in np.arange(num_instances))
            with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "wb") as handle:
                pickle.dump(env_list, handle)

        # time_limit = 0.5 * 60 * 60
        # print()
        # print(f"START RANDOM K = {K}, N = {N}\n")
        #
        # problem_type = f"sp_random_new_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
        # algorithm_r(K, env_list[0], problem_type=problem_type, time_limit=time_limit)
        # Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
        #                                                    time_limit=time_limit) for i in np.arange(num_instances))

        time_limit = 0.5 * 60 * 60
        print()
        print(f"START STRATEGY K = {K}, N = {N}")
        for rf_num in np.arange(3):
            problem_type = f"sp_suc_pred_strategy_K{K}_N{N}_rf{rf_num}"
            success_model_name = f"ResultsSucPred/RFModels/rf_model_sp_suc_pred_K{K}_N{N}_rf{rf_num}.joblib"
            Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series,
                                                               success_model_name=success_model_name,
                                                               problem_type=problem_type,
                                                               time_limit=time_limit)
                                          for i in np.arange(num_instances))

