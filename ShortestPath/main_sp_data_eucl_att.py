from ShortestPath.Environment.Env import Graph
from EuclAttributes.DataGen import data_and_train
from EuclAttributes.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import itertools
import pickle

thread_count = 8
num_instances = 16
num_data_instances = 5
gamma_perc = 0.3
first_stage_ratio = 0.1
depth = 1
width = 50
train_on = np.arange(num_data_instances) + 1

obtain_data = True

# ONLINE LEARNING
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

max_depth = 6
n_back_track = 3
# todo: for N = [30, 40, 50] and K = [2, 3, 4]
for N in [30]:
    for K in [3, 4]:
        # # OBTAIN DATA AND TRAIN RANDOM FOREST
        # time_limit = 10 * 60
        # env_list_train = Parallel(n_jobs=max(8, thread_count))(delayed(Graph)(N=N, gamma_perc=gamma_perc,
        #                                                                       first_stage_ratio=first_stage_ratio,
        #                                                                       max_degree=5,
        #                                                                       throw_away_perc=0.3,
        #                                                                       inst_num=i)
        #                                                        for i in np.arange(num_data_instances))
        # problem_type = f"sp_suc_pred_data_K{K}_N{N}"
        # data_and_train(K, env_list_train, att_series, n_back_track, time_limit, problem_type,
        #                                     thread_count, max_depth, width, depth, train_on)

        # LOAD TEST INSTANCES
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

        # TEST
        time_limit = 0.5 * 60 * 60
        print()
        print(f"START STRATEGY K = {K}, N = {N}")

        with open(f"ResultsSucPred/RFModels/train_combinations_sp_suc_pred_data_K{K}_N{N}.pickle", "rb") as handle:
            train_combinations = pickle.load(handle)

        for comb in train_combinations:
            problem_type = f"sp_suc_pred_strategy_K{K}_N{N}_rf{comb}"
            success_model_name = f"ResultsSucPred/RFModels/rf_model_sp_suc_pred_data_K{K}_N{N}_rf{comb}.joblib"
            Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series,
                                                               success_model_name=success_model_name,
                                                               problem_type=problem_type,
                                                               time_limit=time_limit)
                                          for i in np.arange(num_instances))
