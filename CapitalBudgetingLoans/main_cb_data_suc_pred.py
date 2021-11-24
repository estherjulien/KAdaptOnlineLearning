from CapitalBudgetingLoans.Environment.Env import ProjectsInstance
from SuccessPrediction.DataGenMax import data_and_train_max
from SuccessPrediction.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import itertools
import pickle

thread_count = 8
num_instances = 16
data_instances = np.arange(10, 30)
depth = 1
width = 50
train_on = None
K_train = 3
N_train = 10
obtain_data = True
xi_dim = 4


# ONLINE LEARNING
att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]
max_depth = 6
n_back_track = 3
# OBTAIN DATA AND TRAIN RANDOM FOREST
time_limit = 3 * 60
env_list_train = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N_train, xi_dim=xi_dim, inst_num=i)
                                                       for i in data_instances)
problem_type = f"cb_suc_pred_data_K{K_train}_N{N_train}"
data_and_train_max(K_train, env_list_train, att_series, n_back_track, time_limit, problem_type,
                   thread_count, max_depth, width, depth, train_on, init_data=True)


for N in [10]:
    for K in [3, 4]:
        # LOAD TEST INSTANCES
        try:
            with open(f"Results/Instances/env_list_cb_N{N}_{num_instances}.pickle", "rb") as handle:
                env_list = pickle.load(handle)
        except FileNotFoundError:
            env_list = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N, xi_dim=xi_dim, inst_num=i)
                                                             for i in np.arange(num_instances))
            with open(f"Results/Instances/env_list_cb_N{N}_{num_instances}.pickle", "wb") as handle:
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

        train_combinations = [10, 30, 50]

        for comb in train_combinations:
            problem_type = f"cb_suc_pred_strategy_K{K}_N{N}_rf{comb}"
            success_model_name = f"ResultsSucPred/RFModels/rf_model_cb_suc_pred_data_K{K_train}_N{N_train}_rf{comb}.joblib"
            Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series,
                                                               success_model_name=success_model_name,
                                                               problem_type=problem_type,
                                                               time_limit=time_limit)
                                          for i in np.arange(num_instances))
