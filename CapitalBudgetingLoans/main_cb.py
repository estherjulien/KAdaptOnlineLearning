from CapitalBudgetingLoans.Environment.Env import ProjectsInstance

from KAdaptabilityAlgorithm.Random import algorithm as algorithm_r
from SuccessPrediction.Strategy import algorithm as algorithm_s
from SuccessPrediction.OnlineLearningMax import algorithm_max as algorithm_oa
from joblib import Parallel, delayed
import numpy as np
import pickle

xi_dim = 4

thread_count = 8
num_instances = 16

# print("START ONLINE LEARNING \n")
att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]

for N in [10, 20, 30]:
    try:
        with open(f"Results/Instances/env_list_cb_N{N}_{num_instances}.pickle", "rb") as handle:
            env_list = pickle.load(handle)
    except FileNotFoundError:
        env_list = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N, xi_dim=xi_dim, inst_num=i)
                                                         for i in np.arange(num_instances))
        with open(f"Results/Instances/env_list_cb_N{N}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(env_list, handle)

    # online learning
    for K in [4]:
        # online learning for all instances
        time_limit = 10 * 60
        for i in np.arange(num_instances):
            print(f"START ONLINE LEARNING I = {i}, K = {K}, N = {N}")
            problem_type = f"cb_online_learning_suc_pred_sub_tree_K{K}_N{N}"
            if K == 2:
                max_depth = 10
                n_back_track = 5
            elif K == 3:
                max_depth = 7
                n_back_track = 4
            elif K == 4:
                max_depth = 6
                n_back_track = 3

            algorithm_oa(K, env_list[i], att_series, problem_type=problem_type, time_limit=time_limit, depth=3,
                         width=100, max_depth=max_depth, n_back_track=n_back_track)
        try:
            results_random = dict()
            for i in np.arange(num_instances):
                with open(f"Results/Decisions/Random/final_results_cb_random_K{K}_N{N}_inst{i}.pickle", "rb") as handle:
                    pickle.load(handle)
            del results_random
        except FileNotFoundError:
            print(f"START RANDOM K = {K}, N = {N} \n")
            time_limit = 30 * 60
            problem_type = f"cb_random_K{K}_N{N}"
            Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
                                                               time_limit=time_limit) for i in np.arange(num_instances))

        print(f"START STRATEGY K = {K}, N = {N}")
        time_limit = 0.5 * 60 * 60
        problem_type = f"cb_suc_pred_strategy_K{K}_N{N}_online_strategy"
        success_model_name = [f"Results/NNModels/nn_model_alt_cb_online_learning_suc_pred_sub_tree_K{K}_N{N}_D3_W100" \
                              f"_inst{i}.h5" for i in np.arange(num_instances)]
        Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series,
                                                           success_model_name=success_model_name[i],
                                                           problem_type=problem_type,
                                                           time_limit=time_limit)
                                      for i in np.arange(num_instances))
