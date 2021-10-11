from ShortestPath.Environment.Env import Graph
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_r
from KAdaptabilityAlgorithm.Strategy import algorithm as algorithm_s
from KAdaptabilityAlgorithm.OnlineLearning import algorithm as algorithm_o
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
N = 50
gamma_perc = 0.3
first_stage_ratio = 0.1


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

K = 4
time_limit = 1*60*60

att_series = ["coords", "obj_static", "y_static", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]


# print("START ONLINE LEARNING \n")
# problem_type = f"spip_online_learning_random_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# for i in np.arange(num_instances):
#     algorithm_o(K, env_list[i], att_series, problem_type=problem_type)

print("START STRATEGY \n")
problem_type = f"spip_strategy_sub_tree_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series, problem_type=problem_type,
                                                   time_limit=time_limit) for i in np.arange(num_instances))
#
# print("START RANDOM \n")
#
# problem_type = f"spip_random_new_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# # algorithm_r(K, env_list[0], problem_type=problem_type, time_limit=time_limit)
# Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
#                                                    time_limit=time_limit) for i in np.arange(num_instances))