from ShortestPath.Environment.Env import Graph
from KAdaptabilityAlgorithmOld.OnlineLearning import algorithm as algorithm_ol
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_r
from KAdaptabilityAlgorithm.Strategy import algorithm as algorithm_s
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
N = 100
gamma_perc = 0.3
first_stage_ratio = 0.3
depth = 1
width = 20

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
time_limit = 2*60*60

print("START RANDOM NEW\n")

problem_type = f"sp_random_new_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# algorithm_r(K, env_list[0], problem_type=problem_type, time_limit=time_limit)
Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
                                                   time_limit=time_limit) for i in np.arange(num_instances))

# print("START STRATEGY OTHER\n")
#
# att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]
# problem_type = f"sp_strategy_sub_tree_other_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series, problem_type=problem_type,
#                                                    time_limit=time_limit) for i in np.arange(num_instances))
#
# print("START ONLINE LEARNING RANDOM\n")
#
# problem_type = f"sp_online_learning_random_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# for i in np.arange(num_instances):
#     algorithm_ol(K, env_list[i], att_series, sub_tree=False, problem_type=problem_type, time_limit=30*60, depth=1, width=50)
#
# problem_type = f"sp_strategy_random_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
# model_names = [f"Results/Models/SubTree/nn_model_sp_online_learning_random_K{K}_N{N}_g{int(gamma_perc*100)}_"
#                f"fs{int(first_stage_ratio*100)}_D1_W50_inst{i}.h5" for i in np.arange(num_instances)]
#
# print("START STRATEGY RANDOM\n")
# Parallel(n_jobs=thread_count)(delayed(algorithm_s)(K, env_list[i], att_series, problem_type=problem_type,
#                                                    weight_model_name=model_names[i],
#                                                    time_limit=time_limit) for i in np.arange(num_instances))

