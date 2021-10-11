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
N = 30
K = 4
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

time_limit = 60*60
for i in np.arange(2):
    env = Graph(N=N, gamma_perc=gamma_perc, first_stage_ratio=first_stage_ratio, max_degree=5, throw_away_perc=0.3, inst_num=i)
    problem_type = f"sp_online_sub_tree_other_K{K}_N{N}"
    algorithm_o(K, env, att_series, time_limit=time_limit)

# RANDOM AND STRATEGY
for N in [30, 40, 50]:
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

    time_limit = 2*60*60

    print("START RANDOM NEW\n")

    problem_type = f"sp_random_new_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
    # algorithm_r(K, env_list[0], problem_type=problem_type, time_limit=time_limit)
    Parallel(n_jobs=thread_count)(delayed(algorithm_r)(K, env_list[i], problem_type=problem_type,
                                                       time_limit=time_limit) for i in np.arange(num_instances))


att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist", ""]
