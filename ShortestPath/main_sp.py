from KAdaptabilityAlgorithm.Attributes import algorithm as algorithm_att
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from KAdaptabilityAlgorithm.Attributes_online_learning import algorithm_main as algorithm_online
from KAdaptabilityAlgorithm.Attributes_MP_online_learning import algorithm_main as algorithm_online_mp
from KAdaptabilityAlgorithm.Attributes_MP_weights import algorithm as algorithm_mp_weights
from KAdaptabilityAlgorithm.Attributes_MP_weights_numpy import algorithm as algorithm_mp_np
from ShortestPath.Environment.Env import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle

thread_count = 8
num_instances = 16
N = 100
gamma_perc = 0.3
first_stage_ratio = 0.3

try:
    with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "rb") as handle:
        env_list = pickle.load(handle)
except:
    env_list = Parallel(n_jobs=max(8, thread_count))(delayed(Graph)(N=N, gamma_perc=gamma_perc,
                                                                 first_stage_ratio=first_stage_ratio,
                                                                 max_degree=5,
                                                                 throw_away_perc=0.3,
                                                                 inst_num=i) for i in np.arange(num_instances))
    with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)


K = 4
time_limit = 2*60*60

print("START RANDOM \n")
# RANDOM
problem_type = f"sp_rand_np_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
Parallel(n_jobs=thread_count)(delayed(algorithm_random)(K, env, time_limit=time_limit, print_info=True,
                                                                  problem_type=problem_type) for env in env_list)
print("\n START LEARNED WEIGHTS (for sure need to learn weights again) \n")
# LEARNED WEIGHTS
with open("Results/Weights/avg_weights_sp_mp_K4_N100_g30_fs30_3_no_coords.pickle", "rb") as handle:
    weights = pickle.load(handle)

att_series = ["slack", "const_to_z_dist", "const_to_const_dist"]
problem_type = f"sp_att_learned_np_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
Parallel(n_jobs=thread_count)(delayed(algorithm_mp_np)(K, env, att_series=att_series,
                                                                  time_limit=time_limit,
                                                                  print_info=True,
                                                                  problem_type=problem_type,
                                                                  weights=weights) for env in env_list)
print("\n START NONLEARNED WEIGHTS \n")

# NONLEARNED WEIGHTS
problem_type = f"sp_att_np_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
Parallel(n_jobs=thread_count)(delayed(algorithm_mp_np)(K, env, att_series=att_series,
                                                                  time_limit=time_limit,
                                                                  print_info=True,
                                                                  problem_type=problem_type,
                                                                  weights=None) for env in env_list)


# problem_type = f"sp_mp_att_online_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
#
# for i in np.arange(5):
#     algorithm_online_mp(K, env_list[i], att_series, time_limit=4*60*60, print_info=True, problem_type=problem_type)
