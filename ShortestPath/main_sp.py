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
print("START OBJECTIVE + VIOLATION DIFFERENCE \n")
# # # algorithm_rand(K, env_list[0], print_info=True)
# algorithm_obj_viol_rule_test(K, env_list[0], print_info=True)
problem_type = f"sp_obj_rule_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"
Parallel(n_jobs=thread_count)(delayed(algorithm_obj_rule)(K, env, time_limit=time_limit,
                                                                    print_info=True,
                                                                    problem_type=problem_type)
                              for env in env_list)


# print("START RANDOM\n")
# # RANDOM
# time_limit = 30*60
# problem_type = [f"sp_rand_np_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_t{t}" for t in np.arange(8)]
# Parallel(n_jobs=thread_count)(delayed(algorithm_random)(K, env, time_limit=time_limit, print_info=True,
#                                                         problem_type=pt)
#                               for env in env_list for pt in problem_type)


att_series = ["slack", "const_to_z_dist", "const_to_const_dist"]
