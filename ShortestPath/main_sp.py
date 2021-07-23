from KAdaptabilityAlgorithm.Attributes import algorithm as algorithm_att
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from ShortestPath.Environment.Env import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle

num_cores = 8
num_instances = int(num_cores)
N = 100
try:
    with open(f"Results/Instances/env_list_sp_N{N}_{num_instances}.pickle", "rb") as handle:
        env_list = pickle.load(handle)
except:
    env_list = Parallel(n_jobs=max(8, num_cores))(delayed(Graph)(N=N, gamma_perc=0.01, first_stage_ratio=0.3, max_degree=5,
                                                            throw_away_perc=0.3, inst_num=i)
                                                            for i in np.arange(num_instances))
    with open(f"Results/Instances/env_list_sp_N{N}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)
K = 4
time_limit = [1*60*60, 4*60*60]     # algorithm att has 4 hours, algorithm random has 1 hour.
algorithm = [algorithm_random, algorithm_att]

att_series = ["coords", "static", "static_obj", "static_y", "nominal", "nominal_obj", "nominal_x", "nominal_y"]
problem_type = f"sp_2s_K{K}_N{N}_csn"

# results = algorithm_att(K, env_list[0],
#                             att_series=att_series,
#                             time_limit=time_limit,
#                             print_info=True,
#                             problem_type=problem_type)

results = Parallel(n_jobs=num_cores)(delayed(algorithm[a])(K, env,
                                                        att_series=att_series,
                                                        time_limit=time_limit[a],
                                                        print_info=True,
                                                        problem_type=problem_type)
                                                        for env in env_list
                                                        for a in [0, 1])
