from KAdaptabilityAlgorithm.Attributes import algorithm as algorithm_att
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from KAdaptabilityAlgorithm.Attributes_online_learning import algorithm_main as algorithm_online
from KAdaptabilityAlgorithm.Attributes_MP_online_learning import algorithm_main as algorithm_online_mp
from ShortestPath.Environment.Env import Graph
from joblib import Parallel, delayed
import numpy as np
import pickle

num_cores = 5
num_instances = int(num_cores)
N = 100
gamma_perc = 0.3
first_stage_ratio = 0.3
try:
    with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "rb") as handle:
        env_list = pickle.load(handle)
except:
    env_list = Parallel(n_jobs=max(8, num_cores))(delayed(Graph)(N=N, gamma_perc=gamma_perc,
                                                                 first_stage_ratio=first_stage_ratio,
                                                                 max_degree=5,
                                                                 throw_away_perc=0.3,
                                                                 inst_num=i) for i in np.arange(num_instances))
    with open(f"Results/Instances/env_list_sp_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(env_list, handle)
K = 4
time_limit = [1*60*60, 4*60*60]
algorithm = [algorithm_random, algorithm_att]

att_series = ["slack", "const_to_z_dist", "const_to_const_dist"]
# att_series = ["coords", "static", "static_obj", "static_y", "nominal", "nominal_obj", "nominal_x", "nominal_y"]
problem_type = f"sp_mp_K{K}_N{N}_g{int(gamma_perc*100)}_fs{int(first_stage_ratio*100)}"

for i in np.arange(5):

    results = algorithm_online_mp(K, env_list[i],
                                  att_series=att_series,
                                  time_limit=1/2*60*60,
                                  print_info=True,
                                  problem_type=problem_type,
                                  thread_count=8)
