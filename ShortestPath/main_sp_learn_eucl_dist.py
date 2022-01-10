from ShortestPath.Environment.Env import Graph
from Method.EuclDist.DataGen import data_and_train
from joblib import Parallel, delayed
import numpy as np

thread_count = 8
num_instances = 16
data_instances = np.arange(20, 100)
gamma_perc = 0.3
first_stage_ratio = 0.1
depth = 1
width = 50
train_on = [25, 50, 75, 100]
K_train = 2
N_train = 30
obtain_data = True

# ONLINE LEARNING
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

max_depth = 8
n_back_track = 5
# OBTAIN DATA AND TRAIN RANDOM FOREST
time_limit = 3 * 60
env_list_train = Parallel(n_jobs=max(8, thread_count))(delayed(Graph)(N=N_train, gamma_perc=gamma_perc,
                                                                      first_stage_ratio=first_stage_ratio,
                                                                      max_degree=5,
                                                                      throw_away_perc=0.3,
                                                                      inst_num=i)
                                                       for i in data_instances)
problem_type = f"sp_suc_pred_data_K{K_train}_N{N_train}"
data_and_train(K_train, env_list_train, att_series, n_back_track, time_limit, problem_type,
               thread_count, max_depth, width, depth, train_on, init_data=True)