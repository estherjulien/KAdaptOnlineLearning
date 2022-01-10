from CapitalBudgetingLoans.Environment.Env import ProjectsInstance
from Method.SucPred.DataGenMax import data_and_train_max
from joblib import Parallel, delayed
import numpy as np

thread_count = 8
num_instances = 16
data_instances = np.arange(30)
depth = 1
width = 50
train_on = None
K_train = 3
N_train = 10
obtain_data = True
xi_dim = 4

# SUCCESS PREDICTION
att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]
max_depth = 6
n_back_track = 3
# OBTAIN DATA AND TRAIN RANDOM FOREST
time_limit = 3 * 60
env_list_train = Parallel(n_jobs=max(8, thread_count))(delayed(ProjectsInstance)(N=N_train, xi_dim=xi_dim, inst_num=i)
                                                       for i in data_instances)
problem_type = f"cb_suc_pred_K{K_train}_N{N_train}"
data_and_train_max(K_train, env_list_train, att_series, n_back_track, time_limit, problem_type,
                   thread_count, max_depth, width, depth, train_on, init_data=True)

