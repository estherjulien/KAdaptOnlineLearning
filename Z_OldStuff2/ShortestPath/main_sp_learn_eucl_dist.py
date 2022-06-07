from Z_OldStuff2.Method.EuclDist.DataGen import data_and_train
import pickle

thread_count = 8
num_instances = 16

train_on = [5, 10, 50, 100, 500]

depth = 1
width = 50

N_train = 40

# ONLINE LEARNING
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

max_depth = 8
n_back_track = 5

# OBTAIN DATA AND TRAIN RANDOM FOREST
# load train instances
with open(f"Data/Instances/train_env_list_sp_N{N_train}_g30_fs10_{train_on[-1]}.pickle", "rb") as handle:
    env_list_train = pickle.load(handle)

time_limit = 3 * 60
for K_train in [2, 3, 4]:
    problem_type = f"sp_eucl_dist_K{K_train}_N{N_train}"
    data_and_train(K_train, env_list_train, att_series, n_back_track, time_limit, problem_type,
                   thread_count, max_depth, width, depth, train_on, init_data=None)
