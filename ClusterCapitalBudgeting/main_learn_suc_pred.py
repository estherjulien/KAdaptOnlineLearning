from Method.SucPredData import data_gen_fun_max

import pickle
import sys


def main(i, N_train, K_train):
    max_depth = 6
    n_back_track = 3
    time_limit = 2 * 60
    thread_count = 16

    problem_type = f"cb_suc_pred_N{N_train}_K{K_train}"

    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/train_env_cb_N{N_train}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    data_gen_fun_max(K_train, env, att_series, n_back_track=n_back_track, time_limit=time_limit, problem_type=problem_type,
                     thread_count=thread_count, max_depth=max_depth)


if __name__ == "__main__":
    i = int(sys.argv[1])
    N_train = int(sys.argv[2])
    K_train = int(sys.argv[3])
    main(i, N_train, K_train)
