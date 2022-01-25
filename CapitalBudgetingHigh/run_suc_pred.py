from Method.SucPred import algorithm
import pickle
import sys


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    max_level = int(sys.argv[4])
    if len(sys.argv) == 6:
        time_limit = int(sys.argv[5])*60
    else:
        time_limit = 15*60

    # load environment
    with open(f"CapitalBudgetingHigh/Data/Instances/inst_results/cb_test_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgetingHigh/Data/Models/nn_class_cb_N10_K4_balanced_all_D2_W100.h5"

    # run algorithm with threshold
    problem_type = f"cb_suc_pred_nn_N{N}_K{K}_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name, thresh=0.4,
              time_limit=time_limit)

    # run algorithm without threshold
    problem_type = f"cb_suc_pred_nn_nt_N{N}_K{K}_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name,
              time_limit=time_limit)

