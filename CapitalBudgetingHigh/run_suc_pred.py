from Method.SucPred import algorithm
import pickle
import sys


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    max_level = int(sys.argv[4])
    thresh = int(sys.argv[5])

    # load environment
    with open(f"CapitalBudgetingHigh/Data/Instances/inst_results/cb_test_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgetingHigh/Data/Models/rf_class_cb_p5_N10_K4_ct70_all.joblib"

    # run algorithm with threshold
    if thresh:
        problem_type = f"cb_suc_pred_rf_p5_N{N}_K{K}_L{max_level}"
        algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name, thresh=0.1)
    else:
        problem_type = f"cb_suc_pred_rf_p5_nt_N{N}_K{K}_L{max_level}"
        algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

