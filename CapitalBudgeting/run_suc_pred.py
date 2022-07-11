from Method.SucPred import algorithm
import pickle
import sys

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    K_ML = int(sys.argv[4])
    max_level = int(sys.argv[5])
    minutes = int(sys.argv[6])
    ct = int(sys.argv[7])
    nodes = int(sys.argv[8])

    # load environment
    with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgeting/Data/Models/rf_class_cb_N10_K{K_ML}_min{minutes}_nodes{nodes}_ct{ct}_bal.joblib"

    # run algorithm with threshold
    problem_type = f"cb_suc_pred_ML[N10_K{K_ML}_m{minutes}_nodes{nodes}_ct{ct}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

