from Method.SucPredFromStart import algorithm
import pickle
import sys

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    K_ML = int(sys.argv[4])
    max_level = int(sys.argv[5])

    if len(sys.argv) == 7:
        num_insts = int(sys.argv[6])
    else:
        num_insts = 100

    # load environment
    with open(f"CapitalBudgeting/Data/Instances/inst_results/cb_env_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    success_model_name = f"CapitalBudgeting/Data/Models/rf_class_cb_N10_K{K_ML}_I{num_insts}_ct70_bal.joblib"

    # run algorithm with threshold
    problem_type = f"cb_suc_pred_C[FromStartRandom]_ML[N10_K{K_ML}_I{num_insts}]_T[N{N}_K{K}]_L{max_level}"
    algorithm(K, env, max_level=max_level, problem_type=problem_type, success_model_name=success_model_name)

