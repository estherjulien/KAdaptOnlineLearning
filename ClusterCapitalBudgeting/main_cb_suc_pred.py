from Method.SucPred import algorithm as algorithm_s
import pickle
import sys


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    # rf_name = f"N{N_train}_K{K_train}_rf{data}"
    rf_name = str(sys.argv[4])

    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/test_env_cb_N{N}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    problem_type = f"cb_suc_pred_ML[{rf_name}]_T[K{K}_N{N}]"
    success_model_name = f"Data/ResultsSucPred/RFModels/rf_model_cb_suc_pred_data_{rf_name}.joblib"
    time_limit = 0.5 * 60 * 60

    print(f"START SUCCESS PREDICTION METHOD K = {K}, N = {N}, ML = {rf_name}")
    algorithm_s(K, env, att_series,
                success_model_name=success_model_name,
                problem_type=problem_type,
                time_limit=time_limit)
