from Strategy import algorithm
import pickle
import sys

if __name__ == "__main__":
    inst_num = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    time_limit = int(sys.argv[4])
    K_train = int(sys.argv[5])
    N_train = int(sys.argv[6])
    rf_num = int(sys.argv[7])

    with open(f"ClusterDataGenShortestPath/Instances/sp_env_N{N}_{inst_num}.pickle", "rb") as handle:
        env = pickle.load(handle)

    att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist", "const_to_const_dist"]

    with open(f"ClusterDataGenShortestPath/RFModels/train_combinations_sp_suc_pred_data_K{K_train}_N{N_train}.pickle", "rb") as handle:
        comb = pickle.load(handle)[rf_num]

    problem_type = f"sp_strategy_K{K}_N{N}_rf{comb}"
    success_model_name = f"ClusterDataGenShortestPath/RFModels/rf_model_sp_suc_pred_data_K{K_train}_N{N_train}_rf{comb}.joblib"

    algorithm(K, env, att_series, success_model_name, time_limit=time_limit, problem_type=problem_type)
