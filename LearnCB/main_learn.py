from learn_models import *

import pickle

N = 10

for K in [4]:
    print(f"K = {K}")
    with open(f"../ClusterResults_24-01-22/Summarized/input_train_N{N}_K{K}.pickle", "rb") as handle:
        X_train = pickle.load(handle)

    with open(f"../ClusterResults_24-01-22/Summarized/output_train_N{N}_K{K}.pickle", "rb") as handle:
        Y_train = pickle.load(handle)

    with open(f"../ClusterResults_24-01-22/Summarized/input_validation_N{N}_K{K}.pickle", "rb") as handle:
        X_val = pickle.load(handle)

    with open(f"../ClusterResults_24-01-22/Summarized/output_validation_N{N}_K{K}.pickle", "rb") as handle:
        Y_val = pickle.load(handle)

    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    features = ["theta_node", "theta_pre", "zeta_node", "zeta_pre", "depth", *att_series]

    # DIFFERENT MODELS
    # train_suc_pred_xgboost_class(X_train, Y_train, X_val, Y_val, problem_type=problem_type)
    # train_suc_pred_rf_class(X_train, Y_train, X_val, Y_val, features, problem_type=problem_type)
    for bal in [True, False]:
        if bal:
            problem_type = f"cb_N{N}_K{K}_balanced_all"
        else:
            problem_type = f"cb_N{N}_K{K}_all"
        for w in [10, 20, 50, 100, 200]:
            for d in [1, 2, 5, 10, 20, 50]:
                print(f"Balanced = {bal}, WIDTH = {w}, DEPTH = {d}")

                train_suc_pred_nn_class(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=w, depth=d, balanced=bal)

    # train_suc_pred_rf_regr(X_train, Y_train, X_val, Y_val, features, problem_type=problem_type)
    # train_suc_pred_nn_regr(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=5)

