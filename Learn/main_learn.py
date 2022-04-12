from learn_models import *

import pickle

N = 50

class_perc = 5
Results_day = f"ShortestPathResults/Data"
for I in [100, 200, 500]:
    for K in [2, 3, 4]:
        with open(f"../{Results_day}/Results/TrainData/input_sp_p{class_perc}_N{N}_K{K}_I{I}.pickle", "rb") as handle:
            X = pickle.load(handle)

        with open(f"../{Results_day}/Results/TrainData/output_sp_p{class_perc}_N{N}_K{K}_I{I}.pickle", "rb") as handle:
            Y = pickle.load(handle)

        # att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
        #               "const_to_const_dist"]
        att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist",
                      "const_to_const_dist"]

        features = ["theta_node", "theta_pre", "zeta_node", "zeta_pre", "depth", *att_series]

        # DIFFERENT MODELS

        # CLASSIFICATION
        for ct in [None, 0.3, 0.7]:
            print(f"I = {I}, K = {K}, ct = {ct}")
            if ct:
                problem_type = f"sp_p{class_perc}_N{N}_K{K}_I{I}_ct{int(ct*100)}_all"
            else:
                problem_type = f"sp_p{class_perc}_N{N}_K{K}_I{I}_all"
            train_suc_pred_rf_class(X, Y, features, problem_type=problem_type, save_map=Results_day, class_thresh=ct)
    # REGRESSION
    # print("Regression")
    # problem_type = f"sp_p{class_perc}_N{N}_K{K}_all"
    # train_suc_pred_rf_regr(X, Y, features, problem_type=problem_type, save_map=Results_day)

    # train_suc_pred_nn_regr(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=5)
    # train_suc_pred_nn_class(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=2,
    # balanced=bal, save_map=Results_day)
