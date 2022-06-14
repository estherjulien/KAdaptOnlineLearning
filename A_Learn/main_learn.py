from learn_models import *

import pickle

att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]
# att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist",
#               "const_to_const_dist"]
features = ["theta_node", "theta_pre", "zeta_node", "zeta_pre", "depth", *att_series]

N = 10
for K, I in [(6, 2000)]:
    try:
        with open(f"../CBData/train_data_cb_N10_K{K}_I{I}.pickle", "rb") as handle:
            res = pickle.load(handle)
        X = res["X"]
        Y = res["Y"]
    except FileNotFoundError:
        continue

    # CLASSIFICATION
    for ct in [0.3, 0.7]:
        if ct:
            problem_type = f"cb_N{N}_K{K}_I{I}_ct{int(ct*100)}"
        else:
            problem_type = f"cb_N{N}_K{K}_I{I}"
        for balanced in [False, True]:
            print(f"I = {I}, K = {K}, N = {N}, ct = {ct}, balanced = {balanced}")
            if balanced:
                problem_type += "_bal"
            df_X = pd.DataFrame(X)
            df_Y = pd.Series(Y)
            train_suc_pred_rf_class(df_X, df_Y, features, problem_type=problem_type, save_map="CBModels", class_thresh=ct,
                                    balanced=balanced)
# REGRESSION
# print("Regression")
# problem_type = f"sp_p{class_perc}_N{N}_K{K}_all"
# train_suc_pred_rf_regr(X, Y, features, problem_type=problem_type, save_map=Results_day)

# train_suc_pred_nn_regr(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=5)
# train_suc_pred_nn_class(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=2,
# balanced=bal, save_map=Results_day)
