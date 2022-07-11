from learn_models import *

import pickle

# att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
#               "const_to_const_dist"]
att_series = ["coords", "obj_det", "y_det", "slack", "const_to_z_dist",
              "const_to_const_dist"]
features = ["theta_node", "theta_pre", "zeta_node", "zeta_pre", "depth", *att_series]

N = 20

for ct in [0.05]:
    for K in [2, 3, 4, 5, 6]:
        for minutes in [5, 10, 15, 20]:
            for nodes in [1, 2, 5, 10]:
                try:
                    with open(f"../SPData/train_data_sp_sphere_N{N}_K{K}_min{minutes}_nodes{nodes}.pickle", "rb") as handle:
                        res = pickle.load(handle)
                    X = res["X"]
                    Y = res["Y"]
                except FileNotFoundError:
                    continue

                # CLASSIFICATION
                problem_type = f"sp_sphere_N{N}_K{K}_min{minutes}_nodes{nodes}_ct{int(ct*100)}"
                for balanced in [True]:
                    try:
                        with open(f"../SPModels/Info/rf_class_info_sp_sphere_N20_K{K}_min{minutes}_"
                                  f"nodes{nodes}_ct{ct}_bal.pickle", "rb") as handle:
                            a = pickle.load(handle)
                        continue
                    except:
                        pass

                    print(f"nodes = {nodes}, minutes = {minutes}, K = {K}, N = {N}, ct = {ct}, balanced = {balanced}")
                    if balanced:
                        problem_type += "_bal"
                    df_X = pd.DataFrame(X)
                    df_Y = pd.Series(Y)
                    train_suc_pred_rf_class(df_X, df_Y, features, problem_type=problem_type, save_map="SPModels", class_thresh=ct,
                                            balanced=balanced)

# REGRESSION
# print("Regression")
# problem_type = f"sp_p{class_perc}_N{N}_K{K}_all"
# train_suc_pred_rf_regr(X, Y, features, problem_type=problem_type, save_map=Results_day)

# train_suc_pred_nn_regr(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=5)
# train_suc_pred_nn_class(X_train, Y_train, X_val, Y_val, problem_type=problem_type, width=100, depth=2,
# balanced=bal, save_map=Results_day)
