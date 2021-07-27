import pandas as pd
import numpy as np
from ShortestPath.ProblemMILPs.functions_2s import *
# from sklearn.neighbors import KNeighborsClassifier
import copy


# def knn_on_attributes(df_att, scen_att, att_series, xi_dim, knn_k=3):
#     leftover_subsets = list(df_att.subset.unique())
#     order = []
#     while len(leftover_subsets):
#         df_att_tmp = df_att.loc[df_att["subset"].isin(leftover_subsets)]
#         neigh = KNeighborsClassifier(n_neighbors=min(knn_k, len(df_att_tmp)), weights="distance")
#         # everything except for label
#         if "coords" in att_series:
#             X = df_att_tmp.drop("subset", axis=1)
#             X_scen = scen_att.drop("subset")
#         else:
#             X = df_att_tmp.drop([*[("xi", f"xi{i}") for i in np.arange(xi_dim)], "subset"], axis=1)
#             X_scen = scen_att.drop([*[("xi", f"xi{i}") for i in np.arange(xi_dim)], "subset"])
#         neigh.fit(X, df_att_tmp["subset"])
#         subset = neigh.predict(np.array(X_scen).reshape(1, len(X_scen)))[0]
#         order.append(int(subset))
#         leftover_subsets.remove(subset)
#     return order


def avg_dist_on_attributes(df_att, scen_att, att_series, xi_dim, weights=[]):
    K = len(df_att[("subset", 0)].unique())
    if "coords" in att_series:
        X = df_att
        X_scen = scen_att
    else:
        X = df_att.drop([*[("xi", f"xi{i}") for i in np.arange(xi_dim)]], axis=1)
        X_scen = scen_att.drop([*[("xi", f"xi{i}") for i in np.arange(xi_dim)]])

    # average values of X
    X_avg = pd.DataFrame(index=np.arange(K), columns=X.columns, dtype=np.float32)
    for k in np.arange(K):
        X_avg.loc[k] = X[X[("subset", 0)] == k].mean()
    X_avg = X_avg.drop(("subset", 0), axis=1)
    X_scen = X_scen.drop(("subset", 0))

    # take weighted euclidean distance from rows of X and X_scen
    distance_array = np.zeros(K)
    if len(weights) == 0:
        weights = pd.Series(1, X_scen.index)
    for col in X_avg.columns:
        distance_array += weights[col]*(X_avg[col] - X_scen[col])**2

    order = np.array(distance_array.sort_values(ascending=False).index)

    return order


def attribute_per_scen(scen, env, att_series, scen_nom_model=None, scen_stat_model=None, x_static=None):
    # create list of attributes
    sr_att = pd.Series({("xi", i): scen[i] for i in np.arange(len(scen))})
    sr_att[("subset", 0)] = -1
    # only do computation for static and/or nominal once
    if "static" in att_series:
        theta_static, y_static = scenario_fun_static_update(env, scen, x_static, scen_stat_model)
    if "nominal" in att_series:
        theta_nominal, x_nominal, y_nominal = scenario_fun_nominal_update(env, scen, scen_nom_model)

    if "static_obj" in att_series:
        sr_att[("static_obj", 0)] = theta_static
    if "static_y" in att_series:
        for i in np.arange(env.y_dim):
            sr_att[("static_y", i)] = y_static[i]
    if "nominal_obj" in att_series:
        sr_att[("nominal_obj", 0)] = theta_nominal
    if "nominal_x" in att_series:
        for i in np.arange(env.x_dim):
            sr_att[("nominal_x", i)] = x_nominal[i]
    if "nominal_y" in att_series:
        for i in np.arange(env.xi_dim):
            sr_att[("nominal_y", i)] = y_nominal[i]
    return sr_att


def init_weights(env, att_series):
    # create list of attributes
    weight_id = []
    weight_id += [("xi", i) for i in np.arange(env.xi_dim)]
    # only do computation for static and/or nominal once
    if "static" in att_series:
        x_static = static_solution_rc(env)
    else:
        x_static = None

    if "static_obj" in att_series:
        weight_id += [("static_obj", 0)]
    if "static_y" in att_series:
        weight_id += [("static_y", i) for i in np.arange(env.y_dim)]
    if "nominal_obj" in att_series:
        weight_id += [("nominal_obj", 0)]
    if "nominal_x" in att_series:
        weight_id += [("nominal_x", i) for i in np.arange(env.x_dim)]
    if "nominal_y" in att_series:
        weight_id += [("nominal_y", i) for i in np.arange(env.y_dim)]

    index = pd.MultiIndex.from_tuples(weight_id, names=["att_type", "att"])
    weights = pd.Series(1.0, index=index)
    return weights, x_static