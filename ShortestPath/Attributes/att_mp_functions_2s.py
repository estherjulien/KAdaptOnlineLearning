import pandas as pd
import numpy as np


def avg_dist_on_attributes(df_att, scen_att, att_series, xi_dim, weights=[]):
    K = len(df_att[("subset", 0)].unique())
    if "coords" in att_series:
        X = df_att
        X_scen = scen_att
    else:
        X = df_att.drop([*[("xi", i) for i in np.arange(xi_dim)]], axis=1)
        X_scen = scen_att.drop([*[("xi", i) for i in np.arange(xi_dim)]])

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


def attribute_per_scen(K, scen, env, att_series, tau, theta, x, y):
    # create list of attributes
    sr_att = pd.Series({("xi", i): scen[i] for i in np.arange(len(scen))})
    sr_att[("subset", 0)] = -1

    if "slack" in att_series:
        slack = slack_fun(K, scen, env, theta, x, y)
        for i in np.arange(K+1):
            sr_att[("slack", i)] = slack[i]
    if "const_to_z_dist" in att_series:
        c_to_z = const_to_z_fun(K, scen, env, theta, x, y)
        for i in np.arange(K+1):
            sr_att[("c_to_z", i)] = c_to_z[i]
    if "const_to_const_dist" in att_series:
        c_to_c = const_to_const_fun(K, scen, env, tau)
        for k in np.arange(K):
            sr_att[("c_to_c", k)] = c_to_c[k]

    return sr_att


def slack_fun(K, scen, env, theta, x, y):
    slack = dict()
    slack[0] = sum((1 + scen[a] / 2) * env.distances_array[a] * x[a] for a in np.arange(env.num_arcs)) \
               - env.max_first_stage
    for k in np.arange(K):
        slack[k+1] = sum((1 + scen[a] / 2) * env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)) - theta
    return slack


def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist = dict()
    # CONST 0
    # define coefficients
    coeff_0 = [1/2*env.distances_array[a]*x[a] for a in np.arange(env.num_arcs)]
    # define constant
    const_0 = sum([env.distances_array[a]*x[a] for a in np.arange(env.num_arcs)]) - env.max_first_stage
    # take distance
    dist[0] = (sum([coeff_0[a]*scen[a] for a in np.arange(env.num_arcs)]) + const_0) / \
              (np.sqrt(sum(coeff_0[a]**2 for a in np.arange(env.num_arcs))))

    for k in np.arange(K):
        # CONST 1
        # define coefficients
        coeff_1 = [1 / 2 * env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)]
        # define constant
        const_1 = sum([env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)]) - theta
        # take distance
        dist[k+1] = (sum([coeff_1[a] * scen[a] for a in np.arange(env.num_arcs)]) + const_1) / \
                  (np.sqrt(sum(coeff_1[a] ** 2 for a in np.arange(env.num_arcs))))

    return dist


def const_to_const_fun(K, scen, env, tau):
    # cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    # variable x and y
    # take minimum distance from existing scenarios
    cos = dict()
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos[k] = 0
            continue
        cos_tmp = []
        scen_vec_0 = [(1 + scen[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
        for xi in tau[k]:
            # CONST
            xi_vec_0 = [(1 + xi[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
            similarity = (sum([xi_vec_0[a]*scen_vec_0[a] for a in np.arange(env.num_arcs)])) / \
                         ((np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))) *
                          (np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k] = max(cos_tmp)

    return cos


def init_weights(K, env, att_series):
    # create list of attributes
    weight_id = []
    weight_id += [("xi", i) for i in np.arange(env.xi_dim)]

    if "slack" in att_series:
        weight_id += [("slack", i) for i in np.arange(K+1)]
    if "const_to_z_dist" in att_series:
        weight_id += [("c_to_z", i) for i in np.arange(K+1)]
    if "const_to_const_dist" in att_series:
        weight_id += [("c_to_c", k) for k in np.arange(K)]

    index = pd.MultiIndex.from_tuples(weight_id, names=["att_type", "att"])
    weights = pd.Series(1.0, index=index)
    N_set_att_init = pd.DataFrame(columns=index)
    return weights, N_set_att_init
