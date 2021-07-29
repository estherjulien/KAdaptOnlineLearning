import pandas as pd
import numpy as np


def avg_dist_on_attributes(X, X_scen, weights=[]):
    K = len(X[("subset", 0)].unique())

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
        for k in np.arange(K):
            sr_att[("slack", f"K{k}")] = slack[k]
    if "const_to_z_dist" in att_series:
        c_to_z = const_to_z_fun(K, scen, env, theta, x, y)
        for k in np.arange(K):
            sr_att[("c_to_z", f"K{k}")] = c_to_z[k]
    if "const_to_const_dist" in att_series:
        c_to_c = const_to_const_fun(K, scen, env, tau)
        for k in np.arange(K):
            sr_att[("c_to_c", f"K{k}")] = c_to_c[k]

    return sr_att


def slack_fun(K, scen, env, theta, x, y):
    slack = dict()
    for k in np.arange(K):
        slack[k] = sum((1 + scen[a] / 2) * env.distances_array[a] * y[k][a] for a in np.arange(env.num_arcs)) - theta

    slack_final = dict()
    sum_slack = sum(list(slack.values()))
    for k in np.arange(K):
        slack_final[k] = slack[k]/sum_slack

    return slack_final


def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist = dict()
    for k in np.arange(K):
        # CONST 1
        # define coefficients
        coeff_1 = [1 / 2 * env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]
        # define constant
        const_1 = sum([env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]) - theta
        # take distance
        dist[k] = (sum([coeff_1[a] * scen[a] for a in np.arange(env.num_arcs)]) + const_1) / \
                  (np.sqrt(sum(coeff_1[a] ** 2 for a in np.arange(env.num_arcs))))

    dist_final = dict()
    sum_dist = sum(list(dist.values()))
    for k in np.arange(K):
        dist_final[k] = dist[k]/sum_dist

    return dist_final


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

    cos_final = dict()
    sum_cos = sum(list(cos.values()))
    for k in np.arange(K):
        cos_final[k] = cos[k]/sum_cos

    return cos_final


def init_weights_fun(K, env, att_series, init_weights=None):
    # create list of attributes
    weight_id = []
    weight_val = []

    if "coords" in att_series:
        try:
            weight_val += [init_weights["xi"] for i in np.arange(env.xi_dim)]
        except:
            weight_val += [1.0 for i in np.arange(env.xi_dim)]
        weight_id += [("xi", i) for i in np.arange(env.xi_dim)]
    if "slack" in att_series:
        try:
            weight_val += [init_weights["slack_K"] for k in np.arange(K)]
        except:
            weight_val += [1.0 for k in np.arange(K)]
        weight_id += [("slack", f"K{k}") for k in np.arange(K)]
    if "const_to_z_dist" in att_series:
        try:
            weight_val += [init_weights["c_to_z_K"] for k in np.arange(K)]
        except:
            weight_val += [1.0 for k in np.arange(K)]
        weight_id += [("c_to_z", f"K{k}") for k in np.arange(K)]
    if "const_to_const_dist" in att_series:
        try:
            weight_val += [init_weights["c_to_c_K"] for k in np.arange(K)]
        except:
            weight_val += [1.0 for k in np.arange(K)]
        weight_id += [("c_to_c", f"K{k}") for k in np.arange(K)]

    index = pd.MultiIndex.from_tuples(weight_id, names=["att_type", "att"])
    weights = pd.Series(weight_val, index=index)
    N_set_att_init = pd.DataFrame(columns=index)
    return weights, N_set_att_init


def update_weights(K, env, init_weights, df_rand, df_att, lr_w, att_series):
    # average values of df_att
    X_att = pd.DataFrame(columns=df_att.columns, dtype=np.float32)

    for k in np.arange(K):
        X_att.loc[k] = df_att[df_att[("subset", 0)] == k].var()
        # delete att and make it into one
        if "slack" in att_series:
            X_att.loc[k, ("slack", "K")] = X_att.loc[k, ("slack", f"K{k}")]
        if "const_to_z_dist" in att_series:
            X_att.loc[k, ("c_to_z", "K")] = X_att.loc[k, ("c_to_z", f"K{k}")]
        if "const_to_const_dist" in att_series:
            X_att.loc[k, ("c_to_c", "K")] = X_att.loc[k, ("c_to_c", f"K{k}")]

    # difference
    X_att = X_att.drop(("subset", 0), axis=1).fillna(0).mean()

    # average values of df_rand
    X_rand = pd.DataFrame(columns=df_rand.columns, dtype=np.float32)
    for k in np.arange(K):
        X_rand.loc[k] = df_rand[df_rand[("subset", 0)] == k].var()
        # delete att and make it into one
        if "slack" in att_series:
            X_rand.loc[k, ("slack", "K")] = X_rand.loc[k, ("slack", f"K{k}")]
        if "const_to_z_dist" in att_series:
            X_rand.loc[k, ("c_to_z", "K")] = X_rand.loc[k, ("c_to_z", f"K{k}")]
        if "const_to_const_dist" in att_series:
            X_rand.loc[k, ("c_to_c", "K")] = X_rand.loc[k, ("c_to_c", f"K{k}")]
    # difference
    X_rand = X_rand.drop(("subset", 0), axis=1).fillna(0).mean()

    # delete other K dependent attribute
    try:
        X_att = X_att.drop([("slack", f"K{k}") for k in np.arange(K)])
        X_rand = X_rand.drop([("slack", f"K{k}") for k in np.arange(K)])
    except:
        pass

    try:
        X_att = X_att.drop([("c_to_z", f"K{k}") for k in np.arange(K)])
        X_rand = X_rand.drop([("c_to_z", f"K{k}") for k in np.arange(K)])
    except:
        pass

    try:
        X_att = X_att.drop([("c_to_c", f"K{k}") for k in np.arange(K)])
        X_rand = X_rand.drop([("c_to_c", f"K{k}") for k in np.arange(K)])
    except:
        pass

    # update weights
    weights = pd.Series(0.0, index=init_weights.index)
    for att_type, att_all in X_att.groupby(level=0):
        # only add coords together (at least for MP)
        if att_type == "xi":
            weight_change = 0
            for att in att_all.index:
                weight_change += (X_att[att] * (init_weights[att] * X_att[att] - X_rand[att]))
            for att in att_all.index:
                weights[att] = init_weights[att] - lr_w * weight_change
        else:
            for att in att_all.index:
                for k in np.arange(K):
                    k_att = (att[0], att[1] + f"{k}")
                    weights[k_att] = init_weights[k_att] - lr_w * (X_att[att] * (init_weights[k_att] * X_att[att] - X_rand[att]))

    performance = ((X_att - X_rand) ** 2).mean()
    return weights, performance


def final_weights(att_series, init_weights):
    # create list of attributes
    weight_id = []
    weight_val = []

    if "coords" in att_series:
        weight_id.append("xi")
        # all are the same..
        weight_val.append(init_weights[("xi", 0)])
    if "slack" in att_series:
        try:
            weight_val.append(init_weights[("slack", 0)])
            weight_id.append("slack_0")
        except:
            pass
        try:
            weight_val.append(init_weights[("slack", f"K0")])
            weight_id.append("slack_K")
        except:
            pass
    if "const_to_z_dist" in att_series:
        try:
            weight_val.append(init_weights[("c_to_z", 0)])
            weight_id.append("c_to_z_0")
        except:
            pass
        try:
            weight_val.append(init_weights[("c_to_z", f"K0")])
            weight_id.append("c_to_z_K")
        except:
            pass
    if "const_to_const_dist" in att_series:
        weight_id.append("c_to_c_K")
        weight_val.append(init_weights[("c_to_c", f"K0")])

    weights = pd.Series(weight_val, index=weight_id)
    return weights
