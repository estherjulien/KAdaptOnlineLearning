from NOT_USED.ShortestPath.ProblemMILPs.functions_2s import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import numpy as np


def predict_subset(K, X, X_scen, weight_model, att_index, state_features):
    num_att = len(X_scen)
    # predict weights
    weight_type = weight_model.predict(state_features.reshape([1, -1]))[0]
    # print(f"Instance {instance}: weights = {weight_type}")
    num_att_type = len(weight_type)
    # return predicted weights into weight vector
    weights = np.zeros(num_att)
    for i in np.arange(num_att_type):
        weights[att_index[i]] = weight_type[i]

    # take weighted euclidean distance from rows of X and X_scen
    X_dist = dict()
    for k in np.arange(K):
        X_dist[k] = np.zeros([len(X[k]), num_att])
        for scen in np.arange(len(X[k])):
            X_dist[k][scen] = weights*(X[k][scen] - X_scen)**2

    # first take WED per scenario and new scenario, then average it per subset
    distance_array = np.zeros(K)
    for k in np.arange(K):
        distance_array[k] = np.mean(np.sqrt(np.sum(X_dist[k], axis=1)))

    order = np.argsort(distance_array)

    return order


def attribute_per_scen(K, scen, env, att_series, tau, theta, x, y, x_static=None, stat_model=None, det_model=None):
    # create list of attributes
    # subset is first value
    sr_att = []

    if "coords" in att_series:
        for i in np.arange(env.xi_dim):
            sr_att.append(scen[i])
    if "slack" in att_series:
        slack = slack_fun(K, scen, env, theta, x, y)
        sr_att += slack
    if "const_to_z_dist" in att_series:
        c_to_z = const_to_z_fun(K, scen, env, theta, x, y)
        sr_att += c_to_z
    if "const_to_const_dist" in att_series:
        c_to_c = const_to_const_fun(K, scen, env, tau)
        for k in np.arange(K):
            sr_att.append(c_to_c[k])

    # based on other problems
    # deterministic problem
    if "obj_det" in att_series or "x_det" in att_series or "y_det" in att_series:
        theta_det, x_det, y_det = scenario_fun_deterministic_update(env, scen, det_model)
    if "obj_det" in att_series:
        sr_att.append(theta_det)
    if "x_det" in att_series:
        sr_att += x_det
    if "y_det" in att_series:
        sr_att += y_det

    # static problem
    if "obj_stat" in att_series or "y_stat" in att_series:
        theta_stat, y_stat = scenario_fun_static_update(env, scen, x_static, stat_model)
    if "obj_stat" in att_series:
        sr_att.append(theta_stat)
    if "y_stat" in att_series:
        sr_att += y_stat

    return np.array(sr_att)


def slack_fun(K, scen, env, theta, x, y):
    slack_final = []
    slack = []
    for k in np.arange(K):
        slack.append(abs(sum((1 + scen[a] / 2) * env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)) - theta))

    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final.append(0)
        else:
            slack_final.append(slack[k]/sum_slack)

    return slack_final


def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi

    dist_final = []
    dist = []
    for k in np.arange(K):
        # define coefficients
        coeff_1 = [1 / 2 * env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)]
        # define constant
        const_1 = sum([env.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env.num_arcs)]) - theta
        # take distance
        dist.append((sum([coeff_1[a] * scen[a] for a in np.arange(env.num_arcs)]) + const_1) / (np.sqrt(sum(coeff_1[a] ** 2 for a in np.arange(env.num_arcs)))))
    sum_dist = sum(dist)
    for k in np.arange(K):
        dist_final.append(dist[k]/sum_dist)

    return dist_final


def const_to_const_fun(K, scen, env, tau):
    # same as 1 stage shortest path
    # cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    # variable x and y
    # take minimum distance from existing scenarios
    cos_final = []

    # CONST 1 (ONLY ONE)
    cos = []
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos.append(0)
            continue
        cos_tmp = []
        scen_vec_0 = [(1 + scen[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
        for xi in tau[k]:
            # CONST 1
            xi_vec_0 = [(1 + xi[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
            similarity = (sum([xi_vec_0[a]*scen_vec_0[a] for a in np.arange(env.num_arcs)])) / \
                         ((np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))) *
                          (np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos.append(max(cos_tmp))

    for k in np.arange(K):
        cos_final.append(cos[k])

    return cos_final


def state_features(K, env, theta, zeta, x, y, tot_nodes, tot_nodes_i, df_att, theta_i, zeta_i, att_index):
    features = []
    # objective
    features.append(theta/theta_i)
    # violation
    features.append(zeta/zeta_i)
    # depth
    features.append(tot_nodes/tot_nodes_i)

    # get state features for subsets >> makes it dependent on K

    return np.array(features)


def weight_labels(K, X, X_scen, k_new, att_index):
    num_att = len(X_scen)
    num_att_type = len(att_index)

    X_dist = dict()
    for k in np.arange(K):
        X_dist[k] = np.zeros([len(X[k]), num_att])
        for scen in np.arange(len(X[k])):
            X_dist[k][scen] = (X[k][scen] - X_scen)**2

    distance_array = np.zeros([K, num_att])
    for k in np.arange(K):
        distance_array[k] = np.mean(X_dist[k], axis=0)

    # group together per attribute type
    att_type = np.zeros([K, num_att_type])
    for i in np.arange(num_att_type):
        for k in np.arange(K):
            att_type[k, i] = np.mean(distance_array[k][att_index[i]])

    # compare attributes of k_new to other subsets
    rel_att_type = np.zeros(num_att_type)
    for i in np.arange(num_att_type):
        if att_type[k_new, i] == 0:
            rel_att_type[i] = 1
        else:
            rel_att_type[i] = np.sum([att_type[k, i]/att_type[k_new, i] for k in np.arange(K) if k != k_new])

    weights = np.zeros(num_att_type)
    weights[np.argmax(rel_att_type)] = 1
    return weights


def init_weights_fun(K, env, att_series, init_weights=None):
    # create list of attributes
    weight_val = []
    att_index = []

    if "coords" in att_series:
        # weights
        try:
            weight_val += [init_weights["xi"] for i in np.arange(env.xi_dim)]
        except:
            weight_val += [1.0 for i in np.arange(env.xi_dim)]
        # index
        att_index.append(np.arange(env.xi_dim))
    if "slack" in att_series:
        # weights
        try:
            weight_val += [init_weights["slack_K"] for k in np.arange(K + 1)]
        except:
            weight_val += [1.0 for k in np.arange(K + 1)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + K))
        except:
            att_index.append(np.arange(K + 1))
    if "const_to_z_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_z_K"] for k in np.arange(K)]
        except:
            weight_val += [1.0 for k in np.arange(K + 1)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + K))
        except:
            att_index.append(np.arange(K + 1))
    if "const_to_const_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_c_K"] for k in np.arange(K)]
        except:
            weight_val += [1.0 for k in np.arange(K)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + K))
        except:
            att_index.append(np.arange(K))
    if "obj_det" in att_series:
        # weights
        try:
            weight_val += [init_weights["obj_det"]]
        except:
            weight_val += [1.0]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "x_det" in att_series:
        # weights
        try:
            weight_val += [init_weights["x_det"] for i in np.arange(env.x_dim)]
        except:
            weight_val += [1.0 for i in np.arange(env.y_dim)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.x_dim))
        except:
            att_index.append(np.arange(env.x_dim))
    if "y_det" in att_series:
        # weights
        try:
            weight_val += [init_weights["y_det"] for i in np.arange(env.y_dim)]
        except:
            weight_val += [1.0 for i in np.arange(env.y_dim)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.y_dim))
        except:
            att_index.append(np.arange(env.y_dim))
    if "obj_stat" in att_series:
        # weights
        try:
            weight_val += [init_weights["obj_stat"]]
        except:
            weight_val += [1.0]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "y_stat" in att_series:
        # weights
        try:
            weight_val += [init_weights["y_stat"] for i in np.arange(env.y_dim)]
        except:
            weight_val += [1.0 for i in np.arange(env.y_dim)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.y_dim))
        except:
            att_index.append(np.arange(env.y_dim))

    weights = np.array(weight_val)
    return weights, att_index


def update_weights_fun(state_features, weight_data, depth=1, width=10, weight_model_name="test"):
    # maybe try different architectures? only report the best one
    n_features = np.shape(state_features)[1]
    n_labels = np.shape(weight_data)[1]

    try:
        weight_model = load_model(weight_model_name)
        # increase learning rate as model continues?
        opt = SGD(learning_rate=0.001, momentum=0.9)
        # compile the model
        weight_model.compile(optimizer=opt, loss='categorical_crossentropy')
        # fit the model on new data
        weight_model.fit(state_features, weight_data, epochs=5, batch_size=32, verbose=0)
    except:
        weight_model = Sequential()
        # first layer after input layer
        weight_model.add(Dense(width, activation='relu', input_dim=n_features))
        for d in np.arange(depth-1):
            weight_model.add(Dense(width, activation='relu'))
        # output layer
        weight_model.add(Dense(n_labels, activation='softmax'))
        # define the optimization algorithm
        opt = SGD(learning_rate=0.01, momentum=0.9)
        # compile the model
        weight_model.compile(optimizer=opt, loss='categorical_crossentropy')
        # fit the model on data
        weight_model.fit(state_features, weight_data, epochs=5, batch_size=32, verbose=0)

    # save weight model
    weight_model.save(weight_model_name)

