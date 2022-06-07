from NOT_USED.ShortestPath.ProblemMILPs.functions import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import numpy as np
import joblib


def predict_subset(K, X, X_scen, weight_model, att_index, state_features):
    num_att = len(X_scen)
    # predict weights
    try:
        weight_type = weight_model.predict_proba(state_features.reshape([1, -1]))[0]
    except: # then it has to be a nn
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
    sr_att_k = {k: [] for k in np.arange(K)}

    if "coords" in att_series:
        for i in np.arange(env.xi_dim):
            sr_att.append(scen[i])

    # based on other problems
    # deterministic problem
    if "obj_det" in att_series or "x_det" in att_series or "y_det" in att_series:
        theta_det, y_det = scenario_fun_deterministic_update(env, scen, det_model)
    if "obj_det" in att_series:
        sr_att.append(theta_det / theta)
    if "y_det" in att_series:
        sr_att += y_det

    # static problem doesn't exist for shortest path

    # k dependent
    if "slack" in att_series:
        slack = slack_fun(K, scen, env, theta, x, y)
        for k in np.arange(K):
            sr_att_k[k] += slack[k]
    if "const_to_z_dist" in att_series:
        c_to_z = const_to_z_fun(K, scen, env, theta, x, y)
        for k in np.arange(K):
            sr_att_k[k] += c_to_z[k]
    if "const_to_const_dist" in att_series:
        c_to_c = const_to_const_fun(K, scen, env, tau)
        for k in np.arange(K):
            sr_att_k[k] += c_to_c[k]
    return sr_att, sr_att_k


def slack_fun(K, scen, env, theta, x, y):
    slack = []
    for k in np.arange(K):
        slack.append(abs(sum((1 + scen[a] / 2) * env.distances_array[a] * y[k][a] for a in np.arange(env.num_arcs)) - theta))

    slack_final = {k: [] for k in np.arange(K)}
    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final[k].append(0)
        else:
            slack_final[k].append(slack[k]/sum_slack)

    return slack_final


def const_to_z_fun(K, scen, env, theta, x, y):
    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist = []
    for k in np.arange(K):
        # CONST 1
        # define coefficients
        coeff_1 = [1 / 2 * env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]
        # define constant
        const_1 = sum([env.distances_array[a] * (y[k][a]) for a in np.arange(env.num_arcs)]) - theta
        # take distance
        dist.append((sum([coeff_1[a] * scen[a] for a in np.arange(env.num_arcs)]) + const_1) / (np.sqrt(sum(coeff_1[a] ** 2 for a in np.arange(env.num_arcs)))))

    dist_final = {k: [] for k in np.arange(K)}
    sum_dist = sum(dist)
    for k in np.arange(K):
        if sum_dist == 0:
            dist_final[k].append(1)
        else:
            dist_final[k].append(dist[k]/sum_dist)

    return dist_final


def const_to_const_fun(K, scen, env, tau):
    # cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    # variable x and y
    # take minimum distance from existing scenarios
    cos = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos[k].append(0)
            continue
        cos_tmp = []
        scen_vec_0 = [(1 + scen[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
        for xi in tau[k]:
            # CONST
            xi_vec_0 = [(1 + xi[a]/2)*env.distances_array[a] for a in np.arange(env.num_arcs)]
            similarity = (sum([xi_vec_0[a]*scen_vec_0[a] for a in np.arange(env.num_arcs)])) / \
                         ((np.sqrt(sum(xi_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))) *
                          (np.sqrt(sum(scen_vec_0[a] ** 2 for a in np.arange(env.num_arcs)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k].append(max(cos_tmp))

    return {k: list(np.nan_to_num(cos[k], nan=0.0)) for k in np.arange(K)}


def state_features(K, env, theta, zeta, x, y, tot_nodes, tot_nodes_i, df_att, theta_i, zeta_i, att_index):
    features = []
    # objective
    features.append(theta/theta_i)
    # violation
    features.append(zeta/zeta_i)
    # depth
    features.append(tot_nodes/tot_nodes_i)

    return np.array(features)


def weight_labels(K, X, scen_att, scen_att_k, k_new, att_index):
    X_scen = np.array(scen_att + scen_att_k[k_new]).reshape([1, -1])
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
    if "slack" in att_series:
        # weights
        try:
            weight_val += [init_weights["slack_K"]]
        except:
            weight_val += [1.0]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "const_to_z_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_z_K"]]
        except:
            weight_val += [1.0]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "const_to_const_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_c_K"]]
        except:
            weight_val += [1.0]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))

    weights = np.array(weight_val)
    return weights, att_index


def update_weights_fun_nn(state_features, weight_data, depth=1, width=10, weight_model_name="test"):
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


def update_weights_fun(state_features, weight_data, depth=1, width=10, weight_model_name="test"):
    X_train, X_val, Y_train, Y_val = train_test_split(state_features, weight_data, test_size=0.1)

    # random forest
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)

    print(f"\nRF validation accuracy = {score_rf} \n")
    # save
    joblib.dump(rf, weight_model_name)

