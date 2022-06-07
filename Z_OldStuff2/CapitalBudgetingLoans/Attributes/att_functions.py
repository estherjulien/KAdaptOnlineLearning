from NOT_USED.CapitalBudgetingLoans.ProblemMILPs.functions import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

import numpy as np


def predict_subset(K, X, X_scen, weight_model, init_weights, att_index, state_features):
    num_att = len(X_scen)
    if init_weights is None:
        # predict weights
        weight_type = weight_model.predict(state_features.reshape([1, -1]))[0]
        # print(f"Instance {instance}: weights = {weight_type}")
        num_att_type = len(weight_type)
        # return predicted weights into weight vector
        weights = np.zeros(num_att)
        for i in np.arange(num_att_type):
            weights[att_index[i]] = weight_type[i]
    else:
        weights = init_weights
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
        theta_det, x_det, y_det = scenario_fun_deterministic_update(env, scen, det_model)
    if "obj_det" in att_series:
        sr_att.append(theta_det / theta)
    if "x_det" in att_series:
        sr_att += [x_det[0] / env.max_loan]
        sr_att += x_det[1]
    if "y_det" in att_series:
        sr_att += [y_det[0] / env.max_loan]
        sr_att += y_det[1]

    # static problem
    if "obj_stat" in att_series or "y_stat" in att_series:
        theta_stat, y_stat = scenario_fun_static_update(env, scen, x_static, stat_model)
    if "obj_stat" in att_series:
        sr_att.append(theta_stat / theta)
    if "y_stat" in att_series:
        sr_att += [y_stat[0] / env.max_loan]
        sr_att += y_stat[1]

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


def slack_fun(K, scen, env, theta, x_input, y_input):
    x_0, x = x_input
    y_0, y = y_input

    projects = env.projects
    N = env.N

    slack_final = {k: [] for k in np.arange(K)}

    # CONST 1
    slack = []
    for k in np.arange(K):
        slack.append(abs((sum(rev_fun(projects[p], scen) * (x[p] + env.kappa * y[k][p]) for p in np.arange(N)))
                         - env.lam * (x_0 + env.mu * y_0[k]) + theta))
    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final[k].append(0)
        else:
            slack_final[k].append(slack[k] / sum_slack)

    # CONST 2
    slack = []
    for k in np.arange(K):
        slack.append(abs((sum(cost_fun(projects[p], scen) * (x[p] + y[k][p]) for p in np.arange(N)))
                         - (env.budget + x_0 + y_0[k])))
    sum_slack = sum(slack)
    for k in np.arange(K):
        if sum_slack == 0:
            slack_final[k].append(0)
        else:
            slack_final[k].append(slack[k] / sum_slack)

    return slack_final


def const_to_z_fun(K, scen, env, theta, x_input, y_input):
    x_0, x = x_input
    y_0, y = y_input

    projects = env.projects
    N = env.N

    # point-to-plane: https://mathworld.wolfram.com/Point-PlaneDistance.html
    # variable xi
    dist_final = {k: [] for k in np.arange(K)}

    # CONST 1
    dist = []
    for k in np.arange(K):
        # define coefficients
        coeff = [1/2*sum(projects[p].rev_nom*projects[p].psi[j]*(x[p] + env.kappa*y[k][p]) for p in np.arange(N))
                 for j in np.arange(env.xi_dim)]
        # define constant
        const = sum(projects[p].rev_nom*(x[p] + env.kappa*y[k][p]) for p in np.arange(N)) - env.lam*(x_0 + env.mu*y_0[k]) - theta
        # take distance
        dist.append(abs((sum([coeff[j] * scen[j] for j in np.arange(env.xi_dim)]) + const) / (np.sqrt(sum(coeff[j] ** 2 for j in np.arange(env.xi_dim))))))

    sum_dist = sum(dist)
    for k in np.arange(K):
        dist_final[k].append(dist[k]/sum_dist)

    # CONST 2
    dist = []
    for k in np.arange(K):
        # define coefficients
        coeff = [1/2*sum(projects[p].cost_nom*projects[p].phi[j]*(x[p] + y[k][p]) for p in np.arange(N))
                 for j in np.arange(env.xi_dim)]
        # define constant
        const = sum(projects[p].cost_nom*(x[p] + y[k][p]) for p in np.arange(N)) - env.budget - x_0 - y_0[k]
        # take distance
        dist.append(abs((sum([coeff[j] * scen[j] for j in np.arange(env.xi_dim)]) + const) / (np.sqrt(sum(coeff[j] ** 2 for j in np.arange(env.xi_dim))))))

    sum_dist = sum(dist)
    for k in np.arange(K):
        dist_final[k].append(dist[k] / sum_dist)

    return dist_final


def const_to_const_fun(K, scen, env, tau):
    projects = env.projects
    N = env.N
    # cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
    # variable x and y
    # take minimum distance from existing scenarios, so maximum similarity
    cos = {k: [] for k in np.arange(K)}

    # CONST 1
    scen_vec_0 = [sum(scen[j]*projects[p].psi[j]*projects[p].rev_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos[k].append(0)
            continue
        cos_tmp = []
        for xi in tau[k]:
            xi_vec_0 = [sum(xi[j]*projects[p].psi[j]*projects[p].rev_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
            try:
                similarity = (sum([xi_vec_0[j]*scen_vec_0[j] for j in np.arange(env.xi_dim)])) / \
                             ((np.sqrt(sum(xi_vec_0[j] ** 2 for j in np.arange(env.xi_dim)))) *
                              (np.sqrt(sum(xi_vec_0[j] ** 2 for j in np.arange(env.xi_dim)))))
            except RuntimeWarning:
                similarity = 0
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k].append(max(cos_tmp))

    # CONST 2
    scen_vec_0 = [sum(scen[j]*projects[p].phi[j]*projects[p].cost_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
    for k in np.arange(K):
        if len(tau[k]) == 0:
            cos[k].append(0)
            continue
        cos_tmp = []
        for xi in tau[k]:
            xi_vec_0 = [sum(xi[j]*projects[p].phi[j]*projects[p].cost_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
            similarity = (sum([xi_vec_0[j]*scen_vec_0[j] for j in np.arange(env.xi_dim)])) / \
                         ((np.sqrt(sum(xi_vec_0[j] ** 2 for j in np.arange(env.xi_dim)))) *
                          (np.sqrt(sum(xi_vec_0[j] ** 2 for j in np.arange(env.xi_dim)))))
            cos_tmp.append(similarity)

        # select cos with most similarity, so max cos
        cos[k].append(max(cos_tmp))

    return {k: list(np.nan_to_num(cos[k], nan=0.0)) for k in np.arange(K)}


# def state_features(K, env, theta, zeta, x, y, tot_nodes, tot_nodes_i, df_att, theta_i, zeta_i, att_index):
#     features = []
#     # objective
#     features.append(theta/theta_i)
#     # violation
#     features.append(zeta/zeta_i)
#     # depth
#     features.append(tot_nodes/tot_nodes_i)
#
#     return np.array(features)


def weight_labels(K, X, scen_att, scen_att_k, k_new, att_index):
    X_scen = np.array(scen_att + scen_att_k[k_new])
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
    if "slack" in att_series:
        # weights
        try:
            weight_val += [init_weights["slack_K"] for k in np.arange(2)]
        except:
            weight_val += [1.0 for k in np.arange(2)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))
    if "const_to_z_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_z_K"] for k in np.arange(2)]
        except:
            weight_val += [1.0 for k in np.arange(2)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))
    if "const_to_const_dist" in att_series:
        # weights
        try:
            weight_val += [init_weights["c_to_c_K"] for k in np.arange(2)]
        except:
            weight_val += [1.0 for k in np.arange(2)]
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))

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

