from ProblemFunctions.functions_milp import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import joblib


def predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, state_features, nn_used=False):
    X = input_fun(K, state_features, tau_att, scen_att, scen_att_k, att_index)

    if nn_used:
        success_prediction = success_model.predict(X)[:, 0]
    else:
        pred_tmp = success_model.predict_proba(X)
        success_prediction = np.array([i[1] for i in pred_tmp])
    order = np.argsort(success_prediction)
    return order[::-1], success_prediction


def state_features(theta, zeta, depth, depth_i, theta_i, zeta_i, theta_pre, zeta_pre):
    # objective, objective compared to previous, violation
    features = [theta / theta_i, theta / theta_pre, zeta / zeta_i]
    # violation compared to previous
    try:
        features.append(zeta/zeta_pre)
    except ZeroDivisionError:
        features.append(1)
    # depth
    features.append(depth/depth_i)

    return np.array(features)


def input_fun(K, state_features, tau_att, scen_att_pre, scen_att_k, att_index):
    att_num = len(att_index)
    scen_att = {k: np.array(scen_att_pre + scen_att_k[k]) for k in np.arange(K)}

    diff_info = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        for att_type in np.arange(att_num):
            diff_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0) - scen_att[k][att_index[att_type]]) / len(att_index[att_type]))

    X = np.array([np.hstack([state_features, diff_info[k]]) for k in np.arange(K)])

    return X


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
        if sum_dist == 0:
            dist_final[k].append(0)
        else:
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
    if np.sum(scen_vec_0) == 0:
        for k in np.arange(K):
            cos[k].append(0)
    else:
        for k in np.arange(K):
            if len(tau[k]) == 0:
                cos[k].append(0)
                continue
            cos_tmp = []
            for xi in tau[k]:
                xi_vec_0 = [sum(xi[j]*projects[p].psi[j]*projects[p].rev_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
                if np.sum(xi_vec_0) == 0:
                    cos_tmp.append(0)
                else:
                    similarity = (sum([xi_vec_0[p]*scen_vec_0[p] for p in np.arange(N)])) / \
                                 ((np.sqrt(sum(xi_vec_0[p] ** 2 for p in np.arange(N)))) *
                                  (np.sqrt(sum(scen_vec_0[p] ** 2 for p in np.arange(N)))))

                    cos_tmp.append(similarity)

            # select cos with most similarity, so max cos
            cos[k].append(max(cos_tmp))

    # CONST 2
    scen_vec_0 = [sum(scen[j]*projects[p].phi[j]*projects[p].cost_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
    if np.sum(scen_vec_0) == 0:
        for k in np.arange(K):
            cos[k].append(0)
    else:
        for k in np.arange(K):
            if len(tau[k]) == 0:
                cos[k].append(0)
                continue
            cos_tmp = []
            for xi in tau[k]:
                xi_vec_0 = [sum(xi[j]*projects[p].phi[j]*projects[p].cost_nom for j in np.arange(env.xi_dim)) for p in np.arange(N)]
                if np.sum(xi_vec_0) == 0:
                    cos_tmp.append(0)
                else:
                    similarity = (sum([xi_vec_0[p]*scen_vec_0[p] for p in np.arange(N)])) / \
                                 ((np.sqrt(sum(xi_vec_0[p] ** 2 for p in np.arange(N)))) *
                                  (np.sqrt(sum(scen_vec_0[p] ** 2 for p in np.arange(N)))))

                    cos_tmp.append(similarity)

            # select cos with most similarity, so max cos
            cos[k].append(max(cos_tmp))

    return {k: list(np.nan_to_num(cos[k], nan=0.0)) for k in np.arange(K)}


def att_index_maker(env, att_series):
    # create list of attributes
    att_index = []

    if "coords" in att_series:
        # index
        att_index.append(np.arange(env.xi_dim))
    if "obj_det" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "x_det" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.x_dim))
        except:
            att_index.append(np.arange(env.x_dim))
    if "y_det" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.y_dim))
        except:
            att_index.append(np.arange(env.y_dim))

    if "obj_stat" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 1))
        except:
            att_index.append(np.arange(1))
    if "y_stat" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + env.y_dim))
        except:
            att_index.append(np.arange(env.y_dim))
    if "slack" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))
    if "const_to_z_dist" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))
    if "const_to_const_dist" in att_series:
        # index
        try:
            last_index = att_index[-1][-1]
            att_index.append(np.arange(last_index + 1, last_index + 1 + 2))
        except:
            att_index.append(np.arange(2))

    return att_index
