from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import joblib


def predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, state_features):
    X = input_fun(K, state_features, tau_att, scen_att, scen_att_k, att_index)

    pred_tmp = success_model.predict_proba(X)

    success_prediction = np.array([i[1] for i in pred_tmp])
    # print(f"Success prediction = {[np.round(s, 3) for s in success_prediction]}")
    order = np.argsort(success_prediction)
    return order[::-1]


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


def success_label(K, k_new):
    label = np.zeros(K)
    label[k_new] = 1
    return label


def input_fun(K, state_features, tau_att, scen_att_pre, scen_att_k, att_index):
    att_num = len(att_index)
    scen_att = {k: np.array(scen_att_pre + scen_att_k[k]) for k in np.arange(K)}

    diff_info = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        for att_type in np.arange(att_num):
            diff_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0) - scen_att[k][att_index[att_type]]) / len(att_index[att_type]))

    X = np.array([np.hstack([state_features, diff_info[k]]) for k in np.arange(K)])

    return X


def update_model_fun(X, Y, success_model_name="test"):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)

    # save
    joblib.dump(rf, success_model_name)

    return score_rf, rf.feature_importances_
