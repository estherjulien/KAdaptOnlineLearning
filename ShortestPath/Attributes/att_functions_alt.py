from ShortestPath.Environment.Env import *
from ShortestPath.Attributes.att_functions import *
from ShortestPath.ProblemMILPs.functions import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np
import joblib
import copy


def predict_subset(K, tau_att, scen_att, scen_att_k, success_model, att_index, state_features):
    X = input_fun(K, state_features, tau_att, scen_att, scen_att_k, att_index)

    pred_tmp = success_model.predict_proba(X)

    success_prediction = np.array([i[1] for i in pred_tmp])
    # print(f"Success prediction = {[np.round(s, 3) for s in success_prediction]}")
    order = np.argsort(success_prediction)
    return order[::-1]


def state_features(K, env, theta, zeta, x, y, depth, depth_i, df_att, theta_i, zeta_i, att_index, theta_pre, zeta_pre):
    features = []
    # objective
    features.append(theta/theta_i)
    # objective compared to previous
    features.append(theta/theta_pre)
    # violation
    features.append(zeta/zeta_i)
    # violation compared to before
    try:
        features.append(zeta/zeta_pre)
    except ZeroDivisionError:
        print(f"Division by 0, zeta = {zeta}, zeta_pre = {zeta_pre}")
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

    # scen_info = []
    # for att_type in np.arange(att_num):
    #     scen_info.append(np.linalg.norm(scen_att[k][att_index[att_type]], axis=0) / len(att_index[att_type]))

    # sub_info = {k: [] for k in np.arange(K)}
    # for k in np.arange(K):
    #     for att_type in np.arange(att_num):
    #         sub_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0)) / len(att_index[att_type]))

    diff_info = {k: [] for k in np.arange(K)}
    for k in np.arange(K):
        for att_type in np.arange(att_num):
            diff_info[k].append(np.linalg.norm(np.mean(tau_att[k][:, att_index[att_type]], axis=0) - scen_att[k][att_index[att_type]]) / len(att_index[att_type]))

    # X = np.array([np.hstack([state_features, scen_info, sub_info[k], diff_info[k]]) for k in np.arange(K)])
    X = np.array([np.hstack([state_features, diff_info[k]]) for k in np.arange(K)])

    return X


def update_model_fun_test(X, Y, expert_data_num=0, depth=1, width=10, success_model_name="test"):
    # maybe try different architectures? only report the best one
    print("\nTRAINING STARTED\n")
    n_features = np.shape(X)[1] - 1
    n_labels = 1

    path_data = [X[X[:, -1] == p] for p in np.unique(X[:, -1])]
    num_path = len(path_data)
    # score 1: based on difference from begin to end, (divided by path length)
    path_score_first = np.array([p[0, 0] * np.max([p[0, 1] + 1, 1]) for p in path_data])
    path_score_last = np.array([p[-1, 0] * np.max([p[-1, 1] + 1, 1]) for p in path_data])
    tmp_1 = ((path_score_first)/path_score_last - 1) / np.array([len(p) for p in path_data])
    path_score_1 = tmp_1 / sum(tmp_1)
    # score 2: only based on end result
    path_score_2 = path_score_last / sum(path_score_last)

    # select 40 percent best
    all_score_1 = np.array([path_score_1[p] for p in np.arange(num_path) for l in np.arange(len(path_data[p]))])
    all_score_2 = np.array([path_score_2[p] for p in np.arange(num_path) for l in np.arange(len(path_data[p]))])

    sel_1_q = np.quantile(path_score_1, q=0.6)
    sel_2_q = np.quantile(path_score_2, q=0.6)

    path_1_sel = np.where(all_score_1 > sel_1_q)[0]
    path_2_sel = np.where(all_score_2 > sel_2_q)[0]
    print(f"\nMethod 1: {len(path_1_sel)} data points \nMethod 2: {len(path_2_sel)} data points\n")

    # p_1_score = np.array([path_score_1[p] for p in np.arange(num_path) for l in np.arange(len(path_data[p]))])
    # p_1_score = p_1_score / sum(p_1_score)
    # p_2_score = np.array([path_score_2[p] for p in np.arange(num_path) for l in np.arange(len(path_data[p]))])
    # p_2_score = p_2_score / sum(p_2_score)
    #
    # p_1_id = np.where(p_1_score > 0)[0]
    # p_2_id = np.where(p_2_score > 0)[0]
    #
    # path_1_sel = np.random.choice(p_1_id, size=min(100, len(p_1_id)), p=p_1_score[p_1_id])
    # path_2_sel = np.random.choice(p_2_id, size=min(100, len(p_2_id)), p=p_2_score[p_2_id])

    X_1 = X[path_1_sel, :-1]
    Y_1 = Y[path_1_sel]
    X_2 = X[path_2_sel, :-1]
    Y_2 = Y[path_2_sel]

    X_1_train, X_1_val, Y_1_train, Y_1_val = train_test_split(X_1, Y_1, test_size=0.1)
    X_2_train, X_2_val, Y_2_train, Y_2_val = train_test_split(X_2, Y_2, test_size=0.1)

    # Path 1 score
    # print("\nPath 1 score model NN: \n ")
    success_model_1 = Sequential()
    # first layer after input layer
    success_model_1.add(Dense(width, activation='relu', input_dim=n_features))
    for d in np.arange(depth-1):
        success_model_1.add(Dense(width, activation='relu'))
    # output layer
    success_model_1.add(Dense(n_labels, activation='sigmoid'))
    # define the optimization algorithm
    # compile the model
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    success_model_1.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    # fit the model on data
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    success_model_1.fit(X_1_train, Y_1_train, epochs=500, batch_size=32, verbose=0, callbacks=[es],
                      validation_data=(X_1_val, Y_1_val))
    # save weight model
    success_model_1.save(success_model_name)

    # Path 2 score
    # print("\nPath 2 score model NN: \n ")
    success_model = Sequential()
    # first layer after input layer
    success_model.add(Dense(width, activation='relu', input_dim=n_features))
    for d in np.arange(depth-1):
        success_model.add(Dense(width, activation='relu'))
    # output layer
    success_model.add(Dense(n_labels, activation='sigmoid'))
    # define the optimization algorithm
    # compile the model
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    success_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    # fit the model on data
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    success_model.fit(X_2_train, Y_2_train, epochs=500, batch_size=32, verbose=0, callbacks=[es],
                      validation_data=(X_2_val, Y_2_val))    # save weight model
    success_model.save(success_model_name)

    # random forest
    # Path model  1

    clf1 = RandomForestClassifier()
    clf1.fit(X_1_train, Y_1_train)

    clf2 = RandomForestClassifier()
    clf2.fit(X_2_train, Y_2_train)

    # evaluation of everything
    score_rf_1 = clf1.score(X_1_val, Y_1_val)
    score_rf_2 = clf1.score(X_2_val, Y_2_val)
    _, acc_nn_1 = success_model_1.evaluate(X_1_val, Y_1_val)
    _, acc_nn_2 = success_model_1.evaluate(X_2_val, Y_2_val)

    print(f"\nPath 1 score model RF: validation accuracy = {score_rf_1} ")
    print(f"Path 2 score model RF: validation accuracy = {score_rf_2} ")
    print(f"Path 1 score model NN: validation accuracy = {acc_nn_1} ")
    print(f"Path 2 score model NN: validation accuracy = {acc_nn_2} \n")


def update_model_fun(X, Y, expert_data_num=0, depth=1, width=10, success_model_name="test"):
    # # select "best" data
    # path_data = [X[X[:, -1] == p] for p in np.unique(X[:, -1])]
    # num_path = len(path_data)
    # # score 1: based on difference from begin to end, (divided by path length)
    # path_score_first = np.array([p[0, 0] * np.max([p[0, 1] + 1, 1]) for p in path_data])
    # path_score_last = np.array([p[-1, 0] * np.max([p[-1, 1] + 1, 1]) for p in path_data])
    # tmp = ((path_score_first)/path_score_last - 1) / np.array([len(p) for p in path_data])
    # path_score = tmp / sum(tmp)
    #
    # # select 40 percent best
    # all_score_1 = np.array([path_score[p] for p in np.arange(num_path) for l in np.arange(len(path_data[p]))])
    #
    # sel_q = np.quantile(path_score, q=0.6)
    #
    # path_sel = np.where(all_score_1 > sel_q)[0]
    # print(f"TRAINING STARTED: {len(path_sel)} data points")
    #
    # X_sel = X[path_sel, :-1]
    # Y_sel = Y[path_sel]
    # X_sel_train, X_sel_val, Y_sel_train, Y_sel_val = train_test_split(X_sel, Y_sel, test_size=0.1)
    #
    # # random forest
    # rf = RandomForestClassifier()
    # rf.fit(X_sel_train, Y_sel_train)
    #
    # # evaluation
    # score_rf = rf.score(X_sel_val, Y_sel_val)
    #
    # print(f"\nRF selection validation accuracy = {score_rf} ")

    # random forest
    # del rf

    X_train, X_val, Y_train, Y_val = train_test_split(X[:, :-1], Y, test_size=0.1)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)
    feat_imp = rf.feature_importances_
    # save
    joblib.dump(rf, success_model_name)

    return score_rf, feat_imp
