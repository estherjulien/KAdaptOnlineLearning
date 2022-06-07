from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np
import joblib


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


def update_model_fun_nn(X, Y, expert_data_num=0, depth=1, width=10, success_model_name="test"):
    # maybe try different architectures? only report the best one
    n_features = np.shape(X)[1]
    n_labels = 1

    try:
        success_model = load_model(success_model_name)
        # change learning rate depending on number of expert data num and total?
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        # compile the model
        success_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        # fit the model on new data
        score = X[:, 0] / np.max([X[:, 1] + 1, np.ones(len(X[:, 0]))], axis=0)
        # select everything from ... percent quanitle as validation
        min_score = np.quantile(score, q=0.9)
        score_index = np.where(score > min_score)[0]
        X_val = X[score_index]
        Y_val = Y[score_index]
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        # BOOTSTRAP "GOOD" SOLUTIONS, based on score
        boot_index = np.random.choice(score_index, int(0.4*len(score_index)))
        X_new = np.vstack([X[-expert_data_num:], X[boot_index]])
        Y_new = np.hstack([Y[-expert_data_num:], Y[boot_index]])
        success_model.fit(X_new, Y_new, epochs=100, batch_size=32, verbose=1,
                          callbacks=[es], validation_data=(X_val, Y_val))
    except:
        success_model = Sequential()
        # first layer after input layer
        success_model.add(Dense(width, activation='relu', input_dim=n_features))
        for d in np.arange(depth-1):
            success_model.add(Dense(width, activation='relu'))
        # output layer
        success_model.add(Dense(n_labels, activation='sigmoid'))
        # define the optimization algorithm
        # compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        success_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        # fit the model on data
        es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
        success_model.fit(X, Y, epochs=500, batch_size=32, verbose=1, callbacks=[es])
        # save weight model
        success_model.save(success_model_name)


def update_model_fun(X, Y, expert_data_num=0, depth=1, width=10, success_model_name="test"):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)

    # save
    joblib.dump(rf, success_model_name)

    return score_rf, rf.feature_importances_
