from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import time


# REGRESSION
def train_suc_pred_rf_regr(X, Y, features, problem_type="test", instances=1000, save_map="Results"):
    model_name = f"../{save_map}/Models/rf_regr_{problem_type}.joblib"

    X, Y, before = clean_data(X, Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.01)
    start_time = time.time()
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)

    print(f"\nRF regression validation Rsquared = {score_rf}")

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    print("Feature importance:\n")
    print(feature_importance)

    # save
    joblib.dump(rf, model_name)
    feature_importance.to_pickle(f"../{save_map}/Models/Info/rf_regr_feat_imp_{problem_type}.pickle")

    # save model training info
    model_info = pd.Series([instances, len(X_train), score_rf, *before, time.time() - start_time],
                           index=["instances", "datapoints", "Rsquared",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "runtime"])
    model_info.to_pickle(f"../{save_map}/Models/Info/rf_regr_info_{problem_type}.pickle")

    return score_rf


def train_suc_pred_nn_regr(X_train, Y_train, X_val, Y_val, depth=5, width=100, max_epochs=200, patience_epochs=20,
                      batch_size=64, problem_type="test", instances=1000):
    model_name = f"../CapitalBudgetingHigh/Data/Models/nn_regr_{problem_type}_D{depth}_W{width}.h5"

    X_train, Y_train, X_val, Y_val, before = clean_data(X_train, Y_train, X_val, Y_val)

    start_time = time.time()

    n_features = np.shape(X_train)[1]
    n_labels = 1

    nn_model = Sequential()
    # first layer after input layer
    nn_model.add(Dense(width, activation="relu", input_dim=n_features))

    # hidden layers
    for d in np.arange(depth - 1):
        nn_model.add(Dense(width, activation="relu"))

    # output layer
    nn_model.add(Dense(n_labels, activation="linear"))

    # compile the model
    nn_model.compile(optimizer="adam", loss="mean_squared_error")

    # fit the model on data based on validation
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_epochs,
                                          restore_best_weights=True)

    history = nn_model.fit(X_train, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=1,
                           validation_data=(X_val, Y_val), callbacks=[es])
    # plot training history
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()

    plt.savefig(f"../CapitalBudgetingHigh/Data/Models/Info/nn_regr_plot_{problem_type}_D{depth}_W{width}.png")
    plt.close()

    # evaluation
    score = nn_model.evaluate(X_val, Y_val)

    print(f"NN regression validation MSE = {score}")

    # save model
    nn_model.save(model_name)
    # save model training info
    model_info = pd.Series([instances, len(X_train), score, len(history.epoch), depth, width, *before, time.time() - start_time],
                           index=["instances", "datapoints", "mse", "epochs", "D", "W",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "runtime"])
    model_info.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/nn_regr_info_{problem_type}_D{depth}_W{width}.pickle")


# CLASSIFICATION
def train_suc_pred_nn_class(X, Y, depth=8, width=100, max_epochs=200, patience_epochs=20,
                      batch_size=64, problem_type="test", instances=1000, balanced=True, class_thresh=False,
                            save_map="Results"):
    model_name = f"../{save_map}/Models/nn_class_{problem_type}_D{depth}_W{width}.h5"

    X_train, Y_train, X_val, Y_val, before, after = clean_data(X, Y, classification=True, class_thresh=class_thresh)

    X_train, Y_train, X_val, Y_val = train_test_split(X, Y, stratify=Y)

    # NN MODEL
    start_time = time.time()
    X_train = X_train.to_numpy()
    n_features = np.shape(X_train)[1]
    n_labels = 1

    nn_model = Sequential()
    # first layer after input layer
    nn_model.add(Dense(width, activation="relu", input_dim=n_features))
    # nn_model.add(Conv1D(width, 2, activation="relu", input_shape=X_train.shape))
    # nn_model.add(LSTM(50, input_dim=n_features))
    # hidden layers
    for d in np.arange(depth - 1):
        nn_model.add(Dense(width, activation="relu"))

    # output layer
    nn_model.add(Dense(n_labels, activation="sigmoid"))

    # compile the model
    nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "mse"])

    # fit the model on data based on validation
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_epochs,
                                          restore_best_weights=True)

    history = nn_model.fit(X_train, Y_train, epochs=max_epochs, batch_size=batch_size, verbose=1,
                           validation_data=(X_val, Y_val), callbacks=[es])
    # plot training history
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()

    plt.savefig(f"../{save_map}/Models/Info/nn_class_plot_{problem_type}_D{depth}_W{width}.png")
    plt.close()

    # evaluation
    score = nn_model.evaluate(X_val, Y_val)
    data_0_index = Y_val[Y_val == 0].index
    score_0 = nn_model.evaluate(X_val.loc[data_0_index], Y_val[data_0_index])
    data_1_index = Y_val[Y_val == 1].index
    score_1 = nn_model.evaluate(X_val.loc[data_1_index], Y_val[data_1_index])

    print(f"NN classification validation accuracy = {score[1]}")
    print(f"NN classification validation accuracy (Y = 0) = {score_0[1]}")
    print(f"NN classification validation accuracy (Y = 1) = {score_1[1]}")

    # save model
    nn_model.save(model_name)

    # save model training info
    model_info = pd.Series([instances, len(X_train), score[1], score_0[1], score_1[1], len(history.epoch), depth, width, *before, *after, time.time() - start_time],
                           index=["instances", "datapoints", "accuracy_all", "accuracy_0", "accuracy_1", "epochs", "D", "W",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "0", "1", "runtime"])
    model_info.to_pickle(f"../{save_map}/Models/Info/nn_class_info_{problem_type}_D{depth}_W{width}.pickle")

    return score[1]


def train_suc_pred_rf_class(X, Y, features, problem_type="test", estimators=100, instances=1000,
                            save_map="Results", class_thresh=False, balanced=False):
    model_name = f"../{save_map}/rf_class_{problem_type}.joblib"

    X, Y, df_X, df_Y, before, after = clean_data(X, Y, classification=True, class_thresh=class_thresh, balanced=balanced)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, stratify=Y, test_size=0.01)
    print(f"TRAIN DATA = {len(X_train)}")
    # MODEL
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=estimators)
    rf.fit(X_train, Y_train)

    # evaluation
    score = rf.score(X_val, Y_val)
    data_0_index = np.where(Y_val == 0)[0]
    type_1 = 1 - rf.score(X_val[data_0_index], Y_val[data_0_index])
    data_1_index = np.where(Y_val == 1)[0]
    type_2 = 1 - rf.score(X_val[data_1_index], Y_val[data_1_index])

    print(f"RF classification validation accuracy = {score}")
    print(f"RF classification validation type I error = {type_1}")
    print(f"RF classification validation type II error = {type_2}")

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    print("Feature importance:\n")
    print(feature_importance)

    # save
    joblib.dump(rf, model_name)
    feature_importance.to_pickle(f"../{save_map}/Info/rf_class_feat_imp_{problem_type}.pickle")

    model_info = pd.Series([instances, len(X_train), class_thresh, score, type_1, type_2, estimators, *before, *after, time.time() - start_time],
                           index=["instances", "datapoints", "class_threshold", "accuracy", "type_1", "type_2", "estimators",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "0", "1", "runtime"])
    model_info.to_pickle(f"../{save_map}/Info/rf_class_info_{problem_type}.pickle")

    return score


def train_suc_pred_xgboost_class(X_train, Y_train, X_val, Y_val, problem_type="test", instances=1000):
    model_name = f"../CapitalBudgetingHigh/Data/Models/xgboost_class_{problem_type}.json"

    # clean data
    X_train, Y_train, X_val, Y_val, before, after = clean_data(X_train, Y_train, X_val, Y_val, classification=True)

    start_time = time.time()
    model = XGBRFClassifier()

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_val)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_val, predictions)
    print(f"\nXGboost classification validation accuracy = {accuracy}")
    # save model
    model.save_model(model_name)
    model_info = pd.Series([instances, len(X_train), accuracy, *before, *after, time.time() - start_time],
                           index=["instances", "datapoints", "accuracy",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "0", "1", "runtime"])
    model_info.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/xgboost_class_info_{problem_type}.pickle")
    return accuracy


def clean_data(X, Y, classification=False, class_thresh=False, balanced=False):
    # clean train data
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    Y = Y.replace([np.inf, -np.inf], np.nan).dropna()

    # find similar indices
    same_indices = X.index.intersection(Y.index)
    X = X.loc[same_indices]
    Y = Y[same_indices]

    # check the classes
    before = []
    for p in np.arange(0.1, 1.1, 0.1):
        perc = ((Y >= p-0.1) & (Y <= p)).sum()/len(Y)
        before.append(perc)
    print(before)

    if classification:
        if class_thresh:
            Y[Y < class_thresh] = 0
            Y[Y >= class_thresh] = 1
        else:
            # change Y to integers
            Y = Y.astype(int)

        if balanced:
            X["class"] = Y
            g = X.groupby('class')
            g = pd.DataFrame(g.apply(lambda x: x.sample(int(g.size().mean()), replace=True).reset_index(drop=False)))
            X = g
            X.index = X["index"]
            X.drop(["index", "class"], axis=1, inplace=True)
            Y = Y[X.index]

        after = []
        for p in np.arange(0, 2):
            perc = (Y == int(p)).sum() / len(Y)
            after.append(perc)
        print(after)
    return X.to_numpy(), Y.to_numpy(), X, Y, before, after

