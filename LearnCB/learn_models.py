from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM

import matplotlib.pyplot as plt
import autokeras as ak
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import time


# REGRESSION
def train_suc_pred_rf_regr(X_train, Y_train, X_val, Y_val, features, problem_type="test", instances=1000):
    model_name = f"../CapitalBudgetingHigh/Data/Models/rf_regr_{problem_type}.joblib"

    X_train, Y_train, X_val, Y_val, before = clean_data(X_train, Y_train, X_val, Y_val)

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
    feature_importance.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/rf_regr_feat_imp_{problem_type}.pickle")

    # save model training info
    model_info = pd.Series([instances, len(X_train), score_rf, *before, time.time() - start_time],
                           index=["instances", "datapoints", "Rsquared",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "runtime"])
    model_info.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/rf_regr_info_{problem_type}.pickle")

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
def train_suc_pred_nn_class(X_train, Y_train, X_val, Y_val, depth=8, width=100, max_epochs=200, patience_epochs=20,
                      batch_size=64, problem_type="test", instances=1000, balanced=True):
    model_name = f"../CapitalBudgetingHigh/Data/Models/nn_class_{problem_type}_D{depth}_W{width}.h5"

    X_train, Y_train, X_val, Y_val, before, after = clean_data(X_train, Y_train, X_val, Y_val, classification=True,
                                                               balanced=balanced)

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

    plt.savefig(f"../CapitalBudgetingHigh/Data/Models/Info/nn_class_plot_{problem_type}_D{depth}_W{width}.png")
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
    model_info.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/nn_class_info_{problem_type}_D{depth}_W{width}.pickle")

    return score[1]


def auto_nn_class(X_train, Y_train, X_val, Y_val, max_epochs=20, trials=10, problem_type="test"):
    model_name = f"../CapitalBudgetingHigh/Data/Models/nn_auto_regr_{problem_type}.h5"

    # clean data
    X_train, Y_train, X_val, Y_val = clean_data(X_train, Y_train, X_val, Y_val, classification=True)

    # initialize the structured data regressor
    reg = ak.StructuredDataClassifier(overwrite=True, max_trials=trials)

    # train data
    reg.fit(X_train, Y_train, epochs=max_epochs)

    score = reg.evaluate(X_val, Y_val)

    # save model
    reg.export_model().save(model_name)

    return score


def train_suc_pred_rf_class(X_train, Y_train, X_val, Y_val, features, problem_type="test", estimators=100, instances=1000):
    model_name = f"../CapitalBudgetingHigh/Data/Models/rf_class_{problem_type}.joblib"

    X_train, Y_train, X_val, Y_val, before, after = clean_data(X_train, Y_train, X_val, Y_val, classification=True, balanced=True)

    # MODEL
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=estimators)
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)

    print(f"\nRF classification validation accuracy = {score_rf}")

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=features)
    print("Feature importance:\n")
    print(feature_importance)

    # save
    joblib.dump(rf, model_name)
    feature_importance.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/rf_class_feat_imp_{problem_type}.pickle")

    model_info = pd.Series([instances, len(X_train), score_rf, estimators, *before, *after, time.time() - start_time],
                           index=["instances", "datapoints", "accuracy", "estimators",
                                  "0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
                                  "0.9-1", "0", "1", "runtime"])
    model_info.to_pickle(f"../CapitalBudgetingHigh/Data/Models/Info/rf_class_info_{problem_type}.pickle")

    return score_rf


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


def clean_data(X_train, Y_train, X_val, Y_val, classification=False, balanced=False):
    # clean train data
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    Y_train = Y_train.replace([np.inf, -np.inf], np.nan).dropna()

    # find similar indices
    same_indices = X_train.index.intersection(Y_train.index)
    X_train = X_train.loc[same_indices]
    Y_train = Y_train[same_indices]

    # clean train data
    X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    Y_val = Y_val.replace([np.inf, -np.inf], np.nan).dropna()

    # find similar indices
    same_indices = X_val.index.intersection(Y_val.index)
    X_val = X_val.loc[same_indices]
    Y_val = Y_val[same_indices]

    # check the classes
    before = []
    for p in np.arange(0.1, 1.1, 0.1):
        perc = ((Y_train >= p-0.1) & (Y_train <= p)).sum()/len(Y_train)
        before.append(perc)

    if classification:
        # change Y to integers
        Y_train = Y_train.astype(int)
        Y_val = Y_val.astype(int)

        if balanced:
            X_train["class"] = Y_train
            g = X_train.groupby('class')
            g = pd.DataFrame(g.apply(lambda x: x.sample(int(g.size().mean()), replace=True).reset_index(drop=False)))
            X_train = g
            X_new_index = np.array([g.level_0.to_numpy(), g.level_1.to_numpy()]).transpose()
            X_train.index = [tuple(l) for l in X_new_index]
            # X_train.index = g["index"]
            X_train.drop(["level_0", "level_1", "class"], axis=1, inplace=True)
            Y_train = Y_train[X_train.index]

        after = []
        for p in np.arange(0, 2):
            perc = (Y_train == int(p)).sum() / len(Y_train)
            after.append(perc)

        return X_train, Y_train, X_val, Y_val, before, after

    return X_train, Y_train, X_val, Y_val, before

