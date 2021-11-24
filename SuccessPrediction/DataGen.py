from ShortestPath.ProblemMILPs.functions import *
from ShortestPath.Attributes.att_functions_alt import *

from SuccessPrediction.OnlineLearning import explore_pass, sub_tree_pass, init_pass

from tensorflow.keras.models import load_model
from joblib import Parallel, delayed
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import itertools
import pickle
import copy
import time


def data_gen_fun(K, env, att_series, n_back_track=2, time_limit=20 * 60, problem_type="test",
              thread_count=8, max_depth=5):
    # Initialize
    iteration = 0
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    inc_tau = dict()
    inc_x = dict()
    inc_y = dict()
    inc_tot_nodes = dict()
    inc_tot_nodes[0] = 0
    prev_save_time = 0
    explore_results = dict()
    pass_score = []
    success_data = []
    input_data = []

    # FOR STATIC ATTRIBUTE
    try:
        x_static = static_solution_rc(env)
    except:
        x_static = None

    # initialize N_set
    theta_init, x_i, y_i, N_set, tau_i, N_att_set, tot_nodes_new, init_tot_scens, zeta_init = init_pass(K, env, att_series,
                                                                                                     x_static)
    theta_i = copy.copy(theta_init)
    # save stuff
    tot_nodes = copy.copy(tot_nodes_new)
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes_new] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    path_num = 0
    _, att_index = init_weights_fun(K, env, att_series)

    now = datetime.now().time()
    print("Instance DG {}: started at {}".format(env.inst_num, now))

    while N_set and time.time() - start_time < time_limit:
        # PASSES
        pass_num = min(thread_count, len(N_set))

        results = dict()
        N_set_new = dict()
        N_att_set_new = dict()

        passes = np.random.randint(0, len(N_set)-1, pass_num)
        # parallel
        time_before_run = time.time() - start_time
        tot_results = Parallel(n_jobs=thread_count)(delayed(explore_pass)(K, env, att_series,
                                                                          N_set[i], N_att_set[i],
                                                                          theta_i, theta_init, zeta_init,
                                                                          init_tot_scens, att_index,
                                                                          x_static=x_static)
                                                    for i in passes)
        for i in np.arange(pass_num):
            results[i], N_set_new[i], N_att_set_new[i] = tot_results[i]
        del tot_results

        # ADD NEW NODES
        for i in np.arange(pass_num):
            N_set += N_set_new[i]
            N_att_set += N_att_set_new[i]

        for i in passes:
            try:
                del N_set[i]
                del N_att_set[i]
            except:
                print(f"N_set len {len((N_set))}, N_att_set len {len(N_att_set)}, i = {i}")

        # SCORE PERFORMANCE RANDOM RUNS
        explore_theta = []
        theta_i_old = copy.copy(theta_i)
        for i in np.arange(pass_num):
            if results[i]["zeta"] < 1e-4 and results[i]["theta"] - theta_i < -1e-8:
                # RESULT ROBUST
                theta_i, x_i, y_i = (results[i]["theta"], results[i]["x"], results[i]["y"])
                tau_i = copy.deepcopy(results[i]["tau"])
                tot_nodes_new = results[i]["tot_nodes"]
                # save results
                runtime = results[i]["runtime"]
                inc_thetas_t[time_before_run + runtime] = theta_i
                inc_thetas_n[tot_nodes + tot_nodes_new] = theta_i
                inc_tau[time_before_run + runtime] = tau_i
                inc_x[time_before_run + runtime] = x_i
                inc_y[time_before_run + runtime] = y_i
                inc_tot_nodes[time_before_run + runtime] = tot_nodes + tot_nodes_new
            # Check performance
            tot_nodes += results[i]["tot_nodes"]
            num_explore_passes = len(explore_results)
            explore_results[num_explore_passes] = results[i]
            score = (results[i]["theta"] / theta_i_old) * max([results[i]["zeta"] / zeta_init + 1, 1])
            pass_score.append(score)
            explore_theta.append(results[i]["theta"])
            # print results
            theta_tmp = np.round(results[i]["theta"], 4)
            zeta_tmp = np.round(results[i]["zeta"], 4)
            run_tmp = np.round(results[i]["runtime"], 4)
            tau_tmp = results[i]["tau"]
            now = datetime.now().time()
            print(
                f"Instance DG {env.inst_num}: [{num_explore_passes}] it {iteration} explore pass finished ({run_tmp}) "
                f"(time {now})      theta = {theta_tmp},    zeta = {zeta_tmp},     pass_score = {np.round(score, 4)}   "
                f"Xi{[len(t) for t in tau_tmp.values()]}")
            iteration += 1

        # SELECT EXPERTS
        new_experts_list = []
        for i in np.arange(len(pass_score) - pass_num, len(pass_score)):
            if pass_score[i] < 1.02 and len(explore_results[i]["input_data"]) > 1:
                # change state and weight data for exploring "expert" results of sub-tree
                new_experts_list.append(i)

        time_before_run = time.time() - start_time
        tmp_expert_results = Parallel(n_jobs=thread_count)(delayed(sub_tree_pass)(K, env, att_series,
                                                                                  explore_results[i]["sub_tree"],
                                                                                  n_back_track, theta_i_old,
                                                                                  theta_init, zeta_init, init_tot_scens,
                                                                                  att_index, i, max_depth,
                                                                                  x_static=x_static)
                                                           for i in new_experts_list)

        expert_results = []
        new_experts = 0

        for results_all, runtime, tot_nodes_new, i_explore in tmp_expert_results:
            for results in results_all:
                if results["zeta"] < 1e-4:
                    new_experts += 1
                    expert_results.append(results)
                    if results["theta"] - theta_i < -1e-8:
                        # NEW INCUMBENT SOLUTION
                        now = datetime.now().time()
                        theta_i, x_i, y_i = (results["theta"], results["x"], results["y"])
                        tau_i = copy.deepcopy(results["tau"])
                        tot_nodes_new = results["tot_nodes"]
                        print("Instance DG {}, {}: SUB TREE PASS ROBUST ({}) (time {})   :theta = {},    "
                              "zeta = {}   Xi{}".format(env.inst_num, i_explore, np.round(time_before_run + runtime, 4),
                                                        now, np.round(theta_i, 4), np.round(results["zeta"], 4),
                                                        [len(t) for t in tau_i.values()]))
                        # save results
                        inc_thetas_t[time_before_run + runtime] = theta_i
                        inc_thetas_n[tot_nodes + tot_nodes_new] = theta_i
                        inc_tau[time_before_run + runtime] = tau_i
                        inc_x[time_before_run + runtime] = x_i
                        inc_y[time_before_run + runtime] = y_i
                        inc_tot_nodes[time_before_run + runtime] = tot_nodes + tot_nodes_new
            # todo: change this
            tot_nodes += tot_nodes_new

        for r in expert_results:
            path_num_vec = (np.ones(len(r["input_data"]))*path_num).reshape([-1, 1])
            new_input_data = np.hstack([r["input_data"], path_num_vec])
            path_num += 1
            try:
                input_data = np.vstack([input_data, new_input_data])
            except ValueError:
                input_data = new_input_data
            success_data = np.hstack([success_data, r["success_data"]])

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10 * 60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
                           "input_data": input_data, "success_data": success_data}
            with open("ResultsSucPred/Data/tmp_results_{}_rf{}.pickle".format(problem_type, env.inst_num),
                      "wb") as handle:
                pickle.dump([env, tmp_results], handle)
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()

    print("Instance DG {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime / 60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
               "input_data": input_data, "success_data": success_data}

    with open("ResultsSucPred/Data/final_results_{}_rf{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    return results


# THIS ON OWN LAPTOP, STRATEGY ON CLUSTER
def data_and_train(K, env_list, att_series, n_back_track=2, time_limit=20 * 60, problem_type="test",
                   thread_count=8, max_depth=5, width=50, depth=1, train_on=None, init_data=False):
    # DATA GEN
    if init_data:
        with open(f"ResultsSucPred/Data/all_data_{problem_type}_20.pickle", "rb") as handle:
            X_list, Y_list = pickle.load(handle).values()
        num_envs = len(env_list) + len(X_list)
    else:
        X_list = []
        Y_list = []
        num_envs = len(env_list)

    results = []
    for env in env_list:
        results.append(data_gen_fun(K, env, att_series, n_back_track, time_limit, problem_type,
                                thread_count, max_depth))
        X_list.append(results[-1]["input_data"])
        Y_list.append(results[-1]["success_data"])

    with open(f"ResultsSucPred/Data/all_data_{problem_type}_{num_envs}.pickle", "wb") as handle:
        pickle.dump({"X": X_list, "Y": Y_list}, handle)
    # train on
    if train_on is None:
        train_on = [num_envs]

    for num_data in train_on:
        X = X_list[0]
        Y = Y_list[0]
        for c in np.arange(1, num_data):
            X = np.vstack([X, X_list[c]])
            Y = np.hstack([Y, Y_list[c]])

        success_model_name = f"ResultsSucPred/RFModels/rf_model_{problem_type}_rf{num_data}.joblib"
        acc, feat_imp = update_model_fun(X, Y, depth, width, success_model_name=success_model_name)
        print(f"{num_data}: num_data = {len(Y)}, validation accuracy = {acc}")
        feature_description = ["theta", "theta_pre", "zeta", "zeta_pre", "depth"] + att_series
        feature_importance = pd.Series(feat_imp, index=feature_description)
        print(f"RF feature importance: \n{feature_importance}")
        model_info = {"data_len": len(Y), "accuracy": acc, "num_instances": num_data,
                      "feature_importance": feature_importance}
        with open(f"ResultsSucPred/RFModels/rf_info_{problem_type}_rf{num_data}.pickle", "wb") as handle:
            pickle.dump(model_info, handle)

