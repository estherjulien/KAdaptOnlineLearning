# CHANGE THIS FOR NEW PROBLEMS
from ShortestPath.ProblemMILPs.functions import *
from ShortestPath.Attributes.att_functions import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, att_series, time_limit=20*60, problem_type="test", thread_count=8):
    # Initialize
    iteration = 0
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    inc_tau = dict()
    inc_x = dict()
    inc_y = dict()
    prune_count = 0
    inc_tot_nodes = dict()
    inc_tot_nodes[0] = 0
    prev_save_time = 0
    explore_results = dict()
    pass_score = []
    strategy_results = dict()
    weight_model = None
    weight_data = []
    state_data = []
    # K-branch and bound algorithm
    now = datetime.now().time()

    # initialize N_set
    theta_i, x_i, y_i, N_set, tau_i, N_att_set, tot_nodes_i = init_pass(K, env, att_series)
    # save stuff
    tot_nodes = copy.copy(tot_nodes_i)
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes_i] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    _, att_index = init_weights_fun(K, env, att_series)

    print("Instance OL {},  started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        # PASSES
        num_strategy = 0
        explore_bool = [True for i in np.arange(thread_count) if i > num_strategy]
        results = dict()
        N_set_new = dict()
        N_att_set_new = dict()

        # iterative
        for i in np.arange(thread_count):
            results[i], N_set_new[i], N_att_set_new[i] = \
                parallel_pass(K, env, att_series, N_set[i], N_att_set[i], theta_i, weight_model, start_time, att_index,
                              explore_bool=explore_bool[i])
        # parallel
        tot_results = Parallel(n_jobs=thread_count)(delayed(parallel_pass)(K, env, att_series,
                                                                           N_set[i], N_att_set[i],
                                                                           theta_i, weight_model,
                                                                           start_time, att_index,
                                                                           explore_bool=explore_bool[i])
                                                    for i in np.arange(thread_count))
        for i in np.arange(thread_count):
            results[i], N_set_new[i], N_att_set_new[i] = tot_results[i]
        del tot_results

        # ADD NEW NODES
        for i in np.arange(thread_count):
            N_set += N_set_new[i]
            N_att_set += N_att_set_new[i]

        N_set = N_set[thread_count:]
        N_att_set = N_att_set[thread_count:]

        # SAVE RESULTS
        num_explore = 0
        for i in np.arange(8):
            if results[i]["zeta"] < 1e-4 and results[i]["theta"] - theta_i > -1e-8 :
                # RESULT ROBUST
                theta_i, x_i, y_i = (results[i]["theta"], results[i]["x"], results[i]["y"])
                tau_i = copy.deepcopy(results[i]["tau"])
                tot_nodes_i = results[i]["tot_nodes"]
                # save results
                runtime = results[i]["runtime"]
                inc_thetas_t[runtime] = theta_i
                inc_thetas_n[tot_nodes + tot_nodes_i] = theta_i
                inc_tau[runtime] = tau_i
                inc_x[runtime] = x_i
                inc_y[runtime] = y_i
                inc_tot_nodes[runtime] = tot_nodes + tot_nodes_i
            # for both
            tot_nodes += results[i]["tot_nodes"]
            if results[i]["explore"]:
                num_explore += 1
                num_explore_passes = len(explore_results)
                explore_results[num_explore_passes] = results[i]
                pass_score.append((results[i]["theta"]/theta_i)*max([results[i]["zeta"], 1]) *
                                  (tot_nodes_i*results[i]["tot_nodes"]))
            else:
                num_strategy_passes = len(strategy_results)
                strategy_results[num_strategy_passes] = results[i]

        # SELECT EXPERTS
        best_score = np.min(pass_score[:-num_explore])
        new_experts = 0
        for i in np.arange(len(pass_score) - num_explore, len(pass_score)):
            if pass_score[i]/best_score < 1.01:     # if score diverges 1 percent from best one, add it
                # add state and weight data
                state_data = np.vstack([state_data, explore_results[i]["state_data"]])
                weight_data = np.vstack([weight_data, explore_results[i]["weight_data"]])
                new_experts += 1
        if new_experts > 0:
            weight_model = update_weights_fun(state_data[:-new_experts], weight_data[:-new_experts],
                                              weight_model=weight_model)

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i,  "inc_thetas_t": inc_thetas_t, "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes}
            with open("Results/Decisions/tmp_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
                pickle.dump([env, tmp_results], handle)
        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()
    print("Instance OL {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i,  "inc_thetas_t": inc_thetas_t, "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes}

    with open("Results/Decisions/final_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type=problem_type)
    except:
        pass
    return results


def parallel_pass(K, env, att_series, tau, tau_att, theta_i, weight_model, start_time, att_index, explore_bool):
    if explore_bool:
        return explore_pass(K, env, att_series, tau, tau_att, theta_i, start_time, att_index)
    else:
        return strategy_pass(K, env, att_series, tau, tau_att, theta_i, weight_model, start_time, att_index)


def init_pass(K, env, att_series):
    # Initialize
    start_time = time.time()
    tot_nodes = 0

    # K-branch and bound algorithm
    now = datetime.now().time()
    new_model = True

    # initialize N_set with actual scenario
    tau = init_scen(K, env)
    N_set = [tau]
    N_att_set = []
    df_att = []
    print("Instance OL {} instance pass started at {}".format(env.inst_num, now))
    while True:
        # MASTER PROBLEM
        if new_model:
            # take new node
            tau = N_set.pop(0)
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # make new tau from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k
            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance OL {}: INIT PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            k_new = np.random.randint(K)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        # NEW ATTRIBUTE NODE
        try:
            df_att = np.vstack([df_att, scen_att])
        except:
            df_att = scen_att.reshape([1, -1])
        for k in K_set:
            if k == k_new:
                continue
            # NEW NODE
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau_tmp[k] = adj_tau_k
            # add to node set
            N_set.append(tau_tmp)

            # NEW ATT NODE
            df_att[-1, 0] = copy.deepcopy(k)
            N_set.append(tau_tmp)

    return theta, x, y, N_set, tau, N_att_set, tot_nodes


def explore_pass(K, env, att_series, tau, tau_att, theta_i, start_time, att_index):
    # Initialize
    tot_nodes = 0

    # K-branch and bound algorithm
    now = datetime.now().time()
    new_model = True

    # initialize N_set with actual scenario
    N_set = [tau]
    N_att_set = []

    state_data = []
    weight_data = []
    zeta = 0
    df_att = tau_att.reshape([1, -1])
    print("Instance OL {} explore pass started at {}".format(env.inst_num, now))
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # NEW NODE from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k

            # NEW ATT NODE from k_new
            df_att = np.vstack([df_att, scen_att])
            df_att[-1, 0] = k_new
            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            break
        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # STATE FEATURES (based on master and sub problem)
        tau_s = state_features(K, env, theta, zeta, df_att, theta_i, att_index)
        try:
            state_data = np.vstack([state_data, tau_s])
        except:
            state_data = tau_s.reshape([1, -1])

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance OL {}: EXPLORE PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            k_new = np.random.randint(K)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        # WEIGHT DATA
        tau_w = weight_labels(K, df_att, scen_att, k_new, att_index)
        try:
            weight_data = np.vstack([weight_data, tau_w])
        except:
            weight_data = tau_w.reshape([1, -1])

        # NEW ATTRIBUTE NODE
        df_att_tmp = np.vstack([df_att, scen_att])
        for k in K_set:
            if k == k_new:
                continue
            # NEW NODE
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau_tmp[k] = adj_tau_k
            # add to node set
            N_set.append(tau_tmp)

            # NEW ATT NODE
            df_att_tmp[-1, 0] = copy.deepcopy(k)
            N_att_set.append(df_att_tmp)

    runtime = time.time() - start_time
    results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes,
               "state_data": state_data, "weight_data": weight_data, "explore": True, "runtime": runtime}
    return results, N_set, N_att_set


def strategy_pass(K, env, att_series, tau, tau_att, theta_i, weight_model, start_time, att_index):
    # Initialize
    tot_nodes = 0

    # K-branch and bound algorithm
    now = datetime.now().time()
    new_model = True

    # initialize N_set with actual scenario
    N_set = [tau]
    N_att_set = []

    weight_data = []

    df_att = tau_att.reshape([1, -1])
    print("Instance OL {} explore pass started at {}".format(env.inst_num, now))
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # NEW NODE from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k

            # NEW ATT NODE from k_new
            df_att = np.vstack([df_att, scen_att])
            df_att[-1, 0] = k_new
            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            break
        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # STATE FEATURES (based on master and sub problem)
        tau_s = state_features(K, env, theta, zeta, df_att, theta_i, att_index)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance OL {}: EXPLORE PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            # predict subset
            k_new = predict_subset(df_att, scen_att, weight_model, tau_s)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        # WEIGHT DATA
        tau_w = weight_labels(K, df_att, scen_att, k_new, att_index)
        try:
            weight_data = np.vstack([weight_data, tau_w])
        except:
            weight_data = tau_w.reshape([1, -1])

        # NEW ATTRIBUTE NODE
        df_att_tmp = np.vstack([df_att, scen_att])
        for k in K_set:
            if k == k_new:
                continue
            # NEW NODE
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau_tmp[k] = adj_tau_k
            # add to node set
            N_set.append(tau_tmp)

            # NEW ATT NODE
            df_att_tmp[-1, 0] = copy.deepcopy(k)
            N_att_set.append(df_att_tmp)

    runtime = time.time() - start_time
    results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes, "explore": False,
               "runtime": runtime}
    return results, N_set, N_att_set


def init_scen(K, env):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    _, xi_new = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    tau = {k: [] for k in np.arange(K)}
    tau[0] = xi_new.reshape([1, -1])

    return tau