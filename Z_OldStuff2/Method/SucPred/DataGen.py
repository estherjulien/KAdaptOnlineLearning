from NOT_USED.ShortestPath.Attributes.att_functions_alt import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
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
    print("Instance SP {}: started at {}".format(env.inst_num, now))

    while N_set and time.time() - start_time < time_limit:
        # PASSES
        pass_num = min(thread_count, len(N_set))

        results = dict()
        N_set_new = dict()
        N_att_set_new = dict()

        passes = np.random.choice(len(N_set), pass_num, replace=False)

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

        N_set = [i for j, i in enumerate(N_set) if j not in passes]
        N_att_set = [i for j, i in enumerate(N_att_set) if j not in passes]

        # ADD NEW NODES
        for i in np.arange(pass_num):
            N_set += N_set_new[i]
            N_att_set += N_att_set_new[i]

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
                f"Instance SP {env.inst_num}: [{num_explore_passes}] it {iteration} explore pass finished ({run_tmp}) "
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
                                                                                  explore_results[i]["input_data"],
                                                                                  explore_results[i]["success_data"],
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
                        print("Instance SP {}, {}: SUB TREE PASS ROBUST ({}) (time {})   :theta = {},    "
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
            with open(f"Data/ResultsSucPred/Data/inst_results/tmp_results_{problem_type}_rf{env.inst_num}.pickle",
                      "wb") as handle:
                pickle.dump(tmp_results, handle)
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()

    print("Instance SP {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime / 60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
               "input_data": input_data, "success_data": success_data}

    with open(f"Data/ResultsSucPred/Data/inst_results/final_results_{problem_type}_rf{env.inst_num}.pickle", "wb") as handle:
        pickle.dump(results, handle)

    return results


def init_pass(K, env, att_series, x_static=None):
    # Initialize
    start_time = time.time()
    tot_nodes = 0

    # K-branch and bound algorithm
    new_model = True

    # initialize N_set with actual scenario
    tau, tau_att, zeta_init = init_scen(K, env, att_series, x_static)
    N_set = []
    N_att_set = []
    try:
        stat_mode = scenario_fun_static_build(env, x_static)
    except:
        stat_mode = None

    det_model = scenario_fun_deterministic_build(env)
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

            # NEW ATTRIBUTE NODE
            tau_att = copy.deepcopy(tau_att)
            try:
                tau_att[k_new] = np.vstack([tau_att[k_new], scen_att + scen_att_k[k_new]])
            except:
                tau_att[k_new] = np.array(scen_att + scen_att_k[k_new]).reshape([1, -1])

            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance SP {}: INIT PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                          stat_model=stat_mode, det_model=det_model)

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

            # NEW ATTRIBUTE NODE
            tau_att_tmp = copy.deepcopy(tau_att)
            try:
                tau_att_tmp[k] = np.vstack([tau_att_tmp[k], scen_att + scen_att_k[k]])
            except:
                tau_att_tmp[k] = np.array(scen_att + scen_att_k[k]).reshape([1, -1])
            N_att_set.append(tau_att_tmp)
    tot_scens = np.sum([len(t) for t in tau.values()])
    return theta, x, y, N_set, tau, N_att_set, tot_nodes, tot_scens, zeta_init


def init_scen(K, env, att_series, x_static=None):
    try:
        stat_mode = scenario_fun_static_build(env, x_static)
    except:
        stat_mode = None

    det_model = scenario_fun_deterministic_build(env)

    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta_init, xi_new = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    tau = {k: [] for k in np.arange(K)}
    tau[0] = xi_new.reshape([1, -1])
    tau_att = {k: [] for k in np.arange(K)}

    first_att_part, k_att_part = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y, x_static=x_static,
                                                    stat_model=stat_mode, det_model=det_model)
    tau_att[0] = np.array(first_att_part + k_att_part[0]).reshape([1, -1])
    return tau, tau_att, zeta_init


def explore_pass(K, env, att_series, tau, tau_att, theta_i, theta_init, zeta_init, tot_scens_init, att_index,
                 x_static=None):
    # Initialize
    tot_nodes = 0

    # K-branch and bound algorithm
    start_time = time.time()
    new_model = True

    # initialize N_set with actual scenario
    N_set = []
    N_att_set = []

    pass_track = []

    input_data = []
    success_data = []

    try:
        stat_mode = scenario_fun_static_build(env, x_static)
    except:
        stat_mode = None

    det_model = scenario_fun_deterministic_build(env)
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
            theta_pre, zeta_pre = [1, 1]
        else:
            # theta and zeta pre
            theta_pre = copy.copy(theta)
            zeta_pre = copy.copy(zeta)
            # NEW NODE from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k

            # NEW ATTRIBUTE NODE
            tau_att = copy.deepcopy(tau_att)
            try:
                tau_att[k_new] = np.vstack([tau_att[k_new], scen_att + scen_att_k[k_new]])
            except:
                tau_att[k_new] = np.array(scen_att + scen_att_k[k_new]).reshape([1, -1])
            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)
        scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                      stat_model=stat_mode, det_model=det_model)
        pass_track.append([tau, tau_att])

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            k_new = np.random.randint(K)
            # STATE DATA
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, tot_scens_init, tau_att, theta_init, zeta_init,
                                   att_index, theta_pre, zeta_pre)
            # SUCCESS DATA
            tau_w = success_label(K, k_new)
            success_data = np.hstack([success_data, tau_w])
            # MODEL DATA
            new_input_data = input_fun(K, tau_s, tau_att, scen_att, scen_att_k, att_index)
            try:
                input_data = np.vstack([input_data, new_input_data])
            except:
                input_data = new_input_data
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        # PRUNE
        if theta - theta_i > -1e-8:
            break

        if zeta < 1e-04:
            now = datetime.now().time()
            print("Instance SP {}: EXPLORE PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                env.inst_num, np.round(time.time() - start_time, 3), now,
                np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False

        # NEW ATTRIBUTE NODE
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

            # NEW ATTRIBUTE NODE
            tau_att_tmp = copy.deepcopy(tau_att)
            try:
                tau_att_tmp[k] = np.vstack([tau_att_tmp[k], scen_att + scen_att_k[k]])
            except:
                tau_att_tmp[k] = np.array(scen_att + scen_att_k[k]).reshape([1, -1])
            N_att_set.append(tau_att_tmp)

    tot_scens = np.sum([len(t) for t in tau.values()])
    runtime = time.time() - start_time

    results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes, "tot_scens": tot_scens,
               "input_data": input_data, "success_data": success_data, "explore": True, "runtime": runtime,
               "sub_tree": pass_track}
    return results, N_set, N_att_set


def sub_tree_pass(K, env, att_series, pass_track, input_data_init, success_data_init, n_back_track, theta_i, theta_init, zeta_init,
                  tot_scens_i, att_index, i_explore, max_depth=5, x_static=None):
    # Initialize
    results = []

    tot_nodes = 0
    theta_i_old = copy.copy(theta_i)
    # K-branch and bound algorithm
    start_time = time.time()

    try:
        tau, tau_att = pass_track[-n_back_track]
        try:
            input_data_init = input_data_init[:-n_back_track]
            success_data_init = success_data_init[:-n_back_track]
        except IndexError:
            input_data_init = []
            success_data_init = []
    except IndexError:
        tau, tau_att = pass_track[0]
        input_data_init = []
        success_data_init = []

    # node information
    node_index = (0,)

    input_data_dict = {node_index: input_data_init}
    success_data_dict = {node_index: success_data_init}

    full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
    if len(full_list) == 0:
        K_set = [0]
    elif len(full_list) == K:
        K_set = np.arange(K)
    else:
        K_prime = min(K, full_list[-1] + 2)
        K_set = np.arange(K_prime)

    level_next = {node_index: tuple(K_set)}
    tau_dict = {node_index: tau}
    tau_att_dict = {node_index: tau_att}
    xi_dict = dict()
    theta_dict = dict()
    zeta_dict = dict()

    # INITIAL MODEL
    try:
        stat_mode = scenario_fun_static_build(env, x_static)
    except:
        stat_mode = None

    det_model = scenario_fun_deterministic_build(env)

    theta, x, y, model = scenario_fun_build(K, tau, env)
    zeta, xi = separation_fun(K, x, y, theta, env, tau)
    xi_dict[node_index] = xi
    theta_dict[node_index] = theta
    zeta_dict[node_index] = zeta

    scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                              stat_model=stat_mode, det_model=det_model)

    # NEW ATTRIBUTE NODE
    for k in K_set:
        # NEW NODE
        tau_tmp = copy.deepcopy(tau)
        adj_tau_k = copy.deepcopy(tau_tmp[k])
        try:
            adj_tau_k = np.vstack([adj_tau_k, xi])
        except:
            adj_tau_k = xi.reshape([1, -1])
        tau_tmp[k] = adj_tau_k
        # add to node set
        tau_dict[node_index + (k,)] = tau_tmp

        # NEW ATTRIBUTE NODE
        tau_att_tmp = copy.deepcopy(tau_att)
        try:
            tau_att_tmp[k] = np.vstack([tau_att_tmp[k], scen_att + scen_att_k[k]])
        except:
            tau_att_tmp[k] = np.array(scen_att + scen_att_k[k]).reshape([1, -1])
        tau_att_dict[node_index + (k,)] = tau_att_tmp

        # NEW LEVELS
        level_next[node_index + (k,)] = tuple(K_set)

    # model storage
    while len(level_next) and len(node_index) < max_depth:
        node_index = list(level_next.keys())[0]
        while len(level_next[node_index]):
            k_new = level_next[node_index][0]
            level_next[node_index] = level_next[node_index][1:]
            new_node_index = node_index + (k_new,)
            tot_nodes += 1
            # ALGORITHM
            tau = tau_dict[new_node_index]
            tau_att = tau_att_dict[new_node_index]
            # MASTER PROBLEM
            theta, x, y = scenario_fun_update_sub_tree(K, new_node_index, xi_dict, env, model)

            # SUBPROBLEM
            zeta, xi = separation_fun(K, x, y, theta, env, tau)
            scen_att, scen_att_k = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                                      stat_model=stat_mode, det_model=det_model)
            full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
            if len(full_list) == 0:
                K_set = [0]
            elif len(full_list) == K:
                K_set = np.arange(K)
                # STATE DATA
                tot_scens = np.sum([len(t) for t in tau.values()])
                tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, tot_scens_i, tau_att, theta_init, zeta_init,
                                       att_index, theta_dict[node_index], zeta_dict[node_index])
                # WEIGHT DATA
                tau_w = success_label(K, k_new)
                try:
                    success_data_dict[new_node_index] = np.hstack([success_data_dict[node_index], tau_w])
                except:
                    success_data_dict[new_node_index] = tau_w
                # INPUT DATA
                new_input_data = input_fun(K, tau_s, tau_att, scen_att, scen_att_k, att_index)
                try:
                    input_data_dict[new_node_index] = np.vstack([input_data_dict[node_index], new_input_data])
                except:
                    input_data_dict[new_node_index] = new_input_data
            else:
                K_prime = min(K, full_list[-1] + 2)
                K_set = np.arange(K_prime)
            # prune if theta higher than current robust theta
            if theta - theta_i > -1e-8:
                # delete stuff
                try:
                    del level_next[new_node_index]
                except KeyError:
                    print("KeyError in theta prune")
                    runtime = time.time() - start_time
                    return results, runtime, tot_nodes, i_explore
                continue

            # check if robust
            if zeta < 1e-04:
                theta_i = copy.deepcopy(theta)
                # SAVE PASS
                try:
                    input_data_final = np.vstack([input_data_init, input_data_dict[new_node_index]])
                except:
                    input_data_final = input_data_dict[new_node_index]

                try:
                    success_data_final = np.hstack([success_data_init, success_data_dict[new_node_index]])
                except:
                    success_data_final = success_data_dict[new_node_index]

                tot_scens = np.sum([len(t) for t in tau.values()])
                results.append({"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes,
                                "input_data": input_data_final, "success_data": success_data_final,
                                "tot_scens": tot_scens})
                # delete stuff
                del level_next[new_node_index]

                continue

            xi_dict[new_node_index] = xi
            theta_dict[new_node_index] = theta
            zeta_dict[new_node_index] = zeta

            # NEW ATTRIBUTE NODE
            for k in K_set:
                # NEW NODE
                tau_tmp = copy.deepcopy(tau)
                adj_tau_k = copy.deepcopy(tau_tmp[k])
                try:
                    adj_tau_k = np.vstack([adj_tau_k, xi])
                except:
                    adj_tau_k = xi.reshape([1, -1])
                tau_tmp[k] = adj_tau_k
                # add to node set
                tau_dict[new_node_index + (k,)] = tau_tmp

                # NEW ATTRIBUTE NODE
                tau_att_tmp = copy.deepcopy(tau_att)
                try:
                    tau_att_tmp[k] = np.vstack([tau_att_tmp[k], scen_att + scen_att_k[k]])
                except:
                    tau_att_tmp[k] = np.array(scen_att + scen_att_k[k]).reshape([1, -1])
                tau_att_dict[new_node_index + (k,)] = tau_att_tmp

                # NEW LEVELS
                level_next[new_node_index + (k,)] = tuple(K_set)
        # after each set of children
        del level_next[node_index]

    # termination actions
    runtime = time.time() - start_time
    return results, runtime, tot_nodes, i_explore


# THIS ON OWN LAPTOP, STRATEGY ON CLUSTER
def data_and_train(K, env_list, att_series, n_back_track=2, time_limit=20 * 60, problem_type="test",
                   thread_count=8, max_depth=5, width=50, depth=1, train_on=None, init_data=False):
    # DATA GEN
    if init_data:
        with open(f"Data/ResultsSucPred/Data/all_data_{problem_type}_20.pickle", "rb") as handle:
            X_list, Y_list = pickle.load(handle).values()
        num_envs = len(env_list) + len(X_list)
    else:
        X_list = []
        Y_list = []
        num_envs = len(env_list)

    for env in env_list:
        results = data_gen_fun(K, env, att_series, n_back_track, time_limit, problem_type,
                                        thread_count, max_depth)
        with open(f"Data/ResultsSucPred/Data/inst_results/data_{problem_type}_{env.inst_num}.pickle", "wb") as handle:
            pickle.dump({"X": results["input_data"], "Y": results["success_data"]}, handle)

    for i in np.arange(num_envs):
        with open(f"Data/ResultsSucPred/Data/inst_results/data_{problem_type}_{i}.pickle", "wb") as handle:
            inst_data = pickle.load(handle)
        X_list.append(inst_data["X"])
        Y_list.append(inst_data["Y"])

    with open(f"Data/ResultsSucPred/Data/all_data_{problem_type}_{num_envs}.pickle", "wb") as handle:
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

        # train ml model
        success_model_name = f"Data/ResultsSucPred/RFModels/rf_model_{problem_type}_rf{num_data}.joblib"
        acc, feat_imp = update_model_fun(X, Y, depth, width, success_model_name=success_model_name)
        print(f"{num_data}: num_data = {len(Y)}, validation accuracy = {acc}")

        # feature importance
        feature_description = ["theta", "theta_pre", "zeta", "zeta_pre", "depth"]
        feature_importance = pd.Series(feat_imp, index=feature_description)
        print(f"RF feature importance: \n{feature_importance}")

        # save trained ml model
        model_info = {"data_len": len(Y), "accuracy": acc, "num_instances": num_data,
                      "feature_importance": feature_importance}
        with open(f"Data/ResultsSucPred/RFModels/rf_info_{problem_type}_rf{num_data}.pickle", "wb") as handle:
            pickle.dump(model_info, handle)
