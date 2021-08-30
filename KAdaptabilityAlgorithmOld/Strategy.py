# CHANGE THIS FOR NEW PROBLEMS
from ShortestPath.ProblemMILPs.functions import *
from ShortestPath.Attributes.att_functions import *

from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, att_series, weight_model_name=None, time_limit=20*60, print_info=True, problem_type="test",
              depth=2, width=50):
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
    mp_time = 0
    sp_time = 0
    new_model = True
    # initialize N_set
    theta_i, x_i, y_i, N_set, tau_i, N_att_set, tot_nodes_new, init_tot_scens, init_zeta = init_pass(K, env, att_series)
    # scale values by normalizing
    # save stuff
    tot_nodes = copy.copy(tot_nodes_new)
    runtime = time.time() - start_time
    inc_thetas_t[runtime] = theta_i
    inc_thetas_n[tot_nodes_new] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    _, att_index = init_weights_fun(K, env, att_series)

    if weight_model_name is None:
        weight_model_name = f"Results/Models/SubTreeMP/nn_model_K4_sp_online_learning_sub_tree_mp_K4_N100_g30_fs30_D1_W50_inst{env.inst_num}.h5"
    weight_model = load_model(weight_model_name)

    # K-branch and bound algorithm
    now = datetime.now().time()
    print("Instance S {}: started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if new_model:
            try:
                del model
            except:
                pass
            # take new node
            tau = N_set.pop(0)
            # make new df_att
            df_att = N_att_set.pop(0)
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # new node from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k
            # new att node from k_new
            df_att = np.vstack([df_att, scen_att])
            df_att[-1, 0] = k_new

            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
            mp_time += time.time() - start_mp

        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            new_model = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance S {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    zeta = {}   Xi{},   "
                      "prune count = {}".format(
                       env.inst_num, iteration, np.round(time.time()-start_time, 3), now, np.round(theta, 4),
                       np.round(zeta, 4), [len(t) for t in tau.values()], prune_count))
            # try:
            #     env.plot_graph_solutions(K, y, tau, x=x, tmp=True, it=iteration, alg_type=problem_type)
            # except:
            #     pass
            theta_i, x_i, y_i = (copy.deepcopy(theta), copy.deepcopy(x), copy.deepcopy(y))
            tau_i = copy.deepcopy(tau)
            inc_thetas_t[time.time() - start_time] = theta_i
            inc_thetas_n[tot_nodes] = theta_i
            inc_tau[time.time() - start_time] = tau_i
            inc_x[time.time() - start_time] = x_i
            inc_y[time.time() - start_time] = y_i
            prune_count += 1
            new_model = True
            continue
        else:
            new_model = False
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            # predict subset
            # STATE FEATURES (based on master and sub problem)
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, init_tot_scens, df_att, theta_i, init_zeta, att_index)
            K_set = predict_subset(K, df_att, scen_att, weight_model, att_index, tau_s)
            k_new = K_set[0]
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
            N_set.append(tau_tmp)

            # NEW ATT NODE
            df_att_tmp = copy.deepcopy(np.vstack([df_att, scen_att]))
            df_att_tmp[-1, 0] = k
            N_att_set.append(df_att_tmp)

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
                           "mp_time": mp_time, "sp_time": sp_time}
            with open(f"Results/Decisions/tmp_results_{problem_type}_inst{env.inst_num}.pickle", "wb") as handle:
                pickle.dump([env, tmp_results], handle)
        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()
    print("Instance S {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": runtime, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
               "mp_time": mp_time, "sp_time": sp_time}

    with open(f"Results/Decisions/final_results_{problem_type}_inst{env.inst_num}.pickle", "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type=problem_type)
    except:
        pass
    return results


def init_pass(K, env, att_series):
    # Initialize
    start_time = time.time()
    tot_nodes = 0

    # K-branch and bound algorithm
    now = datetime.now().time()
    new_model = True

    # initialize N_set with actual scenario
    tau, tau_att, init_zeta = init_scen(K, env, att_series)
    N_set = []
    N_att_set = []
    df_att = tau_att.reshape([1, -1])
    # print("Instance OL {}: initial pass started at {}".format(env.inst_num, now))

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
            try:
                df_att = copy.deepcopy(np.vstack([df_att, scen_att]))
            except:
                df_att = copy.deepcopy(scen_att.reshape([1, -1]))
            df_att[-1, 0] = k_new
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
            try:
                df_att_tmp = copy.deepcopy(np.vstack([df_att, scen_att]))
            except:
                df_att_tmp = copy.deepcopy(scen_att.reshape([1, -1]))
            df_att_tmp[-1, 0] = k
            N_att_set.append(df_att_tmp)
    tot_scens = np.sum([len(t) for t in tau.values()])
    return theta, x, y, N_set, tau, N_att_set, tot_nodes, tot_scens, init_zeta


def init_scen(K, env, att_series):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    init_zeta, xi_new = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    tau = {k: [] for k in np.arange(K)}
    tau[0] = xi_new.reshape([1, -1])
    tau_att = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y)
    tau_att[0] = 0

    return tau, tau_att, init_zeta
