# CHANGE THIS FOR NEW PROBLEMS
from CapitalBudgetingLoans.ProblemMILPs.functions_loans import *
from CapitalBudgetingLoans.Attributes.att_functions import *

from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import pickle
import copy
import time


def algorithm(K, env, att_series, equal_weights=False, weight_model_name=None, time_limit=20 * 60, print_info=True,
              problem_type="test"):
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
    # FOR STATIC ATTRIBUTE
    try:
        x_static = static_solution_rc(env)
        stat_model = scenario_fun_static_build(env, x_static)
    except:
        x_static = None
        stat_model = None

    det_model = scenario_fun_deterministic_build(env)

    # initialize N_set
    theta_init, x_i, y_i, tau_i, N_set, scen_all, att_all, tot_nodes_new, init_tot_scens, zeta_init = init_pass(K, env,
                                                                                                             att_series,
                                                                                                             x_static,
                                                                                                             stat_model,
                                                                                                             det_model)
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

    init_weights, att_index = init_weights_fun(K, env, att_series)

    if equal_weights:
        weight_model = None
    else:
        # weight_model = load_model(weight_model_name)
        weight_model = joblib.load(weight_model_name)
        _, att_index = init_weights_fun(K, env, att_series)
        init_weights = None

    new_xi_num = len(scen_all) - 1

    # K-branch and bound algorithm
    now = datetime.now().time()
    print("Instance S {}: started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if new_model:
            tot_nodes += 1
            # take new node
            new_pass = np.random.randint(len(N_set))
            placement = N_set.pop(new_pass)
            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # new node from k_new
            tot_nodes += 1
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
            mp_time += time.time() - start_mp

            placement[k_new].append(new_xi_num)
            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            new_model = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, placement)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance S {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    zeta = {}   Xi{},   "
                      "prune count = {}".format(
                    env.inst_num, iteration, np.round(time.time() - start_time, 3), now, np.round(theta, 4),
                    np.round(zeta, 4), [len(t) for t in placement.values()], prune_count))

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
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                          stat_model=stat_model, det_model=det_model)
            att_all = np.vstack([att_all, scen_att])

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            # predict subset
            # STATE FEATURES (based on master and sub problem)
            tot_scens = np.sum([len(t) for t in placement.values()])
            tau_att = {k: att_all[placement[k]] for k in np.arange(K)}
            tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, init_tot_scens, tau_att, theta_init, zeta_init,
                                   att_index)
            K_set = predict_subset(K, tau_att, scen_att, weight_model, init_weights, att_index, tau_s)
            k_new = K_set[0]
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        for k in K_set:
            if k == k_new:
                continue
            # add to node set
            placement_tmp = copy.deepcopy(placement)
            placement_tmp[k].append(new_xi_num)
            N_set.append(placement_tmp)

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10 * 60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
                           "mp_time": mp_time, "sp_time": sp_time}
            with open(f"ResultsEuclAtt/Decisions/tmp_results_{problem_type}_inst{env.inst_num}.pickle", "wb") as handle:
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
    print("Instance S {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime / 60))

    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": runtime, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
               "mp_time": mp_time, "sp_time": sp_time, "scen_all": scen_all, "att_all": att_all}

    with open(f"ResultsEuclAtt/Decisions/final_results_{problem_type}_inst{env.inst_num}.pickle", "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type=problem_type)
    except AttributeError:
        pass
    return results


def init_pass(K, env, att_series, x_static=None, stat_model=None, det_model=None):
    # Initialize
    start_time = time.time()
    tot_nodes = 0

    # K-branch and bound algorithm
    new_model = True

    # initialize N_set with actual scenario
    xi_init, att_init, zeta_init = init_scen(K, env, att_series, x_static=x_static, stat_model=stat_model,
                                             det_model=det_model)
    N_set = [{k: [] for k in np.arange(K)}]
    N_set[0][0].append(0)
    scen_all = xi_init.reshape([1, -1])
    att_all = att_init.reshape([1, -1])
    new_xi_num = 0
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            placement = N_set.pop(0)
            tau = {k: scen_all[placement[k]] for k in np.arange(K)}
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            # NEW NODE from k_new
            tot_nodes += 1
            # master problem
            placement[k_new].append(new_xi_num)
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, placement)

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance S {}: INIT PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in placement.values()]))
            break
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y, x_static=x_static,
                                          stat_model=stat_model, det_model=det_model)
            att_all = np.vstack([att_all, scen_att])

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
            # add to node set
            placement_tmp = copy.deepcopy(placement)
            placement_tmp[k].append(new_xi_num)
            N_set.append(placement_tmp)

    tot_scens = np.sum([len(t) for t in placement.values()])
    tau = {k: scen_all[placement[k]] for k in np.arange(K)}
    return theta, x, y, tau, N_set, scen_all, att_all, tot_nodes, tot_scens, zeta_init


def init_scen(K, env, att_series, x_static=None, stat_model=None, det_model=None):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta_init, xi_init = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    att_init = attribute_per_scen(K, xi_init, env, att_series, tau, theta, x, y, x_static=x_static,
                                  stat_model=stat_model, det_model=det_model)

    return xi_init, att_init, zeta_init
