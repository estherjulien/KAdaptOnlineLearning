# CHANGE THIS FOR NEW PROBLEMS
from ShortestPath.ProblemMILPs.functions import *

from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, time_limit=20*60, print_info=False, problem_type="test"):

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
    cum_tot_nodes = dict()
    tot_nodes = 0
    inc_tot_nodes[0] = 0
    cum_tot_nodes[0] = 0
    prev_save_time = 0
    mp_time = 0
    sp_time = 0

    # initialization of lower and upper bounds
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    inc_lb = dict()
    inc_lb[0] = 0
    # K-branch and bound algorithm
    now = datetime.now().time()
    select_node = True
    # initialize N_set with actual scenario
    tau_i = init_k_adapt(K, env)
    N_set = [tau_i]

    print("Instance OD {} started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if select_node:
            # take new node
            tot_nodes += 1
            tau = N_set.pop(0)
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # just update tau, master problem actually already solved
            tot_nodes += 1
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k_new])
            try:
                adj_tau_k = np.vstack([adj_tau_k, xi])
            except:
                adj_tau_k = xi.reshape([1, -1])
            tau[k_new] = adj_tau_k
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            select_node = True
            continue

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance OD {}: ROBUST at iteration {} ({}) (time {}) (node {})  :theta = {},    Xi{},   prune count = {}".format(
                    env.inst_num, iteration, np.round(time.time()-start_time, 3), tot_nodes, now, np.round(theta, 4), [len(t) for t in tau.values()], prune_count))
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
            select_node = True
            if K == 1:
                break
            else:
                continue
        else:
            select_node = False

        if K == 1:
            N_set = [1]
        else:
            full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
            if len(full_list) == 0:
                K_set = [0]
                k_new = 0
                # solve master problem here
                start_mp = time.time()
                theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
                mp_time += time.time() - start_mp

            elif len(full_list) == K:
                K_set = np.arange(K)
                # try all master problems and select one with highest objective
                theta_list = np.zeros(K)
                x_list = dict()
                y_list = dict()
                model_list = dict()
                start_mp = time.time()
                for k in K_set:
                    # decide on best tau based on lowest objective
                    theta_list[k], x_list[k], y_list[k], model_list[k] = scenario_fun_update(K, k, xi, env, model)
                mp_time += time.time() - start_mp
                # select k_new based on theta
                scores = [np.max([theta_list[k] - theta, 0.00001]) for k in K_set]
                tmp_prob = (scores/np.sum(scores))**(-1)
                k_prob = tmp_prob/np.sum(tmp_prob)
                k_new = np.random.choice(K_set, p=k_prob)
                theta, x, y, model = (theta_list[k_new], x_list[k_new], y_list[k_new], model_list[k_new])
                del model_list
            else:
                K_prime = min(K, full_list[-1] + 2)
                K_set = np.arange(K_prime)
                k_new = K_set[-1]
                # solve master problem here
                start_mp = time.time()
                theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)
                mp_time += time.time() - start_mp

            for k in K_set:
                if k == k_new:
                    continue
                tau_tmp = copy.deepcopy(tau)
                adj_tau_k = copy.deepcopy(tau_tmp[k])
                try:
                    adj_tau_k = np.vstack([adj_tau_k, xi])
                except:
                    adj_tau_k = xi.reshape([1, -1])
                tau_tmp[k] = adj_tau_k
                N_set.append(tau_tmp)

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = len(N_set)
            cum_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t, "inc_thetas_n": inc_thetas_n, "inc_x": inc_x,
                            "inc_y": inc_y, "inc_tau": inc_tau, "runtime": time.time() - start_time,
                            "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time}
            with open("Results/Decisions/tmp_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
                pickle.dump([env, tmp_results], handle)
        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = len(N_set)
    cum_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()
    print("Instance OD {} completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i,  "inc_thetas_t": inc_thetas_t, "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time}
    with open("Results/Decisions/final_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type=problem_type)
    except:
        pass
    return results


def init_k_adapt(K, env):
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

