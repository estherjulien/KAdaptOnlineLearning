from ShortestPath.Attributes.att_mp_functions import *
from ShortestPath.ProblemMILPs.functions import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm_main(K, env, att_series, lr_w=1, att_crit=.001, thread_count=8,
                   time_limit=20 * 60, print_info=False, problem_type="test"):
    # Initialize
    N_set = [{k: [] for k in np.arange(K)}]
    N_set[0][0].append(env.init_uncertainty)
    tau_i = copy.deepcopy(N_set[0])
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
    att_time = 0

    N_set_att = []
    # store per random node
    theta_rand = []
    num_nodes_rand = []
    violation_rand = []
    robust_rand = []
    df_rand_list = []

    # store per att node
    theta_att = []
    num_nodes_att = []
    violation_att = []
    robust_att = []
    df_att_list = []

    # initialization of lower and upper bounds
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    inc_lb = dict()
    inc_lb[0] = 0

    # K-branch and bound algorithm
    now = datetime.now().time()

    # initialize weights
    weights, N_set_att_init = init_weights(K, env, att_series)
    N_set_att = [N_set_att_init]

    print("Instance AOMP {} started at {}".format(env.inst_num, now))
    # Algorithm
    while time.time() - start_time < time_limit:
        # attribute to random runs ratio
        if len(theta_att):
            obj_perf = np.mean(theta_att[-att_run_num:]) - np.mean(theta_rand[-rand_run_num:])
            now = datetime.now().time()
            try:
                print(f"Instance AOMP {env.inst_num} ({now}), it = {iteration}, att perf = {att_perf}, obj perf = {obj_perf}")
            except UnboundLocalError:
                print(f"Instance AOMP {env.inst_num} ({now}), it = {iteration}, obj perf = {obj_perf}")

            if np.mean(att_new_scens) > 1:
                if obj_perf < att_crit:
                    rand_run_num = max(rand_run_num - 1, 1)
                else:
                    rand_run_num = min(rand_run_num + 1, thread_count - 1)
        else:
            rand_run_num = thread_count - 1
        att_run_num = thread_count - rand_run_num
        att_bool_list = [*[False] * rand_run_num, *[True] * att_run_num]

        # NODE RUNS
        results_list = Parallel(n_jobs=thread_count)(
            delayed(run_function)(att_bool_list[n], K, env, N_set[n], theta_i, weights, att_series, df_att=N_set_att[n], it=iteration + n)
            for n in np.arange(min(thread_count, len(N_set))))
        # delete nodes
        del N_set[:min(thread_count, len(N_set))]
        del N_set_att[:min(thread_count, len(N_set_att))]
        # analyze att nodes
        tot_new_nodes = 0
        n = 0

        att_new_scens = []
        for results in results_list:
            if results["att_bool"]:
                if results["robust"]:
                    if results["theta"] - theta_i < -1e-8:
                        if print_info:
                            now = datetime.now().time()
                            print(
                                "Instance AOMP {}: ATT ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
                                    env.inst_num, iteration, np.round(time.time() - start_time, 3), now,
                                    np.round(results["theta"], 4),
                                    [len(t) for t in results["tau"].values()], prune_count))
                            try:
                                env.plot_graph_solutions(K, results["y"], results["tau"], x=results["x"],
                                                         alg_type="online_att", tmp=True, it=iteration)
                            except:
                                pass
                        # store results
                        theta_i, x_i, y_i = (copy.deepcopy(results["theta"]),
                                             copy.deepcopy(results["x"]),
                                             copy.deepcopy(results["y"]))
                        tau_i = copy.deepcopy(results["tau"])
                        inc_thetas_t[time.time() - start_time] = theta_i
                        inc_thetas_n[tot_nodes + results["num_nodes"]] = theta_i
                        inc_tau[time.time() - start_time] = tau_i
                        inc_x[time.time() - start_time] = x_i
                        inc_y[time.time() - start_time] = y_i
                    else:
                        prune_count += 1
                else:
                    prune_count += 1
                # store results per run for learning
                theta_att.append(results["theta"])
                num_nodes_att.append(results["num_nodes"])
                robust_att.append(results["robust"])
                df_att_list.append(results["df_att"])
                violation_att.append(results["violation"])
                # append new taus to N_set
                N_set += results["N_set"]
                N_set_att += results["N_set_att"]
                # update times
                mp_time += results["mp_time"]
                sp_time += results["sp_time"]
                att_time += results["att_time"]
                tot_new_nodes += results["num_nodes"]
                att_new_scens.append(results["new_scens"])
            else:
                if results["robust"]:
                    if results["theta"] - theta_i < -1e-8:
                        if print_info:
                            now = datetime.now().time()
                            print(
                                "Instance AOMP {}: RANDOM ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
                                    env.inst_num, iteration, np.round(time.time() - start_time, 3), now,
                                    np.round(results["theta"], 4),
                                    [len(t) for t in results["tau"].values()], prune_count))
                            try:
                                env.plot_graph_solutions(K, results["y"], results["tau"], x=results["x"],
                                                         alg_type="online_rand", tmp=True, it=iteration)
                            except:
                                pass
                        # store results
                        theta_i, x_i, y_i = (copy.deepcopy(results["theta"]),
                                             copy.deepcopy(results["x"]),
                                             copy.deepcopy(results["y"]))
                        tau_i = copy.deepcopy(results["tau"])
                        inc_thetas_t[time.time() - start_time] = theta_i
                        inc_thetas_n[tot_nodes + results["num_nodes"]] = theta_i
                        inc_tau[time.time() - start_time] = tau_i
                        inc_x[time.time() - start_time] = x_i
                        inc_y[time.time() - start_time] = y_i
                    else:
                        prune_count += 1
                else:
                    prune_count += 1

                # store results per run for learning
                theta_rand.append(results["theta"])
                num_nodes_rand.append(results["num_nodes"])
                robust_rand.append(results["robust"])
                df_rand_list.append(results["df_att"])
                violation_rand.append(results["violation"])
                # append new taus to N_set
                N_set += results["N_set"]
                N_set_att += results["N_set_att"]
                # update times
                mp_time += results["mp_time"]
                sp_time += results["sp_time"]
                tot_new_nodes += results["num_nodes"]
            # indexing
            n += 1
            iteration += 1
        tot_nodes += tot_new_nodes

        # CHANGE WEIGHTS
        if np.mean(att_new_scens) > 1:
            try:
                # analyze from which random results we learn from
                # metric of best random node depends on: {objective, number of nodes, robustness}
                best_rand_run = np.argmin([(theta_rand[i] * num_nodes_rand[i] + violation_rand[i])
                                           for i in np.arange(len(robust_rand))])
                # print([(theta_rand[i] * num_nodes_rand[i] + violation_rand[i])
                #                            for i in np.arange(len(robust_rand))])
                # select last tau rand and last df_att
                df_rand_best = df_rand_list[best_rand_run]
                df_att_best = df_att_list[-1]

                weights, att_perf = update_weights(K, env, weights, df_rand=df_rand_best,
                                                   df_att=df_att_best,
                                                   lr_w=lr_w,
                                                   att_series=att_series)

                # for testing
                weights_test = final_weights(att_series, weights)
                print(f"Instance AOMP {env.inst_num}, weights test = {weights_test}")
            except:
                print(f"Instance AOMP {env.inst_num}, it = {iteration} finished")
        else:
            print(f"Instance AOMP {env.inst_num}, it = {iteration} finished")

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10 * 60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = len(N_set)
            cum_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_weights = final_weights(att_series, weights)
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "weights": tmp_weights,
                           "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes,
                           "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
                           "att_time": att_time}
            with open("Results/Decisions/tmp_results_online_mp_{}_inst{}.pickle".format(problem_type, env.inst_num),
                      "wb") as handle:
                pickle.dump([env, tmp_results], handle)
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = len(N_set)
    cum_tot_nodes[runtime] = tot_nodes

    weights = final_weights(att_series, weights)

    now = datetime.now().time()
    print("Instance AOMP {} completed at {}, solved in {} minutes".format(env.inst_num, now, runtime / 60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "weights": weights, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes,
               "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
               "att_time": att_time}

    with open("Results/Decisions/final_results_online_mp_{}_inst{}.pickle".format(problem_type, env.inst_num),
              "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type="online")
    except:
        pass
    return results


def run_function(att_bool, K, env, tau, theta_i, weights, att_series, df_att, time_limit=60*60, it=0):
    if att_bool:
        return run_att(K, env, tau, theta_i, weights, att_series, df_att, time_limit=time_limit, it=it)
    else:
        return run_random(K, env, tau, theta_i, att_series, df_att, time_limit=time_limit, it=it)


def run_random(K, env, tau, theta_i, att_series, df_rand, time_limit=20*60, it=0):
    # initialize
    mp_time = 0
    sp_time = 0
    att_time = 0
    N_set = []
    N_set_att = []
    initXi = [len(t) for t in tau.values()]
    robust_bool = False
    zeta = 10
    xi_new = None
    k_new = None
    new_scens = 0
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if xi_new is None:
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # make new tau
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            adj_tau_k.append(xi_new)
            tau[k_new] = adj_tau_k
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi_new, env, model)
            mp_time += time.time() - start_mp

        if theta - theta_i > -1e-8:
            zeta, _ = separation_fun(K, x, y, theta, env, tau)
            if zeta < 1e-4:
                robust_bool = True
            else:
                robust_bool = False
            break

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi_new = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp
        if zeta <= 1e-04:
            robust_bool = True
            break
        else:
            new_scens += 1
            # ATTRIBUTES PER SCENARIO
            start_att = time.time()
            scen_att_new = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y)
            att_time += time.time() - start_att

        # choose k_new and add other tau's to N_set
        full_list = [k for k in np.arange(K) if tau[k]]
        if not full_list:
            K_set = [0]
        elif len(full_list) == K:
            K_set = np.arange(K)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
        k_new = np.random.randint(len(K_set))
        new_att = len(df_rand)
        df_rand.loc[new_att] = scen_att_new
        for k in K_set:
            if k == k_new:
                continue
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            adj_tau_k.append(xi_new)
            tau_tmp[k] = adj_tau_k
            N_set.append(tau_tmp)
            # N_set_att prep
            df_rand.loc[new_att, ("subset", 0)] = k
            N_set_att.append(copy.deepcopy(df_rand))        # check if this goes right

        # add scen to df_att with subset = k_new
        df_rand.loc[new_att, ("subset", 0)] = k_new

    num_nodes = sum([len(t) for t in tau.values()])
    print(
        f"Instance AOMP {env.inst_num}; RANDOM RUN {it} finished in {np.round(time.time() - start_time, 4)}, #Nodes = {num_nodes}, Robust = {robust_bool}, theta = {theta}, "
        f"initXi = {initXi}, finalXi = {[len(t) for t in tau.values()]}")

    return {"theta": theta, "x": x, "y": y, "tau": tau, "df_att": df_rand, "robust": robust_bool, "num_nodes": num_nodes,
            "N_set": N_set, "N_set_att": N_set_att, "mp_time": mp_time, "sp_time": sp_time, "att_time": att_time,
            "att_bool": False, "violation": zeta, "new_scens": new_scens}


def run_att(K, env, tau, theta_i, weights, att_series, df_att, time_limit=20*60, it=0):
    # initialize
    mp_time = 0
    sp_time = 0
    att_time = 0
    N_set = []
    N_set_att = []
    initXi = [len(t) for t in tau.values()]
    robust_bool = False
    zeta = 10
    xi_new = None
    k_new = None
    new_scens = 0
    start_time = time.time()
    # ALGORITHM
    while time.time() - start_time < time_limit:
        # MASTER PROBLEM
        if xi_new is None:
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # make new tau
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            adj_tau_k.append(xi_new)
            tau[k_new] = adj_tau_k
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi_new, env, model)
            mp_time += time.time() - start_mp

        if theta - theta_i > -1e-8:
            zeta, _ = separation_fun(K, x, y, theta, env, tau)
            if zeta < 1e-4:
                robust_bool = True
            else:
                robust_bool = False
            break

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi_new = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp
        if zeta <= 1e-04:
            robust_bool = True
            break
        else:
            new_scens += 1
            # ATTRIBUTES PER SCENARIO
            start_att = time.time()
            scen_att_new = attribute_per_scen(K, xi_new, env, att_series, tau, theta, x, y)
            att_time += time.time() - start_att

        # choose k_new and add other tau's to N_set
        full_list = [k for k in np.arange(K) if tau[k]]
        if not full_list:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            start_att = time.time()
            K_set = avg_dist_on_attributes(df_att, scen_att_new, weights)
            k_new = K_set[0]
            att_time += time.time() - start_att
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            # select empty bag as next bag
            k_new = K_set[-1]

        new_att = len(df_att)
        df_att.loc[new_att] = scen_att_new
        for k in K_set:
            if k == k_new:
                continue
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            adj_tau_k.append(xi_new)
            tau_tmp[k] = adj_tau_k
            N_set.append(tau_tmp)
            # N_set_att prep
            df_att.loc[new_att, ("subset", 0)] = k
            N_set_att.append(copy.deepcopy(df_att))  # check if this goes right

        # add scen to df_att with subset = k_new
        df_att.loc[new_att, ("subset", 0)] = k_new

    num_nodes = sum([len(t) for t in tau.values()])
    print(
        f"Instance AOMP {env.inst_num}; ATT RUN {it} finished in {np.round(time.time() - start_time, 4)}, #Nodes = {num_nodes}, Robust = {robust_bool}, theta = {theta}, "
        f"initXi = {initXi}, finalXi = {[len(t) for t in tau.values()]}")

    return {"theta": theta, "x": x, "y": y, "tau": tau, "df_att": df_att, "robust": robust_bool, "num_nodes": num_nodes,
            "N_set": N_set, "N_set_att": N_set_att, "mp_time": mp_time, "sp_time": sp_time, "att_time": att_time, "att_bool": True,
            "violation": zeta, "new_scens": new_scens}
