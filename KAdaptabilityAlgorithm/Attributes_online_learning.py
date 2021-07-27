from ShortestPath.Attributes.att_functions_2s import *
from ShortestPath.ProblemMILPs.functions_2s import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm_main(K, env, att_series, lr_w=.5, lr_t=0.9, att_crit=.00005, weight_group=False, thread_count=8,
                   time_limit=20*60, print_info=False, problem_type="test"):
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
    # store per random node
    theta_rand = []
    num_nodes_rand = []
    violation_rand = []
    robust_rand = []
    tau_rand = []
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

    weights, x_static = init_weights(env, att_series)

    print("Instance AO {} started at {}".format(env.inst_num, now))

    while time.time() - start_time < time_limit:
        try:
            # adjust learning rate lr_t
            # more att if it is better than random
            if len(theta_att):
                obj_perf = np.mean(theta_att[-att_run_num:]) - np.mean(theta_rand[-rand_run_num:])
                print(f"obj_perf = {obj_perf}")
                lr_t -= 1/thread_count*(obj_perf < att_crit)
        except:
            pass
        rand_run_num = max(1, int(np.floor(thread_count*lr_t)))
        att_run_num = thread_count - rand_run_num
        att_bool_list = [*[False]*rand_run_num, *[True]*att_run_num]
        # ATTRIBUTE NODE RUNS
        results_list = Parallel(n_jobs=thread_count)(delayed(run_function)(att_bool_list[n], K, env, N_set[n], theta_i, weights, att_series,
                                                                     x_static=x_static, it=iteration+n)
                                                    for n in np.arange(min(thread_count, len(N_set))))
        # delete att nodes
        del N_set[:thread_count]
        # analyze att nodes
        tot_new_nodes = 0
        n = 0
        for results in results_list:
            if results["att_bool"]:
                if results["robust"]:
                    if results["theta"] - theta_i < -1e-8:
                        if print_info:
                            now = datetime.now().time()
                            print("Instance AO {}: ATT ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
                                    env.inst_num, iteration, np.round(time.time() - start_time, 3), now, np.round(results["theta"], 4),
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
                        inc_thetas_n[tot_nodes+results["num_nodes"]] = theta_i
                        inc_tau[time.time() - start_time] = tau_i
                        inc_x[time.time() - start_time] = x_i
                        inc_y[time.time() - start_time] = y_i
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
                # update times
                mp_time += results["mp_time"]
                sp_time += results["sp_time"]
                att_time += results["att_time"]
                tot_new_nodes += results["num_nodes"]
            else:
                if results["robust"]:
                    if results["theta"] - theta_i < -1e-8:
                        if print_info:
                            now = datetime.now().time()
                            print(
                                "Instance AO {}: RANDOM ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
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
                # store results per run for learning
                theta_rand.append(results["theta"])
                num_nodes_rand.append(results["num_nodes"])
                robust_rand.append(results["robust"])
                tau_rand.append(results["tau"])
                violation_rand.append(results["violation"])
                # append new taus to N_set
                N_set += results["N_set"]
                # update times
                mp_time += results["mp_time"]
                sp_time += results["sp_time"]
                tot_new_nodes += results["num_nodes"]
            # indexing
            n += 1
            iteration += 1
        tot_nodes += tot_new_nodes

        try:  # CHANGE WEIGHTS
            # analyze from which random results we learn from
            # metric of best random node depends on: {objective, number of nodes, robustness}
            best_rand_run = np.argmin([(theta_rand[i] * num_nodes_rand[i] + violation_rand[i])
                                       for i in np.arange(len(robust_rand))])
            # print([(theta_rand[i] * num_nodes_rand[i] + violation_rand[i])
            #                            for i in np.arange(len(robust_rand))])
            # select last tau rand and last df_att
            tau_rand_best = tau_rand[best_rand_run]
            df_att_best = df_att_list[-1]

            weights, att_perf = update_weights(K, env, weights, tau_rand=tau_rand_best,
                                     df_att=df_att_best,
                                     lr_w=lr_w,
                                     att_series=att_series,
                                     weight_group=weight_group,
                                     x_static=x_static)
            print(f"Instance AO {env.inst_num}, it = {iteration}, att perf = {att_perf}")
        except:
            print(f"Instance AO {env.inst_num}, it = {iteration} finished")

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = len(N_set)
            cum_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "weights": weights, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes,
                           "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
                           "att_time": att_time}
            with open("Results/Decisions/tmp_results_online_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
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
    print("Instance A {} completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "weights": weights, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes,
                           "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
                           "att_time": att_time}

    with open("Results/Decisions/final_results_online_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type="online")
    except:
        pass
    return results


def update_weights(K, env, init_weights, tau_rand, df_att, lr_w, att_series, x_static=None, weight_group=False):
    # make df_rand
    if "nominal" in att_series:
        scen_nom_model = scenario_fun_nominal_build(env)
    else:
        scen_nom_model = None
    if "static" in att_series:
        scen_stat_model = scenario_fun_static_build(env, x_static)
    else:
        scen_stat_model = None

    # initialize df_att
    df_rand = pd.DataFrame()
    i = 0
    for k in np.arange(K):
        for scen in tau_rand[k]:
            new_scen_att = attribute_per_scen(scen, env, att_series,
                                              scen_nom_model=scen_nom_model,
                                              scen_stat_model=scen_stat_model,
                                              x_static=x_static)
            try:
                df_rand.loc[i] = new_scen_att
            except ValueError:
                index = pd.MultiIndex.from_tuples(new_scen_att.index, names=["att_type", "att"])
                df_rand = pd.DataFrame(columns=index)
                df_att.loc[i] = new_scen_att
            df_rand.loc[i, ("subset", 0)] = k
            i += 1

    if "coords" not in att_series:
        df_att = df_att.drop([*[("xi", i) for i in np.arange(env.xi_dim)]], axis=1)
        df_rand = df_rand.drop([*[("xi", i) for i in np.arange(env.xi_dim)]], axis=1)

    # average values of df_att
    X_att = pd.DataFrame(columns=df_att.columns, dtype=np.float32)
    for k in np.arange(K):
        X_att.loc[k] = df_att[df_att[("subset", 0)] == k].var()
    # difference
    X_att = X_att.drop(("subset", 0), axis=1).mean()

    # average values of df_rand
    X_rand = pd.DataFrame(columns=df_rand.columns, dtype=np.float32)
    for k in np.arange(K):
        X_rand.loc[k] = df_rand[df_rand[("subset", 0)] == k].var()
    # difference
    X_rand = X_rand.drop(("subset", 0), axis=1).mean()

    # update weights
    if weight_group:
        # here we have df_att and series together. So use this series as update, then expand for all dimensions again
        weights = pd.Series(0.0, index=init_weights.index)
        for att_type, att_all in X_att.groupby(level=0):
            weight_change = 0
            for att in att_all.index:
                weight_change += (X_att[att]*(init_weights[att]*X_att[att] - X_rand[att]))
            for att in att_all.index:
                weights[att] = init_weights[att] + lr_w*weight_change
    else:
        weights = init_weights + lr_w*(X_att*(init_weights*X_att - X_rand))
    performance = ((X_att - X_rand)**2).mean()
    return weights, performance


def run_function(att_bool, K, env, tau, theta_i, weights, att_series, x_static=None, time_limit=20*60, it=0):
    if att_bool:
        return run_att(K, env, tau, theta_i, weights, att_series, x_static=x_static, time_limit=time_limit, it=it)
    else:
        return run_random(K, env, tau, theta_i, time_limit=time_limit, it=it)


def run_random(K, env, tau, theta_i, time_limit=20*60, it=0):
    # initialize
    mp_time = 0
    sp_time = 0
    N_set = []

    xi_new = None
    k_new = None
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
            robust_bool = False
            break

        # SUBPROBLEM
        start_sp = time.time()
        zeta, xi_new = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp
        if zeta <= 1e-04:
            robust_bool = True
            break

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

        for k in K_set:
            if k == k_new:
                continue
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            adj_tau_k.append(xi_new)
            tau_tmp[k] = adj_tau_k
            N_set.append(tau_tmp)

    num_nodes = sum([len(t) for t in tau.values()])
    print(f"Instance AO {env.inst_num}; RANDOM RUN {it} finished in {np.round(time.time()-start_time, 4)}, #Nodes = {num_nodes}, Robust = {robust_bool}, theta = {theta}")

    return {"theta": theta, "x": x, "y": y, "tau": tau, "robust": robust_bool, "num_nodes": num_nodes, "N_set": N_set,
            "mp_time": mp_time, "sp_time": sp_time, "att_bool": False, "violation": zeta}


def run_att(K, env, tau, theta_i, weights, att_series, x_static=None, time_limit=20*60, it=0):
    # initialize
    mp_time = 0
    sp_time = 0
    att_time = 0
    N_set = []

    if "nominal" in att_series:
        scen_nom_model = scenario_fun_nominal_build(env)
    else:
        scen_nom_model = None
    if "static" in att_series:
        scen_stat_model = scenario_fun_static_build(env, x_static)
    else:
        x_static = None
        scen_stat_model = None

    xi_new = None
    k_new = None
    start_time = time.time()
    # initialize df_att
    df_att = pd.DataFrame()
    i = 0
    for k in np.arange(K):
        for scen in tau[k]:
            new_scen_att = attribute_per_scen(scen, env, att_series,
                                              scen_nom_model=scen_nom_model,
                                              scen_stat_model=scen_stat_model,
                                              x_static=x_static)
            try:
                df_att.loc[i] = new_scen_att
            except ValueError:
                index = pd.MultiIndex.from_tuples(new_scen_att.index, names=["att_type", "att"])
                df_att = pd.DataFrame(columns=index)
                df_att.loc[i] = new_scen_att
            df_att.loc[i, ("subset", 0)] = k
            i += 1

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
            # ATTRIBUTES PER SCENARIO
            start_att = time.time()
            scen_att_new = attribute_per_scen(xi_new, env, att_series,
                                              scen_nom_model=scen_nom_model,
                                              scen_stat_model=scen_stat_model,
                                              x_static=x_static)
            att_time += time.time() - start_att

        # choose k_new and add other tau's to N_set
        full_list = [k for k in np.arange(K) if tau[k]]
        if not full_list:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            start_att = time.time()
            K_set = avg_dist_on_attributes(df_att, scen_att_new, att_series, env.xi_dim, weights)
            k_new = K_set[0]
            att_time += time.time() - start_att
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = np.random.randint(len(K_set))
        # add scen to df_att with subset = k_new
        start_att = time.time()
        new_att = len(df_att)
        df_att.loc[new_att] = scen_att_new
        df_att.loc[new_att, ("subset", 0)] = k_new
        att_time += time.time() - start_att

        for k in K_set:
            if k == k_new:
                continue
            tau_tmp = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau_tmp[k])
            adj_tau_k.append(xi_new)
            tau_tmp[k] = adj_tau_k
            N_set.append(tau_tmp)

    num_nodes = sum([len(t) for t in tau.values()])
    print(f"Instance AO {env.inst_num}; ATT RUN {it} finished in {np.round(time.time()-start_time, 4)}, #Nodes = {num_nodes}, Robust = {robust_bool}, theta = {theta}")

    return {"theta": theta, "x": x, "y": y, "tau": tau, "df_att": df_att, "robust": robust_bool, "num_nodes": num_nodes,
            "N_set": N_set, "mp_time": mp_time, "sp_time": sp_time, "att_time": att_time, "att_bool": True, "violation": zeta}