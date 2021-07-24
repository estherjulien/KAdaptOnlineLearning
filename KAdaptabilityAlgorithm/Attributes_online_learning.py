from ShortestPath.Attributes.att_functions_2s import *
from ShortestPath.ProblemMILPs.functions_2s import *

from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm_main(K, env, att_series, lr_w=.5, lr_t=0.9, sign_lev=.1, weight_group=False,
                   time_limit=20*60, print_info=False, problem_type="test"):
    thread_count = 8
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
    rand_run_time = 0
    att_run_time = 0
    mp_time = 0
    sp_time = 0
    att_time = 0
    # initialization of lower and upper bounds
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    inc_lb = dict()
    inc_lb[0] = 0
    # K-branch and bound algorithm
    now = datetime.now().time()

    num_att = 0
    if "coords" in att_series:
        num_att += env.xi_dim
    if "nominal" in att_series:
        if "nominal_obj" in att_series:
            num_att += 1
        if "nominal_x" in att_series:
            num_att += env.x_dim
        if "nominal_y" in att_series:
            num_att += env.y_dim
    else:
        scen_nom_model = None
    if "static" in att_series:
        x_static = static_solution_rc(env)
        if "static_obj" in att_series:
            num_att += 1
        if "static_y" in att_series:
            num_att += env.y_dim
    else:
        x_static = None
        scen_stat_model = None

    # initialize weights (all 1)
    weights = np.ones(num_att)

    print("Instance AO {} started at {}".format(env.inst_num, now))

    # todo: finish this online learning part
    while time.time() - start_time < time_limit:
        # RANDOM NODE RUNS
        rand_run_num = int(np.floor(thread_count*lr_t))
        # results = run_random(K, env, N_set[0], theta_i)
        results_rand = Parallel(n_jobs=thread_count)(delayed(run_random)(K, env, N_set[n], theta_i)
                                                     for n in np.arange(min(rand_run_num, len(N_set))))
        # delete random nodes
        del N_set[:rand_run_num]
        # analyze random nodes
        theta_rand = []
        num_nodes = []
        robust = []
        tau_rand = []
        for results in results_rand:
            theta_rand.append(results["theta"])
            num_nodes.append(results["num_nodes"])
            robust.append(results["robust"])
        # append new taus to N_set
        # change weights
        # if not weight_group:
        #     weights = update_weights(K, env, weights, tau_rand, df_att, lr_w, att_series)
        # else:
        #     pass
        # ATTRIBUTE NODE RUNS
        att_run_num = thread_count - rand_run_num
        results_att = run_att(K, env, N_set[0], theta_i, att_series, weights, x_static=x_static)
        # results_att = Parallel(n_jobs=thread_count)(delayed(run_att)(K, env, N_set[n], theta_i, att_series, weights,
        #                                                              x_static=x_static) for n in att_run_num)
        # delete att nodes
        del N_set[:att_run_num]
        # analyze att nodes

        # append new taus to N_set

    # analyze best solutions

    return None


def update_weights(K, env, init_weights, tau_rand, df_att, lr_w, att_series):
    # make df_rand
    df_rand = pd.DataFrame()
    i = 0
    for k in np.arange(K):
        for scen in tau_rand[k]:
            new_scen_att = attribute_per_scen(scen, env, att_series)
            try:
                df_rand.iloc[i] = attribute_per_scen(scen, env, att_series)
            except IndexError:
                df_rand = pd.DataFrame(columns=new_scen_att.index)
                df_rand.loc[i] = new_scen_att
            df_rand.loc[i, "subset"] = k
            i += 1

    K = len(df_att["subset"].unique())
    if "coords" in att_series:
        df_att = df_att.drop([*["xi{}".format(i) for i in np.arange(env.xi_dim)]], axis=1)
        df_rand = df_rand.drop([*["xi{}".format(i) for i in np.arange(env.xi_dim)]], axis=1)

    # average values of df_att
    X_att = pd.DataFrame(index=np.arange(K), columns=df_att.columns, dtype=np.float32)
    for k in np.arange(K):
        X_att.loc[k] = df_att[df_att["subset"] == k].mean()
    X_att = X_att.drop("subset", axis=1)

    # average values of df_rand
    X_rand = pd.DataFrame(index=np.arange(K), columns=df_att.columns, dtype=np.float32)
    for k in np.arange(K):
        X_rand.loc[k] = df_rand[df_rand["subset"] == k].mean()
    X_rand = X_rand.drop("subset", axis=1)

    # update weights
    weights = init_weights + lr_w*(X_att*(init_weights*X_att - X_rand))

    return weights


def run_random(K, env, tau, theta_i, scen_model_init=None,
               time_limit=20*60):
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
    return {"theta": theta, "x": x, "y": y, "tau": tau, "robust": robust_bool, "num_nodes": num_nodes, "N_set": N_set,
            "mp_time": mp_time, "sp_time": sp_time}


def run_att(K, env, tau, att_series, theta_i, weights, scen_model_init=None,
            scen_nom_model=None, scen_stat_model=None, x_static=None,
            start_time=0, time_limit=20*60):
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
                df_att.iloc[i] = attribute_per_scen(scen, env, att_series)
            except IndexError:
                df_att = pd.DataFrame(columns=new_scen_att.index)
                df_att.loc[i] = new_scen_att
            df_att.loc[i, "subset"] = k
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
        df_att.loc[new_att] = k_new
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
    return {"theta": theta, "x": x, "y": y, "tau": tau, "df_att": df_att, "robust": robust_bool,
            "num_nodes": num_nodes, "N_set": N_set, "mp_time": mp_time, "sp_time": sp_time, "att_time": att_time}
