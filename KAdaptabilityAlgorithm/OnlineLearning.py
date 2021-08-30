# CHANGE THIS FOR NEW PROBLEMS
from ShortestPath.ProblemMILPs.functions import *
from ShortestPath.Attributes.att_functions import *

from tensorflow.keras.models import load_model
from joblib import Parallel, delayed
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, att_series, sub_tree=True, n_back_track=2, time_limit=20*60, problem_type="test", thread_count=8, depth=2, width=50, max_depth=5):
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
    weight_data = []
    state_data = []
    # K-branch and bound algorithm
    now = datetime.now().time()

    weight_model_name = f"nn_model_K{K}_{problem_type}_D{depth}_W{width}_inst{env.inst_num}.h5"
    # initialize N_set
    theta_i, x_i, y_i, tau_i, N_set, scen_all, att_all, tot_nodes_new, init_tot_scens, zeta_init = init_pass(K, env, att_series)
    theta_i_init = copy.copy(theta_i)
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

    new_experts = 0
    strat_rand_ratio = 1
    new_xi_num = len(scen_all)-1

    print("Instance OL {}: started at {}".format(env.inst_num, now))
    num_strategy = 0
    while N_set and time.time() - start_time < time_limit:
        # PASSES
        pass_num = min(thread_count, len(N_set))
        if new_experts > 0:
            if num_strategy == 0:
                num_strategy = 1
            elif strat_rand_ratio < 1:
                num_strategy = min(int(pass_num/2), num_strategy + 1)
            elif strat_rand_ratio < 1.01:
                pass
            else:
                num_strategy = max(0, num_strategy - 1)
        else:
            pass
        explore_bool = [*[True]*(thread_count - num_strategy), *[False]*num_strategy]
        results = dict()
        N_set_new = dict()
        N_att_set_new = dict()

        # iterative
        # for i in np.arange(pass_num):
        #     results[i], N_set_new[i], N_att_set_new[i] = \
        #         parallel_pass(K, env, att_series, N_set[i], N_att_set[i], theta_i, init_zeta, init_tot_scens, start_time, att_index,
        #                       explore_bool=explore_bool[i], weight_model_name=weight_model_name, sub_tree=sub_tree)
        # parallel
        time_before_run = time.time() - start_time
        # make tau_set and tau_att_set
        tau_set = [{k: scen_all[placement[k]] for k in np.arange(K)} for placement in N_set[:pass_num]]
        tau_att_set = [{k: att_all[placement[k]] for k in np.arange(K)} for placement in N_set[:pass_num]]

        tot_results = Parallel(n_jobs=thread_count)(delayed(parallel_pass)(K, env, att_series,
                                                                           tau_set[i], tau_att_set[i],
                                                                           theta_i, zeta_init,
                                                                           init_tot_scens, att_index,
                                                                           explore_bool=explore_bool[i],
                                                                           weight_model_name=weight_model_name,
                                                                           sub_tree=sub_tree)
                                                    for i in np.arange(pass_num))
        for i in np.arange(pass_num):
            results[i], N_set_new[i], scen_all_new, att_all_new = tot_results[i]
        del tot_results

        # ADD NEW NODES
        for i in np.arange(pass_num):
            N_set += N_set_new[i]
            scen_all = np.vstack([scen_all, scen_all_new])
        N_set = N_set[pass_num:]
        N_att_set = N_att_set[pass_num:]

        # SAVE RESULTS
        num_explore = 0
        explore_theta = []
        strategy_theta = []
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
            # for both
            tot_nodes += results[i]["tot_nodes"]
            if results[i]["explore"]:
                num_explore += 1
                num_explore_passes = len(explore_results)
                explore_results[num_explore_passes] = results[i]
                score = (results[i]["theta"]/theta_i_init)*max([results[i]["zeta"]/init_zeta+1, 1])
                # score = results[i]["theta"]/theta_i     # maybe only if robust?
                pass_score.append(score)
                explore_theta.append(results[i]["theta"])
                # print results
                theta_tmp = np.round(results[i]["theta"], 4)
                zeta_tmp = np.round(results[i]["zeta"], 4)
                run_tmp = np.round(results[i]["runtime"], 4)
                tau_tmp = results[i]["tau"]
                now = datetime.now().time()
                print(f"Instance OL {env.inst_num}: [{num_explore_passes}] it {iteration} explore pass finished ({run_tmp}) (time {now})   "
                      f":theta = {theta_tmp},    zeta = {zeta_tmp},     pass_score = {np.round(score, 4)}   Xi{[len(t) for t in tau_tmp.values()]}")
            else:
                num_strategy_passes = len(strategy_results)
                strategy_results[num_strategy_passes] = results[i]
                strategy_theta.append(results[i]["theta"])
                # print results
                theta_tmp = np.round(results[i]["theta"], 4)
                zeta_tmp = np.round(results[i]["zeta"], 4)
                run_tmp = np.round(results[i]["runtime"], 4)
                tau_tmp = results[i]["tau"]
                now = datetime.now().time()
                print(f"Instance OL {env.inst_num}: it {iteration} strategy pass finished ({run_tmp}) (time {now})   "
                      f":theta = {theta_tmp},    zeta = {zeta_tmp}   Xi{[len(t) for t in tau_tmp.values()]}")
            iteration += 1

        if num_strategy > 0:
            strat_rand_ratio = np.mean(strategy_theta)/np.mean(explore_theta)
        else:
            strat_rand_ratio = 1

        # SELECT EXPERTS
        if sub_tree:
            new_experts_list = []
            for i in np.arange(len(pass_score) - num_explore, len(pass_score)):
                if pass_score[i] < 1.01 and len(explore_results[i]["state_data"]) > 1:
                    # change state and weight data for exploring "expert" results of sub-tree
                    new_experts_list.append(i)

            time_before_run = time.time() - start_time
            # tmp_expert_results = []
            # for i in new_experts_list:
            #     tmp_expert_results.append(sub_tree_pass(K, env, att_series, explore_results[i]["sub_tree"],
            #                                             explore_results[i]["state_data"],
            #                                             explore_results[i]["weight_data"],
            #                                             n_back_track, explore_results[i]["theta"], init_zeta,
            #                                             init_tot_scens, att_index, i))

            tmp_expert_results = Parallel(n_jobs=thread_count)(delayed(sub_tree_pass)(K, env, att_series,
                                                                                      explore_results[i]["sub_tree"],
                                                                                      explore_results[i]["state_data"],
                                                                                      explore_results[i]["weight_data"],
                                                                                      n_back_track, theta_i_old,
                                                                                      zeta_init, init_tot_scens,
                                                                                      att_index, i, max_depth)
                                                               for i in new_experts_list)

            expert_results = []
            new_experts = 0
            print(f"solved: {new_experts_list}, obtained: {[res[3] for res in tmp_expert_results]}")
            for results, runtime, tot_nodes_new, i_explore in tmp_expert_results:
                if results is None:
                    # SCORE
                    if pass_score[i_explore] < 1.001:
                        new_experts += 1
                        expert_results.append(explore_results[i_explore])
                    tot_nodes += tot_nodes_new
                    continue
                # SCORE
                score = (results["theta"] / theta_i_init) * max([results["zeta"] / init_zeta + 1, 1])
                if score < 1.001:
                    new_experts += 1
                    expert_results.append(results)
                    if results["zeta"] < 1e-4 and results["theta"] - theta_i < -1e-8:
                        # RESULT ROBUST
                        now = datetime.now().time()
                        theta_i, x_i, y_i = (results["theta"], results["x"], results["y"])
                        tau_i = copy.deepcopy(results["tau"])
                        tot_nodes_new = results["tot_nodes"]
                        print("Instance OL {}, {}: SUB TREE PASS ROBUST ({}) (time {})   :theta = {},    "
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
                # always
                tot_nodes += tot_nodes_new

            for results in expert_results:
                try:
                    state_data = np.vstack([state_data, results["state_data"]])
                except ValueError:
                    state_data = results["state_data"]
                try:
                    weight_data = np.vstack([weight_data, results["weight_data"]])
                except ValueError:
                    weight_data = results["weight_data"]

        else:
            for i in np.arange(len(pass_score) - num_explore, len(pass_score)):
                if pass_score[i] < 1.1 and len(explore_results[i]["state_data"]) > 0:
                    # if score diverges 10 percent from best one, add it
                    # add state and weight data
                    try:
                        state_data = np.vstack([state_data, explore_results[i]["state_data"]])
                    except ValueError:
                        state_data = explore_results[i]["state_data"]
                    try:
                        weight_data = np.vstack([weight_data, explore_results[i]["weight_data"]])
                    except ValueError:
                        weight_data = explore_results[i]["weight_data"]
                    new_experts += 1

        # update weights
        if new_experts > 0 and len(state_data[-new_experts:]):
            print(f"Instance OL {env.inst_num}: update weights with {new_experts} new experts. "
                  f"ratio = {strat_rand_ratio}, tot_data_points = {len(state_data)}")
            update_weights_fun(state_data[-new_experts:], weight_data[-new_experts:], depth=depth, width=width,
                               weight_model_name=weight_model_name)

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i,  "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
                           "state_data": state_data, "weight_data": weight_data}
            with open("Results/Decisions/tmp_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
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

    print("Instance OL {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i,  "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
               "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
               "state_data": state_data, "weight_data": weight_data}

    with open("Results/Decisions/final_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i, alg_type=problem_type)
    except:
        pass
    return results


def parallel_pass(K, env, att_series, tau, tau_att, theta_i, zeta_i, tot_scens_i, att_index, explore_bool, weight_model_name, sub_tree=False):
    if explore_bool:
        return explore_pass(K, env, att_series, tau, tau_att, theta_i, zeta_i, tot_scens_i, att_index, sub_tree=sub_tree)
    else:
        return strategy_pass(K, env, att_series, tau, tau_att, theta_i, zeta_i, tot_scens_i, weight_model_name, att_index)


def init_pass(K, env, att_series):
    # Initialize
    start_time = time.time()
    tot_nodes = 0

    # K-branch and bound algorithm
    new_model = True

    # initialize N_set with actual scenario
    xi_init, att_init, zeta_init = init_scen(K, env, att_series)
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
                "Instance OL {}: INIT PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                    env.inst_num, np.round(time.time() - start_time, 3), now,
                    np.round(theta, 4), np.round(zeta, 4), [len(t) for t in placement.values()]))
            break
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)
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


def explore_pass(K, env, att_series, tau, tau_att, theta_i, zeta_i, tot_scens_i, att_index, sub_tree=False):
    # Initialize
    tot_nodes = 0

    # K-branch and bound algorithm
    start_time = time.time()
    new_model = True

    # initialize N_set with actual scenario
    N_set = []
    N_att_set = []

    pass_track = []

    state_data = []
    weight_data = []
    df_att = tau_att

    now = datetime.now().time()
    # print("Instance OL {}: explore pass started at {}".format(env.inst_num, now))
    while True:
        # MASTER PROBLEM
        if new_model:
            try:
                del model
            except:
                pass
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
            df_att = copy.deepcopy(np.vstack([df_att, scen_att]))
            df_att[-1, 0] = k_new
            # master problem
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)
        scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

        if sub_tree:
            pass_track.append([tau, df_att])

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == 0:
            K_set = [0]
            k_new = 0
        elif len(full_list) == K:
            K_set = np.arange(K)
            k_new = np.random.randint(K)
            # STATE DATA
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, tot_scens_i, df_att, theta_i, zeta_i, att_index)
            try:
                state_data = np.vstack([state_data, tau_s])
            except:
                state_data = tau_s.reshape([1, -1])
            # WEIGHT DATA
            tau_w = weight_labels(K, df_att, scen_att, k_new, att_index)
            try:
                weight_data = np.vstack([weight_data, tau_w])
            except:
                weight_data = tau_w.reshape([1, -1])
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

        # PRUNE
        if theta - theta_i > -1e-8:
            break

        if zeta < 1e-04:
            now = datetime.now().time()
            print("Instance OL {}: EXPLORE PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
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

            # NEW ATT NODE
            df_att_tmp = copy.deepcopy(np.vstack([df_att, scen_att]))
            df_att_tmp[-1, 0] = k
            N_att_set.append(df_att_tmp)
    tot_scens = np.sum([len(t) for t in tau.values()])
    runtime = time.time() - start_time

    results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes, "tot_scens": tot_scens,
               "state_data": state_data, "weight_data": weight_data, "explore": True, "runtime": runtime,
               "sub_tree": pass_track}
    return results, N_set, N_att_set


def strategy_pass(K, env, att_series, tau, tau_att, theta_i, zeta_i, tot_scens_i, weight_model_name, att_index):
    # Initialize
    tot_nodes = 0

    # weight model
    weight_model = load_model(weight_model_name)
    start_time = time.time()
    # K-branch and bound algorithm
    now = datetime.now().time()
    new_model = True

    # initialize N_set with actual scenario
    N_set = []
    N_att_set = []

    zeta = 0

    df_att = tau_att
    # print("Instance OL {}: strategy pass started at {}".format(env.inst_num, now))
    while True:
        # MASTER PROBLEM
        if new_model:
            try:
                del model
            except:
                pass
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

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            print(
                "Instance OL {}: STRATEGY PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
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
            # predict subset
            # STATE FEATURES (based on master and sub problem)
            tot_scens = np.sum([len(t) for t in tau.values()])
            tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, tot_scens_i, df_att, theta_i, zeta_i, att_index)
            K_set = predict_subset(K, df_att, scen_att, weight_model, att_index, tau_s)
            k_new = K_set[0]
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
            k_new = K_set[-1]

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

            # NEW ATT NODE
            df_att_tmp = copy.deepcopy(np.vstack([df_att, scen_att]))
            df_att_tmp[-1, 0] = k
            N_att_set.append(df_att_tmp)
    tot_scens = np.sum([len(t) for t in tau.values()])

    runtime = time.time() - start_time

    results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes,
               "tot_scens": tot_scens, "explore": False, "runtime": runtime}
    return results, N_set, N_att_set


def init_scen(K, env, att_series):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta_init, xi_init = separation_fun(K, x, y, theta, env, tau)

    # new tau to be saved in N_set
    att_init = attribute_per_scen(K, xi_init, env, att_series, tau, theta, x, y)

    return xi_init, att_init, zeta_init


def sub_tree_pass(K, env, att_series, pass_track, state_data_init, weight_data_init, n_back_track, theta_i, zeta_i,
                  tot_scens_i, att_index, i_explore, max_depth=5):
    # Initialize
    results = None

    tot_nodes = 0
    theta_i_old = copy.copy(theta_i)
    # K-branch and bound algorithm
    start_time = time.time()

    state_data_dict = dict()
    weight_data_dict = dict()

    try:
        tau, tau_att = pass_track[-n_back_track]
        try:
            state_data_init = state_data_init[:-n_back_track]
            weight_data_init = weight_data_init[:-n_back_track]
        except IndexError:
            state_data_init = []
            weight_data_init = []
    except IndexError:
        tau, tau_att = pass_track[0]
        state_data_init = []
        weight_data_init = []

    # node information
    node_index = (0,)
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

    # INITIAL MODEL
    theta, x, y, model = scenario_fun_build(K, tau, env)
    zeta, xi = separation_fun(K, x, y, theta, env, tau)
    xi_dict[node_index] = xi
    scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)
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

        # NEW ATT NODE
        df_att_tmp = copy.deepcopy(np.vstack([tau_att, scen_att]))
        df_att_tmp[-1, 0] = k
        tau_att_dict[node_index + (k,)] = df_att_tmp

        # NEW LEVELS
        level_next[node_index + (k,)] = tuple(K_set)

    # model storage
    now = datetime.now().time()
    # print(f"Instance OL {env.inst_num}: [{i_explore}] sub-tree pass started at {now}")

    while len(level_next) and len(node_index) < max_depth:
        node_index = list(level_next.keys())[0]
        while len(level_next[node_index]):
            k_new = level_next[node_index][0]
            level_next[node_index] = level_next[node_index][1:]
            new_node_index = node_index + (k_new,)
            tot_nodes += 1
            # ALGORITHM
            tau = tau_dict[new_node_index]
            df_att = tau_att_dict[new_node_index]
            # MASTER PROBLEM
            theta, x, y = scenario_fun_update_sub_tree(K, new_node_index, xi_dict, env, model)

            # SUBPROBLEM
            zeta, xi = separation_fun(K, x, y, theta, env, tau)
            scen_att = attribute_per_scen(K, xi, env, att_series, tau, theta, x, y)

            full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
            if len(full_list) == 0:
                K_set = [0]
            elif len(full_list) == K:
                K_set = np.arange(K)
                # STATE DATA
                tot_scens = np.sum([len(t) for t in tau.values()])
                tau_s = state_features(K, env, theta, zeta, x, y, tot_scens, tot_scens_i, df_att, theta_i, zeta_i,
                                       att_index)
                try:
                    state_data_dict[new_node_index] = np.vstack([state_data_dict[node_index], tau_s])
                except:
                    state_data_dict[new_node_index] = tau_s.reshape([1, -1])
                # WEIGHT DATA
                tau_w = weight_labels(K, df_att, scen_att, k_new, att_index)
                try:
                    weight_data_dict[new_node_index] = np.vstack([weight_data_dict[node_index], tau_w])
                except:
                    weight_data_dict[new_node_index] = tau_w.reshape([1, -1])
            else:
                K_prime = min(K, full_list[-1] + 2)
                K_set = np.arange(K_prime)
            # print(f"{i_explore}: {new_node_index}   theta = {theta}, zeta = {zeta}, theta_i = {theta_i}")
            # prune if theta higher than current robust theta
            if theta - theta_i > -1e-8:
                # delete stuff
                del level_next[new_node_index]
                del state_data_dict[new_node_index]
                del weight_data_dict[new_node_index]
                del tau_dict[new_node_index]
                del tau_att_dict[new_node_index]
                continue

            # check if robust
            if zeta < 1e-04:
                theta_i = copy.deepcopy(theta)
                # SAVE PASS
                try:
                    state_data_final = np.vstack([state_data_init, state_data_dict[new_node_index]])
                except:
                    state_data_final = state_data_dict[new_node_index]
                try:
                    weight_data_final = np.vstack([weight_data_init, weight_data_dict[new_node_index]])
                except:
                    weight_data_final = weight_data_dict[new_node_index]
                tot_scens = np.sum([len(t) for t in tau.values()])
                results = {"theta": theta, "x": x, "y": y, "tau": tau, "zeta": zeta, "tot_nodes": tot_nodes,
                           "state_data": state_data_final, "weight_data": weight_data_final, "tot_scens": tot_scens}
                # delete stuff
                del level_next[new_node_index]
                del state_data_dict[new_node_index]
                del weight_data_dict[new_node_index]
                del tau_dict[new_node_index]
                del tau_att_dict[new_node_index]
                continue

            xi_dict[new_node_index] = xi

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

                # NEW ATT NODE
                df_att_tmp = copy.deepcopy(np.vstack([df_att, scen_att]))
                df_att_tmp[-1, 0] = k
                tau_att_dict[new_node_index + (k,)] = df_att_tmp

                # NEW LEVELS
                level_next[new_node_index + (k,)] = tuple(K_set)
        # after each set of children
        del level_next[node_index]

    if results is None:
        print(f"Instance OL {env.inst_num}, [{i_explore}]: No better solution found in sub tree pass, theta = {theta_i_old}")
    else:
        print(f"Instance OL {env.inst_num}, [{i_explore}]: better solution found in sub tree pass, theta = {theta_i}, "
              f"theta_old = {theta_i_old}")

    # termination actions
    runtime = time.time() - start_time
    return results, runtime, tot_nodes, i_explore
