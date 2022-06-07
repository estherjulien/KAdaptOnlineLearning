# CHANGE THIS FOR NEW PROBLEMS
from NOT_USED.CapitalBudgetingLoans.Attributes.att_functions import *

from joblib import Parallel, delayed
from datetime import datetime
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, att_series, sub_tree=True, n_back_track=2, time_limit=20 * 60, problem_type="test",
              thread_count=8, depth=1, width=50, max_depth=5):
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
    strategy_results = dict()
    weight_data = []
    state_data = []
    # K-branch and bound algorithm
    now = datetime.now().time()

    weight_model_name = f"nn_model_{problem_type}_D{depth}_W{width}_inst{env.inst_num}.h5"

    # FOR STATIC ATTRIBUTE
    try:
        x_static = static_solution_rc(env)
    except:
        x_static = None

    # initialize N_set
    theta_i, x_i, y_i, N_set, tau_i, N_att_set, tot_nodes_new, init_tot_scens, init_zeta = init_pass(K, env, att_series,
                                                                                                     x_static)
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
    print("Instance OL {}: started at {}".format(env.inst_num, now))
    num_strategy = 0
    while N_set and time.time() - start_time < time_limit:
        # PASSES
        pass_num = min(thread_count, len(N_set))
        if new_experts > 0:
            if num_strategy == 0:
                num_strategy = 1
            elif strat_rand_ratio > 1:
                num_strategy = min(int(pass_num / 2), num_strategy + 1)
            elif strat_rand_ratio > 0.99:
                pass
            else:
                num_strategy = max(0, num_strategy - 1)
        else:
            pass
        explore_bool = [*[True] * (thread_count - num_strategy), *[False] * num_strategy]
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
        tot_results = Parallel(n_jobs=thread_count)(delayed(parallel_pass)(K, env, att_series,
                                                                           N_set[i], N_att_set[i],
                                                                           theta_i, init_zeta,
                                                                           init_tot_scens, att_index,
                                                                           explore_bool=explore_bool[i],
                                                                           weight_model_name=weight_model_name,
                                                                           sub_tree=sub_tree,
                                                                           x_static=x_static)
                                                    for i in np.arange(pass_num))
        for i in np.arange(pass_num):
            results[i], N_set_new[i], N_att_set_new[i] = tot_results[i]
        del tot_results

        # ADD NEW NODES
        for i in np.arange(pass_num):
            N_set += N_set_new[i]
            N_att_set += N_att_set_new[i]

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
                score = (results[i]["theta"] / theta_i_old) * max([results[i]["zeta"] / init_zeta + 1, 1])
                # score = results[i]["theta"]/theta_i     # maybe only if robust?
                pass_score.append(score)
                explore_theta.append(results[i]["theta"])
                # print results
                theta_tmp = np.round(results[i]["theta"], 4)
                zeta_tmp = np.round(results[i]["zeta"], 4)
                run_tmp = np.round(results[i]["runtime"], 4)
                tau_tmp = results[i]["tau"]
                now = datetime.now().time()
                print(
                    f"Instance OL {env.inst_num}: [{num_explore_passes}] it {iteration} explore pass finished ({run_tmp}) (time {now})   "
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
            strat_rand_ratio = np.mean(strategy_theta) / np.mean(explore_theta)
        else:
            strat_rand_ratio = 1

        # SELECT EXPERTS
        if sub_tree:
            new_experts_list = []
            for i in np.arange(len(pass_score) - num_explore, len(pass_score)):
                if pass_score[i] > 0.98 and len(explore_results[i]["state_data"]) > 1:
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
                                                                                      n_back_track,
                                                                                      theta_i_old,
                                                                                      init_zeta, init_tot_scens,
                                                                                      att_index, i, max_depth,
                                                                                      x_static=x_static)
                                                               for i in new_experts_list)

            expert_results = []
            new_experts = 0
            print(f"solved: {new_experts_list}, obtained: {[res[3] for res in tmp_expert_results]}")
            for results, runtime, tot_nodes_new, i_explore in tmp_expert_results:
                if results is None:
                    # SCORE
                    if pass_score[i_explore] > 0.999:
                        new_experts += 1
                        expert_results.append(explore_results[i_explore])
                    tot_nodes += tot_nodes_new
                    continue
                # SCORE
                score = (results["theta"] / theta_i_old) * max([results["zeta"] / init_zeta + 1, 1])
                if score > 0.999:
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
                if pass_score[i] > 0.999 and len(explore_results[i]["state_data"]) > 1:
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
        if time.time() - start_time - prev_save_time > 10 * 60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
                           "inc_thetas_n": inc_thetas_n, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                           "runtime": time.time() - start_time, "inc_tot_nodes": inc_tot_nodes, "tot_nodes": tot_nodes,
                           "state_data": state_data, "weight_data": weight_data}
            with open("Results/Decisions/tmp_results_{}_inst{}.pickle".format(problem_type, env.inst_num),
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

    print("Instance OL {}, completed at {}, solved in {} minutes".format(env.inst_num, now, runtime / 60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas_t": inc_thetas_t,
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
