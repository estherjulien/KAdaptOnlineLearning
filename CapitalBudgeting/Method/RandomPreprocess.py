from ProblemFunctions.functions_milp import *

from datetime import datetime
import numpy as np
import pickle
import copy
import time


def algorithm(K, env, time_limit=30*60, print_info=True, problem_type="test", num_runs=5):
    # Initialize
    iteration = 0
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas_t = dict()
    inc_thetas_n = dict()
    prune_count = 0
    inc_tot_nodes = dict()
    tot_nodes = 0
    inc_tot_nodes[0] = 0
    mp_time = 0
    sp_time = 0
    # initialization of lower and upper bounds
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    # K-branch and bound algorithm
    now = datetime.now().time()
    xi_new, k_new = None, None

    new_model = True
    # initialize N_set with actual scenario
    N_set, tau, scen_all, placement, theta, x, y, model = fill_subsets(K, env)

    new_xi_num = K - 1

    for i in np.arange(num_runs):
        theta, tot_scens, new_nodes, num_per_subset = random_pass(K, env, tau)
        tot_nodes += new_nodes
        if theta - theta_i > -1e-8:
            continue
        else:   # store incumbent theta
            if print_info:
                now = datetime.now().time()
                print("Instance R {}: ROBUST DURING PREPROCESSING at iteration {} ({}) (time {})   :theta = {},  Xi{},   "
                      "prune count = {}".format(env.inst_num, iteration, np.round(time.time()-start_time, 3), now,
                                                np.round(theta, 4), num_per_subset, prune_count))

            theta_i = copy.deepcopy(theta)
            inc_thetas_t[time.time() - start_time] = theta_i
            inc_thetas_n[tot_nodes] = theta_i

    print(f"Instance R {env.inst_num}: started at {now}")
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
            # make new tau from k_new
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
        zeta, xi = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance R {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    zeta = {}   Xi{},   "
                      "prune count = {}".format(env.inst_num, iteration, np.round(time.time()-start_time, 3), now,
                                                np.round(theta, 4), np.round(zeta, 4),
                                                [len(t) for t in placement.values()], prune_count))
            theta_i, x_i, y_i = (copy.deepcopy(theta), copy.deepcopy(x), copy.deepcopy(y))
            inc_thetas_t[time.time() - start_time] = theta_i
            inc_thetas_n[tot_nodes] = theta_i
            prune_count += 1
            new_model = True
            continue
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])

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

        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas_t[time.time() - start_time] = theta_i
    inc_thetas_n[tot_nodes] = theta_i

    now = datetime.now().time()
    now_nice = f"{now.hour}:{now.minute}:{now.second}"
    print(f"Instance R {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")
    results = {"theta": theta_i, "inc_thetas_t": inc_thetas_t,
               "inc_thetas_n": inc_thetas_n, "runtime": time.time() - start_time,
               "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

    with open(f"CapitalBudgeting/Data/Results/Decisions/inst_results/final_results_{problem_type}_"
              f"inst{env.inst_num}.pickle", "wb") as handle:
        pickle.dump(results, handle)

    return results


def fill_subsets(K, env, progress=False):
    # Initialize
    start_time = time.time()

    # K-branch and bound algorithm
    new_model = True

    xi_init = init_scen(K, env)

    N_set = [{k: [] for k in np.arange(K)}]
    N_set[0][0].append(0)
    scen_all = xi_init.reshape([1, -1])
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
            # master problem
            placement[k_new].append(new_xi_num)
            theta, x, y, model = scenario_fun_update(K, k_new, xi, env, model)

            tau = {k: scen_all[placement[k]] for k in np.arange(K)}

        # SUBPROBLEM
        zeta, xi = separation_fun(K, x, y, theta, env, tau)

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
        if len(full_list) == K:
            N_set.append(placement)
            break

        # check if robust
        if zeta < 1e-04:
            now = datetime.now().time()
            if progress:
                print(
                    "Instance SP {}: ROBUST IN FILLING SUBSETS RUN ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False
            new_xi_num += 1
            scen_all = np.vstack([scen_all, xi])

        full_list = [k for k in np.arange(K) if len(tau[k]) > 0]
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

    return N_set, tau, scen_all, placement, theta, x, y, model


def init_scen(K, env):
    tau = {k: [] for k in np.arange(K)}
    tau[0].append(env.init_uncertainty)

    # run master problem
    theta, x, y, _ = scenario_fun_build(K, tau, env)

    # run sub problem
    zeta, xi_new = separation_fun(K, x, y, theta, env, tau)

    return xi_new


def random_pass(K, env, tau, progress=False):
    # Initialize
    start_time = time.time()
    new_model = True
    tot_nodes = 1
    while True:
        # MASTER PROBLEM
        if new_model:
            # master problem
            theta, x, y, model = scenario_fun_build(K, tau, env)
        else:
            tot_nodes += 1
            # NEW NODE from k_new
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
            if progress:
                now = datetime.now().time()
                print(
                    "Instance R {}: RAND PASS ROBUST ({}) (time {})   :theta = {},    zeta = {}   Xi{}".format(
                        env.inst_num, np.round(time.time() - start_time, 3), now,
                        np.round(theta, 4), np.round(zeta, 4), [len(t) for t in tau.values()]))
            break
        else:
            new_model = False

        # choose new k randomly
        k_new = np.random.randint(K)

    num_per_subset = [len(t) for t in tau.values()]
    tot_scens = np.sum(num_per_subset)
    return theta, tot_scens, tot_nodes,  num_per_subset
