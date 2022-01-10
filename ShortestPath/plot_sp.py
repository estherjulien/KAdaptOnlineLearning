import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle


def plot_stuff(K, N, num_inst, rf_num):
    env_rand = dict()
    env_strat = dict()
    results_random = dict()
    results_strategy = dict()
    for i in np.arange(num_inst):
        with open(f"Data/Results/Decisions/Random/final_results_sp_random_K{K}_N{N}_g30_fs10_inst{i}.pickle", "rb") as handle:
            env_rand[i], results_random[i] = pickle.load(handle)
        theta_rand = results_random[i]["theta"]
        with open(f"ResultsSucPred/Decisions/final_results_sp_suc_pred_strategy_K{K}_N{N}_rf{rf_num}_inst{i}.pickle", "rb") as handle:
            env_strat[i], results_strategy[i] = pickle.load(handle)
        theta_strat = results_strategy[i]["theta"]
        print(f"Instance {i}: R {theta_rand}    S {theta_strat}")
    # PLOT RESULTS OVER RUNTIME
    rt_tmp = np.max([np.round(results_random[i]["runtime"]) for i in np.arange(num_inst)])
    run_time = rt_tmp - rt_tmp % 1800
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, run_time+15, 15)])
    num_grids = len(t_grid)
    obj_rand = pd.DataFrame(index=t_grid, columns=np.arange(num_inst), dtype=np.float)
    obj_strat = pd.DataFrame(index=t_grid, columns=np.arange(num_inst), dtype=np.float)
    for i in np.arange(num_inst):
        # normal
        t_rand = np.zeros(num_grids)
        theta_final = results_random[i]["theta"]
        for t, theta in results_random[i]["inc_thetas_t"].items():
            t_rand[t_grid > t] = theta/theta_final
        obj_rand[i] = t_rand

        # normal
        t_strat = np.zeros(num_grids)
        for t, theta in results_strategy[i]["inc_thetas_t"].items():
            t_strat[t_grid > t] = theta/theta_final
        obj_strat[i] = t_strat

    obj_rand = np.array(obj_rand)
    obj_rand[obj_rand == 0] = np.nan
    obj_strat = np.array(obj_strat)
    obj_strat[obj_strat == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    avg_norm = np.quantile(obj_rand, 0.5, axis=1)
    norm_10 = np.quantile(obj_rand, 0.1, axis=1)
    norm_90 = np.quantile(obj_rand, 0.9, axis=1)
    plt.fill_between(t_grid, norm_10, norm_90, color="k", alpha=0.5)
    plt.plot(t_grid, avg_norm, "k", label="Random")

    avg_norm = np.quantile(obj_strat, 0.5, axis=1)
    norm_10 = np.quantile(obj_strat, 0.1, axis=1)
    norm_90 = np.quantile(obj_strat, 0.9, axis=1)
    plt.fill_between(t_grid, norm_10, norm_90, color="r", alpha=0.5)
    plt.plot(t_grid, avg_norm, "r", label="Success Prediction")

    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.ylim([0.998, 1.005])
    plt.legend()
    plt.savefig(f"plot_runtime_suc_pred_sp_K{K}_N{N}_rf{rf_num}_{num_inst}")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results_strategy[i]["tot_nodes"] for i in np.arange(num_inst)])
    n_grid = np.arange(0, n_grid_max+10, 10)
    num_grids = len(n_grid)
    obj_rand = pd.DataFrame(index=n_grid, columns=np.arange(num_inst), dtype=np.float16)
    obj_strat = pd.DataFrame(index=n_grid, columns=np.arange(num_inst), dtype=np.float16)

    for i in np.arange(num_inst):
        # normal
        n_rand = np.zeros(num_grids)
        theta_final = results_random[i]["theta"]
        for n, theta in results_random[i]["inc_thetas_n"].items():
            n_rand[n_grid > n] = theta/theta_final
        obj_rand[i] = n_rand

        n_rand = np.zeros(num_grids)
        for n, theta in results_strategy[i]["inc_thetas_n"].items():
            n_rand[n_grid > n] = theta/theta_final
        obj_strat[i] = n_rand

    obj_rand = np.array(obj_rand)
    obj_rand[obj_rand == 0] = np.nan
    obj_strat = np.array(obj_strat)
    obj_strat[obj_strat == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    avg_norm = np.quantile(obj_rand, 0.5, axis=1)
    norm_10 = np.quantile(obj_rand, 0.1, axis=1)
    norm_90 = np.quantile(obj_rand, 0.9, axis=1)
    plt.fill_between(n_grid, norm_10, norm_90, color="k", alpha=0.5)
    plt.plot(n_grid, avg_norm, "k", label="Random")

    avg_norm = np.quantile(obj_strat, 0.5, axis=1)
    norm_10 = np.quantile(obj_strat, 0.1, axis=1)
    norm_90 = np.quantile(obj_strat, 0.9, axis=1)
    plt.fill_between(n_grid, norm_10, norm_90, color="r", alpha=0.5)
    plt.plot(n_grid, avg_norm, "r", label="Success Prediction")

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.ylim([0.998, 1.005])
    plt.legend()
    plt.savefig(f"plot_nodes_suc_pred_sp_K{K}_N{N}_rf{rf_num}_{num_inst}")
    plt.close()

    return env_rand, env_strat


num_inst = 16

for K in [3, 4, 5]:
    for N in [30]:
        for rf_num in [5, 10, 20]:
            try:
                plot_stuff(K, N, num_inst, rf_num)
            except FileNotFoundError:
                pass
