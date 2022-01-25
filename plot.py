import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle


def plot_stuff(problem_type, K, N, num_inst):
    results = [{}]
    alg_types = ["Random"]
    num_algs = len(results)
    for i in np.arange(num_inst):

        with open(f"Results_25-1-22/Decisions/inst_results/final_results_cb_random_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
            results[0][i] = pickle.load(handle)
        # with open(f"Results_24-1-22_p2/Decisions/inst_results/final_results_cb_suc_pred_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
        #     results[1][i] = pickle.load(handle)
        # with open(f"CapitalBudgetingHigh/Data/Results/Decisions/inst_results/final_results_cb_suc_pred_nn_no_thresh_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
        #     results[2][i] = pickle.load(handle)

    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 30*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=np.arange(num_inst), dtype=np.float))

    for a in np.arange(num_algs):
        for i in np.arange(num_inst):
            # random
            t_alg = np.zeros(num_grids)
            theta_final = results[a][i]["theta"]
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])

    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=4)
    plt.savefig(f"plot_runtime_{problem_type}_K{K}_N{N}_{num_inst}")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in np.arange(num_inst)])
    n_grid = np.arange(0, n_grid_max+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=np.arange(num_inst), dtype=np.float))

    for a in np.arange(num_algs):
        for i in np.arange(num_inst):
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[a][i]["theta"]
            for n, theta in results[a][i]["inc_thetas_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.legend(loc=4)
    plt.savefig(f"plot_nodes_{problem_type}_K{K}_N{N}_{num_inst}")
    plt.close()


num_inst = 128

for K in [3, 4, 5]:
    for N in [10]:
        plot_stuff("cb", K, N, num_inst)