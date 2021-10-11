import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle


def plot_stuff(problem_type, K, N, num_inst):
    results = dict()
    for i in np.arange(num_inst):
        with open(f"Results/{problem_type.upper()}/Decisions/final_results_{problem_type}_random_K{K}_N{N}_inst{i}.pickle", "rb") as handle:
            results[i] = pickle.load(handle)
        # with open(f"ShortestPath/Results/Decisions/final_results_spip_random_new_K4_N50_g30_fs10_inst{i}.pickle", "rb") as handle:
        #     results[i] = pickle.load(handle)[1]
    # PLOT RESULTS OVER RUNTIME
    rt_tmp = np.max([np.round(results[i]["runtime"])])
    run_time = rt_tmp - rt_tmp % 1800
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, run_time+15, 15)])
    num_grids = len(t_grid)
    obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
    for i in np.arange(num_inst):
        # normal
        t_norm = np.zeros(num_grids)
        theta_final = results[i]["theta"]
        for t, theta in results[i]["inc_thetas_t"].items():
            t_norm[t_grid > t] = theta/theta_final
        obj_norm[f"norm_{i}"] = t_norm

    obj_norm = np.array(obj_norm)
    obj_norm[obj_norm == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    avg_norm = np.quantile(obj_norm, 0.5, axis=1)
    norm_10 = np.quantile(obj_norm, 0.1, axis=1)
    norm_90 = np.quantile(obj_norm, 0.9, axis=1)
    plt.fill_between(t_grid, norm_10, norm_90, color="k", alpha=0.5)
    plt.plot(t_grid, avg_norm, "k", label="Random")

    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend()
    plt.savefig(f"plot_runtime_{problem_type}_random_K{K}_N{N}_{num_inst}")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[i]["tot_nodes"] for i in np.arange(num_inst)])
    n_grid = np.arange(0, n_grid_max+10, 10)
    num_grids = len(n_grid)
    obj_norm = pd.DataFrame(index=n_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float16)

    for i in np.arange(num_inst):
        # normal
        n_norm = np.zeros(num_grids)
        theta_final = results[i]["theta"]
        for n, theta in results[i]["inc_thetas_n"].items():
            n_norm[n_grid > n] = theta/theta_final
        obj_norm[f"norm_{i}"] = n_norm

    obj_norm = np.array(obj_norm)
    obj_norm[obj_norm == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    avg_norm = np.quantile(obj_norm, 0.5, axis=1)
    norm_10 = np.quantile(obj_norm, 0.1, axis=1)
    norm_90 = np.quantile(obj_norm, 0.9, axis=1)
    plt.fill_between(n_grid, norm_10, norm_90, color="k", alpha=0.5)
    plt.plot(n_grid, avg_norm, "k", label="Random")

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.legend()
    plt.savefig(f"plot_nodes_{problem_type}_random_K{K}_N{N}_{num_inst}")
    plt.close()


num_inst = 16

for problem_type in ["cb"]:
    for K in [4]:
        for N in [30]:
            try:
                plot_stuff(problem_type, K, N, num_inst)
            except:
                pass