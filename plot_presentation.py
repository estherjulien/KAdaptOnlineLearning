import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_stuff(problem_type, K, N, num_inst, level=5):
    results = [{}, {}]
    alg_types = ["Random", "Success Prediction"]
    num_algs = len(results)
    insts = []
    for i in np.arange(num_inst):
        try:
            with open(f"CapitalBudgetingResults/RandomResults/final_results_cb_random_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
                results[0][i] = pickle.load(handle)
            with open(f"CapitalBudgetingResults/Results_27-1-22/Decisions/inst_results/final_results_cb_suc_pred_rf_"
                      f"p5_nt_N{N}_K{K}_L{level}_inst{i}.pickle", "rb") as handle:
                results[1][i] = pickle.load(handle)
            insts.append(i)
        except:
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*30+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sns.set()
    cols = ["r", "b"]
    sns.set_style("darkgrid")
    sns.set_context("talk", rc={"grid.linewidth": 0.6})
    plt.style.use("dark_background")
    sns.set_palette(sns.color_palette(cols))
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.3, color=[cols[a]])
        plt.plot(t_grid, avg_random, label=alg_types[a])

    plt.ylim([0.8, 1.05])
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=4)
    plt.savefig(f"plot_presentation_runtime_{problem_type}_nt_K{K}_N{N}_L{level}_{len(insts)}", dpi=400, bbox_inches="tight", transparent=True)
    plt.close()


num_inst = 112
N = 10
# K = 5
for K in [3, 4, 5]:
    plot_stuff("cb", K, N, num_inst, level=30)
