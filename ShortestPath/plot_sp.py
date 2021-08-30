import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 16
K = 4
N = 100
results = dict()
results_strat_1 = dict()
results_strat_2 = dict()

for i in np.arange(num_inst):
    with open(f"Results/Decisions/RandomNew/final_results_sp_random_new_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/StrategySTSameInstance/final_results_sp_strategy_sub_tree_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results_strat_1[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/StrategySTOther/final_results_sp_strategy_sub_tree_other_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results_strat_2[i] = pickle.load(handle)[1]
    theta_rand = results[i]["theta"]
    theta_strat_1 = results_strat_1[i]["theta"]
    theta_strat_2 = results_strat_2[i]["theta"]
    print(f"Instance {i}: Random = {theta_rand},       Strategy Same = {theta_strat_1},       Strategy Other = {theta_strat_2} ")

# # PLOT RESULTS OVER RUNTIME
t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 2*60*60+15, 15)])
num_grids = len(t_grid)
obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_strat_1 = pd.DataFrame(index=t_grid, columns=[f"strat_1_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_strat_2 = pd.DataFrame(index=t_grid, columns=[f"strat_2_{i}" for i in np.arange(num_inst)], dtype=np.float)
for i in np.arange(num_inst):
    # normal
    t_norm = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for t, theta in results[i]["inc_thetas_t"].items():
        t_norm[t_grid > t] = theta/theta_final
    obj_norm[f"norm_{i}"] = t_norm

    t_strat_1 = np.zeros(num_grids)
    for t, theta in results_strat_1[i]["inc_thetas_t"].items():
        t_strat_1[t_grid > t] = theta/theta_final
    obj_strat_1[f"strat_1_{i}"] = t_strat_1

    t_strat_2 = np.zeros(num_grids)
    for t, theta in results_strat_2[i]["inc_thetas_t"].items():
        t_strat_2[t_grid > t] = theta/theta_final
    obj_strat_2[f"strat_2_{i}"] = t_strat_2

obj_norm[obj_norm == 0] = np.nan
obj_strat_1[obj_strat_1 == 0] = np.nan
obj_strat_2[obj_strat_2 == 0] = np.nan

sn.set_style("whitegrid")
# plot results
avg_strat_1 = np.mean(obj_strat_1, axis=1)
strat_1_10 = np.quantile(obj_strat_1, 0.1, axis=1)
strat_1_90 = np.quantile(obj_strat_1, 0.9, axis=1)
start = np.where(strat_1_10 > 0)[0][0]
plt.fill_between(t_grid, strat_1_10, strat_1_90, color="r", alpha=0.5)
plt.plot(t_grid[start:], avg_strat_1[start:], "r", label="Node selection Same")

avg_strat_2 = np.mean(obj_strat_2, axis=1)
strat_2_10 = np.quantile(obj_strat_2, 0.1, axis=1)
strat_2_90 = np.quantile(obj_strat_2, 0.9, axis=1)
start = np.where(strat_2_10 > 0)[0][0]
plt.fill_between(t_grid, strat_2_10, strat_2_90, color="b", alpha=0.5)
plt.plot(t_grid[start:], avg_strat_2[start:], "b", label="Node selection Other")

avg_norm = np.mean(obj_norm, axis=1)
norm_10 = np.quantile(obj_norm, 0.1, axis=1)
norm_90 = np.quantile(obj_norm, 0.9, axis=1)
start = np.where(norm_10 > 0)[0][0]
plt.fill_between(t_grid, norm_10, norm_90, color="k", alpha=0.5)
plt.plot(t_grid[start:], avg_norm[start:], "k", label="Random")

# plt.ylim([0.95, 1.05])
plt.xlabel("Runtime (s)")
plt.legend()
plt.show()
plt.savefig(f"plot_time_sp_strategy_sub_tree_both_K{K}_N{N}_{num_inst}")
plt.close()

# PLOT RESULTS OVER NODES
n_grid_max = np.max([results_strat_1[i]["tot_nodes"] for i in np.arange(num_inst)])
n_grid = np.arange(0, n_grid_max+10, 10)
num_grids = len(n_grid)
obj_norm = pd.DataFrame(index=n_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float16)
obj_strat_1 = pd.DataFrame(index=n_grid, columns=[f"strat_1_{i}" for i in np.arange(num_inst)], dtype=np.float16)
obj_strat_2 = pd.DataFrame(index=n_grid, columns=[f"strat_2_{i}" for i in np.arange(num_inst)], dtype=np.float16)

for i in np.arange(num_inst):
    # normal
    n_norm = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for n, theta in results[i]["inc_thetas_n"].items():
        n_norm[n_grid > n] = theta/theta_final
    obj_norm[f"norm_{i}"] = n_norm

    n_strat_1 = np.zeros(num_grids)
    for n, theta in results_strat_1[i]["inc_thetas_n"].items():
        n_strat_1[n_grid > n] = theta/theta_final
    obj_strat_1[f"strat_1_{i}"] = n_strat_1

    n_strat_2 = np.zeros(num_grids)
    for n, theta in results_strat_2[i]["inc_thetas_n"].items():
        n_strat_2[n_grid > n] = theta/theta_final
    obj_strat_2[f"strat_2_{i}"] = n_strat_2

obj_norm[obj_norm == 0] = np.nan
obj_strat_1[obj_strat_1 == 0] = np.nan
obj_strat_2[obj_strat_2 == 0] = np.nan

sn.set_style("whitegrid")
# plot results
avg_strat_1 = np.mean(obj_strat_1, axis=1)
strat_1_10 = np.quantile(obj_strat_1, 0.1, axis=1)
strat_1_90 = np.quantile(obj_strat_1, 0.9, axis=1)
start = np.where(strat_1_10 > 0)[0][0]
plt.fill_between(n_grid, strat_1_10, strat_1_90, color="r", alpha=0.5)
plt.plot(n_grid[start:], avg_strat_1[start:], "r", label="Node selection Same")

avg_strat_2 = np.mean(obj_strat_2, axis=1)
strat_2_10 = np.quantile(obj_strat_2, 0.1, axis=1)
strat_2_90 = np.quantile(obj_strat_2, 0.9, axis=1)
start = np.where(strat_2_10 > 0)[0][0]
plt.fill_between(n_grid, strat_2_10, strat_2_90, color="b", alpha=0.5)
plt.plot(n_grid[start:], avg_strat_2[start:], "b", label="Node selection Other")

avg_norm = np.mean(obj_norm, axis=1)
norm_10 = np.quantile(obj_norm, 0.1, axis=1)
norm_90 = np.quantile(obj_norm, 0.9, axis=1)
start = np.where(norm_10 > 0)[0][0]
plt.fill_between(n_grid, norm_10, norm_90, color="k", alpha=0.5)
plt.plot(n_grid[start:], avg_norm[start:], "k", label="Random")


# plt.ylim([0.95, 1.05])
plt.xlabel("Nodes")
plt.legend()
plt.show()
plt.savefig(f"plot_nodes_sp_strategy_sub_tree_both_K{K}_N{N}_{num_inst}")
