import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 16
K = 4
N = 100
results = dict()
results_ol = dict()

for i in np.arange(num_inst):
    with open(f"Results/Decisions/Random/final_results_sp_rand_np_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/OnlineLearningMP/final_results_sp_online_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results_ol[i] = pickle.load(handle)[1]
    # theta_rand = results[i]["theta"]
    # theta_ol = results_ol[i]["theta"]
    # print(f"Random: {theta_rand}      Online Learning: {theta_ol}")
    node_rand = list(results[i]["tot_nodes"].values())[-1]
    node_ol = results_ol[i]["tot_nodes"]
    print(f"Random: {node_rand}      Online Learning: {node_ol}")

# PLOT RESULTS OVER RUNTIME
# t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 2*60*60+15, 15)])
# num_grids = len(t_grid)
# obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
# obj_r1 = pd.DataFrame(index=t_grid, columns=[f"r1_{i}" for i in np.arange(num_inst)], dtype=np.float)
# for i in np.arange(num_inst):
#     # normal
#     t_norm = np.zeros(num_grids)
#     theta_final = results[i]["theta"]
#     for t, theta in results[i]["inc_thetas_t"].items():
#         t_norm[t_grid > t] = theta/theta_final
#     obj_norm[f"norm_{i}"] = t_norm
#
# obj_norm[obj_norm == 0] = np.nan
#
# sn.set_style("whitegrid")
# # plot results
# # avg_r1 = np.mean(obj_r1, axis=1)
# # r1_10 = np.quantile(obj_r1, 0.1, axis=1)
# # r1_90 = np.quantile(obj_r1, 0.9, axis=1)
# # start = np.where(r1_10 > 0)[0][0]
# # plt.fill_between(t_grid, r1_10, r1_90, color="r", alpha=0.5)
# # plt.plot(t_grid[start:], avg_r1[start:], "r", label="Node selection")
#
# avg_norm = np.mean(obj_norm, axis=1)
# norm_10 = np.quantile(obj_norm, 0.1, axis=1)
# norm_90 = np.quantile(obj_norm, 0.9, axis=1)
# start = np.where(norm_10 > 0)[0][0]
# plt.fill_between(t_grid, norm_10, norm_90, color="k", alpha=0.5)
# plt.plot(t_grid[start:], avg_norm[start:], "k", label="Random")
#
# # plt.ylim([0.95, 1.05])
# plt.xlabel("Runtime (s)")
# plt.legend()
# plt.show()
# plt.savefig(f"plot_time_sp_K{K}_N{N}_{num_inst}")
# plt.close()
#
# # PLOT RESULTS OVER NODES
# n_grid_max = np.min([list(results_r1[i]["inc_thetas_n"].keys())[-1] for i in np.arange(num_inst)])
# n_grid = np.arange(0, n_grid_max+10, 10)
# num_grids = len(n_grid)
# obj_norm = pd.DataFrame(index=n_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float16)
# obj_r1 = pd.DataFrame(index=n_grid, columns=[f"r1_{i}" for i in np.arange(num_inst)], dtype=np.float16)
#
# for i in np.arange(num_inst):
#     # normal
#     n_norm = np.zeros(num_grids)
#     theta_final = results[i]["theta"]
#     for n, theta in results[i]["inc_thetas_n"].items():
#         n_norm[n_grid > n] = theta/theta_final
#     obj_norm[f"norm_{i}"] = n_norm
#
#     n_r1 = np.zeros(num_grids)
#     for n, theta in results_r1[i]["inc_thetas_n"].items():
#         n_r1[n_grid > n] = theta/theta_final
#     obj_r1[f"r1_{i}"] = n_r1
#
# obj_norm[obj_norm == 0] = np.nan
# obj_r1[obj_r1 == 0] = np.nan
#
# sn.set_style("whitegrid")
# # plot results
# avg_r1 = np.mean(obj_r1, axis=1)
# r1_10 = np.quantile(obj_r1, 0.1, axis=1)
# r1_90 = np.quantile(obj_r1, 0.9, axis=1)
# start = np.where(r1_10 > 0)[0][0]
# plt.fill_between(n_grid, r1_10, r1_90, color="r", alpha=0.5)
# plt.plot(n_grid[start:], avg_r1[start:], "r", label="Node selection")
#
# avg_norm = np.mean(obj_norm, axis=1)
# norm_10 = np.quantile(obj_norm, 0.1, axis=1)
# norm_90 = np.quantile(obj_norm, 0.9, axis=1)
# start = np.where(norm_10 > 0)[0][0]
# plt.fill_between(n_grid, norm_10, norm_90, color="k", alpha=0.5)
# plt.plot(n_grid[start:], avg_norm[start:], "k", label="Random")
#
#
# # plt.ylim([0.95, 1.05])
# plt.xlabel("Nodes")
# plt.legend()
# plt.show()
# plt.savefig(f"plot_nodes_sp_K{K}_N{N}_{num_inst}")
