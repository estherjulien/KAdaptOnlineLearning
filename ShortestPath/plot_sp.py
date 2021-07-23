import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 28
K = 4
N = 100
results = dict()
results_knn = dict()
for i in np.arange(num_inst):
    with open(f"Results/Decisions/final_results_sp_K{K}_N{N}_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/final_results_knn_sp_K{K}_N{N}_inst{i}.pickle", "rb") as handle:
        results_knn[i] = pickle.load(handle)[1]
# PLOT RESULTS OVER RUNTIME
t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 1*60*60+15, 15)])
num_grids = len(t_grid)
obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_knn = pd.DataFrame(index=t_grid, columns=[f"knn_{i}" for i in np.arange(num_inst)], dtype=np.float)
for i in np.arange(num_inst):
    # normal
    t_norm = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for t, theta in results[i]["inc_thetas_t"].items():
        t_norm[t_grid > t] = theta/theta_final
    obj_norm[f"norm_{i}"] = t_norm

    t_knn = np.zeros(num_grids)
    for t, theta in results_knn[i]["inc_thetas_t"].items():
        t_knn[t_grid > t] = theta/theta_final
    obj_knn[f"knn_{i}"] = t_knn

obj_norm[obj_norm == 0] = np.nan
obj_knn[obj_knn == 0] = np.nan
sn.set_style("whitegrid")
# plot results
avg_norm = np.mean(obj_norm, axis=1)
# plt.plot(t_grid, obj_norm[f"norm_{0}"], "k", label="Normal")
# for i in np.arange(1, num_inst):
#     plt.plot(t_grid, obj_norm[f"norm_{i}"], "k")
plt.plot(t_grid, avg_norm, "k", label="Random")


avg_knn = np.mean(obj_knn, axis=1)
# plt.plot(t_grid, obj_knn[f"knn_{0}"], "r", label="knnonoi")
# for i in np.arange(1, num_inst):
#     plt.plot(t_grid, obj_knn[f"knn_{i}"], "r")
plt.plot(t_grid, avg_knn, "r", label="KNN")


# plt.ylim([0.95, 1.05])
plt.xlabel("Runtime (s)")
plt.legend()
plt.show()
plt.savefig(f"plot_time_sp_knn_random_K{K}_N{N}_{num_inst}")
plt.close()

# PLOT RESULTS OVER NODES
n_grid_max = np.min([list(results_knn[i]["inc_thetas_n"].keys())[-1] for i in np.arange(num_inst)])
n_grid = np.arange(0, n_grid_max+10, 10)
num_grids = len(n_grid)
obj_norm = pd.DataFrame(index=n_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float16)
obj_knn = pd.DataFrame(index=n_grid, columns=[f"knn_{i}" for i in np.arange(num_inst)], dtype=np.float16)
for i in np.arange(num_inst):
    # normal
    n_norm = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for n, theta in results[i]["inc_thetas_n"].items():
        n_norm[n_grid > n] = theta/theta_final
    obj_norm[f"norm_{i}"] = n_norm

    n_knn = np.zeros(num_grids)
    for n, theta in results_knn[i]["inc_thetas_n"].items():
        n_knn[n_grid > n] = theta/theta_final
    obj_knn[f"knn_{i}"] = n_knn
obj_norm[obj_norm == 0] = np.nan
obj_knn[obj_knn == 0] = np.nan
sn.set_style("whitegrid")
# plot results
avg_norm = np.mean(obj_norm, axis=1)
# plt.plot(n_grid, obj_norm[f"norm_{0}"], "k", label="Random")
# for i in np.arange(1, num_inst):
#     plt.plot(n_grid, obj_norm[f"norm_{i}"], "k")
plt.plot(n_grid, avg_norm, "k", label="Random")


avg_knn = np.mean(obj_knn, axis=1)
# plt.plot(n_grid, obj_knn[f"knn_{0}"], "r", label="KNN")
# for i in np.arange(1, num_inst):
#     plt.plot(n_grid, obj_knn[f"knn_{i}"], "r")
plt.plot(n_grid, avg_knn, "r", label="KNN")


# plt.ylim([0.95, 1.05])
plt.xlabel("Nodes")
plt.legend()
plt.show()
plt.savefig(f"plot_nodes_sp_knn_random_K{K}_N{N}_{num_inst}")
