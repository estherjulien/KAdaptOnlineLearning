import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 16
K = 4
N = 100
results = dict()
results_knn = dict()
results_att = dict()
for i in np.arange(num_inst):
    with open(f"Results/Decisions/Random/final_results_sp_rand_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/AttributesLearned/final_results_att_sp_mp_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results_knn[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/AttributesNotLearned/final_results_att_sp_mp_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        results_att[i] = pickle.load(handle)[1]
# PLOT RESULTS OVER RUNTIME
t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 2*60*60+15, 15)])
num_grids = len(t_grid)
obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_knn = pd.DataFrame(index=t_grid, columns=[f"knn_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_att = pd.DataFrame(index=t_grid, columns=[f"att_{i}" for i in np.arange(num_inst)], dtype=np.float)
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

    t_att = np.zeros(num_grids)
    for t, theta in results_att[i]["inc_thetas_t"].items():
        t_att[t_grid > t] = theta/theta_final
    obj_att[f"att_{i}"] = t_att

obj_norm[obj_norm == 0] = np.nan
obj_knn[obj_knn == 0] = np.nan
obj_att[obj_att == 0] = np.nan

sn.set_style("whitegrid")
# plot results
avg_knn = np.mean(obj_knn, axis=1)
knn_10 = np.quantile(obj_knn, 0.1, axis=1)
knn_90 = np.quantile(obj_knn, 0.9, axis=1)
start = np.where(knn_10 > 0)[0][0]
plt.fill_between(t_grid, knn_10, knn_90, color="r", alpha=0.5)
plt.plot(t_grid[start:], avg_knn[start:], "r", label="Attribute Learned")

# avg_att = np.mean(obj_att, axis=1)
# att_10 = np.quantile(obj_att, 0.1, axis=1)
# att_90 = np.quantile(obj_att, 0.9, axis=1)
# start = np.where(att_10 > 0)[0][0]
# plt.fill_between(t_grid, att_10, att_90, color="b", alpha=0.5)
# plt.plot(t_grid[start:], avg_att[start:], "b", label="Attribute")

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
plt.savefig(f"plot_time_sp_att_weights_random_K{K}_N{N}_{num_inst}")
plt.close()

# PLOT RESULTS OVER NODES
n_grid_max = np.min([list(results_att[i]["inc_thetas_n"].keys())[-1] for i in np.arange(num_inst)])
n_grid = np.arange(0, n_grid_max+10, 10)
num_grids = len(n_grid)
obj_norm = pd.DataFrame(index=n_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float16)
obj_knn = pd.DataFrame(index=n_grid, columns=[f"knn_{i}" for i in np.arange(num_inst)], dtype=np.float16)
obj_att = pd.DataFrame(index=n_grid, columns=[f"att_{i}" for i in np.arange(num_inst)], dtype=np.float16)

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

    n_att = np.zeros(num_grids)
    for n, theta in results_att[i]["inc_thetas_n"].items():
        n_att[n_grid > n] = theta/theta_final
    obj_att[f"att_{i}"] = n_att

obj_norm[obj_norm == 0] = np.nan
obj_knn[obj_knn == 0] = np.nan
obj_att[obj_att == 0] = np.nan

sn.set_style("whitegrid")
# plot results
avg_knn = np.mean(obj_knn, axis=1)
knn_10 = np.quantile(obj_knn, 0.1, axis=1)
knn_90 = np.quantile(obj_knn, 0.9, axis=1)
start = np.where(knn_10 > 0)[0][0]
plt.fill_between(n_grid, knn_10, knn_90, color="r", alpha=0.5)
plt.plot(n_grid[start:], avg_knn[start:], "r", label="Attribute Learned")

# avg_att = np.mean(obj_att, axis=1)
# att_10 = np.quantile(obj_att, 0.1, axis=1)
# att_90 = np.quantile(obj_att, 0.9, axis=1)
# start = np.where(att_10 > 0)[0][0]
# plt.fill_between(n_grid, att_10, att_90, color="b", alpha=0.5)
# plt.plot(n_grid[start:], avg_att[start:], "b", label="Attribute")

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
plt.savefig(f"plot_nodes_sp_att_weights_random_K{K}_N{N}_{num_inst}")
