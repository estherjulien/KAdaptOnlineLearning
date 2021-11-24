import pickle
import numpy as np

num_instances = 16
X_list = []
Y_list = []
num_envs = 20
K = 2
N = 30

results = []
for rf in np.arange(num_envs):
    with open(f"ShortestPath/ResultsSucPred/Data/final_results_sp_suc_pred_data_K{K}_N{N}_rf{rf}.pickle", "rb") as handle:
        results.append(pickle.load(handle)[1])
    X_list.append(results[-1]["input_data"])
    Y_list.append(results[-1]["success_data"])

with open(f"ShortestPath/ResultsSucPred/Data/all_data_sp_suc_pred_data_K{K}_N{N}_{num_envs}.pickle", "wb") as handle:
    pickle.dump({"X": X_list, "Y": Y_list}, handle)
