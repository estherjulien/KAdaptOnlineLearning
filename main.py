import pickle
import numpy as np
import pandas as pd

data = dict()
labels = dict()
info = dict()
finished_instances = dict()
missing_instances = dict()
train = True

# with open("ClusterResults_24-01-22/Summarized/metadata_train_N10_K4.pickle", "rb") as handle:
#     meta = pickle.load(handle)
#
# with open("ClusterResults_24-01-22/Summarized/output_train_N10_K4.pickle", "rb") as handle:
#     Y = pickle.load(handle)
#
# with open("ClusterResults_24-01-22/Summarized/input_train_N10_K4.pickle", "rb") as handle:
#     X = pickle.load(handle)

metadata_all = pd.DataFrame()
N = 10
K = 4
for bal in [True, False]:
    if bal:
        problem_type = f"cb_N{N}_K{K}_balanced_all"
    else:
        problem_type = f"cb_N{N}_K{K}_all"
    for w in [10, 20, 50, 100, 200]:
        for d in [1, 2, 5, 10, 20, 50]:
            meta = pd.read_pickle(f"CapitalBudgetingHigh/Data/Models/Info/nn_class_info_{problem_type}_D{d}_W{w}.pickle")
            meta["balanced"] = bal
            meta["accuracy_sum"] = meta["accuracy_0"] + meta["accuracy_1"]
            if len(metadata_all) == 0:
                metadata_all = pd.DataFrame(columns=meta.index)
            metadata_all.loc[len(metadata_all)] = meta

            print(f"Balanced = {bal}, WIDTH = {w}, DEPTH = {d}")
            print(f"NN accuracy = ", meta["accuracy_all"], ", data points = ", meta["datapoints"], ", runtime = ", meta["runtime"],
                  "\nNN accuracy (Y = 0) = ", meta["accuracy_0"],
                  "\nNN accuracy (Y = 1) = ", meta["accuracy_1"],
                  "\nNN accuracy (sum) = ", meta["accuracy_sum"])

metadata_all.to_pickle("Results_25-1-22/NN_metadata_results.pickle")
# for K in [2, 3, 4]:
#     for N in [10]:
#         finished_instances[(K, N)] = 0
#         missing_instances[(K, N)] = []
#
#         if train:
#             save_name = f"train_norm_N{N}_K{K}"
#             instances = np.arange(500)
#         else:
#             save_name = f"validation_norm_N{N}_K{K}"
#             instances = np.arange(500, 510)
#         for i in instances:
#             try:
#                 with open(f"Data/Results/TrainDataOld/inst_results/data_results_cb_N{N}_K{K}_{i}.pickle", "rb") as handle:
#                     new_data = pickle.load(handle)
#                 with open(f"Data/RunInfoOld/run_info_cb_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
#                     new_info = pickle.load(handle)
#                 finished_instances[(K, N)] += 1
#             except:
#                 missing_instances[(K, N)].append(i)
#                 continue
#
#             X = pd.DataFrame(new_data["X"])
#             X.index = pd.MultiIndex.from_product([[i], X.index])
#             Y = pd.Series(new_data["Y"])
#             Y.index = pd.MultiIndex.from_product([[i], Y.index])
#
#             try:
#                 data[(K, N)] = pd.concat([data[(K, N)], X])
#                 labels[(K, N)] = pd.concat([labels[(K, N)], Y])
#                 info[(K, N)].loc[i] = new_info
#             except:
#                 data[(K, N)] = X
#                 labels[(K, N)] = Y
#                 info[(K, N)] = pd.DataFrame(columns=new_info.index)
#                 info[(K, N)].loc[i] = new_info
#         print(f"K = {K}, N = {N}")
#         print(labels[(K, N)].describe())
#
#         data[(K, N)].to_pickle(f"CapitalBudgetingHigh/Data/TrainData/input_{save_name}.pickle")
#         labels[(K, N)].to_pickle(f"CapitalBudgetingHigh/Data/TrainData/output_{save_name}.pickle")
#         info[(K, N)].to_pickle(f"CapitalBudgetingHigh/Data/TrainData/metadata_{save_name}.pickle")


# with open("CapitalBudgetingHigh/Data/Results/Decisions/inst_results/final_results_cb_random_N10_K2_inst0.pickle", "rb") as handle:
#     res = pickle.load(handle)

# K = 3
# N = 10
# for i in np.arange(10):
#     try:
#         with open(f"CapitalBudgetingHigh/Data/Results/TrainData/inst_results/data_results_cb_not_norm_N{N}_K{K}_{i}.pickle", "rb") as handle:
#             new_data = pickle.load(handle)
#         with open(f"Data/RunInfo/run_info_cb_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
#             new_info = pickle.load(handle)
#     except:
#         continue
#
#     X = pd.DataFrame(new_data["X"])
#     X.index = pd.MultiIndex.from_product([[i], X.index])
#     Y = pd.Series(new_data["Y"])
#     Y.index = pd.MultiIndex.from_product([[i], Y.index])
#
#     try:
#         data = pd.concat([data, X])
#         labels = pd.concat([labels, Y])
#         info.loc[i] = new_info
#     except:
#         data = X
#         labels = Y
#         info = pd.DataFrame(columns=new_info.index)
#         info.loc[i] = new_info
# print(f"K = {K}, N = {N}")
# print(labels.describe())