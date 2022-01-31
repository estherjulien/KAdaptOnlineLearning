import pickle
import numpy as np
import pandas as pd

data = dict()
labels = dict()
info = dict()
finished_instances = dict()
missing_instances = dict()
instances = np.arange(120)
data_mins = 2
class_perc = 5

K = 3

with open(f"Results_27-1-22/TrainData/metadata_cb_p5_N10_K{K}.pickle", "rb") as handle:
    meta = pickle.load(handle)

with open(f"Results_27-1-22/TrainData/output_cb_p5_N10_K{K}.pickle", "rb") as handle:
    Y = pickle.load(handle)

with open(f"Results_27-1-22/TrainData/input_cb_p5_N10_K{K}.pickle", "rb") as handle:
    X = pickle.load(handle)

metadata_models = dict()
for K in [2, 3, 4]:
    new_model = pd.read_pickle(f"Results_27-1-22/Models/Info/rf_class_info_cb_p5_N10_K2_all.pickle")
    metadata_models[K] = pd.DataFrame(columns=new_model.index)
    metadata_models[K].loc[0] = new_model
    # CLASSIFICATION
    for ct in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.8]:
        if ct:
            problem_type = f"cb_p{class_perc}_N10_K{K}_ct{int(ct * 100)}_all"
        else:
            problem_type = f"cb_p{class_perc}_N10_K{K}_all"
        metadata_models[K].loc[len(metadata_models[K])] = pd.read_pickle(f"Results_27-1-22/Models/Info/rf_class_info_{problem_type}.pickle")

# for K in [2, 3, 4]:
#     for N in [10]:
#         finished_instances[(K, N)] = 0
#         missing_instances[(K, N)] = []
#         save_name = f"cb_p{class_perc}_N{N}_K{K}"
#
#         for i in instances:
#             try:
#                 with open(f"Results_27-1-22/TrainData/inst_results/data_results_cb_p{class_perc}_N{N}_K{K}_{i}.pickle", "rb") as handle:
#                     new_data = pickle.load(handle)
#                 with open(f"Results_27-1-22/RunInfo/run_info_cb_p{class_perc}_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
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
#         data[(K, N)].to_pickle(f"Results_27-1-22/TrainData/input_{save_name}.pickle")
#         labels[(K, N)].to_pickle(f"Results_27-1-22/TrainData/output_{save_name}.pickle")
#         info[(K, N)].to_pickle(f"Results_27-1-22/TrainData/metadata_{save_name}.pickle")
