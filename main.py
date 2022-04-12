import pickle
import numpy as np
import pandas as pd

data = dict()
labels = dict()
info = dict()
finished_instances = dict()
missing_instances = dict()
data_mins = 2
class_perc = 5

# K = 3
#
# with open(f"CapitalBudgetingResults/Results_27-1-22/TrainData/metadata_cb_p5_N10_K{K}.pickle", "rb") as handle:
#     meta = pickle.load(handle)
#
# with open(f"CapitalBudgetingResults/Results_27-1-22/TrainData/output_cb_p5_N10_K{K}.pickle", "rb") as handle:
#     Y = pickle.load(handle)
#
# with open(f"CapitalBudgetingResults/Results_27-1-22/TrainData/input_cb_p5_N10_K{K}.pickle", "rb") as handle:
#     X = pickle.load(handle)
#
# metadata_models = dict()
# for K in [2, 3, 4]:
#     new_model = pd.read_pickle(
#         f"CapitalBudgetingResults/Results_27-1-22/Models/Info/rf_class_info_cb_p5_N10_K2_all.pickle")
#     metadata_models[K] = pd.DataFrame(columns=new_model.index)
#     metadata_models[K].loc[0] = new_model
#     # CLASSIFICATION
#     for ct in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.7, 0.8]:
#         if ct:
#             problem_type = f"cb_p{class_perc}_N10_K{K}_ct{int(ct * 100)}_all"
#         else:
#             problem_type = f"cb_p{class_perc}_N10_K{K}_all"
#         metadata_models[K].loc[len(metadata_models[K])] = pd.read_pickle(
#             f"CapitalBudgetingResults/Results_27-1-22/Models/Info/rf_class_info_{problem_type}.pickle")

todays_map = "ShortestPathResults/Data"
for I in [100, 200, 500]:
    for K in [2, 3, 4]:
        for N in [50]:
            finished_instances[(I, K, N)] = 0
            missing_instances[(I, K, N)] = []
            save_name = f"sp_p{class_perc}_N{N}_K{K}_I{I}"

            for i in np.arange(I):
                try:
                    with open(f"{todays_map}/Results/TrainData/inst_results/data_results_sp_p{class_perc}_N{N}_K{K}_{i}.pickle", "rb") as handle:
                        new_data = pickle.load(handle)
                    with open(f"{todays_map}/RunInfo/run_info_sp_p{class_perc}_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
                        new_info = pickle.load(handle)
                    finished_instances[(I, K, N)] += 1
                except:
                    missing_instances[(I, K, N)].append(i)
                    continue

                X = pd.DataFrame(new_data["X"])
                X.index = pd.MultiIndex.from_product([[i], X.index])
                Y = pd.Series(new_data["Y"])
                Y.index = pd.MultiIndex.from_product([[i], Y.index])

                try:
                    data[(K, N, I)] = pd.concat([data[(K, N, I)], X])
                    labels[(K, N, I)] = pd.concat([labels[(K, N, I)], Y])
                    info[(K, N, I)].loc[i] = new_info
                except:
                    data[(K, N, I)] = X
                    labels[(K, N, I)] = Y
                    info[(K, N, I)] = pd.DataFrame(columns=new_info.index)
                    info[(K, N, I)].loc[i] = new_info
            print(f"K = {K}, N = {N}, I = {I}")
            print(labels[(K, N, I)].describe())

            data[(K, N, I)].to_pickle(f"{todays_map}/Results/TrainData/input_{save_name}.pickle")
            labels[(K, N, I)].to_pickle(f"{todays_map}/Results/TrainData/output_{save_name}.pickle")
            info[(K, N, I)].to_pickle(f"{todays_map}/Results/TrainData/metadata_{save_name}.pickle")
