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


def sp_metadata():
    metadata_models = pd.DataFrame()
    for I in [100, 200, 500]:
        for K in [4]:
            for N in [50, 20]:
                for ct in [None, 30, 70]:
                    try:
                        if ct is None:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_p5_N{N}_K{K}_I{I}_all.pickle")
                        else:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_p5_N{N}_K{K}_I{I}_ct{ct}_all.pickle")
                        new_info = pd.read_pickle(
                            f"NOT_USED/SPSphereResults/TrainData/CombData/metadata_sp_p5_N{N}_K{K}_I{I}.pickle")
                    except:
                        continue
                    try:
                        metadata_models.loc[f"I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])
                    except:
                        metadata_models = pd.DataFrame(columns=pd.concat([new_model, new_info.mean()]).index)
                    metadata_models.loc[f"I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])

    for I in [100, 150, 200, 240, 480]:
        for K in [4]:
            for N in [50]:
                for ct in [None, 30, 70]:
                    try:
                        if ct is None:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_long_p5_N{N}_K{K}_I{I}_all.pickle")
                        else:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_long_p5_N{N}_K{K}_I{I}_ct{ct}_all.pickle")
                        new_info = pd.read_pickle(
                            f"NOT_USED/SPSphereResults/TrainData/CombData/metadata_sp_long_p5_N{N}_K{K}_I{I}.pickle")
                    except:
                        continue
                    metadata_models.loc[f"long_I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])
    #
    for I in [50, 60, 100, 120, 200, 300, 360]:
        for K in [4]:
            for N in [50]:
                for ct in [None, 30, 70]:
                    try:
                        if ct is None:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_extra_long_p5_N{N}_K{K}_I{I}_all.pickle")
                        else:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_extra_long_p5_N{N}_K{K}_I{I}_ct{ct}_all.pickle")
                        new_info = pd.read_pickle(
                            f"NOT_USED/SPSphereResults/TrainData/CombData/metadata_sp_extra_long_p5_N{N}_K{K}_I{I}.pickle")
                    except:
                        continue
                    metadata_models.loc[f"extra_long_I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])
    return metadata_models


def sp_metadata_N20():
    metadata_models = pd.DataFrame()
    for I in [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
        for K in [4]:
            for N in [20]:
                for ct in [None, 30, 70]:
                    try:
                        if ct is None:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_p5_N{N}_K{K}_I{I}_all.pickle")
                        else:
                            new_model = pd.read_pickle(
                                f"NOT_USED/SPSphereResults/TrainData/Models/Info/rf_class_info_sp_p5_N{N}_K{K}_I{I}_ct{ct}_all.pickle")
                        new_info = pd.read_pickle(
                            f"NOT_USED/SPSphereResults/TrainData/CombData/metadata_sp_p5_N{N}_K{K}_I{I}.pickle")
                    except:
                        continue
                    try:
                        metadata_models.loc[f"I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])
                    except:
                        metadata_models = pd.DataFrame(columns=pd.concat([new_model, new_info.mean()]).index)
                    metadata_models.loc[f"I{I}_K{K}_N{N}_ct{ct}"] = pd.concat([new_model, new_info.mean()])
    return metadata_models


metadata_sp = sp_metadata_N20()
# metadata_cb = pd.read_pickle("CapitalBudgetingResults/extra_info_cb_models.pickle")
# data_sp = pd.read_pickle("SPSphereResults/TrainData/CombData/metadata_sp_p5_N20_K4_I500.pickle")
# data_cb = pd.read_pickle("CapitalBudgetingResults/ModelResults/TrainData/metadata_cb_p5_N10_K4_I1000.pickle")
# for I in [100, 150, 200, 240, 480]:
#     for K in [4]:
#         for N in [50]:
#             try:
#                 new_info = pd.read_pickle(f"SPSphereResults/TrainData/CombData/metadata_sp_long_p5_N{N}_K{K}_I{I}.pickle")
#                 data = pd.read_pickle(f"SPSphereResults/TrainData/CombData/input_sp_long_p5_N50_K4_I{I}.pickle")
#             except:
#                 continue
#             try:
#                 info_data.loc[f"long_I{I}_K{K}_N{N}_ct{ct}"] = new_info.mean()
#             except:
#                 info_data = pd.DataFrame(columns=new_info.mean().index)
#             info_data.loc[f"long_I{I}_K{K}_N{N}_ct{ct}"] = new_info.mean()
#
# for I in [60, 120]:
#     for K in [4]:
#         for N in [50]:
#             try:
#                 new_info = pd.read_pickle(
#                     f"SPSphereResults/TrainData/CombData/metadata_sp_extra_long_p5_N{N}_K{K}_I{I}.pickle")
#                 data = pd.read_pickle(f"SPSphereResults/TrainData/CombData/input_sp_extra_long_p5_N50_K4_I{I}.pickle")
#             except:
#                 continue
#             try:
#                 info_data.loc[f"extra_long_I{I}_K{K}_N{N}_ct{ct}"] = new_info.mean()
#             except:
#                 info_data = pd.DataFrame(columns=new_info.mean().index)
#             info_data.loc[f"extra_long_I{I}_K{K}_N{N}_ct{ct}"] = new_info.mean()

# todays_map = "SPSphereResults/TrainData"
# for I in [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
#     for K in [4]:
#         for N in [20]:
#             finished_instances[(I, K, N)] = 0
#             missing_instances[(I, K, N)] = []
#             save_name = f"sp_p{class_perc}_N{N}_K{K}_I{I}"
#
#             for i in np.arange(I):
#                 try:
#                     with open(f"{todays_map}/inst_results/data_results_sp_sphere_N{N}_d5_tap90_g7_p{class_perc}_K{K}_{i}.pickle", "rb") as handle:
#                         new_data = pickle.load(handle)
#                     with open(f"{todays_map}/RunInfo/run_info_sp_sphere_N{N}_d5_tap90_g7_p{class_perc}_K{K}_inst{i}.pickle", "rb") as handle:
#                         new_info = pickle.load(handle)
#                     finished_instances[(I, K, N)] += 1
#                 except:
#                     missing_instances[(I, K, N)].append(i)
#                     continue
#
#                 X = pd.DataFrame(new_data["X"])
#                 X.index = pd.MultiIndex.from_product([[i], X.index])
#                 Y = pd.Series(new_data["Y"])
#                 Y.index = pd.MultiIndex.from_product([[i], Y.index])
#
#                 try:
#                     data[(K, N, I)] = pd.concat([data[(K, N, I)], X])
#                     labels[(K, N, I)] = pd.concat([labels[(K, N, I)], Y])
#                     info[(K, N, I)].loc[i] = new_info
#                 except:
#                     data[(K, N, I)] = X
#                     labels[(K, N, I)] = Y
#                     info[(K, N, I)] = pd.DataFrame(columns=new_info.index)
#                     info[(K, N, I)].loc[i] = new_info
#
#             print(f"K = {K}, N = {N}, I = {I}")
#             print(labels[(K, N, I)].describe())
#
#             data[(K, N, I)].to_pickle(f"{todays_map}/input_{save_name}.pickle")
#             labels[(K, N, I)].to_pickle(f"{todays_map}/output_{save_name}.pickle")
#             info[(K, N, I)].to_pickle(f"{todays_map}/metadata_{save_name}.pickle")
#
#
