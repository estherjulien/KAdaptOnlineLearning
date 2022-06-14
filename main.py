import pickle
import numpy as np
import pandas as pd

#
for num_insts in [2000]:
    input_data = dict()
    output_data = dict()
    for K in [6]:
        actual_insts = 0
        for i in np.arange(num_insts):
            try:
                with open(f"CBData/inst_results/data_results_cb_N10_K{K}_{i}.pickle", "rb") as handle:
                    new_results = pickle.load(handle)
                actual_insts += 1
            except FileNotFoundError:
                continue
            try:
                input_data[K] = np.concatenate([input_data[K], new_results["X"]])
                output_data[K] = np.concatenate([output_data[K], new_results["Y"]])
            except:
                input_data[K] = new_results["X"]
                output_data[K] = new_results["Y"]
        print(f"I = {num_insts}, K = {K}, insts = {actual_insts}")
        with open(f"CBData/train_data_cb_N10_K{K}_I{num_insts}.pickle", "wb") as handle:
            pickle.dump({"X": input_data[K], "Y": output_data[K], "actual_insts": actual_insts}, handle)

# model_info = dict()
# for K in [2, 3, 4, 5, 6]:
#     for I in [10, 100, 500, 1000]:
#         try:
#             model_info[f"K{K}_I{I}"] = pd.read_pickle(f"CBModels/Info/rf_class_info_cb_N10_K{K}_I{I}.pickle")
#             model_info[f"K{K}_I{I}_bal"] = pd.read_pickle(f"CBModels/Info/rf_class_info_cb_N10_K{K}_I{I}_bal.pickle")
#             for ct in [70]:
#                 model_info[f"K{K}_I{I}_ct{ct}"] = pd.read_pickle(f"CBModels/Info/rf_class_info_cb_N10_K{K}_I{I}_ct{ct}.pickle")
#                 model_info[f"K{K}_I{I}_ct{ct}_bal"] = pd.read_pickle(f"CBModels/Info/rf_class_info_cb_N10_K{K}_I{I}_ct{ct}_bal.pickle")
#         except FileNotFoundError:
#             continue
#
# df_model_info = pd.DataFrame(model_info).transpose()
# df_good = df_model_info[df_model_info["accuracy"] > 0.95]
# df_very_good = df_good[df_good["type_1"] + df_good["type_1"] < 0.05]