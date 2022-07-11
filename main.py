import pickle
import numpy as np
import pandas as pd

# COMBINE DATA
# info = dict()
# N = 20
# for minutes in [5, 10, 15, 20]:
#     for nodes in [2, 1, 5, 10]:
#         num_insts = int(np.floor(60/minutes)*nodes)
#         for K in [2, 3, 4, 5, 6]:
#             input_data = None
#             output_data = None
#             actual_insts = 0
#             for i in np.arange(num_insts):
#                 try:
#                     with open(f"SPData/inst_results/data_results_sp_sphere_N{N}_K{K}_m{minutes}_{i}.pickle", "rb") as handle:
#                         new_results = pickle.load(handle)
#                     actual_insts += 1
#                     with open(f"SPDataInfo/run_info_sp_sphere_N{N}_K{K}_m{minutes}_inst{i}.pickle", "rb") as handle:
#                         new_info = pickle.load(handle)
#                     info[f"K{K}_m{minutes}_nodes{nodes}_i{i}"] = new_info
#                 except FileNotFoundError:
#                     print(f"MISSING: K = {K}, min = {minutes}, nodes = {nodes}, i = {i}")
#                 try:
#                     input_data = np.concatenate([input_data, new_results["X"]])
#                     output_data = np.concatenate([output_data, new_results["Y"]])
#                 except:
#                     input_data = new_results["X"]
#                     output_data = new_results["Y"]
#             print(f"nodes = {nodes}, minutes = {minutes}, K = {K}, insts = {actual_insts}")
#             with open(f"SPData/train_data_sp_sphere_N{N}_K{K}_min{minutes}_nodes{nodes}.pickle", "wb") as handle:
#                 pickle.dump({"X": input_data, "Y": output_data, "actual_insts": actual_insts}
#                 , handle)
# df_info = pd.DataFrame(info).transpose()

# CHECK MODELS
df_info = dict()
for K in [2, 3, 4, 5, 6]:
    info = dict()
    for minutes in [5, 10, 15, 20]:
        for nodes in [1, 2, 5, 10]:
            num_insts = int(np.floor(60 / minutes) * nodes)
            data_info = dict()
            for i in np.arange(num_insts):
                with open(f"SPDataInfo/run_info_sp_sphere_N20_K{K}_m{minutes}_inst{i}.pickle", "rb") as handle:
                    data_info[i] = pickle.load(handle)
            data_info_all = pd.DataFrame(data_info).transpose().mean()
            for ct in [5]:
                with open(f"SPModels/Info/rf_class_info_sp_sphere_N20_K{K}_min{minutes}_nodes{nodes}_ct{ct}_bal.pickle", "rb") as handle:
                    model_info = pickle.load(handle)
                model_info["instances"] = num_insts
                info[f"K{K}_min{minutes}_nodes{nodes}_ct{ct}"] = pd.concat([model_info, data_info_all])
    df_info[K] = pd.DataFrame(info).transpose()