import pickle
import numpy as np
import pandas as pd

data = []
labels = []
for K in [3]:
    for N in [10]:
        try:
            with open(f"CapitalBudgetingHigh/Data/RunInfo/run_info_cb_N{N}_K{K}_inst0.pickle",
                      "rb") as handle:
                rt_new = pickle.load(handle)
        except FileNotFoundError:
            continue
        for i in np.arange(2):
            with open(f"CapitalBudgetingHigh/Data/Results/TrainData/inst_results/data_results_cb_N{N}_K{K}_{i}.pickle", "rb") as handle:
                new_data = pickle.load(handle)
            try:
                data = np.vstack([data, new_data["X"]])
                labels = np.hstack([labels, new_data["Y"]])
            except:
                data = new_data["X"]
                labels = new_data["Y"]
        print(pd.Series(labels).describe())

        # try:
        #     with open(f"CapitalBudgetingHigh/Data/RunInfo/runtime_overview_cb_N{N}_K{K}_inst0.pickle",
        #               "rb") as handle:
        #         rt_new = pickle.load(handle)
        # except FileNotFoundError:
        #     continue
        # for i in np.arange(10):
        #     with open(f"CapitalBudgetingHigh/Data/RunInfo/runtime_overview_cb_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
        #         rt_new = pickle.load(handle)
        #
        #     with open(f"CapitalBudgetingHigh/Data/RunInfo/run_info_cb_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
        #         info_new = pickle.load(handle)
        #     if i == 0:
        #         columns = [*list(info_new.index), "start_points", "perc_start_point", "mean_runtime", "qt90_runtime", "max_runtime"]
        #         info = pd.DataFrame(columns=columns)
        #     level = len(list(rt_new.columns)[0])
        #     perc_start_point = len(rt_new.loc[0])/(K**level)
        #     mean_runtime = np.mean(np.array(rt_new))
        #     qt90_runtime = np.quantile(np.array(rt_new), 0.9)
        #     max_runtime = np.max(np.array(rt_new))
        #     info.loc[len(info)] = [*list(info_new), len(rt_new.loc[0]), perc_start_point, mean_runtime, qt90_runtime, max_runtime]
        # print(f"K = {K} N = {N}")
        # print(info[["perc_start_point", "mean_runtime", "qt90_runtime", "max_runtime"]].describe())