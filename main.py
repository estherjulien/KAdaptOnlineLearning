import pickle
import numpy as np
import pandas as pd

results_rand = dict()
results_rule = dict()
for i in np.arange(16):
    with open(f"ShortestPath/Results/Decisions/Random/final_results_sp_rand_np_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        _, results_rand[i] = pickle.load(handle)
    with open(f"ShortestPath/Results/Decisions/ObjViolDiff/final_results_sp_obj_viol_diff_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        _, results_rule[i] = pickle.load(handle)