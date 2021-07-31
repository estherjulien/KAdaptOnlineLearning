import pickle
import numpy as np
import pandas as pd

weight_list = list()
for i in np.arange(3):
    with open(f"ShortestPath/Results/Decisions/OnlineMP/final_results_online_mp_sp_mp_att_online_K4_N100_g30_fs30_inst{i}.pickle", "rb") as handle:
        weight_list += [pickle.load(handle)[1]["weights"]]
    print(f"Weights instance {i}: \n {weight_list[i]}")

new_weights = pd.DataFrame(weight_list).mean(axis=0)

with open(f"ShortestPath/Results/Weights/avg_weights_sp_mp_K4_N100_g30_fs30_3_no_coords.pickle", "wb") as handle:
    pickle.dump(new_weights, handle)
