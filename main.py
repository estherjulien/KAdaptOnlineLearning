import pickle
import numpy as np
import pandas as pd

results = dict()
for i in np.arange(112):
    with open(f"Results/SPIP/Decisions/final_results_spip_random_K3_N150_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)
