import numpy as np
import pickle
import sys

if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    # rf_name = f"N{N_train}_K{K_train}_rf{data}"
    rf_name = str(sys.argv[4])

    problem_type = f"cb_suc_pred_ML[{rf_name}]_T[N{N}_K{K}]"

    # combine all results
    results = dict()
    for i in np.arange(num_instances):
        with open(f"ClusterCapitalBudgeting/Data/ResultsSucPred/Decisions/inst_results/final_results_{problem_type}_{i}.pickle", "rb") \
                as handle:
            results[i] = pickle.load(handle)
    with open(f"ClusterCapitalBudgeting/Data/ResultsSucPred/Decisions/FINAL_results_{problem_type}_{num_instances}.pickle", "wb") as handle:
        pickle.dump(results, handle)
