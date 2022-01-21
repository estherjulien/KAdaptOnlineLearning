import numpy as np
import pickle
import sys

if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N_train = int(sys.argv[2])
    K_train = int(sys.argv[3])

    problem_type = f"cb_suc_pred_N{N_train}_K{K_train}"

    results = dict()
    for i in np.arange(num_instances):
        try:
            with open(f"ClusterCapitalBudgeting/Data/ResultsSucPred/TrainData/inst_results/final_results_{problem_type}_"
                      f"{i}.pickle", "rb") as handle:
                results[i] = pickle.load(handle)
        except FileNotFoundError:
            continue

    with open(f"ClusterCapitalBudgeting/Data/ResultsSucPred/TrainData/FINAL_DATA_{problem_type}_"
              f"{num_instances}.pickle", "wb") as handle:
        pickle.dump(results, handle)
    
