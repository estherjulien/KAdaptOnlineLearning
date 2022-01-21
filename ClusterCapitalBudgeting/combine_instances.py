import numpy as np
import pickle
import sys


def combine_data(N, num_instances, test_data=False):
    env_list = []

    for i in np.arange(num_instances):
        if test_data:
            with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/test_env_cb_N{N}_{i}.pickle", "rb") as handle:
                env = pickle.load(handle)
        else:
            with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/train_env_cb_N{N}_{i}.pickle", "rb") as handle:
                env = pickle.load(handle)

        env_list.append(env)

    # save instances in one file
    if test_data:
        with open(f"ClusterCapitalBudgeting/Data/Instances/test_env_list_cb_N{N}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(env_list, handle)
    else:
        with open(f"ClusterCapitalBudgeting/Data/Instances/train_env_cb_N{N}_{num_instances}.pickle", "wb") as handle:
            pickle.dump(env_list, handle)


if __name__ == "__main__":
    num_instances = int(sys.argv[1])
    N = int(sys.argv[2])
    test_data = int(sys.argv[3])
    combine_data(N, num_instances, test_data)