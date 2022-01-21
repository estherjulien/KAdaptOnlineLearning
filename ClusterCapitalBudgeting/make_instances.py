from ProblemFunctions.Env import ProjectsInstance

import pickle
import sys


if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    test_data = int(sys.argv[3])
    env = ProjectsInstance(N, inst_num=i)

    if test_data:
        with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/test_env_cb_N{N}_{i}.pickle", "wb") as handle:
            pickle.dump(env, handle)
    else:
        # save train instances
        with open(f"ClusterCapitalBudgeting/Data/Instances/inst_results/train_env_cb_N{N}_{i}.pickle", "wb") as handle:
            pickle.dump(env, handle)
