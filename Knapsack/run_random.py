from Method.Random import algorithm
import pickle
import sys

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    budget_perc = int(sys.argv[3])
    gamma_perc = int(sys.argv[4])
    K = int(sys.argv[5])

    # load environment
    with open(f"Knapsack/Data/Instances/inst_results/ks_env_N{N}_b{budget_perc}_g{gamma_perc}_{i}.pickle", "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"ks_random_N{N}_b{budget_perc}_g{gamma_perc}_K{K}"
    algorithm(K, env, problem_type=problem_type)

