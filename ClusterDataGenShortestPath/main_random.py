from Random import algorithm
import pickle
import sys

if __name__ == "__main__":
    inst_num = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    time_limit = int(sys.argv[4])

    with open(f"ClusterDataGenShortestPath/Instances/sp_env_N{N}_{inst_num}.pickle", "rb") as handle:
        env = pickle.load(handle)

    problem_type = f"sp_random_K{K}_N{N}"
    algorithm(K, env, time_limit=time_limit, problem_type=problem_type)
