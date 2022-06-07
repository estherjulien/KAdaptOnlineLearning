from Method.Random import algorithm
import pickle
import sys

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    degree = int(sys.argv[3])
    tap = int(sys.argv[4])
    gamma = int(sys.argv[5])
    K = int(sys.argv[6])

    # load environment
    with open(
            f"ShortestPathCluster/Data/Instances/inst_results/sp_env_sphere_N{N}_d{degree}_tap{tap}_g{gamma}_{i}.pickle",
            "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"sp_random_sphere_N{N}_d{degree}_tap{tap}_g{gamma}_K{K}"
    algorithm(K, env, problem_type=problem_type, time_limit=30 * 60)

