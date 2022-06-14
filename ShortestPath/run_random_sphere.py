from Method.Random import algorithm
import pickle
import sys

if __name__ == "__main__":
    i = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])

    # load environment
    with open(f"ShortestPath/Data/Instances/inst_results/sp_env_sphere_N{N}_{i}.pickle",
              "rb") as handle:
        env = pickle.load(handle)

    # run random algorithm
    problem_type = f"sp_random_sphere_N{N}_K{K}"
    algorithm(K, env, problem_type=problem_type, time_limit=30 * 60)
