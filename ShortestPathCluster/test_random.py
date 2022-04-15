from Method.Random import algorithm

from joblib import Parallel, delayed
import pickle
import sys

if __name__ == "__main__":
    num_instances = 32
    K = 4
    tap = 80
    for N in [50, 100, 200]:
        for degree in [2, 3, 4, 5]:
            for gamma in [1, 2, 3, 4]:
                # load environment
                with open(f"ShortestPathCluster/Data/Instances/sp_env_list_sphere_N{N}_d{degree}_tap{tap}"
                          f"_g{gamma}_{num_instances}.pickle", "rb") as handle:
                    env_list = pickle.load(handle)

                # run random algorithm
                problem_type = f"sp_random_N{N}_d{degree}_tap{tap}_g{tap}_K{K}"
                results = Parallel(n_jobs=-1)(delayed(algorithm)(K, env, problem_type=problem_type) for env in env_list)

