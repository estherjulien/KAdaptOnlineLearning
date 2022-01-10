from ShortestPath.Environment import Graph
import pickle
import sys

if __name__ == "__main__":
    inst_num = int(sys.argv[1])
    N = int(sys.argv[2])
    gamma_perc = 0.3
    first_stage_ratio = 0.3

    graph = Graph(N=N, gamma_perc=gamma_perc, first_stage_ratio=first_stage_ratio, max_degree=5, throw_away_perc=0.3,
                  inst_num=inst_num)

    with open(f"ClusterOnlineLearning/Instances/SP/sp_env_N{N}_{inst_num}.pickle", "wb") as handle:
        pickle.dump(graph, handle)

    graph_output = {"N": N, "first_stage_ratio": first_stage_ratio, "gamma_perc": gamma_perc, "inst_num": inst_num,
                    "init_vertices": graph.vertices, "init_distances": graph.distances}

    with open(f"ClusterOnlineLearning/Instances/SP/sp_env_output_N{N}_{inst_num}.pickle", "wb") as handle:
        pickle.dump(graph_output, handle)


