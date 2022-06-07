import copy

from Method.Random import algorithm

from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import itertools
import pickle
import sys

degree_set = np.arange(3, 7)
gamma_set = np.arange(1, 5)

if __name__ == "__main__":
    num_instances = 32
    K = 4
    tap = 80
    for N in [50]:
        for degree in degree_set:
            for gamma in gamma_set:
                print(f"N = {N}, degree = {degree}, gamma = {gamma}")
                problem_type = f"sp_random_N{N}_d{degree}_tap{tap}_g{gamma}_K{K}"
                # # load environment
                # with open(f"Data/Instances/sp_env_sphere_list_N{N}_d{degree}_tap{tap}"
                #           f"_g{gamma}_{num_instances}.pickle", "rb") as handle:
                #     env_list = pickle.load(handle)
                #
                # # run random algorithm
                # algorithm(K, env_list[0], problem_type=problem_type, first_sol=True)
                # load environment
                with open(f"ShortestPathCluster/Data/Instances/sp_env_sphere_list_N{N}_d{degree}_tap{tap}"
                          f"_g{gamma}_{num_instances}.pickle", "rb") as handle:
                    env_list = pickle.load(handle)

                # run random algorithm
                results = Parallel(n_jobs=-1)(delayed(algorithm)(K, env, problem_type=problem_type, first_sol=True) for env in env_list)
                with open(f"ShortestPathCluster/Data/test_results_{problem_type}.pickle", "wb") as handle:
                    pickle.dump(results, handle)

#
# check_plot = False
# make_df = True
# if make_df:
#     # df_good = pd.DataFrame(columns=["N", "degree", "gamma", "num_edges", "tot_scens", "theta",
#     #                                 "edge_10q", "edge_50q", "edge_90q",
#     #                                 *[f"num_edges_path_{k}" for k in np.arange(4)],
#     #                                 *[f"len_path_{k}" for k in np.arange(4)],
#     #                                 "same_len_num", "min_len",
#     #                                 "node_occ_min", "node_occ_mean", "node_occ_max",
#     #                                 *[f"node_{i}" for i in np.arange(8)]], dtype=float)
#     df_good = pd.DataFrame(columns=["N", "degree", "gamma",
#                                     "num_unique_y"])
#     df_bad = copy.deepcopy(df_good)
#     sphere_pos_neg = [i for i in itertools.product(*[np.arange(2)]*3)]
#
# good_degree_gamma = pd.DataFrame(0.0, index=degree_set, columns=gamma_set)
# v_good_degree_gamma = pd.DataFrame(0.0, index=degree_set, columns=gamma_set)
# v_bad_degree_gamma = pd.DataFrame(0.0, index=degree_set, columns=gamma_set)
# num_instances = 32
# K = 4
# tap = 80
# for N in [50]:
#     for degree in degree_set:
#         for gamma in gamma_set:
#             print(f"N = {N}, degree = {degree}, gamma = {gamma}")
#             problem_type = f"sp_random_N{N}_d{degree}_tap{tap}_g{gamma}_K{K}"
#             with open(f"Data/test_results_{problem_type}.pickle", "rb") as handle:
#                 results = pickle.load(handle)
#             for i in np.arange(num_instances):
#                 graph = results[i]["graph"]
#                 if make_df:
#                     tau = results[i]["tau"]
#                     tot_scens = sum([len(t) for t in tau.values()])
#                     if tot_scens <= 3:
#                         continue
#                     # EDGE LENGTHS
#                     # edge_info = np.array([np.quantile(graph.distances_array, 0.10),
#                     #                       np.quantile(graph.distances_array, 0.5),
#                     #                       np.quantile(graph.distances_array, 0.90)])
#                     # num_edge_info = [sum(results[i]["y"][0].values()) for k in np.arange(K)]
#                     # path_length_info = np.zeros(K)
#                     # for k in np.arange(K):
#                     #     if len(tau[k]) == 0:
#                     #         path_length_info[k] = np.nan
#                     #         continue
#                     #     y_k = np.where([np.array(list(results[i]["y"][k].values())) > 0.9])[1]
#                     #     path_k = []
#                     #     for scen in tau[k]:
#                     #         path_k.append(sum(graph.distances_array[a]*(1 + scen[a]/2) for a in y_k))
#                     #     path_length_info[k] = max(path_k)
#                     # path_length_info /= results[i]["theta"]
#                     # same_len_num = sum(path_length_info == 1)
#                     # NODE LOCATIONS
#                     # scen_occupancy = np.zeros(8)
#                     # zone = 0
#                     # for x_zone, y_zone, z_zone in sphere_pos_neg:
#                     #     x_bound = (0, 1) if x_zone else (-1, 0)
#                     #     y_bound = (0, 1) if y_zone else (-1, 0)
#                     #     z_bound = (0, 1) if z_zone else (-1, 0)
#                     #     for x, y, z in graph.vertices:
#                     #         if x_bound[0] <= x <= x_bound[1] and y_bound[0] <= y <= y_bound[1] and z_bound[0] <= z <= z_bound[1]:
#                     #             scen_occupancy[zone] += 1
#                     #     zone += 1
#                     # scen_occupancy /= N
#                     # scen_info = [min(scen_occupancy), np.mean(scen_occupancy), max(scen_occupancy)]
#
#                     # inst_info = [N, degree, gamma, graph.num_arcs, tot_scens, results[i]["theta"],
#                     #              *edge_info, *num_edge_info, *np.sort(path_length_info),
#                     #              same_len_num, min(path_length_info),
#                     #              *scen_info, *scen_occupancy]
#                     y_unique = set()
#                     for k in np.arange(K):
#                         if len(tau[k]):
#                             y_k = np.where([np.array(list(results[i]["y"][k].values())) > 0.9])[1]
#                             path = graph.arcs_array[y_k]
#                             print(f"I{i}K{k}: s = {graph.s}, t = {graph.t}, path = {list(path)}")
#                             y_unique.add(tuple(y_k))
#
#                     inst_info = [N, degree, gamma, len(y_unique)]
#                     if results[i]["good"]:
#                         df_good.loc[len(df_good)] = inst_info
#                         good_degree_gamma.loc[degree, gamma] += 1/num_instances
#
#                     else:
#                         df_bad.loc[len(df_bad)] = inst_info
#                 if check_plot:
#                     label = "good" if results[i]["good"] else "bad"
#                     graph.plot_graph(problem_type=problem_type, name=label)
#
# # ONLY HERE!
#
# good_min = df_good["tot_scens"].quantile(0.9)
# df_v_good = df_good.loc[df_good["tot_scens"] >= good_min]
# for i, columns in df_v_good.iterrows():
#     v_good_degree_gamma.loc[columns["degree"], columns["gamma"]] += 1
# df_des_good = df_good.describe()
#
# bad_max = df_bad["tot_scens"].quantile(0.1)
# df_v_bad = df_bad.loc[df_bad["tot_scens"] <= bad_max]
# for i, columns in df_v_bad.iterrows():
#     v_bad_degree_gamma.loc[columns["degree"], columns["gamma"]] += 1
# df_des_bad = df_v_bad.describe()
