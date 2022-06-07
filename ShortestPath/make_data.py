from ProblemFunctions.EnvSphere import Graph
from Method.SucPredData import data_gen_fun

import numpy as np
import sys

if __name__ == "__main__":
    # array_num = int(sys.argv[1])
    # N = int(sys.argv[2])
    # K = int(sys.argv[3])
    # time_limit = int(sys.argv[4])
    # normalized = float(sys.argv[5])
    #
    # if len(sys.argv) == 7:
    #     num_instances = int(sys.argv[6])
    # else:
    #     num_instances = int(np.floor(60/time_limit))

    array_num = 1
    N = 20
    K = 2
    time_limit = 0.2
    normalized = False
    num_instances = 30

    perc_label = 0.05
    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    if normalized:
        problem_type = f"sp_sphere_N{N}_d5_tap90_g7_norm_p{int(perc_label*100)}_K{K}"
    else:
        problem_type = f"sp_sphere_N{N}_d5_tap90_g7_p{int(perc_label*100)}_K{K}"

    for i in np.arange((array_num - 1)*num_instances, array_num*num_instances):
        env = Graph(N=N, gamma=7, degree=5, throw_away_perc=0.9, inst_num=i)
        data_gen_fun(K, env, att_series=att_series, problem_type=problem_type, time_limit=time_limit*60,
                         perc_label=perc_label, normalized=normalized)
