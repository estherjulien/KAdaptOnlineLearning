from ProblemFunctions.EnvNormal import Graph
from Method.SucPredData import data_gen_fun_max

import numpy as np
import sys

if __name__ == "__main__":
    array_num = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    time_limit = int(sys.argv[4])
    normalized = float(sys.argv[5])

    if len(sys.argv) == 7:
        num_instances = int(sys.argv[6])
    else:
        num_instances = int(np.floor(60/time_limit))

    perc_label = 0.05
    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    if normalized:
        problem_type = f"sp_norm_p{int(perc_label*100)}_N{N}_K{K}"
    else:
        problem_type = f"sp_p{int(perc_label*100)}_N{N}_K{K}"

    for i in np.arange((array_num - 1)*num_instances, array_num*num_instances):
        env = Graph(N=N, gamma=3, inst_num=i)
        data_gen_fun_max(K, env, att_series=att_series, problem_type=problem_type, time_limit=time_limit*60,
                         perc_label=perc_label, normalized=normalized)
