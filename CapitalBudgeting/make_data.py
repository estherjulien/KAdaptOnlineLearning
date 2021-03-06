from ProblemFunctions.Env import ProjectsInstance
from Method.SucPredDataNew import data_gen_fun_max

import numpy as np
import sys

if __name__ == "__main__":
    array_num = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    time_limit = int(sys.argv[4])

    if len(sys.argv) == 6:
        num_instances = int(sys.argv[5])
    else:
        num_instances = int(np.floor(60/time_limit))

    att_series = ["coords", "obj_stat", "y_stat", "obj_det", "x_det", "y_det", "slack", "const_to_z_dist",
                  "const_to_const_dist"]

    problem_type = f"cb_N{N}_K{K}_m{time_limit}"

    for i in np.arange((array_num - 1)*num_instances, array_num*num_instances):
        env = ProjectsInstance(N=N, inst_num=i)
        data_gen_fun_max(K, env, att_series=att_series, problem_type=problem_type, time_limit=time_limit*60)
