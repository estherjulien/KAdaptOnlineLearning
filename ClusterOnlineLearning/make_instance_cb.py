from CapitalBudgetingLoans.Environment import ProjectsInstance
import pickle
import sys

if __name__ == "__main__":
    inst_num = int(sys.argv[1])
    N = int(sys.argv[2])
    xi_dim = 4

    env = ProjectsInstance(N=N, xi_dim=xi_dim, inst_num=inst_num)

    with open(f"ClusterOnlineLearning/Instances/CB/cb_env_N{N}_{inst_num}.pickle", "wb") as handle:
        pickle.dump(env, handle)

    graph_output = {"N": N, "inst_num": inst_num, "init_cost_vector": env.cost_nom_vector,
                    "init_phi_vector": env.phi_vector, "init_psi_vector": env.psi_vector}

    with open(f"ClusterOnlineLearning/Instances/CB/cb_env_output_N{N}_{inst_num}.pickle", "wb") as handle:
        pickle.dump(graph_output, handle)


