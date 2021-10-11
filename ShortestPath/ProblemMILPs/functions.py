import numpy as np
import gurobipy as gp
from gurobipy import GRB


def scenario_fun_update(K, k_new, xi_new, graph, scen_model=None):
    # use same model and just add new constraint
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * y[k_new][a]
                                     for a in np.arange(graph.num_arcs)) <= theta)
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, None, y_sol, scen_model


def scenario_fun_build(K, tau, graph):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = scen_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))
    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for k in np.arange(K):
        for j in np.arange(graph.N):
            if j == 0:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) >= 1)
                continue
            if j == N - 1:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                continue
            scen_model.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 0)

    for k in np.arange(K):
        for xi in tau[k]:
            scen_model.addConstr(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                             for a in np.arange(graph.num_arcs)) <= theta)
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, None, y_sol, scen_model


def scenario_fun_update_sub_tree(K, new_node, xi_dict, graph, scen_model=None):
    # use same model and just add new constraint
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    for node_sec in np.arange(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * y[k_new][a]
                                         for a in np.arange(graph.num_arcs)) <= theta, name=f"const_{new_node[:node_sec]}_{k_new}")
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    # delete constraints
    for node_sec in np.arange(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const_{xi_found}_{k_new}"))
    scen_model.update()

    return theta_sol, None, y_sol


def separation_fun(K, x, y, theta, graph, tau):
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-graph.bigM, name="zeta", vtype=GRB.CONTINUOUS)
    xi = sep_model.addVars(graph.num_arcs, lb=0, ub=1, name="xi", vtype=GRB.CONTINUOUS)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)

    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[a] for a in np.arange(graph.num_arcs)) <= graph.gamma)

    for k in np.arange(K):
        if len(tau[k]) > 0:
            sep_model.addConstr(zeta <= gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                                    for a in np.arange(graph.num_arcs)) - theta)

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol


def scenario_fun_deterministic_build(graph):
    smn = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = smn.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = smn.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y")
    # objective function
    smn.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for j in np.arange(graph.N):
        if j == 0:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) >= 1)
            continue
        if j == N - 1:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 1)
            continue
        smn.addConstr(
            gp.quicksum(y[a] for a in graph.arcs_out[j])
            - gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 0)

    # solve model
    smn.Params.OutputFlag = 0
    smn.optimize()
    return smn


def scenario_fun_deterministic_update(graph, scen, smn):
    y = {a: smn.getVarByName(f"y[{a}]") for a in np.arange(graph.num_arcs)}
    theta = smn.getVarByName("theta")

    # constraint
    smn.addConstr(gp.quicksum((1 + scen[a] / 2) * graph.distances_array[a] * y[a]
                              for a in np.arange(graph.num_arcs)) <= theta, name="new_const")
    smn.update()
    # solve model
    smn.optimize()
    y_sol = {i: var.X for i, var in y.items()}
    theta_sol = smn.getVarByName("theta").X

    # delete new constraint
    smn.remove(smn.getConstrByName("new_const"))
    smn.update()

    return theta_sol, y_sol