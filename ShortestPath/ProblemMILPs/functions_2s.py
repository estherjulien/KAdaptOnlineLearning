import numpy as np
import gurobipy as gp
from gurobipy import GRB


def scenario_fun_update(K, k_new, xi_new, graph, scen_model):
    # use same model and just add new constraints
    x = {a: scen_model.getVarByName("x[{}]".format(a)) for a in np.arange(graph.num_arcs)}
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    # first stage constraint
    scen_model.addConstr(
        gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
        >= graph.max_first_stage)
    # objective constraint
    scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * (y[k_new][a] + x[a])
                                     for a in np.arange(graph.num_arcs)) <= theta)
    scen_model.update()
    # solve model
    scen_model.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, x_sol, y_sol, scen_model


def scenario_fun_build(K, tau, graph):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = scen_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    x = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="x")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))

    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints

    # first stage
    scen_model.addConstr(gp.quicksum(x[a] for a in graph.arcs_out[0]) >= 1)
    # switch point
    scen_model.addConstr(gp.quicksum(gp.quicksum(x[a] for a in graph.arcs_in[j]) - gp.quicksum(x[a] for a in graph.arcs_out[j]) for j in np.arange(1, graph.N-1)) == 1)
    # outside range
    # scen_model.addConstrs(x[a] == 0 for j in graph.outside_range for a in graph.arcs_out[j])

    # second stage
    for k in np.arange(K):
        # only one outgoing for each node
        scen_model.addConstrs(
            gp.quicksum(x[a] + y[k][a] for a in graph.arcs_out[j]) <= 1 for j in np.arange(graph.N - 1))
        # all ys inside range have to be zero
        scen_model.addConstrs(y[k][a] == 0 for j in graph.inside_range for a in graph.arcs_in[j])
        # total sum of arcs is smaller than N-1
        scen_model.addConstr(gp.quicksum(x[a] + y[k][a] for a in np.arange(graph.N)) <= graph.N - 1)
        for j in np.arange(graph.N):
            if j == 0:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) <= 0)
                continue
            if j == N - 1:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                continue
            # normal shortest path constraint
            scen_model.addConstr(
                gp.quicksum(y[k][a] + x[a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] + x[a] for a in graph.arcs_in[j]) >= 0)

            # no vertex can have ingoing y and outgoing x
            scen_model.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_in[j])
                + gp.quicksum(x[a] for a in graph.arcs_out[j]) <= 1)
    # uncertain constraints
    for k in np.arange(K):
        for xi in tau[k]:
            # first stage constraint
            scen_model.addConstr(
                gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
                >= graph.max_first_stage)
            # objective function
            scen_model.addConstr(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * (x[a] + y[k][a])
                                             for a in np.arange(graph.num_arcs)) <= theta)
    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, x_sol, y_sol, scen_model


def separation_fun(K, x, y, theta, graph, tau):
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-graph.bigM, name="zeta", vtype=GRB.CONTINUOUS)
    xi = sep_model.addVars(graph.num_arcs, lb=0, ub=1, name="xi", vtype=GRB.CONTINUOUS)

    num_consts = 2
    z = dict()
    for k in np.arange(K):
        z[k] = sep_model.addVars(num_consts, vtype=GRB.BINARY)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)

    # z constraint
    sep_model.addConstrs(gp.quicksum(z[k][c] for c in np.arange(num_consts)) == 1 for k in np.arange(K))

    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[a] for a in np.arange(graph.num_arcs)) <= graph.gamma)

    for k in np.arange(K):
        if len(tau[k]) > 0:
            # first stage constraint
            sep_model.addConstr(zeta <= -(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs)))
                                 + graph.max_first_stage + graph.bigM * (1 - z[k][0]))
            # objective constraint
            sep_model.addConstr(zeta <= gp.quicksum((1 + xi[a] / 2) *
                                graph.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(graph.num_arcs))
                                - theta + graph.bigM * (1 - z[k][1]))

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol


def scenario_fun_update_sub_tree(K, new_node, xi_dict, graph, scen_model=None):
    # use same model and just add new constraint
    x = {a: scen_model.getVarByName(f"x[{a}]") for a in np.arange(graph.num_arcs)}
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    for node_sec in np.arange(1, len(new_node)):
        xi_new = xi_dict[new_node[:node_sec]]
        k_new = new_node[node_sec]
        # first stage constraint
        scen_model.addConstr(
            gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
            >= graph.max_first_stage, name=f"const1_{new_node[:node_sec]}_{k_new}")
        # objective function
        scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * (x[a] + y[k][a])
                                         for a in np.arange(graph.num_arcs)) <= theta, name=f"const2_{new_node[:node_sec]}_{k_new}")
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    # delete constraints
    for node_sec in np.arange(1, len(new_node)):
        xi_found = new_node[:node_sec]
        k_new = new_node[node_sec]
        scen_model.remove(scen_model.getConstrByName(f"const1_{xi_found}_{k_new}"))
        scen_model.remove(scen_model.getConstrByName(f"const2_{xi_found}_{k_new}"))
    scen_model.update()

    return theta_sol, x_sol, y_sol


# STATIC SOLUTION ROBUST COUNTERPART
def static_solution_rc(graph):
    src = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = src.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    x = src.addVars(graph.num_arcs, vtype=GRB.BINARY, name="x")
    y = src.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y")

    # dual variables
    # objective
    d_1_a = src.addVars(graph.num_arcs)
    d_1_b = src.addVar()
    # first-stage constraint
    d_2_a = src.addVars(graph.num_arcs)
    d_2_b = src.addVar()

    # objective function
    src.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints

    # first stage
    src.addConstr(gp.quicksum(x[a] for a in graph.arcs_out[0]) >= 1)
    # switch point without extra variable
    src.addConstr(gp.quicksum(gp.quicksum(x[a] for a in graph.arcs_in[j]) - gp.quicksum(x[a] for a in graph.arcs_out[j]) for j in np.arange(1, graph.N-1)) == 1)
    # outside range     ONLY HERE
    src.addConstrs(x[a] == 0 for j in graph.outside_range for a in graph.arcs_out[j])

    # second stage
    # only one outgoing for each node
    src.addConstrs(
        gp.quicksum(x[a] + y[a] for a in graph.arcs_out[j]) <= 1 for j in np.arange(graph.N - 1))
    # all ys inside range have to be zero
    src.addConstrs(y[a] == 0 for j in graph.inside_range for a in graph.arcs_in[j])
    # total sum of arcs is smaller than N-1
    src.addConstr(gp.quicksum(x[a] + y[a] for a in np.arange(graph.N)) <= graph.N - 1)
    for j in np.arange(graph.N):
        if j == 0:
            src.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) <= 0)
            continue
        if j == N - 1:
            src.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 1)
            continue
        # normal shortest path constraint
        src.addConstr(
            gp.quicksum(y[a] + x[a] for a in graph.arcs_out[j])
            - gp.quicksum(y[a] + x[a] for a in graph.arcs_in[j]) >= 0)

        # no vertex can have ingoing y and outgoing x
        src.addConstr(
            gp.quicksum(y[a] for a in graph.arcs_in[j])
            + gp.quicksum(x[a] for a in graph.arcs_out[j]) <= 1)

        # objective function
        src.addConstr(
            gp.quicksum((y[a] + x[a]) * graph.distances_array[a] + d_1_a[a] for a in np.arange(graph.num_arcs))
            + d_1_b * graph.gamma <= theta)
        src.addConstrs(graph.distances_array[a] / 2 * y[a] - d_1_a[a] - d_1_b
                       <= 0 for a in np.arange(graph.num_arcs))
        # first stage constraint
        src.addConstr(gp.quicksum((x[a]) * graph.distances_array[a] - d_2_a[a] for a in np.arange(graph.num_arcs))
                      - d_2_b * graph.gamma >= graph.max_first_stage)
        src.addConstrs(graph.distances_array[a] / 2 * y[a] + d_2_a[a] + d_2_b
                       >= 0 for a in np.arange(graph.num_arcs))

    # solve model
    src.Params.OutputFlag = 0
    src.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = {i: var.X for i, var in y.items()}
    theta_sol = src.getVarByName("theta").X

    return x_sol


# SCENARIO FUN STATIC ATTRIBUTES
def scenario_fun_static_build(graph, x):
    sms = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = sms.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = sms.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y")

    # objective function
    sms.setObjective(theta, GRB.MINIMIZE)

    # second stage
    # only one outgoing for each node
    sms.addConstrs(
        gp.quicksum(x[a] + y[a] for a in graph.arcs_out[j]) <= 1 for j in np.arange(graph.N - 1))
    # all ys inside range have to be zero
    sms.addConstrs(y[a] == 0 for j in graph.inside_range for a in graph.arcs_in[j])
    # total sum of arcs is smaller than N-1
    sms.addConstr(gp.quicksum(x[a] + y[a] for a in np.arange(graph.N)) <= graph.N - 1)
    for j in np.arange(graph.N):
        if j == 0:
            sms.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) <= 0)
            continue
        if j == N - 1:
            sms.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 1)
            continue
        # normal shortest path constraint
        sms.addConstr(
            gp.quicksum(y[a] + x[a] for a in graph.arcs_out[j])
            - gp.quicksum(y[a] + x[a] for a in graph.arcs_in[j]) >= 0)

        # no vertex can have ingoing y and outgoing x
        sms.addConstr(
            gp.quicksum(y[a] for a in graph.arcs_in[j])
            + gp.quicksum(x[a] for a in graph.arcs_out[j]) <= 1)

    # solve model
    sms.Params.OutputFlag = 0
    sms.optimize()
    return sms


def scenario_fun_static_update(graph, scen, x, sms):
    y = {a: sms.getVarByName(f"y[{a}]") for a in np.arange(graph.num_arcs)}
    theta = sms.getVarByName("theta")

    # objective constraint
    sms.addConstr(gp.quicksum((1 + scen[a] / 2) * graph.distances_array[a] * (y[a] + x[a])
                                     for a in np.arange(graph.num_arcs)) <= theta, name="new_const")
    sms.update()
    # solve model
    sms.optimize()
    y_sol = {i: var.X for i, var in y.items()}
    theta_sol = sms.getVarByName("theta").X

    # delete new constraint
    sms.remove(sms.getConstrByName("new_const"))
    sms.update()

    return theta_sol, y_sol


# SCENARIO FUN NOMINAL ATTRIBUTES
def scenario_fun_deterministic_build(graph):
    smn = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = smn.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    x = smn.addVars(graph.num_arcs, vtype=GRB.BINARY, name="x")
    y = smn.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y")

    # objective function
    smn.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints

    # first stage
    smn.addConstr(gp.quicksum(x[a] for a in graph.arcs_out[0]) >= 1)
    # switch point without extra variable
    smn.addConstr(gp.quicksum(gp.quicksum(x[a] for a in graph.arcs_in[j]) - gp.quicksum(x[a] for a in graph.arcs_out[j]) for j in np.arange(1, graph.N-1)) == 1)

    # second stage
    # only one outgoing for each node
    smn.addConstrs(
        gp.quicksum(x[a] + y[a] for a in graph.arcs_out[j]) <= 1 for j in np.arange(graph.N - 1))
    # all ys inside range have to be zero
    smn.addConstrs(y[a] == 0 for j in graph.inside_range for a in graph.arcs_in[j])
    # total sum of arcs is smaller than N-1
    smn.addConstr(gp.quicksum(x[a] + y[a] for a in np.arange(graph.N)) <= graph.N - 1)
    for j in np.arange(graph.N):
        if j == 0:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_out[j]) <= 0)
            continue
        if j == N - 1:
            smn.addConstr(gp.quicksum(y[a] for a in graph.arcs_in[j]) >= 1)
            continue
        # normal shortest path constraint
        smn.addConstr(
            gp.quicksum(y[a] + x[a] for a in graph.arcs_out[j])
            - gp.quicksum(y[a] + x[a] for a in graph.arcs_in[j]) >= 0)

        # no vertex can have ingoing y and outgoing x
        smn.addConstr(
            gp.quicksum(y[a] for a in graph.arcs_in[j])
            + gp.quicksum(x[a] for a in graph.arcs_out[j]) <= 1)

    # solve model
    smn.Params.OutputFlag = 0
    smn.optimize()

    return smn


def scenario_fun_deterministic_update(graph, scen, smn):
    x = {a: smn.getVarByName(f"x[{a}]") for a in np.arange(graph.num_arcs)}
    y = {a: smn.getVarByName(f"y[{a}]") for a in np.arange(graph.num_arcs)}
    theta = smn.getVarByName("theta")

    # first stage constraint
    smn.addConstr(
        gp.quicksum((1 + scen[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
        >= graph.max_first_stage, name="new_const")
    # objective constraint
    smn.addConstr(gp.quicksum((1 + scen[a] / 2) * graph.distances_array[a] * (y[a] + x[a])
                              for a in np.arange(graph.num_arcs)) <= theta)
    smn.update()
    # solve model
    smn.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = {i: var.X for i, var in y.items()}
    theta_sol = smn.getVarByName("theta").X

    # delete new constraint
    smn.remove(smn.getConstrByName("new_const"))
    smn.update()

    return theta_sol, x_sol, y_sol