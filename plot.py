import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle


def plot_stuff(K, N, num_inst, level=30, thresh=False, ml_model_type=None):
    results = [{}, {}]
    alg_types = ["Random", "Success Prediction"]
    num_algs = len(results)
    insts = []
    for i in np.arange(num_inst):
        try:
            with open(f"SPSphereResults/inst_results/final_results_sp_random_sphere_N{N}_d5_tap90_g7_K{K}_inst{i}.pickle", "rb") as handle:
                results[0][i] = pickle.load(handle)

            with open(f"SPSphereResults/inst_results/final_results_sp_suc_pred_rf_nt_T[N{N}_K{K}]_ML[{ml_model_type}]_L{level}_inst{i}.pickle", "rb") as handle:
                results[1][i] = pickle.load(handle)
            if results[1][i]["runtime"] > 32*60:
                continue
            if len(results[1][i]["inc_thetas_t"]) <= 2:
                continue
            insts.append(i)
        except:
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])
    plt.xlim([0, 31*60])
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=1)
    if thresh:
        plt.savefig(f"plot_runtime_sp_sphere_K{K}_N{N}_L{level}_{len(insts)}")
    else:
        plt.savefig(f"plot_runtime_sp_sphere_ML[{ml_model_type}]_K{K}_N{N}_L{level}_{len(insts)}")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in insts])
    n_grid = np.arange(0, n_grid_max+10, 10)
    # n_grid = np.arange(0, 1000+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[0][i]["theta"]
            for n, theta in results[a][i]["inc_thetas_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.legend(loc=1)
    if thresh:
        plt.savefig(f"plot_nodes_sp_sphere_K{K}_N{N}_L{level}_{len(insts)}")
    else:
        plt.savefig(f"plot_nodes_sp_sphere_ML[{ml_model_type}]_K{K}_N{N}_L{level}_{len(insts)}")
    plt.close()


def plot_random(problem_type, K, N, num_inst, gamma=None, degree=None, tap=None):
    results = [{}]
    alg_types = ["Random"]
    num_algs = len(results)
    insts = []
    for i in np.arange(num_inst):
        try:
            with open(f"SPShereResults/inst_results/final_results_sp_random_sphere_N{N}_d{degree}_tap{tap}_g{gamma}"
                      f"_K{K}_inst{i}.pickle", "rb") as handle:
                results[0][i] = pickle.load(handle)
            if results[0][i]["theta"] > 1000:
                continue
            insts.append(i)
        except:
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*30+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])

    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=1)
    plt.savefig(f"plot_runtime_{problem_type}_nt_K{K}_N{N}_d{degree}_tap{tap}_g{gamma}_{len(insts)}")
    plt.close()

    # PLOT RESULTS OVER NODES
    n_grid_max = np.max([results[0][i]["tot_nodes"] for i in insts])
    n_grid = np.arange(0, n_grid_max+10, 10)
    num_grids = len(n_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            n_alg = np.zeros(num_grids)
            theta_final = results[0][i]["theta"]
            for n, theta in results[a][i]["inc_thetas_n"].items():
                n_alg[n_grid > n] = theta/theta_final
            obj[a][i] = n_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
        plt.plot(n_grid, avg_random, label=alg_types[a])

    plt.xlabel("Nodes")
    plt.ylabel("Relative Objective")
    plt.legend(loc=1)
    plt.savefig(f"plot_nodes_{problem_type}_nt_K{K}_N{N}_d{degree}_tap{tap}_g{gamma}_{len(insts)}")
    plt.close()
    return results[0]


def show_stuff_sp(normal=True):
    num_inst = 16
    index = [(N, degree, tap, gamma, i) for N in [50]
             for degree in [3, 4, 5, 6]
             for tap in [70, 80, 90]
             for gamma in [1, 2, 3, 4, 5, 6, 7, 8]
             for i in np.arange(num_inst)]
    index_avg = [(N, degree, tap, gamma) for N in [50]
                 for degree in [3, 4, 5, 6]
                 for tap in [70, 80, 90]
                 for gamma in [1, 2, 3, 4, 5, 6, 7, 8]]
    df_results = pd.DataFrame(index=index, columns=["obj_change", "num_res"], dtype=float)
    df_results.index = pd.MultiIndex.from_tuples(df_results.index)
    df_avg_results = pd.DataFrame(index=index_avg, columns=[*["10%", "50%", "90%"], "num_sols"], dtype=float)
    df_avg_results.index = pd.MultiIndex.from_tuples(df_avg_results.index)

    ind = pd.IndexSlice
    for N in [50]:
        for degree in [3, 4, 5, 6]:
            for tap in [70, 80, 90]:
                for gamma in np.arange(1, 9):
                    insts = []
                    for i in np.arange(num_inst):
                        try:
                            if normal:
                                with open(f"SPSphereResults/inst_results/final_results_sp_normal_test_random_N{N}_d{degree}_tap{tap}_g{gamma}_K4_inst{i}.pickle", "rb") as handle:
                                    res = pickle.load(handle)
                            else:
                                try:
                                    with open(f"SPSphereResults/TestResults/final_results_sp_sphere_test_random_N{N}_d{degree}_tap{tap}_g{gamma}_K4_inst{i}.pickle", "rb") as handle:
                                        res = pickle.load(handle)
                                except:
                                    with open(f"SPSphereResults/inst_results/final_results_sp_random_sphere_N{N}_d{degree}_tap{tap}_g{gamma}_K4_inst{i}.pickle", "rb") as handle:
                                        pickle.load(handle)
                            obj_change = list(res["inc_thetas_t"].values())[0] / res["theta"] - 1
                            num_res = len(res["inc_thetas_t"]) - 1
                            df_results.loc[(N, degree, tap, gamma, i)] = [obj_change, num_res]
                            insts.append(i)
                        except:
                            continue
                    if len(insts) == 0:
                        continue
                    change_description = df_results.loc[ind[N, degree, tap, gamma, :]]["obj_change"].describe(percentiles=[0.1, 0.5, 0.9])[["10%", "50%", "90%"]].round(5).to_list()
                    num_sols = df_results.loc[ind[N, degree, tap, gamma, :]]["num_res"].mean().round(2)
                    if change_description[1] > 0.025:
                        print(f"N{N}    degree{degree}      tap{tap}        gamma{gamma}    :   num_sols = {num_sols}   change = {change_description}")
                    df_avg_results.loc[(N, degree, tap, gamma)] = [*change_description, num_sols]
    df_results.index = pd.MultiIndex.from_tuples(df_results.index)
    df_avg_results.index = pd.MultiIndex.from_tuples(df_avg_results.index)
    return df_results, df_avg_results


# df_results = show_stuff_sp()

def diff_cb_random():
    num_instances = 112
    K_list = [2, 3, 4, 5, 6]
    N_list = [10, 20, 30]
    results = pd.DataFrame(index=N_list, columns=K_list, dtype=float)
    insts = pd.DataFrame(0, index=N_list, columns=K_list, dtype=int)

    for K in K_list:
        for N in N_list:
            for i in np.arange(num_instances):
                diff = []
                try:
                    with open(f"CapitalBudgeting/Data/Results/Decisions/inst_results/final_results_cb_random_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
                        res = pickle.load(handle)
                    insts.loc[N, K] += 1
                except FileNotFoundError:
                    continue
                first = list(res["inc_thetas_t"].values())[0]
                last = res["theta"]
                diff.append(abs((first-last)/last))
            results.loc[N, K] = np.mean(diff)*100
    return results, insts, N_list, K_list


def diff_sp_normal_random():
    num_instances = 112
    K_list = [2, 3, 4, 5, 6]
    N_list = [20, 40, 60]
    results = pd.DataFrame(index=N_list, columns=K_list, dtype=float)
    insts = pd.DataFrame(0, index=N_list, columns=K_list, dtype=int)
    for K in K_list:
        for N in N_list:
            for i in np.arange(num_instances):
                diff = []
                try:
                    with open(f"ShortestPath/Data/Results/Decisions/inst_results/final_results_sp_random_normal_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
                        res = pickle.load(handle)
                    insts.loc[N, K] += 1
                except FileNotFoundError:
                    continue
                first = list(res["inc_thetas_t"].values())[0]
                last = res["theta"]
                diff.append(abs((first-last)/last))
            results.loc[N, K] = np.mean(diff)*100
    return results, insts, N_list, K_list


def diff_sp_sphere_random():
    num_instances = 112
    K_list = [2, 3, 4, 5, 6]
    N_list = [20, 40, 60]
    results = pd.DataFrame(index=N_list, columns=K_list, dtype=float)
    insts = pd.DataFrame(0, index=N_list, columns=K_list, dtype=int)
    for K in K_list:
        for N in N_list:
            for i in np.arange(num_instances):
                diff = []
                try:
                    with open(f"ShortestPath/Data/Results/Decisions/inst_results/final_results_sp_random_sphere_N{N}_K{K}_inst{i}.pickle", "rb") as handle:
                        res = pickle.load(handle)
                    insts.loc[N, K] += 1
                except FileNotFoundError:
                    continue
                first = list(res["inc_thetas_t"].values())[0]
                last = res["theta"]
                diff.append(abs((first-last)/last))
            results.loc[N, K] = np.mean(diff)*100
    return results, insts, N_list, K_list


def plot_diff_random(cb=False, sp_normal=False, sp_sphere=False):
    if cb:
        results, _, N_list, K_list = diff_cb_random()
        save_name = "cb"
    elif sp_normal:
        results, _, N_list, K_list = diff_sp_normal_random()
        save_name = "sp_normal"
    elif sp_sphere:
        results, _, N_list, K_list = diff_sp_sphere_random()
        save_name = "sp_sphere"
    sn.set()
    sn.color_palette("rocket", as_cmap=True)
    fig, ax = plt.subplots(figsize=(10,5))

    results.plot(style='o-', ax=ax)
    plt.xticks(N_list)
    plt.legend(title="K")
    plt.xlabel("Instance size ($N$)", fontsize=15)
    plt.ylabel("Total increase of objective ($\%$)", fontsize=15)
    plt.savefig("bla_diff_test")
    plt.savefig(f"diff_results_{save_name}.pdf", dpi=400)


def plot_cb_level(K, I):
    results = [{}, {}]
    alg_types = ["Random", "Success Prediction"]
    num_inst = 112
    num_algs = len(results)
    for L in [5, 10, 30, 20, 40, 50, 60, 70]:
        save_name = f"cb_ML[N10_K{K}_I{I}]_T[N10_K{K}]_L{L}"
        insts = []
        for i in np.arange(num_inst):
            try:
                with open(f"CBDecisions/inst_results/final_results_cb_random_N10_K{K}_inst{i}.pickle", "rb") as handle:
                    results[0][i] = pickle.load(handle)
                with open(f"CBDecisions/inst_results/final_results_cb_suc_pred_ML[N10_K{K}_I{I}]_T[N10_K{K}]_L{L}_inst{i}.pickle", "rb") as handle:
                    results[1][i] = pickle.load(handle)
                insts.append(i)
            except:
                continue
        # PLOT RESULTS OVER RUNTIME
        t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*60+15, 15)])
        num_grids = len(t_grid)
        obj = []
        for a in np.arange(num_algs):
            obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=np.float))

        for a in np.arange(num_algs):
            for i in insts:
                # random
                theta_final = results[0][i]["theta"]
                t_alg = np.zeros(num_grids)
                for t, theta in results[a][i]["inc_thetas_t"].items():
                    t_alg[t_grid > t] = theta/theta_final
                obj[a][i] = t_alg

            obj[a] = np.array(obj[a])
            obj[a][obj[a] == 0] = np.nan

        sn.set_style("whitegrid")
        # plot results
        for a in np.arange(num_algs):
            avg_random = np.quantile(obj[a], 0.5, axis=1)
            random_10 = np.quantile(obj[a], 0.1, axis=1)
            random_90 = np.quantile(obj[a], 0.9, axis=1)
            plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
            plt.plot(t_grid, avg_random, label=alg_types[a])

        plt.ylim([0.85, 1.05])
        plt.xlim([0, 31*60])
        plt.xlabel("Runtime (sec)")
        plt.ylabel("Relative Objective")
        plt.legend(loc=3)
        plt.savefig(f"plot_runtime_{save_name}_{len(insts)}")
        plt.close()

        # PLOT RESULTS OVER NODES
        n_grid_max = np.max([results[a][i]["tot_nodes"] for a in np.arange(num_algs) for i in insts])
        n_grid = np.arange(0, n_grid_max + 10, 10)
        # n_grid = np.arange(0, 1000+10, 10)
        num_grids = len(n_grid)
        obj = []
        for a in np.arange(num_algs):
            obj.append(pd.DataFrame(index=n_grid, columns=insts, dtype=np.float))

        for a in np.arange(num_algs):
            for i in insts:
                # random
                n_alg = np.zeros(num_grids)
                theta_final = results[0][i]["theta"]
                for n, theta in results[a][i]["inc_thetas_n"].items():
                    n_alg[n_grid > n] = theta / theta_final
                obj[a][i] = n_alg

            obj[a] = np.array(obj[a])
            obj[a][obj[a] == 0] = np.nan

        sn.set_style("whitegrid")
        # plot results
        for a in np.arange(num_algs):
            avg_random = np.quantile(obj[a], 0.5, axis=1)
            random_10 = np.quantile(obj[a], 0.1, axis=1)
            random_90 = np.quantile(obj[a], 0.9, axis=1)
            plt.fill_between(n_grid, random_10, random_90, alpha=0.5)
            plt.plot(n_grid, avg_random, label=alg_types[a])

        plt.ylim([0.85, 1.05])
        plt.xlabel("Nodes")
        plt.ylabel("Relative Objective")
        plt.legend(loc=3)
        plt.savefig(f"plot_nodes_{save_name}_{len(insts)}")
        plt.close()


def plot_cb_random(K):
    results = [{}, {}]
    alg_types = ["Random", "Random Preprocess"]
    num_inst = 112
    num_algs = len(results)
    insts = []
    for i in np.arange(num_inst):
        try:
            with open(f"CBDecisions/inst_results/final_results_cb_random_N10_K{K}_inst{i}.pickle", "rb") as handle:
                results[0][i] = pickle.load(handle)
            with open(f"CBDecisions/inst_results/final_results_cb_random_C[preproc]_N10_K{K}_inst{i}.pickle", "rb") as handle:
                results[1][i] = pickle.load(handle)
            insts.append(i)
        except:
            continue
    # PLOT RESULTS OVER RUNTIME
    t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 60*60+15, 15)])
    num_grids = len(t_grid)
    obj = []
    for a in np.arange(num_algs):
        obj.append(pd.DataFrame(index=t_grid, columns=insts, dtype=np.float))

    for a in np.arange(num_algs):
        for i in insts:
            # random
            theta_final = results[0][i]["theta"]
            t_alg = np.zeros(num_grids)
            for t, theta in results[a][i]["inc_thetas_t"].items():
                t_alg[t_grid > t] = theta/theta_final
            obj[a][i] = t_alg

        obj[a] = np.array(obj[a])
        obj[a][obj[a] == 0] = np.nan

    sn.set_style("whitegrid")
    # plot results
    for a in np.arange(num_algs):
        avg_random = np.quantile(obj[a], 0.5, axis=1)
        random_10 = np.quantile(obj[a], 0.1, axis=1)
        random_90 = np.quantile(obj[a], 0.9, axis=1)
        plt.fill_between(t_grid, random_10, random_90, alpha=0.5)
        plt.plot(t_grid, avg_random, label=alg_types[a])

    plt.ylim([0.8, 1.05])
    plt.xlim([0, 31*60])
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Relative Objective")
    plt.legend(loc=3)
    plt.savefig(f"plot_runtime_cb_random_C[PREPROC]_T[N10_K{K}]_{len(insts)}")
    plt.close()


plot_cb_level(4, 500)
