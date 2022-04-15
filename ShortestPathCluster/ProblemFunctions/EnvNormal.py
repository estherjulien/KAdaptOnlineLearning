import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy


class Graph:
    def __init__(self, N, gamma=3, inst_num=0, throw_away_perc=0.7, init_vertices=None, init_distances=None, init_s=None, init_t=None, plot=False):
        self.N = N
        if init_vertices is None:
            self.vertices, init_arcs = self.init_graph()
        else:
            self.vertices = init_vertices
        if init_distances is None:
            self.distances, self.s, self.t = self.update_graph(init_arcs, throw_away_perc)
        else:
            self.distances = init_distances
            self.s = init_s
            self.t = init_t
        self.arcs = self.distances > 1e-5
        self.num_arcs = int(self.arcs.sum())
        self.xi_dim = self.num_arcs
        self.x_dim = self.num_arcs
        self.y_dim = self.num_arcs

        self.arcs_array = np.array([[i, j] for i in np.arange(self.N) for j in np.arange(self.N) if self.arcs[i, j]])
        self.distances_array = np.array([self.distances[i, j] for i, j in self.arcs_array])
        self.arcs_in, self.arcs_out = self.in_out_arcs()

        self.bigM = sum(self.distances_array)*3
        self.upper_bound = sum(self.distances_array)*3
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.num_arcs)

        if gamma is not None:
            self.gamma = gamma
        if plot:
            self.plot_graph()

    def set_gamma(self, gamma):
        self.gamma = gamma

    def vertices_fun(self):
        vertices_set = np.zeros([self.N, 2], dtype=np.float)
        # start and terminal node are the first and last one, on 0,0 and 10,10, respectively
        for n in np.arange(self.N):
            x, y = np.random.uniform(0, 10, 2)
            vertices_set[n] = [x, y]
        return vertices_set

    def init_graph(self):
        vertices = self.vertices_fun()
        N = self.N
        # make initial arcs
        arcs = np.ones([N, N], dtype=np.float)
        for i in np.arange(N):
            for j in np.arange(N):
                if i == j:
                    arcs[i, j] = 0

        return vertices, arcs

    def update_graph(self, arcs, throw_away_perc):
        N = self.N
        # delete arcs with middle in no go zones
        arc_dict = dict()
        for i in np.arange(N):
            for j in np.arange(N):
                if arcs[i, j] < 1e-5:
                    continue
                x_i, y_i = self.vertices[i]
                x_j, y_j = self.vertices[j]
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                arcs[i, j] = distance
                arc_dict[(i, j)] = distance
        # delete long arcs (first sort and then sth percent)
        arc_dict_order = {k: v for k, v in sorted(arc_dict.items(), key=lambda item: -item[1])}
        # first ones are s and t!
        s, t = list(arc_dict_order)[0]
        # delete "throw_away_perc" longest arcs
        throw_away_num = np.floor(len(arc_dict_order)*throw_away_perc)
        del_arc = 0
        while True:
            arcs_copy = copy.deepcopy(arcs)
            while del_arc < throw_away_num:
                # check here if when you delete this, degree out and in will be >= 1 and total degree >= min_degree
                try:
                    i, j = list(arc_dict_order)[del_arc]
                except KeyError:
                    break
                # check in degree of j
                if sum([arcs[:, j] > 1e-5][0]) < 1:
                    continue
                # check out degree of i
                if sum([arcs[i, :] > 1e-5][0]) < 1:
                    continue
                # check each time if you can delete this arc based on connected graph
                arcs_copy[i, j] = 0
                del_arc += 1
            if self.is_connected(arcs_copy, s, t):
                break
            else:
                throw_away_num -= 5
                print(f"Inst {self.inst_num}: again")
        return arcs_copy, s, t

    def is_connected(self, arcs, s, t):
        N = self.N
        # dijkstra on arcs, with node s=0 and t=N
        # initialize stuff
        tmp_bigM = 10**2
        dist = np.ones(N)*tmp_bigM
        dist[s] = 0
        Q = list(np.arange(N))
        prev = dict()
        # algorithm
        connected = False
        while Q and not connected:
            i = Q[np.argmin(dist[Q])]
            Q.remove(i)

            for j in np.arange(N):
                if arcs[i, j] < 1e-5:
                    continue
                alt = dist[i] + arcs[i, j]
                if alt < dist[j]:
                    if j == t:
                        connected = True
                        break
                    dist[j] = alt
                    prev[j] = i
        return connected

    def in_out_arcs(self):
        arcs_in = {i: [] for i in np.arange(self.N)}
        arcs_out = {i: [] for i in np.arange(self.N)}
        for a in np.arange(self.num_arcs):
            i, j = self.arcs_array[a]
            arcs_in[j].append(a)
            arcs_out[i].append(a)
        return arcs_in, arcs_out

    # plot graphs
    def plot_graph(self):
        arcs = self.arcs
        sns.set()
        sns.set_style("whitegrid")
        # plot all arcs and vertices
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if arcs[i, j] < 1e-5:
                    continue
                plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "darkgrey")
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "rx")
        plt.plot(*self.vertices[self.s], "go")
        plt.plot(*self.vertices[self.t], "bo")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(f"init_graph_N{self.N}_inst{self.inst_num}")
        plt.close()
        return plt

    def plot_graph_solutions(self, K, y, tau, x=None, tmp=False, it=0, alg_type="rand"):
        arcs = self.arcs
        sns.set()
        sns.set_style("whitegrid")
        # plot all arcs and vertices
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if arcs[i, j] < 1e-5:
                    continue
                plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "darkgrey")
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "rx")

        # first stage
        if x is not None:
            for a in np.arange(self.num_arcs):
                if x[a] > 0.5:
                    i, j = self.arcs_array[a]
                    plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "k-", linewidth=5)

        # second stage
        cols = ["blue", "purple", "green", "red", "yellow", "orange", "grey", "cornflowerblue", "hotpink"]
        for k in np.arange(K):
            if len(tau[k]) == 0:
                continue
            for a in np.arange(self.num_arcs):
                if y[k][a] > 0.5:
                    i, j = self.arcs_array[a]
                    plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], cols[k])

        plt.xlim(0, 10)
        plt.ylim(0, 10)

        if tmp:
            plt.savefig(f"Results/Plots/tmp_graph_{alg_type}_inst{self.inst_num}_it{it}")
        else:
            plt.savefig(f"Results/Plots/final_graph_{alg_type}_inst{self.inst_num}")
            plt.savefig(f"Results/Plots/final_graph_{alg_type}_inst{self.inst_num}.pdf")
        plt.close()
