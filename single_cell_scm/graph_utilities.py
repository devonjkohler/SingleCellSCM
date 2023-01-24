import networkx as nx
from y0.graph import NxMixedGraph

import matplotlib.pyplot as plt

class CausalGraph:

    """
    A class to create a graph that can be used to find identifiable queries and fit an SCM.

    Parameters
    ----------
    full_graph : networkx.DiGraph
        A full graph with both latent and observed nodes
    latent_nodes : list
        A list of strings indicating which nodes in the full graph were not observed

    Methods
    -------
    remove_latent_nodes()
        Reduces full graph to smaller graph containing only observed nodes

    add_latent_edges()
        Adds latent confounders to reduced graph and converts graph into y0 object
    """

    def __init__(self, full_graph: nx.DiGraph, latent_nodes: list):

        self.full_graph = full_graph
        self.latent_nodes = latent_nodes

        ## Define missing variables
        self.observed_graph = None
        self.causal_graph = None

    def remove_latent_nodes(self):

        """
        Function to remove all latent nodes in a DAG. Removes unobserved confounders, and abstracts out latent nodes on
        causal paths. Prerequisite to identifying latent confounders for indentifiability.

        :return:
        nx.DiGraph
            Reduced graph with only observed nodes
        """

        g = self.full_graph.copy()
        for i in range(len(self.latent_nodes)):

            g0 = g.copy()
            for node, degree in g.degree():
                ## only remove latent nodes
                if node == self.latent_nodes[i]:

                    ## grab edges
                    in_edge = list(g0.in_edges(node))
                    out_edge = list(g0.out_edges(node))

                    ## Dump node and fully reconnect edges
                    g0.remove_node(node)
                    for in_e in range(len(in_edge)):
                        for out_e in range(len(out_edge)):
                            g0.add_edge(in_edge[in_e][0], out_edge[out_e][1])
            g = g0

        for node, degree in g.degree():
            g1 = g.copy()
            if degree == 0:
                g1.remove_node(node)
            g = g1
        self.observed_graph = g

    def add_latent_edges(self):

        """
        Function to determine latent edges of a DAG. Requires two graphs. The first a
        full graph with both the latent and observed nodes. The second a reduced graph
        with only the observed graphs. The function then compares the graphs and
        determines where to add the latent edges.

        Returns
        -------
        y0.graph.NxMixedGraph
            A y0 graph which includes observed nodes and latent confounders
        """

        latent_edges = list()
        node_pairs = list(self.observed_graph.edges())

        for pair in node_pairs:

            temp_g = self.full_graph.copy()

            ## Always break an edge to prevent lowest_common_ancestor from returning itself
            try:
                path = nx.shortest_path(temp_g, pair[0], pair[1])
            except:
                path = None
            while path is not None:
                temp_g.remove_edge(path[0], path[1])
                try:
                    path = nx.shortest_path(temp_g, pair[0], pair[1])
                except:
                    path = None

            # find common ancestor
            confounder = nx.lowest_common_ancestor(temp_g, pair[0], pair[1], default=None)

            if confounder is not None:
                path1 = nx.shortest_path(temp_g, confounder, pair[0])
                path1 = [i for i in path1 if i != pair[0]]
                path2 = nx.shortest_path(temp_g, confounder, pair[1])
                path2 = [i for i in path2 if i != pair[1]]

                if (not any([self.observed_graph.has_node(i) for i in path1])) & \
                        (not any([self.observed_graph.has_node(i) for i in path2])):
                    latent_edges.append((pair[0], pair[1]))
        causal_graph = NxMixedGraph.from_str_edges(directed=list(self.observed_graph.edges),
                                             undirected=latent_edges)

        self.causal_graph = causal_graph

    def plot_latent_graph(self, figure_size=(4, 3), title=None):

        ## Create new graph and specify color and shape of observed vs latent edges
        temp_g = nx.DiGraph()

        for d_edge in list(self.causal_graph.directed.edges):
            temp_g.add_edge(d_edge[0], d_edge[1], color="black", style='-', size=30)
        for u_edge in list(self.causal_graph.undirected.edges):
            if temp_g.has_edge(u_edge[0], u_edge[1]):
                temp_g.add_edge(u_edge[1], u_edge[0], color="red", style='--', size=1)
            else:
                temp_g.add_edge(u_edge[0], u_edge[1], color="red", style='--', size=1)

        ## Extract edge attributes
        pos = nx.spring_layout(temp_g)
        edges = temp_g.edges()
        colors = [temp_g[u][v]['color'] for u, v in edges]
        styles = [temp_g[u][v]['style'] for u, v in edges]
        arrowsizes = [temp_g[u][v]['size'] for u, v in edges]

        ## Plot
        fig, ax = plt.subplots(figsize=figure_size)
        nx.draw_networkx_nodes(temp_g, pos=pos, node_size=1000, margins=[.1, .1], alpha=.7)
        nx.draw_networkx_labels(temp_g, pos=pos, font_weight='bold')
        nx.draw_networkx_edges(temp_g, pos=pos, ax=ax, connectionstyle='arc3, rad = 0.1',
                               edge_color=colors, width=3, style=styles, arrowsize=arrowsizes)
        if title is not None:
            ax.set_title(title)
        plt.show()
