
import pandas as pd
import numpy as np

import itertools
import copy

import matplotlib.pyplot as plt

from y0.graph import NxMixedGraph
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.identify import is_identifiable
from y0.dsl import P
import networkx as nx

from dowhy import gcm
from scipy.stats import linregress

import single_cell_scm.graph_utilities

class CausalGraph:

    """
    A class to create a graph that can be used to find identifiable queries and fit an SCM.

    Parameters
    ----------
    indra_graph : pd.DataFrame
        A pandas DataFrame where the rows are the relationships between proteins

    Methods
    -------
    remove_latent_nodes()
        Reduces full graph to smaller graph containing only observed nodes

    add_latent_edges()
        Adds latent confounders to reduced graph and converts graph into y0 object
    """

    def __init__(self, indra_graph: pd.DataFrame):

        self.indra_graph = indra_graph

        ## Define missing variables
        self.full_graph = None
        self.latent_nodes = None
        self.observed_graph = None
        self.causal_graph = None
        self.identified_edges = None
        self.experimental_data = None

    def indra_to_dag(self,
                     relations: list,
                     drop_bidirect_randomly: bool):

        """
        Function to convert and clean an Indra graph (in Pandas DataFrame format) into a networkx graph. This graph can then
        be converted into latent variable model for causal inference.

        :param relations: list Types of relationships to keep in DAG
        :param drop_bidirect_randomly: bool Nodes which are fully connected to each other (bidirectional edge) will have one
        of their edges randomly dropped
        :return: nx.DiGraph Representation of Indra network in graph form
        """

        filtered_network = self.indra_graph[self.indra_graph["stmt_type"].isin(relations)].loc[:,
                           ["agA_name", "agB_name"]].rename(columns={"agA_name":"From", "agB_name":"To"})
        nx_graph = filtered_network.drop_duplicates(ignore_index=True)

        if drop_bidirect_randomly:
            nx_graph = single_cell_scm.graph_utilities.drop_bidirect(nx_graph)

        G = nx.DiGraph()
        for i in range(len(nx_graph)):
            G.add_edge(nx_graph.loc[i, "From"], nx_graph.loc[i, "To"])

        cycles = nx.simple_cycles(G)
        slice_cycles = list(itertools.islice(cycles, 100))

        # if len(slice_cycles) > 0:
        #     print("Cycles have been found in graph. Current causal inference methods require a DAG. Please remove ",
        #           "edges to remove cycles: ")
        #     print(slice_cycles)

        self.full_graph = G

    ## TODO: Replaced with y0 functionality
    # def remove_latent_nodes(self):
    #
    #     """
    #     Function to remove all latent nodes in a DAG. Removes unobserved confounders, and abstracts out latent nodes on
    #     causal paths. Prerequisite to identifying latent confounders for indentifiability.
    #
    #     :return:
    #     nx.DiGraph
    #         Reduced graph with only observed nodes
    #     """
    #
    #     g = self.full_graph.copy()
    #     for i in range(len(self.latent_nodes)):
    #
    #         g0 = g.copy()
    #         for node, degree in g.degree():
    #             ## only remove latent nodes
    #             if node == self.latent_nodes[i]:
    #
    #                 ## grab edges
    #                 in_edge = list(g0.in_edges(node))
    #                 out_edge = list(g0.out_edges(node))
    #
    #                 ## Dump node and fully reconnect edges
    #                 g0.remove_node(node)
    #                 for in_e in range(len(in_edge)):
    #                     for out_e in range(len(out_edge)):
    #                         g0.add_edge(in_edge[in_e][0], out_edge[out_e][1])
    #         g = g0
    #
    #     for node, degree in g.degree():
    #         g1 = g.copy()
    #         if degree == 0:
    #             g1.remove_node(node)
    #         g = g1
    #     self.observed_graph = g
    #
    # def add_latent_edges(self):
    #
    #     """
    #     Function to determine latent edges of a DAG. Requires two graphs. The first a
    #     full graph with both the latent and observed nodes. The second a reduced graph
    #     with only the observed graphs. The function then compares the graphs and
    #     determines where to add the latent edges.
    #
    #     Returns
    #     -------
    #     y0.graph.NxMixedGraph
    #         A y0 graph which includes observed nodes and latent confounders
    #     """
    #
    #     latent_edges = list()
    #     node_pairs = list(self.observed_graph.edges())
    #
    #     for pair in node_pairs:
    #
    #         temp_g = self.full_graph.copy()
    #
    #         ## Always break an edge to prevent lowest_common_ancestor from returning itself
    #         try:
    #             path = nx.shortest_path(temp_g, pair[0], pair[1])
    #         except:
    #             path = None
    #         while path is not None:
    #             temp_g.remove_edge(path[0], path[1])
    #             try:
    #                 path = nx.shortest_path(temp_g, pair[0], pair[1])
    #             except:
    #                 path = None
    #
    #         # find common ancestor
    #         confounder = nx.lowest_common_ancestor(temp_g, pair[0], pair[1], default=None)
    #
    #         if confounder is not None:
    #             path1 = nx.shortest_path(temp_g, confounder, pair[0])
    #             path1 = [i for i in path1 if i != pair[0]]
    #             path2 = nx.shortest_path(temp_g, confounder, pair[1])
    #             path2 = [i for i in path2 if i != pair[1]]
    #
    #             if (not any([self.observed_graph.has_node(i) for i in path1])) & \
    #                     (not any([self.observed_graph.has_node(i) for i in path2])):
    #                 latent_edges.append((pair[0], pair[1]))
    #     causal_graph = NxMixedGraph.from_str_edges(directed=list(self.observed_graph.edges),
    #                                          undirected=latent_edges)
    #
    #     self.causal_graph = causal_graph

    def identify_latent_nodes(self, experimental_data: pd.DataFrame):
        """
        Takes experimental data of observed nodes and returns list of proteins that were unobserved. Data must be in
        long format with protein names as columns.

        :param experimental_data:
        :return: latent_nodes: list - list of strings indicating nodes that were not measured in experimental data
        """

        self.experimental_data = experimental_data

        obs_nodes = self.experimental_data.columns
        all_nodes = list(self.full_graph)

        latent_nodes = [i for i in all_nodes if i not in obs_nodes and i != "\\n"]
        attrs = {node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes}

        nx.set_node_attributes(self.full_graph, attrs, name="hidden")

        self.latent_nodes = latent_nodes

    def create_latent_graph(self):
        """
        Takes latent nodes, removes them from the causal graph, and adds latent edges in their place. Will return a
        NxMixedGraph from the y0 package which can be used to find identifiable queries.

        Returns
        -------
        y0.graph.NxMixedGraph
            A y0 graph which includes observed nodes and latent confounders
        """
        # try:
        simplified_graph = simplify_latent_dag(copy.deepcopy(self.full_graph), "hidden")
        self.simplified_graph = simplified_graph
        y0_graph = NxMixedGraph()
        y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")

        self.causal_graph = y0_graph
        # except Exception as e:
        #     print("Error when creating latent graph. This is usually caused by the existence of cycles. {0}".format(e))

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

    def find_all_identifiable_pairs(self):

        def get_nodes_combinations(graph):
            all_pairs_it = nx.all_pairs_shortest_path(graph)
            potential_nodes = list()
            for i in all_pairs_it:
                temp_pairs = list(zip([i[0] for _ in range(len(i[1].keys()))], i[1].keys()))
                for j in temp_pairs:
                    if j[0] != j[1]:
                        potential_nodes.append(j)
            return(potential_nodes)

        potential_nodes = get_nodes_combinations(self.causal_graph.directed)

        identify = list()
        not_identify = list()
        i=1
        for pair in potential_nodes:
            is_ident = is_identifiable(self.causal_graph, P(pair[0] @ pair[1]))
            if is_ident:
                identify.append((pair[0], pair[1]))
            else:
                not_identify.append((pair[0], pair[1]))

            print(i)
            i+=1

        self.identified_edges = {"Identifiable": identify, "NonIdentifiable": not_identify}

    def find_queries_of_interest(self, abs_filter=95):

        causal_model = gcm.StructuralCausalModel(self.causal_graph)
        gcm.auto.assign_causal_mechanisms(causal_model, self.experimental_data)
        gcm.fit(causal_model, self.data)

        ace_pairs = dict()
        lm_coef = dict()

        for pair in self.identified_edges["Identifiable"]:
            ## TODO: Figure out why some queries fail (backwards queries maybe?)
            try:
                effect = gcm.average_causal_effect(causal_model,
                                                   str(pair[1]),
                                                   interventions_alternative={str(pair[0]): lambda y: 1},
                                                   interventions_reference={str(pair[0]): lambda y: 0},
                                                   num_samples_to_draw=1000)
                ace_pairs["{0}-{1}".format(pair[0], pair[1])] = effect

                ## TODO: Remove when satisfied with performance compared to regression
                model = linregress(self.experimental_data[str(pair[0])], self.experimental_data[str(pair[1])])
                lm_coef["{0}-{1}".format(pair[0], pair[1])] = model.slope

            except:
                pass
            ## TODO: Warning about failed estimation
            #     print("{0}-{1} Could not be estimated".format(pair[0], pair[1]))

        diff = dict()
        for key in ace_pairs.keys():
            diff[key] = abs(ace_pairs[key] - lm_coef[key])

        sorted_dif = sorted(diff.items(), key=lambda item: item[1], reverse=True)

        high_ace = np.percentile(np.abs(list(ace_pairs.values())), abs_filter)

        pairs_to_keep = [sorted_dif[i] for i in range(len(sorted_dif)) if abs(ace_pairs[sorted_dif[i][0]]) > high_ace]

        return {"ace_pairs": ace_pairs, "lm_coef": lm_coef, "pairs_to_keep": pairs_to_keep}