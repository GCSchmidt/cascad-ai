import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from cascadai.schema.token_piece import Token_Type, Token


MAX_NEIGHBOURS = 6
DISTANCE_FACTOR = 2.5


class EnvironmentGraph:

    def __init__(self, tokens: list[Token]) -> None:
        self.EG = nx.Graph()
        self.build_graph(tokens)

    def add_tokens(self, tokens) -> None:
        for i, t in enumerate(tokens):
            self.EG.add_node(t, type=t.type, x=t.x, y=t.y, width=t.width)

    def get_token_distance_threshold(self):
        """
        determines the maximum distance between tokens for them to be connected in graphs.

        Method 1: Distance = Fixed Factor * AVG Width of tokens 
        Method 2: Distance = Fixed Factor * (minimum distance between detected tokens)

        Args:
            tokens (_type_): _description_
        """
        nodes = list(self.EG.nodes)
        widths = np.array([n.width for n in nodes])
        avg_width = np.average(widths)
        return avg_width * DISTANCE_FACTOR

    def add_edges(self) -> None:
        distance_threshold = self.get_token_distance_threshold()
        nodes = list(self.EG.nodes)
        coords = np.array([(n.x, n.y) for n in nodes])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=distance_threshold, output_type="ndarray")
        dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        order = np.argsort(dists)
        for index in order:
            i, j = pairs[index]
            t1, t2 = nodes[i], nodes[j]
            if self.EG.degree[t1] < MAX_NEIGHBOURS and self.EG.degree[t2] < MAX_NEIGHBOURS:
                self.EG.add_edge(t1, t2, weight=dists[index]) 

    def build_graph(self, tokens):
        self.add_tokens(tokens)
        self.add_edges()
