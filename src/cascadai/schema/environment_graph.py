import networkx as nx
import numpy as np
import math
from collections import deque
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from cascadai.schema.token_piece import Token_Type, Token


MAX_NEIGHBOURS = 6
DISTANCE_FACTOR = 2.25


def angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


class EnvironmentGraph:

    def __init__(self, tokens: list[Token]) -> None:
        self.token_graph = nx.Graph()  # initially true postions of tokens in image, then ideal hex lattics 
        self.primary_axis = 0  # base angle (degrees) for hexagon lattice
        self.build_graph(tokens)

    @property
    def latice_angles(self):
        return np.arange(0, 6) * 60 + self.primary_axis

    def add_tokens(self, tokens) -> None:
        for i, t in enumerate(tokens):
            self.token_graph.add_node(t, type=t.type, x=t.x, y=t.y, width=t.width)

    def get_token_distance_threshold(self):
        """
        determines the maximum distance between tokens for them to be connected in graphs.

        Method 1: Distance = Fixed Factor * AVG Width of tokens 

        Args:
            tokens (list(Token)): list of Token objects
        """
        nodes = list(self.token_graph.nodes)
        widths = np.array([n.width for n in nodes])
        if len(widths) == 0:
            return 0.0
        avg_width = np.average(widths)
        return avg_width * DISTANCE_FACTOR

    def add_edges(self) -> None:
        distance_threshold = self.get_token_distance_threshold()
        nodes = list(self.token_graph.nodes)
        if not nodes:
            return
        coords = np.array([(n.x, n.y) for n in nodes])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=distance_threshold, output_type="ndarray")
        dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        order = np.argsort(dists)
        for index in order:
            i, j = pairs[index]
            t1, t2 = nodes[i], nodes[j]
            if self.token_graph.degree[t1] < MAX_NEIGHBOURS and self.token_graph.degree[t2] < MAX_NEIGHBOURS:
                self.token_graph.add_edge(t1, t2, weight=dists[index]) 

    def build_graph(self, tokens):
        self.add_tokens(tokens)
        self.add_edges()
        self._hex_lattice_orientation_estimation()
        self.fill_token_graph()

    def fill_token_graph(self):
        # loop until all real tokens have 6 neighbours, and graph is fully connected
        while True:
            for n in list(self.token_graph.nodes()):
                # after the 1st loop all real tokens will have 6 neighbours
                if self.token_graph.degree[n] < 6:
                    self.add_missing_neighbours(n)
                    self.token_graph.clear_edges()
                    self.add_edges()
            if nx.is_connected(self.token_graph):
                break

    def _get_angle_between_nodes(self, node1, node2) -> float:
        """
        Gets the angle between 2 tokens / nodes in the token graph.
        The angle is relative to node1.

        Returns:
            float: degrees
        """
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y 
        d_x, d_y = (x2 - x1), (y2 - y1)  # image coords, y increases downward
        angle = math.degrees(np.arctan2(d_y, d_x))
        return angle

    def token_in_line_of_sight(self, node1, node2) -> bool:
        """_summary_

        Args:
            node1 (_type_): _description_
            node2 (_type_): _description_

        Returns:
            bool: _description_
        """
        angle = int(self._get_angle_between_nodes(node1, node2))

        diffs = angle_difference(self.latice_angles, angle)

        diff_min = np.min(diffs)

        if diff_min < 5:
            return True

        return False

    def _get_edge_angles(self) -> np.ndarray:
        angles = []
        for node in self.token_graph.nodes:
            neighbours = neighbours = self.token_graph.neighbors(node)
            for neighbour in neighbours:
                angle = self._get_angle_between_nodes(node, neighbour)
                angles.append(angle)
        return np.array(angles)

    def _hex_lattice_orientation_estimation(self):
        angles = self._get_edge_angles()
        if angles.size == 0:
            return

        directions = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        best_theta = None
        best_score = math.inf

        for theta in np.arange(-30, 30):
            diffs = angle_difference(angles[:, None], theta + directions[None, :])
            score = np.sum(np.min(diffs, axis=1))
            if score < best_score:
                best_score = score
                best_theta = theta

        self.primary_axis = best_theta

    def add_missing_neighbours(self, node):
        n_neigbours = self.token_graph.degree[node]
        if MAX_NEIGHBOURS == n_neigbours:
            return

        neighbours = self.token_graph.neighbors(node)

        neighbour_angles = [
            self._get_angle_between_nodes(node, neighbour) for neighbour in neighbours
        ]

        occupied_angles = np.zeros(n_neigbours, dtype=int)

        for i, angle1 in enumerate(neighbour_angles):
            d = [
                angle_difference(angle1, angle2) for angle2 in self.latice_angles
                ]

            occupied_angles[i] = np.argmin(d)

        mask = np.ones(len(self.latice_angles), dtype=bool)
        mask[occupied_angles] = False

        free_angles = self.latice_angles[mask]

        #  There might be an unconnected node at the free angle
        # check that angle for a node and add edge between them

        width = node.width

        for angle in free_angles:
            angle_radians = np.deg2rad(angle)
            edge_length = width * (DISTANCE_FACTOR - 0.75)
            d_x = np.cos(angle_radians) * edge_length
            d_y = np.sin(angle_radians) * edge_length
            x_new = d_x + node.x
            y_new = d_y + node.y
            blank_token = Token(Token_Type.BLANK, x_new, y_new, node.width)
            self.token_graph.add_node(blank_token, type=blank_token.type, x=blank_token.x, y=blank_token.y, width=blank_token.width)
            # self.token_graph.add_edge(node, blank_token, weight=edge_length) 

    def build_ideal_lattice(self):
        """Snap the token graph onto a perfect unit hexagonal lattice.

        Each node is mapped to an axial lattice coordinate (q, r) by walking
        the measured edges and snapping each edge onto the nearest of the 6
        lattice directions (aligned with ``primary_axis``). Updated node
        positions are exactly 1 unit apart with neighbors 60° apart. Updates the
        original measured graph ``self.token_graph``.
        """

        theta = np.deg2rad(self.primary_axis)
        e1 = np.array([np.cos(theta), np.sin(theta)])
        e2 = np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])

        rel_angles = np.deg2rad(np.arange(0, 360, 60))
        deltas = np.array([
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        ])

        axial = {}
        for component in nx.connected_components(self.token_graph):
            component = list(component)
            xs = np.array([n.x for n in component], dtype=float)
            ys = np.array([n.y for n in component], dtype=float)
            dists_to_centre = (xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2
            root = component[int(np.argmin(dists_to_centre))]

            axial[root] = np.zeros(2, dtype=int)
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for nb in self.token_graph.neighbors(node):
                    if nb in axial:
                        continue
                    phi = np.arctan2(nb.y - node.y, nb.x - node.x)
                    diff = np.abs(np.arctan2(
                        np.sin(phi - (theta + rel_angles)),
                        np.cos(phi - (theta + rel_angles)),
                    ))
                    k = int(np.argmin(diff))
                    axial[nb] = axial[node] + deltas[k]
                    queue.append(nb)

        for node, (q, r) in axial.items():
            px, py = q * e1 + r * e2
            node.x, node.y = float(px), float(py)
            self.token_graph.nodes[node]["x"], self.token_graph.nodes[node]["y"] = node.x, node.y

