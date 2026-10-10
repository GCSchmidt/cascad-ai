import networkx as nx
import numpy as np
import math
from enum import IntEnum
from collections import deque
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from cascadai.schema.token_piece import Token_Type, Token


MAX_NEIGHBOURS = 6
DISTANCE_FACTOR = 2.25


class EdegeDirectionClass(IntEnum):
    CLASS0 = 0
    CLASS1 = 1
    CLASS2 = 2


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

    def _edge_direction_index(self, node1, node2) -> int:
        """Nearest lattice direction index (0..5) for the edge node1-node2."""
        angle = self._get_angle_between_nodes(node1, node2)
        diffs = angle_difference(self.latice_angles, angle)
        return int(np.argmin(diffs))

    def _edge_line_class(self, node1, node2) -> EdegeDirectionClass:
        """Line class 0/1/2. Opposite directions (k, k+3) map to the same class."""
        return EdegeDirectionClass(self._edge_direction_index(node1, node2) % 3)

    def resolve_line_direction(self, direction) -> EdegeDirectionClass:
        """Resolve a direction to a line class (0/1/2).

        An int 0/1/2 (or ``EdegeDirectionClass`` member) is taken as a line
        class relative to ``primary_axis``. Any other value is treated as an
        absolute angle in degrees and snapped to the nearest lattice direction.
        """
        if isinstance(direction, (int, np.integer)) and int(direction) in (0, 1, 2):
            return EdegeDirectionClass(int(direction))
        diffs = angle_difference(self.latice_angles, float(direction))
        return EdegeDirectionClass(int(np.argmin(diffs)) % 3)

    def assign_edge_classes(self, graph=None) -> None:
        """Write ``direction_index`` (0..5) and ``direction_class`` (0..2) onto
        every edge. Must run after the lattice orientation is known and after
        node coordinates change (e.g. ``build_ideal_lattice``).
        """
        graph = self.token_graph if graph is None else graph
        for u, v in graph.edges():
            graph.edges[u, v]["direction_index"] = self._edge_direction_index(u, v)
            graph.edges[u, v]["direction_class"] = self._edge_line_class(u, v)

    def get_lines(self, direction, graph=None, include_singletons=False):
        """Return the straight lines of nodes along ``direction``.

        ``direction`` is a line class (0/1/2, relative to ``primary_axis``) or
        an absolute angle in degrees. Edges whose class differs are dropped;
        each remaining connected component is a maximal straight line, returned
        as a list of nodes ordered end-to-end. Works on any graph or subgraph
        whose edges carry a ``direction_class`` attribute (e.g. a
        ``token_graph.subgraph(cluster)`` view), falling back to computing the
        class from node coordinates when the attribute is absent.
        """
        graph = self.token_graph if graph is None else graph
        cls = self.resolve_line_direction(direction)

        line_graph = nx.Graph()
        line_graph.add_nodes_from(graph.nodes)
        for u, v, data in graph.edges(data=True):
            edge_class = data.get("direction_class")
            if edge_class is None:
                edge_class = self._edge_line_class(u, v)
            else:
                edge_class = EdegeDirectionClass(edge_class)
            if edge_class == cls:
                line_graph.add_edge(u, v)

        lines = []
        for comp in nx.connected_components(line_graph):
            if len(comp) < 2:
                if include_singletons:
                    lines.append([next(iter(comp))])
                continue
            sub = line_graph.subgraph(comp)
            endpoints = [n for n in sub.nodes if sub.degree[n] <= 1]
            start = endpoints[0] if endpoints else next(iter(sub.nodes))
            ordered, prev, cur = [], None, start
            while cur is not None:
                ordered.append(cur)
                nxt = [n for n in sub.neighbors(cur) if n != prev]
                prev, cur = cur, (nxt[0] if nxt else None)
            lines.append(ordered)
        return lines

    def nodes_on_single_line(self, nodes, graph=None) -> bool:
        """Return True if ``nodes`` all lie on one straight lattice line.

        A set of nodes lies on one line if, for some edge class
        (:class:`EdegeDirectionClass`), all nodes fall in a single connected
        component of the subgraph induced by ``nodes`` using only edges of that
        class. Edge classes are read from the ``direction_class`` attribute
        (set by ``assign_edge_classes``); ``get_lines`` falls back to computing
        the class from node coordinates when the attribute is absent. An empty
        or single-node set is trivially on a line.
        """
        graph = self.token_graph if graph is None else graph
        nodes = list(nodes)
        if len(nodes) <= 2:
            return True
        sub = graph.subgraph(nodes)
        return any(
            len(self.get_lines(cls, sub, include_singletons=True)) == 1
            for cls in EdegeDirectionClass
        )

    def get_possible_lines(self, graph=None, max_length=4):
        """Return every possible straight line of 1..``max_length`` nodes.

        ``graph`` may be an ``nx.Graph`` or subgraph view (e.g.
        ``self.token_graph.subgraph(cluster)``) or an iterable of nodes (in
        which case the induced subgraph on those nodes is used); it defaults
        to the full ``token_graph``. Lines never use nodes or edges outside
        the given graph. The nodes are assumed to form a connected subgraph.

        For each of the 3 line classes the maximal straight lines are found
        with ``get_lines``, then every contiguous sub-segment of length
        1..``max_length`` is emitted (a 5-node row yields 14 lines for
        ``max_length=4``). Duplicate sequences (a node shared by several line
        classes reappears as a length-1 segment) are collapsed, and the lines
        are returned longest-first, ties keeping class order (0/1/2).

        Returns:
            list[list]: straight lines, each an ordered list of nodes.
        """
        graph = self.token_graph if graph is None else graph
        if not isinstance(graph, nx.Graph):
            graph = self.token_graph.subgraph(graph)

        lines = []
        for cls in EdegeDirectionClass:
            for maximal in self.get_lines(cls, graph):
                m = len(maximal)
                if m < 1:
                    continue
                for length in range(1, min(m, max_length) + 1):
                    for start in range(0, m - length + 1):
                        lines.append(maximal[start:start + length])

        seen = set()
        unique = []
        for line in sorted(lines, key=len, reverse=True):
            key = tuple(line)
            if key not in seen:
                seen.add(key)
                unique.append(line)
        return unique

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
        self.assign_edge_classes()

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
        self.assign_edge_classes()

