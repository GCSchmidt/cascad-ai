import numpy as np
import networkx as nx
import argparse
import os
import cv2
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
from ultralytics import YOLO
from enum import Enum
from itertools import combinations

from cascadai.utils import token_detector
from cascadai.schema.token_piece import Token_Type, Token
from cascadai.schema.environment_graph import EnvironmentGraph, EdegeDirectionClass
from cascadai.utils import environment_graph_utils

MODEL_PATH = "src/cascadai/models/token_detector_yolo11n/TokenNet.pt"


class Score_Card(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


def find_all_tokens_of_type(graph: nx.Graph, toke_type: Token_Type):
    """
    Find all nodes with a specific type.

    Returns a list of nodes
    """
    nodes_of_type = [n for n, attrs in graph.nodes(data=True) if attrs.get("type") == toke_type]
    return nodes_of_type


def find_clusters(graph: nx.Graph, toke_type: Token_Type):
    """
    Find all clusters (connected components) of nodes with a specific type.

    Returns a list of clusters, where each cluster is a set of node ids,
    sorted by size (largest first).
    """
    nodes_of_type = find_all_tokens_of_type(graph, toke_type)

    # Build the induced subgraph on just those nodes
    # only edges between same-type nodes survive
    subgraph = graph.subgraph(nodes_of_type)

    # Connected components of that subgraph = clusters
    clusters = list(nx.connected_components(subgraph))

    # Sort by size, descending
    clusters.sort(key=len, reverse=True)

    return clusters


def prune_to_max_degree_2(subgraph: nx.Graph):
    """
    Greedily remove one high-degree node at a time until no node touches
    more than 2 other nodes.

    Args:
        subgraph (nx.Graph): the cluster subgraph to prune.

    Returns:
        tuple: (list of remaining node ids, count of remaining nodes)
    """
    g = subgraph.copy()

    while True:
        high = [n for n in g.nodes if g.degree[n] > 2]
        if not high:
            break
        g.remove_node(high[0])

    remaining_nodes = list(g.nodes)
    return remaining_nodes, len(remaining_nodes)


def optimize_elk_groups(groups, score_by_size):
    """Select the highest-scoring set of non-overlapping elk groups.

    Args:
        groups: list of groups, each a list of nodes.
        score_by_length: dict mapping a group's size to its score.

    Returns:
        tuple: (selected_groups, optimal_score) where no node is used by more
        than one selected group and ``optimal_score`` is the maximum total score.
    """
    if not groups:
        return [], 0

    # Objective: maximize sum of scores (milp minimizes, so negate).
    c = np.array([-score_by_size[len(group)] for group in groups], dtype=float)

    # Each node may be used by at most one selected group.
    all_nodes = list(set(node for group in groups for node in group))
    node_rows = {
        node: [i for i, group in enumerate(groups) if node in group]
        for node in all_nodes
    }
    A = np.zeros((len(all_nodes), len(groups)))
    for row, node in enumerate(all_nodes):
        for i in node_rows[node]:
            A[row, i] = 1.0

    constraints = (
        LinearConstraint(A, -np.inf, 1.0) if all_nodes else None
    )

    result = milp(
        c,
        integrality=np.ones(len(groups)),
        bounds=Bounds(0, 1),
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    selected_groups = [
        groups[i] for i in range(len(groups)) if result.x[i] > 0.5
    ]
    return selected_groups, int(round(-result.fun))


def score_bears_A(graph: nx.Graph) -> int:
    """
    Mating pair scoring. Score for number of pairs of Bears with no other bears next to them.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.BEAR)

    if len(clusters) == 0:
        return 0

    n_pairs = 0

    for c in clusters:
        if len(c) == 2:
            n_pairs += 1

    match n_pairs:
        case 0:
            return 0
        case 1:
            return 4
        case 2:
            return 11
        case 3:
            return 19
        case _:
            # >= 4 
            return 27

    return 0


def score_bears_B(graph: nx.Graph) -> int:
    """
    Mother and Cubs scoring. Score per group of 3 Bears with no other bears next to it.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.BEAR)

    if len(clusters) == 0:
        return 0

    n_mom_and_cubs = 0

    for c in clusters:
        if len(c) == 3:
            n_mom_and_cubs += 1

    return n_mom_and_cubs * 10


def score_bears_C(graph: nx.Graph) -> int:
    """
    Family scoring. Score per group of with no other bears next to it.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.BEAR)

    if len(clusters) == 0:
        return 0

    has_single = False
    has_pair = False
    has_family = False

    score = 0

    for c in clusters:
        fam_size = len(c)
        match fam_size:
            case 1:
                score += 2
                has_single = True
            case 2:
                score += 5
                has_pair = True
            case 3:
                score += 8
                has_family = True

    if has_single and has_pair and has_family:
        score += 3

    return score


def score_bears_D(graph: nx.Graph) -> int:
    """
    Big Group scoring. Score per group of with no other bears next to it.

    Args:
        EG (EnvironmentGraph):

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.BEAR)

    if len(clusters) == 0:
        return 0

    score = 0 

    for c in clusters:
        group_size = len(c)
        match group_size:
            case 2:
                score += 5
            case 3:
                score += 8
            case 4:
                score += 13

    return score


def score_elk_A(env_graph: EnvironmentGraph) -> int:
    """
    Score for groups of elk in a staright line.
    Groups may be touching. 

    Args:
        EG (EnvironmentGraph): an EnvironmentGraph

    Returns:
        int: score for elks
    """
    clusters = find_clusters(env_graph.token_graph, Token_Type.ELK)

    if len(clusters) == 0:
        return 0

    score_by_length = {
        1: 2,
        2: 5,
        3: 9,
        4: 13,
    }

    def score_elk_in_line(n_elk: int) -> int:
        """
        Helper function for elk score card A
        Args:
            n_elk (int): number of elks in a straight line
        Returns:
            int: score for elks in straight line
        """
        line_score = (n_elk // 4) * score_by_length[4]
        line_score += score_by_length[n_elk % 4]
        return line_score

    score = 0

    for c in clusters:

        group_size = len(c)

        straight_line = env_graph.nodes_on_single_line(c)

        if straight_line:
            score += score_elk_in_line(group_size)
        else:
            if group_size == 3:
                score += (2 + 5)  # 1 elk + 2 elk scores 
            else:
                lines = env_graph.get_possible_lines(c)
                _, optimal_score = optimize_elk_groups(lines, score_by_length)
                score += optimal_score

    return score


def is_triangle(graph: nx.Graph, nodes) -> bool:
    """Return True if the nodes form a 3-elk triangle."""
    return len(nodes) == 3 and graph.subgraph(nodes).number_of_edges() == 3


def is_diamond(graph: nx.Graph, nodes) -> bool:
    """Return True if the nodes form a 4-elk diamond."""
    if len(nodes) != 4:
        return False

    subgraph = graph.subgraph(nodes)
    if subgraph.number_of_edges() != 5:
        return False

    degrees = sorted(dict(subgraph.degree()).values())
    return degrees == [2, 2, 3, 3]


def score_elk_B(token_graph: nx.Graph) -> int:
    """    Score for groups of elk in specific shapes.
    Groups may be touching.

    Args:
        EG (EnvironmentGraph): an EnvironmentGraph

    Returns:
        int: score for elks
    """
    clusters = find_clusters(token_graph, Token_Type.ELK)

    if len(clusters) == 0:
        return 0

    score_by_size = {
        1: 2,
        2: 5,
        3: 9,
        4: 13,
    }

    score = 0

    for cluster in clusters:
        nodes = list(cluster)

        # Single elk and adjacent pairs are always valid groups.
        candidate_groups = [[node] for node in nodes]
        candidate_groups.extend(
            list(pair)
            for pair in combinations(nodes, 2)
            if token_graph.has_edge(*pair)
        )

        candidate_groups.extend(
            list(triangle)
            for triangle in combinations(nodes, 3)
            if is_triangle(token_graph, triangle)
        )

        candidate_groups.extend(
            list(diamond)
            for diamond in combinations(nodes, 4)
            if is_diamond(token_graph, diamond)
        )

        _, optimal_score = optimize_elk_groups(candidate_groups, score_by_size)
        score += optimal_score

    return score


def score_elk_C(graph: nx.Graph) -> int:
    clusters = find_clusters(graph, Token_Type.ELK)

    if len(clusters) == 0:
        return 0

    score = 0

    for c in clusters:
        herd_size = len(c)
        match herd_size:
            case 1 | 9 | 17:
                score += 2
            case 2 | 10 | 18:
                score += 4
            case 3 | 11 | 19:
                score += 7
            case 4 | 12 | 20:
                score += 10
            case 5 | 13:
                score += 14
            case 6 | 14:
                score += 18
            case 7 | 15:
                score += 23
            case 8 | 16:
                score += 28

    return score


def score_elk_D(env_graph: EnvironmentGraph) -> int:
    """    Score for groups of elk in a circular shape.
    Groups may be touching.

    Args:
        EG (EnvironmentGraph): an EnvironmentGraph

    Returns:
        int: score for elks
    """
    graph = env_graph.token_graph
    elk_tokens = find_all_tokens_of_type(graph, Token_Type.ELK)

    if len(elk_tokens) == 0:
        return 0

    # score based on number of elk touching each other around a single node
    score_by_size = {
        1: 2,
        2: 5,
        3: 8,
        4: 12,
        5: 16,
        6: 21
    }

    elk_set = set(elk_tokens)

    # A circular group is centred on a node with elk surrounding it. The centre
    # may itself be an elk, another token, or an empty tile.
    candidate_centers = set(elk_tokens)
    for elk in elk_tokens:
        candidate_centers.update(graph.neighbors(elk))

    candidate_groups = []
    seen_groups = set()

    for center in candidate_centers:
        elk_neighbours = [n for n in graph.neighbors(center) if n in elk_set]

        for size in range(1, min(len(elk_neighbours), 6) + 1):
            for subset in combinations(elk_neighbours, size):
                if not nx.is_connected(graph.subgraph(subset)):
                    continue

                group_key = frozenset(subset)
                if group_key in seen_groups:
                    continue

                seen_groups.add(group_key)
                candidate_groups.append(list(subset))

    if len(candidate_groups) == 0:
        return 0

    _, optimal_score = optimize_elk_groups(candidate_groups, score_by_size)
    return optimal_score


def score_salmon_A(graph: nx.Graph) -> int:
    """
    Score for each salmon in a salmon run. 
    A solmon that touches more than 2 other may not be included in the salmon run.
    Max salmon run length is 7.
    Args:
        EG (EnvironmentGraph): 

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.SALMON)

    if len(clusters) == 0:
        return 0

    score = 0

    for c in clusters: 
        cluster_graph = graph.subgraph(c)

        # there is a salmon adjacent to the run
        if max(dict(cluster_graph.degree()).values()) > 2:
            continue

        run_length = len(cluster_graph)

        match run_length:
            case 1:
                score += 2
            case 2:
                score += 5
            case 3:
                score += 8
            case 4:
                score += 12
            case 5:
                score += 16
            case 6:
                score += 20
            case _:
                score += 25

    return score


def score_salmon_B(graph: nx.Graph) -> int:
    """
    Score for each salmon in a salmon run. 
    A solmon that touches more than 2 other may not be included in the salmon run.
    Max salmon run length is 5.
    Args:
        EG (EnvironmentGraph): 

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.SALMON)

    if len(clusters) == 0:
        return 0

    score = 0

    for c in clusters: 
        cluster_graph = graph.subgraph(c)

        # there is a salmon adjacent to the run
        if max(dict(cluster_graph.degree()).values()) > 2:
            continue

        run_length = len(cluster_graph)

        match run_length:
            case 1:
                score += 2
            case 2:
                score += 4
            case 3:
                score += 9
            case 4:
                score += 11
            case _:
                score += 17

    return score


def score_salmon_C(graph: nx.Graph) -> int:
    """
    Score for each salmon in a salmon run. 
    A solmon that touches more than 2 other may not be included in the salmon run.
    Minimum salmon run length is 3.
    Max salmon run length is 5.
    Args:
        EG (EnvironmentGraph): 

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.SALMON)

    if len(clusters) == 0:
        return 0

    score = 0

    for c in clusters: 
        cluster_graph = graph.subgraph(c)
        max(dict(cluster_graph.degree()).values())
        _, run_length = prune_to_max_degree_2(cluster_graph)

        if run_length < 3:
            continue

        match run_length:
            case 3:
                score += 10
            case 4:
                score += 12
            case _:
                score += 15

    return score


def score_salmon_D(graph: nx.Graph) -> int:
    """
    Score for each salmon in a salmon run and adjacent animal tokens. 
    A solmon that touches more than 2 other may not be included in the salmon run.
    Args:
        EG (EnvironmentGraph): 

    Returns:
        int: score
    """
    clusters = find_clusters(graph, Token_Type.SALMON)

    if len(clusters) == 0:
        return 0

    score = 0

    for c in clusters: 
        cluster_graph = graph.subgraph(c)
        # there is a salmon adjacent to the run
        if max(dict(cluster_graph.degree()).values()) > 2:
            continue

        run_length = len(cluster_graph)

        score += run_length
        adjacent_tokens = set()
        for salmon in cluster_graph:
            neighbours = set(graph.neighbors(salmon))
            adjacent_tokens = (adjacent_tokens | neighbours)

        adjacent_tokens = adjacent_tokens - set(cluster_graph)
        adjacent_tokens = {
            t for t in adjacent_tokens if t.type != Token_Type.BLANK
        }
        score += len(adjacent_tokens)

    return score


def score_hawks_A(graph: nx.Graph) -> int:
    """Scores for each single hawk, which is not touching another.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: _description_
    """

    clusters = find_clusters(graph, Token_Type.HAWK)

    if len(clusters) == 0:
        return 0

    score = 0
    singles = 0

    for c in clusters:
        group_size = len(c)
        if group_size > 1:
            continue

        singles += 1

        if singles < 2:
            score += 2
        elif singles < 6:
            score += 3
        elif singles < 9:
            score += 4

    return score


def score_hawks_B(env_graph: EnvironmentGraph) -> int:
    """
    Score points from each hawk that has at least one other hawk in its line of sight. 
    Hawks may not touch other hawks. 
    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: _description_
    """

    clusters = find_clusters(env_graph.token_graph, Token_Type.HAWK)

    if len(clusters) == 0:
        return 0

    score = 0

    valid_hawks = set()  # hawks not touching others
    for c in clusters:
        group_size = len(c)

        if group_size == 1:
            valid_hawk = next(iter(c))
            valid_hawks.add(valid_hawk)

    if len(valid_hawks) <= 1:
        return 0

    scored_hawks = set()

    for hawk1 in valid_hawks:
        if hawk1 in scored_hawks:
            # Hawk already part of line of sight
            continue

        # check all other valid hawks
        for hawk2 in valid_hawks.difference([hawk1]):

            if not env_graph.token_in_line_of_sight(hawk1, hawk2):
                # angle does not agree with line of sight/latice_angles
                continue

            try:
                shortest_path = nx.shortest_path(env_graph.token_graph, hawk1, hawk2)[1:-1]  # nodes between source and end
            except nx.NetworkXNoPath:
                continue
            
            valid_path = True

            for n in shortest_path:
                if n.type == Token_Type.HAWK:
                    valid_path = False
                    break

            if not valid_path:
                break

            scored_hawks.add(hawk1)
            scored_hawks.add(hawk2)
            break

    n_scoring_hawks = len(scored_hawks)

    match n_scoring_hawks:
        case 0:
            return 0
        case 2:
            score += 5
        case 3:
            score += 9
        case 4:
            score += 12
        case 5:
            score += 16
        case 6:
            score += 20
        case 7:
            score += 24
        case 8:
            score += 29
        case _:
            score += 29

    return score


def score_hawks_C(env_graph: EnvironmentGraph) -> int:
    """Score 3 poitns for each line of sight between hawks.
    Hawks may not touch other hawks.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score for hawk tokens
    """
    clusters = find_clusters(env_graph.token_graph, Token_Type.HAWK)

    if len(clusters) == 0:
        return 0

    score = 0

    valid_hawks = set()  # hawks not touching others
    for c in clusters:
        group_size = len(c)

        if group_size == 1:
            valid_hawk = next(iter(c))
            valid_hawks.add(valid_hawk)

    if len(valid_hawks) <= 1:
        return 0

    possible_hawk_pairs = set(combinations(valid_hawks, 2))
    n_line_of_sights = 0

    for hawk_pair in possible_hawk_pairs:
        hawk1, hawk2 = hawk_pair
        if not env_graph.token_in_line_of_sight(hawk1, hawk2):
            # angle does not agree with line of sight/latice_angles
            continue
        try:
            shortest_path = nx.shortest_path(env_graph.token_graph, hawk1, hawk2)[1:-1]  # nodes between source and end
        except nx.NetworkXNoPath:
            continue
            
        valid_path = True

        for n in shortest_path:
            if n.type == Token_Type.HAWK:
                valid_path = False
                break

        if not valid_path:
            continue

        n_line_of_sights += 1

    score = n_line_of_sights * 3

    return score


def score_hawks_D(env_graph: EnvironmentGraph) -> int:
    """Score for each pair of hawks, an increasing number of points based on
    the number of unique animal types between them (not including other hawks).
    Each hawk can only be part of one pair.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """

    def score_hawk_pair(sub_graph: nx.Graph) -> int:
        if any(node.type == Token_Type.HAWK for node in sub_graph.nodes):
            return -1

        n_unique = len({
            node.type
            for node in sub_graph.nodes
            if node.type not in (Token_Type.HAWK, Token_Type.BLANK)
        })

        match n_unique:
            case 0:
                return 0
            case 1:
                return 4
            case 2:
                return 7
            case _:
                return 9

    hawk_tokens = find_all_tokens_of_type(env_graph.token_graph, Token_Type.HAWK)

    if len(hawk_tokens) < 2:
        return 0

    pair_scores = {}

    for hawk1, hawk2 in combinations(hawk_tokens, 2):
        if not env_graph.token_in_line_of_sight(hawk1, hawk2):
            continue

        try:
            shortest_path = nx.shortest_path(env_graph.token_graph, hawk1, hawk2)
        except nx.NetworkXNoPath:
            continue

        between_hawks = shortest_path[1:-1]

        if not between_hawks:
            continue

        if not env_graph.nodes_on_single_line(shortest_path):
            continue

        pair_score = score_hawk_pair(env_graph.token_graph.subgraph(between_hawks))

        if pair_score > 0:
            pair_scores[(hawk1, hawk2)] = pair_score

    if not pair_scores:
        return 0

    pairs = list(pair_scores)
    scores = np.array([pair_scores[pair] for pair in pairs], dtype=float)
    c = -scores

    hawks = list({hawk for pair in pairs for hawk in pair})
    A = np.zeros((len(hawks), len(pairs)))
    for row, hawk in enumerate(hawks):
        for col, pair in enumerate(pairs):
            if hawk in pair:
                A[row, col] = 1.0

    result = milp(
        c,
        integrality=np.ones(len(pairs)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(A, -np.inf, 1.0),
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return int(round(-result.fun))


def score_foxes_A(graph: nx.Graph) -> int:
    """
    Score for each unique animal type around a fox.
    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    fox_nodes = find_all_tokens_of_type(graph, Token_Type.FOX)

    if len(fox_nodes) == 0:
        return 0

    score = 0

    for fox in fox_nodes:
        neighbours = graph.neighbors(fox)
        n_types = np.array([0]*5)
        for neighbour in neighbours:
            if neighbour.type == Token_Type.BLANK:
                continue
            n_types[neighbour.type.value] = 1
        score += np.sum(n_types)
  
    return score


def score_foxes_B(graph: nx.Graph) -> int:
    """
    Score for each unique animal type pair around a fox.
    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    fox_nodes = find_all_tokens_of_type(graph, Token_Type.FOX)

    if len(fox_nodes) == 0:
        return 0

    score = 0

    for fox in fox_nodes:
        neighbours = graph.neighbors(fox)
        n_types = np.array([0]*4)
        for neighbour in neighbours:
            if neighbour.type == Token_Type.BLANK:
                continue
            if neighbour.type == Token_Type.FOX:
                continue
            n_types[neighbour.type.value] += 1

        n_pairs = (n_types >= 2).sum()

        match n_pairs:
            case 1:
                score += 3
            case 2:
                score += 5
            case 3:
                score += 7
              
    return score


def score_foxes_C(graph: nx.Graph) -> int:
    """
    Score for the most frequent animal type around a fox.
    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    fox_nodes = find_all_tokens_of_type(graph, Token_Type.FOX)

    if len(fox_nodes) == 0:
        return 0

    score = 0

    for fox in fox_nodes:
        neighbours = graph.neighbors(fox)
        n_types = np.array([0]*4)
        for neighbour in neighbours:
            if neighbour.type == Token_Type.BLANK:
                continue
            if neighbour.type == Token_Type.FOX:
                continue
            n_types[neighbour.type.value] += 1
        score += n_types.max()

    return score


def score_foxes_D(graph: nx.Graph) -> int:
    """
    Score for each unique animal type pair around a pair of foxes.
    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    score = 0

    def score_fox_pair(fox1, fox2) -> int:
        if not graph.has_edge(fox1, fox2):
            return 0

        pair_score = 0
        neighbours1 = set(graph.neighbors(fox1))
        neighbours2 = set(graph.neighbors(fox2))
        neighbours = (neighbours1 | neighbours2) - {fox1, fox2}
        n_types = np.array([0]*4)
        for neighbour in neighbours:
            if neighbour.type == Token_Type.BLANK:
                continue
            if neighbour.type == Token_Type.FOX:
                continue
            n_types[neighbour.type.value] += 1
        n_pairs = (n_types >= 2).sum()
        if n_pairs:
            pair_score += n_pairs * 2 + 3

        return pair_score

    def find_optimal_score(pair_scores: dict) -> int:
        score_graph = nx.Graph()

        for (a, b), score in pair_scores.items():
            score_graph.add_edge(a, b, weight=score)

        matching = nx.max_weight_matching(score_graph, maxcardinality=False)
        max_score = sum(score_graph[u][v]["weight"] for u, v in matching)

        return max_score

    clusters = find_clusters(graph, Token_Type.FOX)

    if len(clusters) == 0:
        return 0

    for cluster in clusters:
        if len(cluster) < 2:
            # single fox, no points
            continue
        elif len(cluster) > 2:
            possible_fox_pairs = set(combinations(range(len(cluster)), 2))
            score_per_pair = dict()
            for fox_pair in possible_fox_pairs:
                cluster = list(cluster)
                id1, id2 = fox_pair[0], fox_pair[1]
                fox1, fox2 = cluster[id1], cluster[id2]
                if graph.has_edge(fox1, fox2):
                    score_per_pair[fox_pair] = score_fox_pair(fox1, fox2)

            score += find_optimal_score(score_per_pair)
        else:
            fox1, fox2 = cluster
            score += score_fox_pair(fox1, fox2)

    return score


def score_token_type(env_graph: EnvironmentGraph, SC: Score_Card, token_type: Token_Type) -> int:
    match token_type:
        case Token_Type.BEAR:
            match SC:
                case Score_Card.A:
                    return score_bears_A(env_graph.token_graph)
                case Score_Card.B:
                    return score_bears_B(env_graph.token_graph)
                case Score_Card.C:
                    return score_bears_C(env_graph.token_graph)
                case Score_Card.D:
                    return score_bears_D(env_graph.token_graph)
        case Token_Type.ELK:
            match SC:
                case Score_Card.A:
                    return score_elk_A(env_graph)
                case Score_Card.B:
                    return score_elk_B(env_graph.token_graph)
                case Score_Card.C:
                    return score_elk_C(env_graph.token_graph)
                case Score_Card.D:
                    return score_elk_D(env_graph)
        case Token_Type.SALMON:
            match SC:
                case Score_Card.A:
                    return score_salmon_A(env_graph.token_graph)
                case Score_Card.B:
                    return score_salmon_B(env_graph.token_graph)
                case Score_Card.C:
                    return score_salmon_C(env_graph.token_graph)
                case Score_Card.D:
                    return score_salmon_D(env_graph.token_graph)
        case Token_Type.HAWK:
            match SC:
                case Score_Card.A:
                    return score_hawks_A(env_graph.token_graph)
                case Score_Card.B:
                    return score_hawks_B(env_graph)
                case Score_Card.C:
                    return score_hawks_C(env_graph)
                case Score_Card.D:
                    return score_hawks_D(env_graph)
        case Token_Type.FOX:
            match SC:
                case Score_Card.A:
                    return score_foxes_A(env_graph.token_graph)
                case Score_Card.B:
                    return score_foxes_B(env_graph.token_graph)
                case Score_Card.C:
                    return score_foxes_C(env_graph.token_graph)
                case Score_Card.D:
                    return score_foxes_D(env_graph.token_graph)
        case Token_Type.BLANK:
            return 0
        case _:
            raise Exception("No Toke Type was given")


def main():
    parser = argparse.ArgumentParser(
        description="Detect tokens in an image with a YOLO model"
    )
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    parser.add_argument("-o", "--output", default="output", help="output directory")
    parser.add_argument("-br", "--bear", default="A", help="Bear scoring card")
    parser.add_argument("-ek", "--elk", default="A", help="Elk scoring card")
    parser.add_argument("-sm", "--salmon", default="A", help="Salmon scoring card")
    parser.add_argument("-hk", "--hawk", default="A", help="Hawk scoring card")
    parser.add_argument("-fx", "--fox", default="A", help="Fox scoring card")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)
    tokens = token_detector.detect_tokens(model, args.image)
    EG = EnvironmentGraph(tokens)
    EG.build_ideal_lattice()
    environment_graph_utils.plot_environment_graph_tokens(EG.token_graph)

    score_bear = score_token_type(EG, Score_Card[args.bear], Token_Type.BEAR)
    score_elk = score_token_type(EG, Score_Card[args.elk], Token_Type.ELK)
    score_salmon = score_token_type(EG, Score_Card[args.salmon], Token_Type.SALMON)
    score_hawk = score_token_type(EG, Score_Card[args.hawk], Token_Type.HAWK)
    score_fox = score_token_type(EG, Score_Card[args.fox], Token_Type.FOX)
    total_score = score_bear + score_elk + score_salmon + score_hawk + score_fox
    print(f"Score for Bears [{args.bear}]: {score_bear}")
    print(f"Score for Elk [{args.elk}]: {score_elk}")
    print(f"Score for Salmon [{args.salmon}]: {score_salmon}")
    print(f"Score for Hawks [{args.hawk}]: {score_hawk}")
    print(f"Score for Foxes [{args.fox}]: {score_fox}")
    print(f"Total Score for {total_score}")


if __name__ == "__main__":
    main()
