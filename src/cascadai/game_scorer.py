import numpy as np
import networkx as nx
import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from enum import Enum

from cascadai.utils import token_detector
from cascadai.schema.token_piece import Token_Type, Token
from cascadai.schema.environment_graph import EnvironmentGraph 
from cascadai.utils import environment_graph_utils

MODEL_PATH = "src/cascadai/models/token_detector_yolo11n/TokenNet.pt"

class Score_Card(Enum):
    A = 0
    B = 1
    C = 2
    D = 3


def detect_tokens(image_path):
    model = YOLO(MODEL_PATH)
    detections = token_detector.detect_tokens(model, image_path)
    tokens = []
    for detection in detections:
        x1, y1, x2, y2, token_type, _ = detection
        token_type = Token_Type(token_type)
        x_center = x1 + (x2 - x1) / 2
        y_center = y1 + (y2 - y1) / 2
        width = ((x2 - x1) + (y2 - y1)) / 2 
        t = Token(token_type, x_center, y_center, width)
        tokens.append(t)
    return tokens


def find_all_tokens_of_type(EG: EnvironmentGraph, toke_type: Token_Type):
    """
    Find all nodes with a specific type.

    Returns a list of nodes
    """
    nodes_of_type = [n for n, attrs in EG.EG.nodes(data=True) if attrs.get("type") == toke_type]
    return nodes_of_type


def find_clusters(EG: EnvironmentGraph, toke_type: Token_Type):
    """
    Find all clusters (connected components) of nodes with a specific type.

    Returns a list of clusters, where each cluster is a set of node ids,
    sorted by size (largest first).
    """
    nodes_of_type = find_all_tokens_of_type(EG, toke_type)

    # Build the induced subgraph on just those nodes
    # only edges between same-type nodes survive
    subgraph = EG.EG.subgraph(nodes_of_type)

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


def score_bears_A(EG: EnvironmentGraph) -> int:
    """
    Mating pair scoring. Score for number of pairs of Bears with no other bears next to them.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(EG, Token_Type.BEAR)

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


def score_bears_B(EG: EnvironmentGraph) -> int:
    """
    Mother and Cubs scoring. Score per group of 3 Bears with no other bears next to it.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(EG, Token_Type.BEAR)

    if len(clusters) == 0:
        return 0

    n_mom_and_cubs = 0

    for c in clusters:
        if len(c) == 3:
            n_mom_and_cubs += 1

    return n_mom_and_cubs * 10


def score_bears_C(EG: EnvironmentGraph) -> int:
    """
    Family scoring. Score per group of with no other bears next to it.

    Args:
        EG (EnvironmentGraph): _description_

    Returns:
        int: score
    """
    clusters = find_clusters(EG, Token_Type.BEAR)

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


def score_bears_D(EG: EnvironmentGraph) -> int:
    """
    Big Group scoring. Score per group of with no other bears next to it.

    Args:
        EG (EnvironmentGraph):

    Returns:
        int: score
    """
    clusters = find_clusters(EG, Token_Type.BEAR)

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


def score_elk_A(EG: EnvironmentGraph) -> int:
    clusters = find_clusters(EG, Token_Type.ELK)

    if len(clusters) == 0:
        return 0

    score = 0 

    for c in clusters:
        group_size = len(c)
        match group_size:
            case 1:
                score += 2
            case 2:
                score += 5
            case 3:
                straight = False
                if straight:
                    score += 9
                else:
                    score += 9
            case 4:
                score += 13
    return score


def score_elk_B(EG: EnvironmentGraph) -> int:
    return 0


def score_elk_C(EG: EnvironmentGraph) -> int:
    return 0


def score_elk_D(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_A(EG: EnvironmentGraph) -> int:
    clusters = find_clusters(EG, Token_Type.SALMON)
    score = 0
    return score


def score_salmon_B(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_C(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_D(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_A(EG: EnvironmentGraph) -> int:
    clusters = find_clusters(EG, Token_Type.HAWK)

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


def score_hawks_B(EG: EnvironmentGraph) -> int:
    clusters = find_clusters(EG, Token_Type.HAWK)

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


def score_hawks_C(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_D(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_A(EG: EnvironmentGraph) -> int:
    fox_nodes = find_all_tokens_of_type(EG, Token_Type.FOX)

    if len(fox_nodes) == 0:
        return 0

    score = 0

    for fox in fox_nodes:
        neighbours = EG.EG.neighbors(fox)
        n_types = np.array([0]*5)
        for neighbour in neighbours:
            n_types[neighbour.type.value] = 1
        score += np.sum(n_types)
        
    return score


def score_foxes_B(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_C(EG: EnvironmentGraph) -> int:
    return 0

    score = 0

    for fox in fox_nodes:
        neighbours = EG.EG.neighbors(fox)
        n_types = np.array([0]*4)
        for neighbour in neighbours:
            if neighbour.type == Token_Type.FOX:
                continue
            n_types[neighbour.type.value] += 1
        score += n_types.max()

    return score


def score_foxes_D(EG: EnvironmentGraph) -> int:
    return 0


def score_token_type(EG: EnvironmentGraph, SC: Score_Card, token_type: Token_Type) -> int:
    match token_type:
        case Token_Type.BEAR:
            match SC:
                case Score_Card.A:
                    return score_bears_A(EG)
                case Score_Card.B:
                    return score_bears_B(EG)
                case Score_Card.C:
                    return score_bears_C(EG)
                case Score_Card.D:
                    return score_bears_D(EG)
        case Token_Type.ELK:
            match SC:
                case Score_Card.A:
                    return score_elk_A(EG)
                case Score_Card.B:
                    return score_elk_B(EG)
                case Score_Card.C:
                    return score_elk_C(EG)
                case Score_Card.D:
                    return score_elk_D(EG)               
        case Token_Type.SALMON:
            match SC:
                case Score_Card.A:
                    return score_salmon_A(EG)
                case Score_Card.B:
                    return score_salmon_B(EG)
                case Score_Card.C:
                    return score_salmon_C(EG)
                case Score_Card.D:
                    return score_salmon_D(EG)
        case Token_Type.HAWK:
            match SC:
                case Score_Card.A:
                    return score_hawks_A(EG)
                case Score_Card.B:
                    return score_hawks_B(EG)
                case Score_Card.C:
                    return score_hawks_C(EG)
                case Score_Card.D:
                    return score_hawks_D(EG)
        case Token_Type.FOX:
            match SC:
                case Score_Card.A:
                    return score_foxes_A(EG)
                case Score_Card.B:
                    return score_foxes_B(EG)
                case Score_Card.C:
                    return score_foxes_C(EG)
                case Score_Card.D:
                    return score_foxes_D(EG)
        case _:
            raise Exception("No Toke Type was given")


def main():
    parser = argparse.ArgumentParser(
        description="Detect tokens in an image with a YOLO model"
    )
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    parser.add_argument("-o", "--output", default="output", help="output directory")
    parser.add_argument("-br", "--bear", default="A", help="Bear scoring card")
    parser.add_argument("-dr", "--elk", default="A", help="Elk scoring card")
    parser.add_argument("-sm", "--salmon", default="A", help="Salmon scoring card")
    parser.add_argument("-hk", "--hawk", default="A", help="Hawk scoring card")
    parser.add_argument("-fx", "--fox", default="A", help="Fox scoring card")
    args = parser.parse_args()

    tokens = detect_tokens(args.image)
    EG = EnvironmentGraph(tokens)
    environment_graph_utils.plot_environment_graph_tokens(EG)
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
