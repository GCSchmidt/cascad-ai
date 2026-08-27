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


def find_clusters(EG: EnvironmentGraph, toke_type: Token_Type):
    """
    Find all clusters (connected components) of nodes with a specific type.

    Returns a list of clusters, where each cluster is a set of node ids,
    sorted by size (largest first).
    """
    nodes_of_type = [n for n, attrs in EG.EG.nodes(data=True) if attrs.get("type") == toke_type]

    # Build the induced subgraph on just those nodes
    # only edges between same-type nodes survive
    subgraph = EG.EG.subgraph(nodes_of_type)

    # Connected components of that subgraph = clusters
    clusters = list(nx.connected_components(subgraph))

    # Sort by size, descending
    clusters.sort(key=len, reverse=True)

    return clusters


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


def score_deer_A(EG: EnvironmentGraph) -> int:
    return 0


def score_deer_B(EG: EnvironmentGraph) -> int:
    return 0


def score_deer_C(EG: EnvironmentGraph) -> int:
    return 0


def score_deer_D(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_A(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_B(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_C(EG: EnvironmentGraph) -> int:
    return 0


def score_salmon_D(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_A(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_B(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_C(EG: EnvironmentGraph) -> int:
    return 0


def score_hawks_D(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_A(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_B(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_C(EG: EnvironmentGraph) -> int:
    return 0


def score_foxes_D(EG: EnvironmentGraph) -> int:
    return 0


def score_bears(EG: EnvironmentGraph, SC: Score_Card) -> int:

    clusters = find_clusters(EG, Token_Type.BEAR)

    match SC:
        case Score_Card.A:
            return score_bears_A(EG)
        case Score_Card.B:
            return score_bears_B(EG)
        case Score_Card.C:
            return score_bears_C(EG)
        case Score_Card.D:
            return score_bears_D(EG)
        case _:
            raise Exception("No Score Card provided for Bear")


def score_deer(EG: EnvironmentGraph, SC: Score_Card) -> int:
    clusters = find_clusters(EG, Token_Type.DEER)
    match SC:
        case Score_Card.A:
            return score_deer_A(EG)
        case Score_Card.B:
            return score_deer_B(EG)
        case Score_Card.C:
            return score_deer_C(EG)
        case Score_Card.D:
            return score_deer_D(EG)
        case _:
            raise Exception("No Score Card provided for Deer")


def score_salmon(EG: EnvironmentGraph, SC: Score_Card) -> int:
    clusters = find_clusters(EG, Token_Type.SALMON)
    match SC:
        case Score_Card.A:
            return score_salmon_A(EG)
        case Score_Card.B:
            return score_salmon_B(EG)
        case Score_Card.C:
            return score_salmon_C(EG)
        case Score_Card.D:
            return score_salmon_D(EG)
        case _:
            raise Exception("No Score Card provided for Salmon")


def score_hawks(EG: EnvironmentGraph, SC: Score_Card) -> int:
    clusters = find_clusters(EG, Token_Type.HAWK)
    match SC:
        case Score_Card.A:
            return score_hawks_A(EG)
        case Score_Card.B:
            return score_hawks_B(EG)
        case Score_Card.C:
            return score_hawks_C(EG)
        case Score_Card.D:
            return score_hawks_D(EG)
        case _:
            raise Exception("No Score Card provided for Hawk")


def score_foxes(EG: EnvironmentGraph, SC: Score_Card) -> int:
    clusters = find_clusters(EG, Token_Type.HAWK)
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
            raise Exception("No Score Card provided for Fox")

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
        case Token_Type.DEER:
            match SC:
                case Score_Card.A:
                    return score_deer_A(EG)
                case Score_Card.B:
                    return score_deer_B(EG)
                case Score_Card.C:
                    return score_deer_C(EG)
                case Score_Card.D:
                    return score_deer_D(EG)               
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
    parser.add_argument("-dr", "--deer", default="A", help="Deer scoring card")
    parser.add_argument("-sm", "--salmon", default="A", help="Salmon scoring card")
    parser.add_argument("-hk", "--hawk", default="A", help="Hawk scoring card")
    parser.add_argument("-fx", "--fox", default="A", help="Fox scoring card")
    args = parser.parse_args()

    tokens = detect_tokens(args.image)
    EG = EnvironmentGraph(tokens)
    environment_graph_utils.plot_environment_graph_tokens(EG)
    score = 0
    score += score_token_type(EG, Score_Card[args.bear], Token_Type.BEAR)
    score += score_token_type(EG, Score_Card[args.deer], Token_Type.DEER)
    score += score_token_type(EG, Score_Card[args.salmon], Token_Type.SALMON)
    score += score_token_type(EG, Score_Card[args.hawk], Token_Type.HAWK)
    score += score_token_type(EG, Score_Card[args.fox], Token_Type.FOX)
    print(score)


if __name__ == "__main__":
    main()
