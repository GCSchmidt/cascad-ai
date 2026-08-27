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

    # 2. Build the induced subgraph on just those nodes
    #    (only edges between same-type nodes survive)
    subgraph = EG.EG.subgraph(nodes_of_type)
    
    # 3. Connected components of that subgraph = clusters
    clusters = list(nx.connected_components(subgraph))
    
    # Sort by size, descending
    clusters.sort(key=len, reverse=True)

    print(f"Amount of {toke_type.name} clusters = {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"{toke_type.name} cluster {i+1} size = {len(c)}")
    
    return clusters


def score_bears_A(EG: EnvironmentGraph) -> int:
    return 0


def score_bears_B(EG: EnvironmentGraph) -> int:
    return 0


def score_bears_C(EG: EnvironmentGraph) -> int:
    return 0


def score_bears_D(EG: EnvironmentGraph) -> int:
    return 0


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
    clusters = find_clusters(EG, token_type)
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
    args = parser.parse_args()

    tokens = detect_tokens(args.image)
    EG = EnvironmentGraph(tokens)
    environment_graph_utils.plot_environment_graph_tokens(EG)
    score = 0
    score += score_token_type(EG, Score_Card.A, Token_Type.BEAR)
    score += score_token_type(EG, Score_Card.A, Token_Type.DEER)
    score += score_token_type(EG, Score_Card.A, Token_Type.SALMON)
    score += score_token_type(EG, Score_Card.A, Token_Type.HAWK)
    score += score_token_type(EG, Score_Card.A, Token_Type.FOX)


if __name__ == "__main__":
    main()
