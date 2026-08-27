import os
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from cascadai.schema.tile_piece import TilePiece
from cascadai.schema.environment_graph import EnvironmentGraph
from cascadai.utils.annotation_utils import CLASS_COLORS


def generate_random_graph(n: int, side_length: float = 1.0, seed: int | None = None) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # Side 0 is the rightmost side at 0°, then counterclockwise
    # by 60° increments: side 1 = 60°, side 2 = 120°, etc.
    base_angles = np.array(
        [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]
    )
    dist = side_length * np.sqrt(3)

    # First three tiles form a fixed triangle (the starting piece).
    # All three have theta = 0.
    tile0 = TilePiece(x=0.0, y=0.0, theta=0.0, side_length=side_length)
    G.add_node(0, tile=tile0)
    occupied = {(0.0, 0.0)}

    # Tile 1: attached to tile 0's side 4 (240° — lower-left)
    angle1 = 4 * np.pi / 3
    x1 = dist * np.cos(angle1)
    y1 = dist * np.sin(angle1)
    tile1 = TilePiece(x=x1, y=y1, theta=0.0, side_length=side_length)
    G.add_node(1, tile=tile1)
    G.add_edge(0, 1)
    occupied.add((round(x1, 10), round(y1, 10)))

    # Tile 2: attached to tile 0's side 5 (300° — lower-right)
    angle2 = 5 * np.pi / 3
    x2 = dist * np.cos(angle2)
    y2 = dist * np.sin(angle2)
    tile2 = TilePiece(x=x2, y=y2, theta=0.0, side_length=side_length)
    G.add_node(2, tile=tile2)
    G.add_edge(0, 2)
    occupied.add((round(x2, 10), round(y2, 10)))

    # Tiles 1 and 2 are also adjacent to each other
    G.add_edge(1, 2)

    for node_id in range(3, n):
        placed = False
        attempts = 0
        max_attempts = n * 10

        while not placed and attempts < max_attempts:
            attempts += 1
            parent = rng.integers(0, node_id)
            side = rng.integers(0, 6)

            parent_tile = G.nodes[parent]["tile"]
            angle = parent_tile.theta + base_angles[side]
            nx_ = parent_tile.x + dist * np.cos(angle)
            ny_ = parent_tile.y + dist * np.sin(angle)
            key = (round(nx_, 10), round(ny_, 10))

            if key in occupied:
                continue

            theta_new = rng.integers(0, 6) * np.pi / 3
            tile = TilePiece(x=nx_, y=ny_, theta=theta_new, side_length=side_length)
            G.add_node(node_id, tile=tile)
            G.add_edge(parent, node_id)
            occupied.add(key)

            # Add edges to any other adjacent tiles
            for other in range(node_id):
                if other == parent:
                    continue
                ot = G.nodes[other]["tile"]
                dx = nx_ - ot.x
                dy = ny_ - ot.y
                if abs(np.hypot(dx, dy) - dist) < 1e-8:
                    G.add_edge(node_id, other)

            placed = True

        if not placed:
            raise RuntimeError(
                f"Could not place tile {node_id} after {max_attempts} attempts"
            )

    return G


def plot_environment_graph(
    G: nx.Graph,
    ax=None,
):

    if ax is None:
        _, ax = plt.subplots()

    vertex_offsets = np.array([
        -np.pi / 6, np.pi / 6, np.pi / 2,
        5 * np.pi / 6, 7 * np.pi / 6, 3 * np.pi / 2
    ])

    for n, data in G.nodes(data=True):
        t = data["tile"]
        angles = t.theta + vertex_offsets
        verts = np.column_stack([
            t.x + t.side_length * np.cos(angles),
            t.y + t.side_length * np.sin(angles),
        ])
        ax.add_patch(Polygon(verts, facecolor="lightblue", edgecolor="black", lw=1.5))
        ax.plot([t.x, t.x + t.side_length * np.cos(t.theta)],
                 [t.y, t.y + t.side_length * np.sin(t.theta)],
                 color="gray", lw=2)
        ax.text(t.x, t.y, str(n), ha="center", va="center", fontsize=8)

    for u, v in G.edges():
        tu = G.nodes[u]["tile"]
        tv = G.nodes[v]["tile"]
        ax.plot([tu.x, tv.x], [tu.y, tv.y], color="gray", lw=1, zorder=0)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return ax


def plot_environment_graph_tokens(
    env_graph: EnvironmentGraph,
    out_dir: str | Path = "output",
    filename: str = "env_graph_tokens.png",
):
    """Plot the token environment graph.

    Token coordinates are given in pixel/image space (y grows downward), so the
    y-axis is inverted to match the image orientation and an equal aspect ratio
    keeps x and y scales undistorted. Nodes are colored by token type and sized
    by token width.
    """
    nodes = list(env_graph.EG.nodes())
    widths = np.array([n.width for n in nodes], dtype=float)
    median_width = np.median(widths)
    scale = (40 / median_width) ** 2 if median_width > 0 else 1.0
    node_sizes = [max(25, (w * w) * scale) for w in widths]
    node_colors = [CLASS_COLORS.get(n.type, "#333333") for n in nodes]
    pos = {n: (n.x, n.y) for n in nodes}

    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    fig, ax = plt.subplots(figsize=(10, 10))

    nx.draw_networkx_edges(
        env_graph.EG, pos, ax=ax,
        edge_color="#555555", alpha=0.5, width=1.5,
    )
    ax.scatter(
        xs, ys, s=node_sizes, c=node_colors,
        edgecolors="black", linewidths=1.2, zorder=3,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor=color, markeredgecolor="black", label=name)
        for name, color in [(t.name, CLASS_COLORS[t]) for t in CLASS_COLORS]
    ]
    ax.legend(handles=legend_handles, loc="best", framealpha=0.9, title="Token type")

    ax.set_title("Environment graph tokens")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path