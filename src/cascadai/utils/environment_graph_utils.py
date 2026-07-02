import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from cascadai.schema.tile_piece import TilePiece


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
