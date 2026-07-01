"""
This file contains useful functions for working with the hexagon shaped tile pieces of Cascadia.
"""

import numpy as np
import cv2
import re
from pathlib import Path


def _sort_corners_clockwise(corners):
    """Sort 6 corners in clockwise order, starting from the topmost point."""
    center = np.mean(corners, axis=0)
    index_top = np.argmin(corners[:, 1])
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    shifted = angles + np.pi / 2
    shifted = shifted % (2 * np.pi)
    order = np.argsort(-shifted)
    pos = np.where(order == index_top)[0][0]
    order = np.roll(order, -pos)
    return corners[order]


def _fallback_ellipse_hexagon(contour):
    """Fallback: fit a regular hexagon via ellipse fitting."""
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (ma, mi), angle = ellipse
    area = cv2.contourArea(contour)
    r = np.sqrt(area / (3 * np.sqrt(3) / 2))
    angle_rad = np.deg2rad(angle)
    corners = []
    for i in range(6):
        theta = angle_rad + i * (np.pi / 3)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        corners.append([x, y])
    return np.array(corners, dtype=np.float32)


def find_tile_corners(binary_mask, min_area=500):
    """
    Detect 6 corners of a regular hexagonal tile from a binary mask.

    Uses approxPolyDP to find candidate vertices, determines the
    centroid and average corner radius, then constructs corners
    at exactly 60 degree intervals for a geometrically perfect hexagon.

    Args:
        binary_mask: 2D numpy array (uint8), white=object, black=background
        min_area: minimum contour area to consider

    Returns:
        (6, 2) array of [x, y] corners, sorted clockwise from topmost
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < min_area:
        raise ValueError(f"Contour area below minimum {min_area}")

    perimeter = cv2.arcLength(contour, closed=True)

    best_approx = None
    for eps_mult in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
        epsilon = eps_mult * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) == 6:
            best_approx = approx.reshape(-1, 2).astype(float)
            break

    if best_approx is None:
        return _fallback_ellipse_hexagon(contour)

    centroid = np.mean(best_approx, axis=0)
    angles = np.arctan2(best_approx[:, 1] - centroid[1], best_approx[:, 0] - centroid[0])
    sort_order = np.argsort(angles)
    sorted_corners = best_approx[sort_order]
    sorted_angles = angles[sort_order]

    radii = np.linalg.norm(sorted_corners - centroid, axis=1)
    avg_radius = np.mean(radii)

    start_angle = sorted_angles[0]
    corners = []
    for i in range(6):
        theta = start_angle + i * (np.pi / 3)
        x = centroid[0] + avg_radius * np.cos(theta)
        y = centroid[1] + avg_radius * np.sin(theta)
        corners.append([x, y])

    corners = np.array(corners, dtype=np.float32)
    return _sort_corners_clockwise(corners)


def rotate_and_crop_hexagon(image, corners, output_size=300):
    """_summary_

    Args:
        image (_type_): image of piece
        corners (_type_): location of tile corners in image
        output_size (int, optional): . Defaults to 300.

    Returns:
        _type_: transformed image of peices such that its alligned with center and has default shape.
    """
    h, w = output_size, output_size
    cx = cy = w / 2.0
    r = h / 2.0

    target = np.zeros((6, 2), dtype=np.float32)
    for i in range(6):
        angle = np.pi / 2 - i * np.pi / 3
        target[i, 0] = cx + r * np.cos(angle)
        target[i, 1] = cy - r * np.sin(angle)

    M, _ = cv2.estimateAffine2D(corners.astype(np.float32), target)
    return cv2.warpAffine(image, M, (w, h))