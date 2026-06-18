import cv2
import numpy as np
from pathlib import Path
import random
from enum import Enum
import re


class Piece_Type(Enum):
    STARTER_TILE = 1
    TILE = 2
    TOKEN = 3


script_path = Path(__file__).resolve()


BASE_PROJ_DIR = script_path.parent.parent
DATASET_DIR_PATH = BASE_PROJ_DIR / "datasets"
GAME_PIECES_IMAGE_DIR_PATH = DATASET_DIR_PATH / "original_game_pieces"
ALL_STARTER_TILE_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_starter_tiles_bg_removed.png"
ALL_TILE_COMBIS_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_tile_combis_bg_removed.png"
ALL_TOKENS_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_tokens_bg_removed.png"
BUILDING_BLOCKS_DIR = DATASET_DIR_PATH / "building_blocks"
FOR_MODEL_DIR = DATASET_DIR_PATH / "for_model"


def get_background_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    b, g, r, alpha = cv2.split(image)
    mask = alpha
    _, binary_mask = cv2.threshold(mask, 222, 255, cv2.THRESH_BINARY)
    return binary_mask


def generate_pieces(piece_type, min_area=500):
    match piece_type:
        case Piece_Type.STARTER_TILE:
            image_path = ALL_STARTER_TILE_IMAGE_PATH
            default_file_name = "starter_tile"
        case Piece_Type.TILE:
            image_path = ALL_TILE_COMBIS_IMAGE_PATH
            default_file_name = "tile"
        case Piece_Type.TOKEN:
            image_path = ALL_TOKENS_IMAGE_PATH
            default_file_name = "token"
        case _:
            raise Exception("No valid Piece_Type")

    mask = get_background_mask(image_path)
    image_gbr = cv2.imread(image_path)

    num_labels, labels = cv2.connectedComponents(mask)

    saved = 1
    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        if cv2.countNonZero(component_mask) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(component_mask)

        cropped_color = image_gbr[y:y+h, x:x+w]
        cropped_mask = component_mask[y:y+h, x:x+w]

        b, g, r = cv2.split(cropped_color)
        bgra = cv2.merge([b, g, r, cropped_mask])

        out_path = BUILDING_BLOCKS_DIR / f"{default_file_name}_{saved:03d}.png"
        cv2.imwrite(str(out_path), bgra)
        saved += 1
   
    print(f"Saved {saved} {default_file_name}s to {BUILDING_BLOCKS_DIR}")


def generate_building_block_images():
    building_blocks_path = Path(BUILDING_BLOCKS_DIR)
    building_blocks_path.mkdir(parents=True, exist_ok=True)
    generate_pieces(Piece_Type.STARTER_TILE)
    generate_pieces(Piece_Type.TILE)
    generate_pieces(Piece_Type.TOKEN)


def overlay_centered(background, foreground, max_deviation=10):
    """
    Place foreground centered on background with optional random offset.
    max_deviation: max pixels to randomly shift in any direction (0 to disable)
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    cx = (bg_w - fg_w) // 2
    cy = (bg_h - fg_h) // 2

    if max_deviation > 0:
        cx += random.randint(-max_deviation, max_deviation)
        cy += random.randint(-max_deviation, max_deviation)

    cx = max(0, min(cx, bg_w - fg_w))
    cy = max(0, min(cy, bg_h - fg_h))

    fg_bgr = foreground[:, :, :3]
    alpha = foreground[:, :, 3] / 255.0
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

    bg_region = background[cy:cy+fg_h, cx:cx+fg_w]
    bg_region_bgr = bg_region[:, :, :3]  # only use the 3 colour channels

    bg_region_bgr[:] = (alpha_3ch * fg_bgr + (1 - alpha_3ch) * bg_region_bgr).astype(np.uint8)
    background[cy:cy+fg_h, cx:cx+fg_w, :3] = bg_region_bgr

    return background


def get_building_block_image(piece_type):
    match piece_type:
        case Piece_Type.STARTER_TILE:
            default_file_name = "starter_tile"
        case Piece_Type.TILE:
            default_file_name = "tile"
        case Piece_Type.TOKEN:
            default_file_name = "token"
        case _:
            raise Exception("No valid Piece_Type")
    pattern = re.compile(rf'^{default_file_name}_\d{{3}}\.png$')
    building_blocks_dir = Path(BUILDING_BLOCKS_DIR)
    matches = [f.name for f in building_blocks_dir.iterdir() if pattern.match(f.name)]
    piece_file_path = BUILDING_BLOCKS_DIR / random.choice(matches)
    return piece_file_path


def generate_env_image():   
    tile_path = get_building_block_image(Piece_Type.TILE)
    token_path = get_building_block_image(Piece_Type.TOKEN)
    tile_image = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
    token_image = cv2.imread(token_path, cv2.IMREAD_UNCHANGED)
    print(tile_path, token_path)
    covered_tile = overlay_centered(tile_image, token_image)
    image_path = FOR_MODEL_DIR / "test.png"
    cv2.imwrite(image_path, covered_tile)


if __name__ == "__main__":
    generate_building_block_images()
    generate_env_image()
