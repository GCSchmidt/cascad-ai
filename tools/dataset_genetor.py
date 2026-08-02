import argparse
import cv2
import numpy as np
from pathlib import Path
import random
from enum import Enum
import re
import yaml

from cascadai.utils import tile_piece_utils, environment_graph_utils, annotation_utils
from cascadai.schema import token_piece


class Piece_Type(Enum):
    STARTER_TILE = 1
    TILE = 2
    TOKEN = 3


class Dataset_Type(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


script_path = Path(__file__).resolve()


BASE_PROJ_DIR = script_path.parent.parent
DATASET_DIR_PATH = BASE_PROJ_DIR / "datasets"
GAME_PIECES_IMAGE_DIR_PATH = DATASET_DIR_PATH / "original_game_pieces"
ALL_STARTER_TILE_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_starter_tiles_bg_removed.png"
ALL_TILE_COMBIS_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_tile_combis_bg_removed.png"
ALL_TOKENS_IMAGE_PATH = GAME_PIECES_IMAGE_DIR_PATH / "all_tokens_bg_removed.png"
BUILDING_BLOCKS_DIR = DATASET_DIR_PATH / "building_blocks"
FOR_MODEL_DIR = DATASET_DIR_PATH / "for_model"
SINGLE_PIECE_MASK_DIR = DATASET_DIR_PATH / "single_piece_masks"


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
        piece_mask = np.where(labels == label, 255, 0).astype(np.uint8)

        if cv2.countNonZero(piece_mask) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(piece_mask)

        cropped_color = image_gbr[y:y+h, x:x+w]
        cropped_mask = piece_mask[y:y+h, x:x+w]

        out_path = SINGLE_PIECE_MASK_DIR / f"{default_file_name}_{saved:03d}.png"

        # save masks for testing
        cv2.imwrite(str(out_path), cropped_mask)

        b, g, r = cv2.split(cropped_color)
        bgra = cv2.merge([b, g, r, cropped_mask])

        match piece_type:
            case Piece_Type.STARTER_TILE:
                pass
            case Piece_Type.TILE:
                bgra = preprocess_tile_piece(bgra, cropped_mask)
            case Piece_Type.TOKEN:
                bgra = preprocess_token_piece(bgra)

        out_path = BUILDING_BLOCKS_DIR / f"{default_file_name}_{saved:03d}.png"
        cv2.imwrite(str(out_path), bgra)
        saved += 1
   
    print(f"Saved {saved} {default_file_name}s to {BUILDING_BLOCKS_DIR}")


def preprocess_tile_piece(bgra, mask):
    padding = 10
    mask = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    corners = tile_piece_utils.find_tile_corners(mask)
    processed_bgra = tile_piece_utils.rotate_and_crop_hexagon(bgra, corners)
    return processed_bgra


def preprocess_token_piece(bgra):
    scale = 2
    processed_bgra = cv2.resize(bgra, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return processed_bgra


def generate_building_block_images():
    building_blocks_path = Path(BUILDING_BLOCKS_DIR)
    building_blocks_path.mkdir(parents=True, exist_ok=True)
    generate_pieces(Piece_Type.STARTER_TILE)
    generate_pieces(Piece_Type.TILE)
    generate_pieces(Piece_Type.TOKEN)


def overlay_centered(background, foreground, shift=[0, 0]):
    """
    Place foreground centered on background with optional random offset.
    shift: amount of pixels to shift foreground
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    cx = (bg_w - fg_w) // 2
    cy = (bg_h - fg_h) // 2

    cx += shift[0]
    cy += shift[1]

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


def place_token_on_tile(tile_image, token_type, shift=[0, 0]):
    token_path = BUILDING_BLOCKS_DIR / f'token_00{token_type.value}.png'
    token_image = cv2.imread(token_path, cv2.IMREAD_UNCHANGED)

    theta = random.randint(0, 359)
    h, w = token_image.shape[:2]
    M = cv2.getRotationMatrix2D(
        (w / 2, h / 2), np.degrees(theta), 1.0
    )
    token_image = cv2.warpAffine(
        token_image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    covered_tile = overlay_centered(tile_image, token_image, shift)
    return covered_tile


def generate_dataset_sample(image_path: str, anno_path: str):

    side_length = 150
    environment_graph = environment_graph_utils.generate_random_graph(23, side_length)

    xs = [data["tile"].x for _, data in environment_graph.nodes(data=True)]
    ys = [data["tile"].y for _, data in environment_graph.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 150
    canvas_w = int(max_x - min_x + 2 * padding)
    canvas_h = int(max_y - min_y + 2 * padding)
    env_image = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255

    offset_x = padding - min_x
    offset_y = padding - min_y

    Annotation = annotation_utils.EnvImageAnnotation(anno_path, env_image.shape)

    for node_id, data in environment_graph.nodes(data=True):
        tile = data["tile"]

        tile_img_path = get_building_block_image(Piece_Type.TILE)
        tile_img = cv2.imread(str(tile_img_path), cv2.IMREAD_UNCHANGED)

        if tile.theta != 0.0:
            h, w = tile_img.shape[:2]
            M = cv2.getRotationMatrix2D(
                (w / 2, h / 2), np.degrees(tile.theta), 1.0
            )
            tile_img = cv2.warpAffine(
                tile_img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

        cx = int(round(tile.x + offset_x))
        cy = int(round(tile.y + offset_y))
        h, w = tile_img.shape[:2]
        x0, y0 = cx - w // 2, cy - h // 2

        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        sx1 = min(w, canvas_w - x0)
        sy1 = min(h, canvas_h - y0)
        dx1 = min(canvas_w, x0 + w)
        dy1 = min(canvas_h, y0 + h)

        if sx0 >= sx1 or sy0 >= sy1:
            continue

        tile_img_region = tile_img[sy0:sy1, sx0:sx1]
        canvas_region = env_image[dy0:dy1, dx0:dx1]

        place_token = random.uniform(0, 1) > 0.1

        if place_token:

            token_type = random.choice(list(token_piece.Token_Type))
            max_deviation = 10
            shift_x = random.randint(-max_deviation, max_deviation)
            shift_y = random.randint(-max_deviation, max_deviation)
            shift = [shift_x, shift_y]
            tile_img = place_token_on_tile(tile_img, token_type, shift)

            # write to annotation file
            token_x = cx + shift_x
            token_y = cy + shift_y
            token_width = 170
            token = token_piece.Token(token_type, token_x, token_y, token_width)
            Annotation.append_annotation(token)

        alpha = tile_img_region[:, :, 3] / 255.0
        alpha_3ch = np.dstack([alpha] * 3)
        canvas_region[:, :, :3] = (
            alpha_3ch * tile_img_region[:, :, :3]
            + (1 - alpha_3ch) * canvas_region[:, :, :3]
        ).astype(np.uint8)
        canvas_region[:, :, 3] = np.maximum(
            canvas_region[:, :, 3], tile_img_region[:, :, 3]
        )

    cv2.imwrite(image_path, env_image)
    Annotation.write()


def generate_dataset_yaml(tag: str = ""):
    yaml_path = FOR_MODEL_DIR / f"config{tag}.yml"
    tokens_json = {tt.value: tt.name for tt in token_piece.Token_Type}

    data = {
        "path": yaml_path.name,
        "train": f"train{tag}",
        "val": f"val{tag}",
        "test": f"test{tag}",
        "names": tokens_json
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Generated dataset config to {yaml_path}")


def generate_dataset_images(dataset_type: Dataset_Type, N: int = 1, tag: str = ""):

    dirname = ""
    match dataset_type:
        case Dataset_Type.TRAIN:
            dirname = "train"
        case Dataset_Type.VALIDATION:
            dirname = "val"
        case Dataset_Type.TEST:
            dirname = "test"
    dirname += tag

    FOR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = FOR_MODEL_DIR / dirname / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    anno_dir = FOR_MODEL_DIR / dirname / "labels"
    anno_dir.mkdir(parents=True, exist_ok=True) 

    for i in range(1, 1+N):
        image_path = image_dir / f"{i}.png"
        anno_path = anno_dir / f"{i}.txt"
        generate_dataset_sample(str(image_path), str(anno_path))

    print(f"Generated {N} images in {FOR_MODEL_DIR / dirname}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Cascadia dataset (images + YOLO labels)."
    )
    parser.add_argument(
        "-train",
        type=int,
        required=True,
        help="Number of training images to generate",
    )
    parser.add_argument(
        "-val",
        type=int,
        default=None,
        help="Number of validation images to generate (default: 20% of train)",
    )
    args = parser.parse_args()

    val_count = args.val if args.val is not None else int(round(0.2 * args.train))

    generate_building_block_images()
    generate_dataset_images(Dataset_Type.TRAIN, args.train)
    generate_dataset_images(Dataset_Type.VALIDATION, val_count)
    generate_dataset_yaml()


if __name__ == "__main__":
    main()
