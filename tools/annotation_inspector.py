#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt

from cascadai.utils.annotation_utils import plot_annotations


def main():
    parser = argparse.ArgumentParser(
        description="Inspect YOLO annotations on images"
    )

    parser.add_argument(
        "-i", "--image-path",
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "-a", "--anno-path",
        required=True,
        help="Path to the YOLO annotation file",
    )

    args = parser.parse_args()

    plot_annotations(args.image_path, args.anno_path, save=True)


if __name__ == "__main__":
    main()
