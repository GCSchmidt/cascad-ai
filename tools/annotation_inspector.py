#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt

from cascadai.utils.annotation_utils import PlotAnnotations


def main():
    parser = argparse.ArgumentParser(
        description="Inspect YOLO annotations on images"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot annotations on the image",
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
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save the annotated plot to output/",
    )

    args = parser.parse_args()

    if args.plot:
        fig = PlotAnnotations(args.image_path, args.anno_path, save=args.save)
        plt.show()


if __name__ == "__main__":
    main()
