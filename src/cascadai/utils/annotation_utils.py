import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from cascadai.schema import token_piece


CLASS_COLORS = {
    token_piece.Token_Type.BEAR: "#352c02",
    token_piece.Token_Type.FOX: "#ff7b00",  
    token_piece.Token_Type.HAWK: "#50a2e6",
    token_piece.Token_Type.ELK: "#e0c466",
    token_piece.Token_Type.SALMON: "#e972df",
}


def plot_annotations(image_path: str, anno_path: str, save: bool = False):
    img = plt.imread(image_path)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    with open(anno_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xn, yn, wn, hn = map(float, parts)
        cls, xn, yn = int(cls), xn, yn
        wp = wn * w
        hp = hn * h
        xc = xn * w
        yc = yn * h
        x1 = xc - (wp / 2)
        y1 = yc - (hp / 2)

        token_type = token_piece.Token_Type(cls)
        color = CLASS_COLORS.get(token_type, "#333333")
        label = token_type.name
        
        rect = patches.Rectangle(
            (x1, y1), wp, hp, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, label, fontsize=10, color=color,
            weight="bold", bbox=dict(facecolor="white", alpha=0.7, pad=1),
        )

    ax.set_title(f"Annotations — {os.path.basename(anno_path)}")
    ax.axis("off")

    if save:
        os.makedirs("output", exist_ok=True)
        stem = os.path.splitext(os.path.basename(anno_path))[0]
        out_path = os.path.join("output", f"{stem}_annotated.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)

    return fig


class EnvImageAnnotation():
    '''
    A class to help build and manage annotations in the YOLO format. 
    '''
    def __init__(self, file_path, image_shape) -> None:
        self.file_path = file_path
        self.image_shape = image_shape
        self.tokens = []

    def write(self) -> None:
        with open(self.file_path, 'w') as f:
            for token in self.tokens:
                line = self.token_to_annotation(token)
                f.write(line + '\n')
        f.close

    def token_to_annotation(self, token: token_piece.Token) -> str:
        scaled_x = token.x / self.image_shape[1]
        scaled_y = token.y / self.image_shape[0]
        scaled_width = token.width / self.image_shape[1]
        scaled_height = token.width / self.image_shape[0]
        return f'{token.type.value} {scaled_x} {scaled_y} {scaled_width} {scaled_height}'

    def append_annotation(self, token: token_piece.Token) -> None:
        self.tokens.append(token)
