"""leuven embedded figures dataset generator."""

import csv
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.ImageOps import invert
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


def get_highest_number(folder_path):
    """return the highest number found in filenames within a folder."""
    filenames = os.listdir(folder_path)
    highest_number = -1

    for filename in filenames:
        numbers = re.findall(r"\d+", filename)

        for number_str in numbers:
            number = int(number_str)
            if number > highest_number:
                highest_number = number

    return highest_number


def load_and_invert(path, canvas_size, background, antialiasing):
    """load an image, invert it, resize and apply background color."""
    try:
        img = invert(Image.open(path).convert("RGB"))

    except UnidentifiedImageError:
        img = np.load(
            path.parent.parent / "shapes_npy" / path.name.replace(".png", ".npy"),
            allow_pickle=True,
        )
        img = Image.fromarray(img)

    img = img.resize(canvas_size)
    img = img.point(lambda x: 255 if x >= 10 else 0)

    img = img.convert("RGB")

    data = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = data[x, y]
            if r == 0 or g == 0 or b == 0:
                if background == "rnd-uniform":
                    background = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                else:
                    data[x, y] = tuple(background)

    return apply_antialiasing(img) if antialiasing else img


@dataclass
class LeuvenEmbeddedConfig(GeneratorConfig):
    """config for leuven embedded figures dataset."""

    output_folder: str = field(
        default="data/shape_and_object_recognition/leuven_embedded_figures",
        metadata={"label": "output folder"},
    )


@register("leuven_embedded", "shape_recognition")
@generator(LeuvenEmbeddedConfig)
def generate_all(config: LeuvenEmbeddedConfig):
    """generate leuven embedded figures dataset with shapes and context stimuli."""
    output_folder = Path(config.output_folder)
    left_ds = Path("mindset/assets") / "leuven_embedded"

    figs_to_take = range(0, 16 * 4, 4)
    all_shapes_path = [
        left_ds / "shapes" / (str(i).zfill(3) + ".png") for i in figs_to_take
    ]
    all_context_path = [
        left_ds / "context" / (str(i).zfill(3) + "a.png") for i in range(0, 64)
    ]

    output_folder_shape = output_folder / "shapes"
    for i, s in enumerate(all_shapes_path):
        (output_folder_shape / str(i)).mkdir(parents=True, exist_ok=True)

    output_folder_context = output_folder / "context"
    for i, s in enumerate(all_context_path):
        (output_folder_context / str(i // 4)).mkdir(parents=True, exist_ok=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "Class", "BackgroundColor"])

        for idx, s in tqdm(enumerate(all_shapes_path)):
            img = load_and_invert(
                s, config.canvas_size, config.background_color, config.antialiasing
            )
            folder = output_folder_shape / str(idx)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow(
                [
                    f"shapes/{str(idx)}/{n + 1}.png",
                    "shapes",
                    idx,
                    config.background_color,
                ]
            )

        for idx, s in enumerate(tqdm(all_context_path, leave=False)):
            img = load_and_invert(
                s, config.canvas_size, config.background_color, config.antialiasing
            )
            folder = output_folder_context / str(idx // 4)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow(
                [
                    f"context/{str(idx // 4)}/{n + 1}.png",
                    "context",
                    idx // 4,
                    config.background_color,
                ]
            )

    return str(output_folder)
