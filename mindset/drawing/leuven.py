import os
import random
import re

import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.ImageOps import invert

from mindset.utils.misc import apply_antialiasing


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
            if (
                r == 0 or g == 0 or b == 0
            ):
                if background == "rnd-uniform":
                    background = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                else:
                    data[x, y] = tuple(background)

    return apply_antialiasing(img) if antialiasing else img
