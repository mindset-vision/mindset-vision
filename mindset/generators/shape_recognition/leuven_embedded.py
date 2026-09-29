"""leuven embedded figures dataset generator."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.ImageOps import invert
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


def load_and_invert(path, canvas_size, background, antialiasing):
    """load an image, invert it, resize and apply background color."""
    try:
        img = invert(Image.open(path).convert("RGB"))
    except UnidentifiedImageError:
        arr = np.load(
            path.parent.parent / "shapes_npy" / path.name.replace(".png", ".npy"),
            allow_pickle=True,
        )
        img = Image.fromarray(arr)

    img = img.resize(canvas_size)
    img_gray = np.array(img.convert("L").point(lambda x: 255 if x >= 10 else 0))

    out = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    out[img_gray == 0] = tuple(background)
    out[img_gray == 255] = [255, 255, 255]
    img = Image.fromarray(out)

    return apply_antialiasing(img) if antialiasing else img


@dataclass
class LeuvenEmbeddedConfig(GeneratorConfig):
    """config for leuven embedded figures dataset."""

    input_folder: str | None = field(
        default=None,
        metadata={"label": "input folder with leuven embedded assets"},
    )
    output_folder: str = field(
        default="data/shape_recognition/leuven_embedded_figures",
        metadata={"label": "output folder"},
    )


@register("leuven_embedded", "shape_recognition")
@generator(LeuvenEmbeddedConfig)
def generate_all(config: LeuvenEmbeddedConfig):
    """generate leuven embedded figures dataset with shapes and context stimuli."""
    output_folder = Path(config.output_folder)
    if config.input_folder is None:
        left_ds = Path(__file__).resolve().parents[2] / "assets" / "leuven_embedded"
    else:
        left_ds = Path(config.input_folder)

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
            img_rel = Path("shapes") / str(idx) / "1.png"
            img.save(output_folder / img_rel)
            writer.writerow(
                [
                    img_rel.as_posix(),
                    "shapes",
                    idx,
                    config.background_color,
                ]
            )

        for idx, s in enumerate(tqdm(all_context_path, leave=False)):
            img = load_and_invert(
                s, config.canvas_size, config.background_color, config.antialiasing
            )
            class_id = idx // 4
            sample_in_class = (idx % 4) + 1
            img_rel = Path("context") / str(class_id) / f"{sample_in_class}.png"
            img.save(output_folder / img_rel)
            writer.writerow(
                [
                    img_rel.as_posix(),
                    "context",
                    class_id,
                    config.background_color,
                ]
            )

    return str(output_folder)
