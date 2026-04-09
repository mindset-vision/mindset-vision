"""2d transformations dataset generator."""

import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from mindset.drawing.affine import get_affine_rnd_fun, my_affine
from mindset.drawing.base import DrawStimuli, resize_image_keep_aspect_ratio
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


class DrawTransform(DrawStimuli):
    """applies affine transformations to linedrawing images."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_image_transformed(self, image_path, tr, rt, sc, sh):
        """load, resize, paste on canvas, apply affine transform."""
        img = Image.fromarray(
            resize_image_keep_aspect_ratio(
                np.array(Image.open(image_path)), self.obj_longest_side
            )
        )
        canvas = self.create_canvas()
        canvas.paste(
            img,
            (
                canvas.size[0] // 2 - img.size[0] // 2,
                canvas.size[1] // 2 - img.size[1] // 2,
            ),
        )
        canvas = my_affine(
            canvas,
            translate=tr,
            angle=rt,
            scale=sc,
            shear=sh,
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        canvas = ImageOps.invert(canvas.convert("L"))
        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class Transformations2DConfig(GeneratorConfig):
    """config for 2d transformations dataset."""

    input_folder: str = field(
        default="mindset/assets/baker_2018_linedrawings/cropped/",
        metadata={"label": "input folder with line drawings"},
    )
    object_longest_side: int = field(
        default=200,
        metadata={
            "min": 50,
            "max": 500,
            "step": 10,
            "label": "object longest side (px)",
        },
    )
    translation_x: list = field(
        default_factory=lambda: [-0.2, 0.2], metadata={"label": "translation X range"}
    )
    translation_y: list = field(
        default_factory=lambda: [-0.2, 0.2], metadata={"label": "translation Y range"}
    )
    scale: list = field(
        default_factory=lambda: [0.5, 0.9], metadata={"label": "scale range"}
    )
    rotation: list = field(
        default_factory=lambda: [0, 360], metadata={"label": "rotation range (degrees)"}
    )
    num_samples: int = field(
        default=50,
        metadata={"min": 1, "max": 10000, "step": 1, "label": "samples per image"},
    )
    background_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "background color (RGB)"},
    )
    output_folder: str = field(
        default="data/shape_and_object_recognition/2d_transformations",
        metadata={"label": "output folder"},
    )


@register("transformations_2d", "shape_recognition")
@generator(Transformations2DConfig)
def generate_all(config: Transformations2DConfig):
    """generate 2d transformations dataset."""
    output_folder = Path(config.output_folder)
    input_folder = Path(config.input_folder)

    all_categories = [p.stem for p in input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawTransform(
        obj_longest_side=config.object_longest_side,
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    image_files = sorted(input_folder.rglob("*.jpg")) + sorted(
        input_folder.rglob("*.png")
    )

    af = get_affine_rnd_fun(
        {
            "translation_X": config.translation_x,
            "translation_Y": config.translation_y,
            "rotation": config.rotation,
            "scale": config.scale,
        }
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "BackgroundColor",
                "Translation_X",
                "Translation_Y",
                "Rotation",
                "Scale",
                "Shear",
                "IterNum",
            ]
        )

        for img_path in tqdm(image_files):
            for i in range(config.num_samples):
                rnd_v = af()
                tr_x, tr_y, rt, sc, sh = (
                    rnd_v["tr"][0],
                    rnd_v["tr"][1],
                    rnd_v["rt"],
                    rnd_v["sc"],
                    rnd_v["sh"],
                )

                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.get_image_transformed(img_path, [tr_x, tr_y], rt, sc, sh)
                uui = uuid.uuid4().hex[:8]
                path = Path(class_name) / f"{image_name}_{uui}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [path, class_name, ds.background, tr_x, tr_y, rt, sc, sh, i]
                )

    return str(output_folder)
