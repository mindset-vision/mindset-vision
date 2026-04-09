"""ebbinghaus illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ImageDraw
from tqdm import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


class DrawEbbinghaus(DrawStimuli):
    """draws ebbinghaus illusion stimuli."""

    def create_ebbinghaus(
        self,
        r_c,
        d=0,
        r2=0,
        n=0,
        shift=0,
        colour_center_circle=(255, 255, 255),
        img=None,
    ):
        """create an ebbinghaus illusion with center circle and flankers."""
        if img is None:
            img = self.create_canvas()
        draw = ImageDraw.Draw(img)
        if d != 0:
            thetas = np.linspace(0, np.pi * 2, n, endpoint=False) + shift
            dd = img.size[0] * d
            vect = [[np.cos(t) * dd, np.sin(t) * dd] for t in thetas]
            [
                self.circle(draw, np.array(vv) + img.size[0] / 2, img.size[0] * r2 / 2)
                for vv in vect
            ]
        self.circle(
            draw,
            np.array(img.size) / 2,
            img.size[0] * r_c / 2,
            fill=colour_center_circle,
        )

        return apply_antialiasing(img) if self.antialiasing else img

    def create_random_ebbinghaus(
        self,
        r_c,
        n=0,
        flankers_size_range=(0.1, 0.5),
        colour_center_circle=(255, 255, 255),
    ):
        """create ebbinghaus with randomly placed flankers."""
        gen_rnd = lambda r: np.random.uniform(*r)

        img = self.create_canvas()
        draw = ImageDraw.Draw(img)

        for i in range(n):
            random_points = [
                np.random.randint(self.canvas_size[0]),
                np.random.randint(self.canvas_size[1]),
            ]
            random_size = self.canvas_size[0] * gen_rnd(
                np.array(flankers_size_range) / 2
            )
            self.circle(draw, np.array(random_points), int(random_size))
        self.circle(
            draw,
            np.array(self.canvas_size) / 2,
            self.canvas_size[0] * r_c / 2,
            fill=colour_center_circle,
        )

        return apply_antialiasing(img) if self.antialiasing else img


@dataclass
class EbbinghausConfig(GeneratorConfig):
    """config for ebbinghaus illusion dataset."""
    num_samples_scrambled: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "scrambled samples"})
    num_samples_illusory: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "illusory samples"})
    output_folder: str = field(default="data/visual_illusions/ebbinghaus", metadata={"label": "output folder"})


@register("ebbinghaus", "visual_illusions")
@generator(EbbinghausConfig)
def generate_all(config: EbbinghausConfig):
    """generate ebbinghaus illusion dataset."""
    output_folder = Path(config.output_folder)
    for d in ["scrambled_circles", "small_flankers", "big_flankers"]:
        (output_folder / d).mkdir(parents=True, exist_ok=True)

    ds = DrawEbbinghaus(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Category", "NormSizeCenterCircle", "NormSizeOtherFlankers", "NumFlankers", "BackgroundColor", "Shift"])

        for i in tqdm(range(config.num_samples_scrambled)):
            r_c = np.random.uniform(0.1, 0.4)
            img = ds.create_random_ebbinghaus(r_c=r_c, n=5, flankers_size_range=(0.04, 0.3), colour_center_circle=(255, 0, 0))
            path = Path("scrambled_circles") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow([path, "scrambled_circles", r_c, "", 5, ds.background, ""])

        for i in tqdm(range(config.num_samples_illusory)):
            r_c = np.random.uniform(0.1, 0.4)
            r2 = np.random.uniform(0.24, 0.3)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(r_c=r_c, d=0.02 + (r_c + r2) / 2, r2=r2, n=5, shift=shift, colour_center_circle=(255, 0, 0))
            path = Path("big_flankers") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow([path, "big_flankers", r_c, r2, 5, ds.background, shift])

            r_c = np.random.uniform(0.1, 0.4)
            r2 = np.random.uniform(0.04, 0.15)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(r_c=r_c, d=0.02 + (r_c + r2) / 2, r2=r2, n=8, shift=shift, colour_center_circle=(255, 0, 0))
            unique_hex = uuid.uuid4().hex[:8]
            path = Path("small_flankers") / f"{r_c:.5f}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, "small_flankers", r_c, r2, 8, ds.background, shift])

    return str(output_folder)
