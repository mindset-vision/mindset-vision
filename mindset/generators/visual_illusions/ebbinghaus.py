"""ebbinghaus illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.ebbinghaus import DrawEbbinghaus
from mindset.generators._base import GeneratorConfig, generator, register


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
