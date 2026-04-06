"""uncrowding dataset generator."""
import csv
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.generate_datasets.low_mid_level_vision.un_crowding.generate_dataset import (
    DrawUncrowding,
    all_test_shapes,
)
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class UncrowdingConfig(GeneratorConfig):
    """config for uncrowding dataset."""
    num_samples_vernier_inside: int = field(default=100, metadata={"min": 1, "max": 10000, "step": 10, "label": "vernier inside samples"})
    num_samples_vernier_outside: int = field(default=100, metadata={"min": 1, "max": 10000, "step": 10, "label": "vernier outside samples"})
    random_size: bool = field(default=True, metadata={"label": "random shape size"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/low_mid_level_vision/un_crowding", metadata={"label": "output folder"})


@register("uncrowding", "low_mid_vision")
@generator(UncrowdingConfig)
def generate_all(config: UncrowdingConfig):
    """generate uncrowding dataset."""
    output_folder = Path(config.output_folder)

    vernier_in_out = ["outside", "inside"]
    vernier_type = [0, 1]

    for v in vernier_in_out:
        for c in vernier_type:
            (output_folder / v / str(c)).mkdir(exist_ok=True, parents=True)

    ds = DrawUncrowding(
        canvas_size=config.canvas_size,
        background=config.background_color,
        antialiasing=config.antialiasing,
        bar_width=1,
    )

    t = all_test_shapes()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "VernierInOut", "VernierType", "BackgroundColor", "ShapeCode", "ShapeSize", "IterNum"])

        for v_in_out in tqdm(vernier_in_out):
            num_requested = (
                config.num_samples_vernier_outside
                if v_in_out == "outside"
                else config.num_samples_vernier_inside
            )
            samples_per_cond = num_requested // len(t)
            if samples_per_cond == 0:
                print(f"in order to have at least one sample per condition, the total number of sample has been increased to {len(t)}")
                samples_per_cond = 1
            if samples_per_cond * len(t) != num_requested:
                print(f"you specified {num_requested} for {v_in_out} but to keep the number of sample per subcategory equal, {samples_per_cond * len(t)} samples will be generated ({len(t)} categories, {samples_per_cond} samples per category)")

            for v in vernier_type:
                for s in tqdm(t, leave=False):
                    for n in range(samples_per_cond):
                        shape_size = (
                            random.randint(int(config.canvas_size[0] * 0.1), config.canvas_size[0] // 7)
                            if config.random_size
                            else config.canvas_size[0] * 0.08
                        )
                        shape_size -= int(shape_size / 6)

                        img = ds.draw_stim(
                            vernier_ext=v_in_out == "outside",
                            shape_matrix=s,
                            shape_size=shape_size,
                            vernier_in=v_in_out == "inside",
                            fixed_position=None,
                            offset=v,
                            offset_size=None,
                            noise_patch=None,
                        )
                        strs = str(s).replace("], ", "nl")
                        shape_code = "".join(i for i in strs if i not in [",", "[", "]", " "])
                        shape_code = shape_code if shape_code != "" else "none"

                        unique_hex = uuid.uuid4().hex[:8]
                        path = Path(v_in_out) / str(v) / f"{shape_code}_{n}_{unique_hex}.png"
                        img.save(output_folder / path)
                        writer.writerow([path, v_in_out, v, ds.background, shape_code, shape_size, n])

    return str(output_folder)
