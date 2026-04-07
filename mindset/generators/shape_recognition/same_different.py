"""same different task dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.generators.shape_recognition._same_different_drawing import (
    DrawSameDifferentStimuli,
    is_overlapping,
    get_regular,
    get_irregular_polygon,
    get_open,
    get_wider_line,
    get_rnd_color,
    get_filled,
    get_open_squares,
    get_rectangles,
    get_straight_lines,
    get_closed_squares,
    is_integer,
)


@dataclass
class SameDifferentConfig(GeneratorConfig):
    """config for same different task dataset."""
    num_samples: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "samples per type per condition"})
    size_shapes: str = field(default="20", metadata={"label": "shape size (int, rnd1, or rnd2)"})
    type_dataset: str = field(default="all", metadata={"label": "dataset type (all or specific name)"})
    output_folder: str = field(default="data/shape_and_object_recognition/same_different_task", metadata={"label": "output folder"})


@register("same_different", "shape_recognition")
@generator(SameDifferentConfig)
def generate_all(config: SameDifferentConfig):
    """generate same-different task dataset with various shape types."""
    output_folder = Path(config.output_folder)

    all_datasets = {
        "regular": get_regular,
        "irregular": get_irregular_polygon,
        "open": get_open,
        "wider_line": get_wider_line,
        "rnd_color": get_rnd_color,
        "filled": get_filled,
        "open_squares": get_open_squares,
        "rectangles": get_rectangles,
        "straight_lines": get_straight_lines,
        "closed_squares": get_closed_squares,
    }

    datasets = all_datasets if config.type_dataset == "all" else {config.type_dataset: all_datasets[config.type_dataset]}

    ds = DrawSameDifferentStimuli(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    labels = ["same", "diff"]
    for label in labels:
        for ds_name in datasets:
            (output_folder / ds_name / label).mkdir(exist_ok=True, parents=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "BackgroundColor", "TypeDataset", "SizeShape1", "SizeShape2", "SameDiff", "SampleNum"])

        for ds_name, dataset_fun in tqdm(datasets.items()):
            use_size2 = ds_name not in ["open_squares", "closed_squares", "rectangles"]
            for n in tqdm(range(config.num_samples), leave=False):
                for label in labels:
                    while True:
                        if is_integer(config.size_shapes):
                            size1, size2 = int(config.size_shapes), int(config.size_shapes)
                        elif config.size_shapes == "rnd1":
                            size1 = np.random.randint(ds.canvas_size[0] // 15, ds.canvas_size[0] // 4)
                            size2 = size1
                        else:
                            size1, size2 = np.random.randint(ds.canvas_size[0] // 15, ds.canvas_size[0] // 4, 2)

                        args = dict(label=1 if label == "same" else 0, size1=size1)
                        if use_size2:
                            args["size2"] = size2

                        img = dataset_fun(ds, **args)
                        if not is_overlapping(np.array(img), ds.background):
                            break

                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = Path(ds_name) / label / f"{unique_hex}.png"
                    img.save(output_folder / img_path)
                    writer.writerow([img_path, config.background_color, ds_name, size1, size2 if use_size2 else None, label, n])

    return str(output_folder)
