"""jastrow illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.jastrow import DrawJastrow, get_random_params
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class JastrowConfig(GeneratorConfig):
    """config for jastrow illusion dataset."""
    num_samples_illusory: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "illusory samples"})
    num_samples_random: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "random samples"})
    num_samples_aligned: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "aligned samples"})
    num_samples_random_same_size: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "random same size samples"})
    output_folder: str = field(default="data/visual_illusions/jastrow", metadata={"label": "output folder"})


@register("jastrow", "visual_illusions")
@generator(JastrowConfig)
def generate_all(config: JastrowConfig):
    """generate jastrow illusion dataset."""
    output_folder = Path(config.output_folder)

    types = ["illusory", "random_same_size", "random", "aligned"]
    on_top_cols = ["red", "blue"]

    for type_name in types:
        for top in on_top_cols:
            subfolder = output_folder / type_name / ("" if "random" in type_name else f"{top}_on_top")
            subfolder.mkdir(exist_ok=True, parents=True)

    ds = DrawJastrow(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    num_samples_map = {
        "illusory": config.num_samples_illusory,
        "aligned": config.num_samples_aligned,
        "random": config.num_samples_random,
        "random_same_size": config.num_samples_random_same_size,
    }

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "BackgroundColor", "Type", "ArcSize", "ArcWidth", "SizeTop", "SizeBottom", "OnTop", "SampleNum", "SizeRed", "SizeBlue", "SizeRedMinusBlue"])

        for type_name in tqdm(types):
            num_samples = num_samples_map[type_name]
            for idx in tqdm(range(num_samples), leave=False):
                for top_color in on_top_cols:
                    arc_curvature, width, size_red, size_blue = get_random_params()
                    size_blue = size_red if type_name == "illusory" else size_blue
                    img = ds.generate_jastrow_illusion(
                        arc_curvature, width, size_red, size_blue, top_color, type_stimulus=type_name,
                    )
                    unique_hex = uuid.uuid4().hex[:8]
                    path = (
                        Path(type_name)
                        / ("" if type_name in ["random", "random_same_size"] else f"{top_color}_on_top")
                        / f"red{size_red:.2f}_blue{size_blue:.2f}_{idx}_{unique_hex}.png"
                    )
                    img.save(output_folder / path)
                    writer.writerow([
                        path, config.background_color, type_name, arc_curvature, width,
                        size_red, size_blue,
                        "none" if type_name in ["random", "random_same_size"] else top_color,
                        idx, size_red, size_blue, size_red - size_blue,
                    ])

    return str(output_folder)
