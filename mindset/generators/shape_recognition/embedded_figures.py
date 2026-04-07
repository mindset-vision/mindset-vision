"""embedded figures dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generate_datasets.shape_and_object_recognition.embedded_figures.utils import DrawEmbeddedFigures
from mindset.generate_datasets.shape_and_object_recognition.embedded_figures.generate_dataset import polys
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class EmbeddedFiguresConfig(GeneratorConfig):
    """config for embedded figures dataset."""
    num_samples: int = field(default=100, metadata={"min": 1, "max": 10000, "step": 1, "label": "samples per polygon"})
    shape_size: float = field(default=45, metadata={"min": 10, "max": 200, "step": 5, "label": "shape size (px)"})
    output_folder: str = field(default="data/shape_and_object_recognition/embedded_figures", metadata={"label": "output folder"})


@register("embedded_figures", "shape_recognition")
@generator(EmbeddedFiguresConfig)
def generate_all(config: EmbeddedFiguresConfig):
    """generate embedded figures dataset."""
    output_folder = Path(config.output_folder)

    shapes = list((n, p) for n, p in enumerate(polys))

    ds = DrawEmbeddedFigures(
        shape_size=config.shape_size,
        canvas_size=config.canvas_size,
        background=config.background_color,
        antialiasing=config.antialiasing,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "PolygonId", "BackgroundColor", "IterNum"])

        for cond in tqdm(["polygon", "embedded_polygon"]):
            n_samples = 1 if cond == "polygon" else config.num_samples

            for s in tqdm(shapes, leave=False):
                shape_name, shape_points = str(s[0]), s[1]
                class_folder = output_folder / cond / shape_name
                class_folder.mkdir(parents=True, exist_ok=True)

                for i in tqdm(range(n_samples)):
                    if cond == "polygon":
                        img = ds.draw_shape(shape_points, extend_lines=False, num_shift_lines=0, num_rnd_lines=0)
                    else:
                        img = ds.draw_shape(shape_points, extend_lines=True, num_shift_lines=10, num_rnd_lines=10)

                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = Path(cond) / shape_name / f"{unique_hex}.png"
                    img.save(output_folder / img_path)
                    writer.writerow([img_path, cond, shape_name, ds.background, i])

    return str(output_folder)
