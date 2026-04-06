"""leuven embedded figures dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generate_datasets.shape_and_object_recognition.leuven_embedded_figures.generate_dataset import (
    load_and_invert,
    get_highest_number,
)
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class LeuvenEmbeddedConfig(GeneratorConfig):
    """config for leuven embedded figures dataset."""
    output_folder: str = field(default="data/shape_and_object_recognition/leuven_embedded_figures", metadata={"label": "output folder"})


@register("leuven_embedded", "shape_recognition")
@generator(LeuvenEmbeddedConfig)
def generate_all(config: LeuvenEmbeddedConfig):
    """generate leuven embedded figures dataset with shapes and context stimuli."""
    output_folder = Path(config.output_folder)
    left_ds = Path("assets") / "leuven_embedded_figures_test"

    figs_to_take = range(0, 16 * 4, 4)
    all_shapes_path = [left_ds / "shapes" / (str(i).zfill(3) + ".png") for i in figs_to_take]
    all_context_path = [left_ds / "context" / (str(i).zfill(3) + "a.png") for i in range(0, 64)]

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
            img = load_and_invert(s, config.canvas_size, config.background_color, config.antialiasing)
            folder = output_folder_shape / str(idx)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow([f"shapes/{str(idx)}/{n + 1}.png", "shapes", idx, config.background_color])

        for idx, s in enumerate(tqdm(all_context_path, leave=False)):
            img = load_and_invert(s, config.canvas_size, config.background_color, config.antialiasing)
            folder = output_folder_context / str(idx // 4)
            n = get_highest_number(folder)
            img.save(folder / f"{n + 1}.png")
            writer.writerow([f"context/{str(idx // 4)}/{n + 1}.png", "context", idx // 4, config.background_color])

    return str(output_folder)
