"""global change dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generate_datasets.shape_and_object_recognition.global_change.generate_dataset import DrawLinedrawings
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class GlobalChangeConfig(GeneratorConfig):
    """config for global change dataset."""
    object_longest_side: int = field(default=120, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    image_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with images"})
    convert_to_silhouettes: int = field(default=0, metadata={"choices": [0, 1], "label": "convert to silhouettes"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/global_change", metadata={"label": "output folder"})


@register("global_change", "shape_recognition")
@generator(GlobalChangeConfig)
def generate_all(config: GlobalChangeConfig):
    """generate global change dataset with whole, fragmented, and frankenstein conditions."""
    output_folder = Path(config.output_folder)
    image_input_folder = Path(config.image_input_folder)

    all_categories = [p.stem for p in image_input_folder.glob("*")]
    conditions = ["whole", "fragmented", "frankenstein"]

    for c in conditions:
        for cat in all_categories:
            (output_folder / c / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        convert_to_silhouettes=config.convert_to_silhouettes,
    )

    image_files = sorted(image_input_folder.rglob("*.jpg")) + sorted(image_input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            for t in conditions:
                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.get_linedrawings(img_path, type=t)
                path = Path(t) / class_name / f"{image_name}.png"
                img.save(output_folder / path)
                writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)
