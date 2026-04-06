"""silhouettes dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generate_datasets.shape_and_object_recognition.silhouettes.generate_dataset import DrawLinedrawings
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class SilhouettesConfig(GeneratorConfig):
    """config for silhouettes dataset."""
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    image_input_folder: str = field(default="assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with images"})
    input_image_type: str = field(default="linedrawings", metadata={"choices": ["linedrawings", "silhouettes"], "label": "input image type"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/silhouettes", metadata={"label": "output folder"})


@register("silhouettes", "shape_recognition")
@generator(SilhouettesConfig)
def generate_all(config: SilhouettesConfig):
    """generate silhouettes dataset from source images."""
    output_folder = Path(config.output_folder)
    image_input_folder = Path(config.image_input_folder)

    all_categories = [p.stem for p in image_input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        input_image_type=config.input_image_type,
    )

    jpg_files = list(image_input_folder.rglob("*.jpg"))
    png_files = list(image_input_folder.rglob("*.png"))
    image_files = jpg_files + png_files

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])

        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            image_name = img_path.stem
            img = ds.get_linedrawings(img_path)
            unique_hex = uuid.uuid4().hex[:8]
            path = Path(class_name) / f"{image_name}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)
