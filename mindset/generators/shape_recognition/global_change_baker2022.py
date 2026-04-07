"""global change baker2022 dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.drawing.baker_stimuli import DrawBakerStimuli
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class GlobalChangeBaker2022Config(GeneratorConfig):
    """config for global change baker2022 dataset."""
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/global_change_baker2022", metadata={"label": "output folder"})


@register("global_change_baker2022", "shape_recognition")
@generator(GlobalChangeBaker2022Config)
def generate_all(config: GlobalChangeBaker2022Config):
    """generate global change baker2022 dataset from baker 2022 silhouettes."""
    output_folder = Path(config.output_folder)
    image_input_folder = Path("mindset/assets/baker_2022")

    all_categories = [p.stem for p in image_input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawBakerStimuli(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    image_files = sorted(image_input_folder.rglob("*.jpg")) + sorted(image_input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            type_image = img_path.parent.parts[-2]
            image_name = img_path.stem
            img = ds.get_linedrawings(img_path)
            unique_hex = uuid.uuid4().hex[:8]
            path = Path(type_image) / class_name / f"{image_name}_{unique_hex}.png"
            (output_folder / path).parent.mkdir(exist_ok=True, parents=True)
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)
