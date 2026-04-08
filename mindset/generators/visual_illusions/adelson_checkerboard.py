"""adelson checkerboard illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.drawing.primitives import add_arrow
from mindset.drawing.base import resize_and_paste
from mindset.utils import apply_antialiasing


@dataclass
class AdelsonCheckerboardConfig(GeneratorConfig):
    """config for adelson checkerboard illusion dataset."""
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    steps_arrow: int = field(default=5, metadata={"min": 1, "max": 100, "step": 1, "label": "arrow step size"})
    grayscale_background: int = field(default=0, metadata={"min": 0, "max": 255, "step": 1, "label": "grayscale background"})
    output_folder: str = field(default="data/visual_illusions/adelson_checkerboard", metadata={"label": "output folder"})


@register("adelson_checkerboard", "visual_illusions")
@generator(AdelsonCheckerboardConfig)
def generate_all(config: AdelsonCheckerboardConfig):
    """generate adelson checkerboard illusion dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "all_images").mkdir(parents=True, exist_ok=True)

    canvas_size = config.canvas_size
    img_path = Path("assets") / "adelson_checkerboard" / "nochars_nobg.png"
    original_image = Image.open(img_path)
    img = Image.new("RGBA", canvas_size, (*(config.grayscale_background,) * 3, 0))
    resize_and_paste(original_image, img)
    img = img.convert("L")

    width, height = img.size
    coordinates = [
        (x, y)
        for x in range(config.steps_arrow, width, config.steps_arrow)
        for y in range(config.steps_arrow, height, config.steps_arrow)
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Target Pixel Color", "Target Pixel Location", "BackgroundColor"])

        for coordinate in tqdm(coordinates, colour="green"):
            pixel_color = img.getpixel(coordinate)
            img_copy = img.copy()
            img_copy = add_arrow(img_copy, tuple(coordinate))
            image_path = Path("all_images") / f"{uuid.uuid4().hex[:8]}.png"
            img_copy = apply_antialiasing(img_copy) if config.antialiasing else img_copy
            img_copy.save(output_folder / image_path)
            writer.writerow([str(image_path), pixel_color, coordinate, config.grayscale_background])

    return str(output_folder)
