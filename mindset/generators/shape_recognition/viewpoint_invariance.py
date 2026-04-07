"""viewpoint invariance dataset generator."""
import csv
import glob
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.drawing_utils import DrawStimuli, resize_image_keep_aspect_ratio
from mindset.utils.misc import apply_antialiasing, check_download_ETH_80_dataset


class DrawETH(DrawStimuli):
    """draws ETH-80 dataset images with background removal."""

    def __init__(self, obj_longest_side, map_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.map_path = map_path

    def create_ETH(self, img_path):
        """load an ETH-80 image, crop to object bounds, and paste on canvas."""
        path_parts = Path(img_path).parts
        desired_path = str(Path(*path_parts[-3:])).rstrip(".png")
        map_path = f"{self.map_path}/{desired_path}-map.png"

        map_pil = Image.open(map_path).convert("L")

        mask_array = np.array(map_pil)

        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        cropped_image = Image.open(img_path).crop((xmin, ymin, xmax, ymax))
        cropped_map_pil = map_pil.crop((xmin, ymin, xmax, ymax))

        canvas_only_obj = self.create_canvas(size=cropped_image.size)
        canvas_only_obj.paste(cropped_image, mask=cropped_map_pil)
        canvas_only_obj = Image.fromarray(
            resize_image_keep_aspect_ratio(
                np.array(canvas_only_obj), self.obj_longest_side
            )
        )

        canvas = self.create_canvas()
        paste_position = (
            (canvas.size[0] - canvas_only_obj.size[0]) // 2,
            (canvas.size[1] - canvas_only_obj.size[1]) // 2,
        )
        canvas.paste(canvas_only_obj, paste_position)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class ViewpointInvarianceConfig(GeneratorConfig):
    """config for viewpoint invariance dataset."""
    eth_80_folder: str = field(default="mindset/assets/ETH_80", metadata={"label": "ETH-80 dataset folder"})
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    azimuth_lim: list = field(default_factory=lambda: [0, 365], metadata={"label": "azimuth limits"})
    inclination_lim: list = field(default_factory=lambda: [30, 90], metadata={"label": "inclination limits"})
    output_folder: str = field(default="data/shape_and_object_recognition/viewpoint_invariance", metadata={"label": "output folder"})


@register("viewpoint_invariance", "shape_recognition")
@generator(ViewpointInvarianceConfig)
def generate_all(config: ViewpointInvarianceConfig):
    """generate viewpoint invariance dataset from ETH-80."""
    output_folder = Path(config.output_folder)

    check_download_ETH_80_dataset(destination_dir=config.eth_80_folder)

    ds = DrawETH(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        map_path=config.eth_80_folder + "/maps/",
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "ObjectID", "Azimuth", "Inclination", "BackgroundColor"])

        all_images = glob.glob(config.eth_80_folder + "/images/*/*/*.png", recursive=True)

        for img_path in tqdm(all_images):
            class_num = Path(img_path).parts[2]
            object_id = int(Path(img_path).parts[3])
            match = re.search(r"([a-zA-Z]+)\d+-0*(\d+)-0*(\d+).png$", Path(img_path).name)

            class_name = match.group(1)
            inclination = int(match.group(2))
            azimuth = int(match.group(3))

            if not (config.inclination_lim[0] <= inclination <= config.inclination_lim[1]):
                continue
            if not (config.azimuth_lim[0] <= azimuth <= config.azimuth_lim[1]):
                continue

            (output_folder / class_name / str(object_id)).mkdir(parents=True, exist_ok=True)
            img = ds.create_ETH(img_path)
            unique_hex = uuid.uuid4().hex[:8]

            img_save_path = Path(class_name) / str(object_id) / f"{Path(img_path).stem}_{unique_hex}.png"
            img.save(output_folder / img_save_path)
            writer.writerow([img_save_path, class_name, object_id, azimuth, inclination, ds.background])

    return str(output_folder)
