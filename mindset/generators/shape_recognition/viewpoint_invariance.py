"""viewpoint invariance dataset generator."""
import csv
import glob
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generate_datasets.shape_and_object_recognition.viewpoint_invariance.generate_dataset import DrawETH
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.misc import check_download_ETH_80_dataset


@dataclass
class ViewpointInvarianceConfig(GeneratorConfig):
    """config for viewpoint invariance dataset."""
    eth_80_folder: str = field(default="assets/ETH_80", metadata={"label": "ETH-80 dataset folder"})
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
            match = re.search(r"([a-zA-Z]+)\d+-0*(\d+)-0*(\d+).png$", os.path.basename(img_path))

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
