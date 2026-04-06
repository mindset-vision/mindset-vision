"""thatcher illusion face dataset generator."""
import csv
import pathlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from tqdm import tqdm

from mindset.generate_datasets.visual_illusions.thatcher_illusion_face.utils import (
    get_image_facial_landmarks,
    get_bounding_rectangle,
    apply_thatcher_effect_on_image,
)
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class ThatcherFaceConfig(GeneratorConfig):
    """config for thatcher illusion face dataset."""
    face_folder: str = field(default="assets/celebA_sample/normal", metadata={"label": "face folder"})
    output_folder: str = field(default="data/visual_illusions/thatcher_face", metadata={"label": "output folder"})


@register("thatcher_face", "visual_illusions")
@generator(ThatcherFaceConfig)
def generate_all(config: ThatcherFaceConfig):
    """generate thatcher illusion face dataset."""
    output_folder = Path(config.output_folder)
    face_folder = Path(config.face_folder)

    conditions = ["straight", "inverted", "thatcherized_straight", "thatcherized_inverted"]
    for cond in conditions:
        (output_folder / cond).mkdir(parents=True, exist_ok=True)

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("assets/lbfmodel.yaml")

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Condition", "FaceId"])

        for idx, image_path in tqdm(enumerate(face_folder.glob("*"))):
            image_facial_landmarks = get_image_facial_landmarks(image_path, facemark)
            image_name = image_path.stem
            if (
                not image_facial_landmarks
                or len(image_facial_landmarks) == 0
                or len(image_facial_landmarks) != 68
            ):
                continue

            left_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[36:42])
            right_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[42:48])
            mouth_rectangle = get_bounding_rectangle(image_facial_landmarks[48:68])

            cv_image = apply_thatcher_effect_on_image(
                str(image_path),
                np.array(left_eye_rectangle).astype(int),
                np.array(right_eye_rectangle).astype(int),
                np.array(mouth_rectangle).astype(int),
            )

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.fromarray(cv_image)
            ).save(output_folder / "thatcherized_straight" / f"{idx}.png")
            writer.writerow([pathlib.Path("thatcherized_straight") / f"{image_name}.png", "thatcherized_straight", idx])

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.fromarray(cv2.flip(cv_image, 0))
            ).save(output_folder / "thatcherized_inverted" / f"{idx}.png")
            writer.writerow([pathlib.Path("thatcherized_inverted") / f"{image_name}.png", "thatcherized_inverted", idx])

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.open(image_path)
            ).save(output_folder / "straight" / f"{image_name}.png")
            writer.writerow([pathlib.Path("straight") / f"{idx}.png", "straight", idx])

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.open(image_path).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            ).save(output_folder / "inverted" / f"{image_name}.png")
            writer.writerow([pathlib.Path("inverted") / f"{image_name}.png", "inverted", idx])

    return str(output_folder)
