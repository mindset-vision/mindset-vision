"""thatcher illusion face dataset generator."""

import csv
import pathlib
from dataclasses import dataclass, field
from math import inf
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
from torchvision import transforms
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register

# ---------------------------------------------------------------------------
# drawing helpers (from GPL 3.0 thatcher-effect-dataset-generator)
# ---------------------------------------------------------------------------


def get_image_facial_landmarks(image_path, facemark):
    """detect facial landmarks from an image using opencv facemark."""
    import matplotlib.pyplot as plt

    face_cascade = cv2.CascadeClassifier(
        "mindset/assets/haarcascade_frontalface_default.xml"
    )

    ret = []

    image = plt.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        ok, landmarks = facemark.fit(gray, faces)

        if ok:
            for landmark in landmarks:
                for x, y in np.squeeze(landmark):
                    ret.append((x, y))

    return ret


def get_bounding_rectangle(points):
    """compute bounding rectangle from a list of points."""
    top_left = [inf, inf]
    bottom_right = [-inf, -inf]
    for point in points:
        top_left[0] = min(top_left[0], point[1])
        top_left[1] = min(top_left[1], point[0])
        bottom_right[0] = max(bottom_right[0], point[1])
        bottom_right[1] = max(bottom_right[1], point[0])
    return [top_left, bottom_right]


def flip_subimage_vertically(image, x1, y1, x2, y2):
    """flip a rectangular subimage vertically in-place."""
    mid_x = (x1 + x2) // 2
    for x in range(x1, mid_x):
        for y in range(y1, y2 + 1):
            image[x][y], image[x1 + x2 - x][y] = (
                image[x1 + x2 - x][y].copy(),
                image[x][y].copy(),
            )


def flip_subimage_ellipse_vertically(image, x1, y1, x2, y2):
    """flip an elliptical subimage region vertically in-place."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1 and x1 + x2 - x > x:
                image[x][y], image[x1 + x2 - x][y] = (
                    image[x1 + x2 - x][y].copy(),
                    image[x][y].copy(),
                )


def blur_ellipse_border(image, x1, y1, x2, y2):
    """blur the border region of an elliptical subimage."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.25 and (dx * dx) / (
                a * a
            ) + (dy * dy) / (b * b) >= 0.75:
                image[x][y] = blurred_image[x][y]


def blur_orthogonal_border(image, blurred_image, x1, y1, x2, y2, border_size):
    """blur an orthogonal border strip of the subimage."""
    if x1 == x2:
        for x in range(x1 - border_size, x1 + border_size + 1):
            for y in range(y1, y2 + 1):
                image[x][y] = blurred_image[x][y]
    if y1 == y2:
        for y in range(y1 - border_size, y1 + border_size + 1):
            for x in range(x1, x2 + 1):
                image[x][y] = blurred_image[x][y]


def blur_rectangle_border(image, x1, y1, x2, y2, border_size=2):
    """blur all four borders of a rectangular subimage."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    blur_orthogonal_border(image, blurred_image, x1, y1, x2, y1, border_size)
    blur_orthogonal_border(image, blurred_image, x1, y2, x2, y2, border_size)
    blur_orthogonal_border(image, blurred_image, x1, y1, x1, y2, border_size)
    blur_orthogonal_border(image, blurred_image, x2, y1, x2, y2, border_size)


def flip_subimage_ellipse_vertically_with_border_softening(image, x1, y1, x2, y2):
    """flip elliptical subimage vertically and soften border."""
    flip_subimage_ellipse_vertically(image, x1, y1, x2, y2)
    blur_ellipse_border(image, x1, y1, x2, y2)


def apply_thatcher_effect_on_image(
    input_image_path,
    left_eye_rectangle,
    right_eye_rectangle,
    mouth_rectangle,
):
    """apply the thatcher effect to eyes and mouth regions of a face image."""
    image = cv2.cvtColor(
        cv2.imread(input_image_path),
        cv2.COLOR_BGR2RGB,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        left_eye_rectangle[0][0] - 5,
        left_eye_rectangle[0][1] - 6,
        left_eye_rectangle[1][0] + 7,
        left_eye_rectangle[1][1] + 3,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        right_eye_rectangle[0][0] - 5,
        right_eye_rectangle[0][1] - 3,
        right_eye_rectangle[1][0] + 7,
        right_eye_rectangle[1][1] + 6,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        mouth_rectangle[0][0] - 4,
        mouth_rectangle[0][1] - 5,
        mouth_rectangle[1][0] + 3,
        mouth_rectangle[1][1] + 5,
    )
    return image


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


@dataclass
class ThatcherFaceConfig(GeneratorConfig):
    """config for thatcher illusion face dataset."""

    face_folder: str = field(
        default="mindset/assets/celebA_sample/normal", metadata={"label": "face folder"}
    )
    output_folder: str = field(
        default="data/visual_illusions/thatcher_face",
        metadata={"label": "output folder"},
    )


@register("thatcher_face", "visual_illusions")
@generator(ThatcherFaceConfig)
def generate_all(config: ThatcherFaceConfig):
    """generate thatcher illusion face dataset."""
    output_folder = Path(config.output_folder)
    face_folder = Path(config.face_folder)

    conditions = [
        "straight",
        "inverted",
        "thatcherized_straight",
        "thatcherized_inverted",
    ]
    for cond in conditions:
        (output_folder / cond).mkdir(parents=True, exist_ok=True)

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("mindset/assets/lbfmodel.yaml")

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
            writer.writerow(
                [
                    pathlib.Path("thatcherized_straight") / f"{image_name}.png",
                    "thatcherized_straight",
                    idx,
                ]
            )

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.fromarray(cv2.flip(cv_image, 0))
            ).save(output_folder / "thatcherized_inverted" / f"{idx}.png")
            writer.writerow(
                [
                    pathlib.Path("thatcherized_inverted") / f"{image_name}.png",
                    "thatcherized_inverted",
                    idx,
                ]
            )

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.open(image_path)
            ).save(output_folder / "straight" / f"{image_name}.png")
            writer.writerow([pathlib.Path("straight") / f"{idx}.png", "straight", idx])

            transforms.CenterCrop((config.canvas_size[1], config.canvas_size[0]))(
                Image.open(image_path).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            ).save(output_folder / "inverted" / f"{image_name}.png")
            writer.writerow(
                [pathlib.Path("inverted") / f"{image_name}.png", "inverted", idx]
            )

    return str(output_folder)
