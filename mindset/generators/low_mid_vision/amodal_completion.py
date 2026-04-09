"""amodal completion dataset generator."""

import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL.ImageDraw import Draw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing, generate_random_color


def vector_length(s, theta):
    """compute vector length for square-circle overlap geometry."""
    theta_rad = math.radians(theta)
    use_cosine_ranges = [(0, 45), (135, 180), (180, 225), (315, 360)]
    for range_start, range_end in use_cosine_ranges:
        if range_start <= theta < range_end:
            return 0.5 * s / abs(math.cos(theta_rad))
    return 0.5 * s / abs(math.sin(theta_rad))


class DrawCompletion(DrawStimuli):
    """draw amodal completion stimuli with circle and square."""

    def draw(
        self,
        center_circle,
        center_square,
        circle_color,
        square_color,
        radius_circle,
        side_square,
        center_notched=None,
        top="s",
    ):
        """render a single amodal completion image."""
        img = self.create_canvas()
        draw = Draw(img)
        x_s, y_s = center_square
        x_c, y_c = center_circle

        if top == "s":
            draw.ellipse(
                [
                    (x_c - radius_circle, y_c - radius_circle),
                    (x_c + radius_circle, y_c + radius_circle),
                ],
                outline=None,
                fill=tuple(circle_color),
            )
            if center_notched is not None:
                x_n, y_n = center_notched
                draw.rectangle(
                    [
                        (x_n - side_square / 2, y_n - side_square / 2),
                        (x_n + side_square / 2, y_n + side_square / 2),
                    ],
                    outline=self.background,
                    fill=self.background,
                )
                center_notched = False

        draw.rectangle(
            [
                (x_s - side_square / 2, y_s - side_square / 2),
                (x_s + side_square / 2, y_s + side_square / 2),
            ],
            outline=None,
            fill=tuple(square_color),
        )

        if top == "c":
            if center_notched is not None:
                x_n, y_n = center_notched
                draw.ellipse(
                    [
                        (x_n - radius_circle, y_n - radius_circle),
                        (x_n + radius_circle, y_n + radius_circle),
                    ],
                    outline=None,
                    fill=self.background,
                )
            draw.ellipse(
                [
                    (x_c - radius_circle, y_c - radius_circle),
                    (x_c + radius_circle, y_c + radius_circle),
                ],
                outline=None,
                fill=tuple(circle_color),
            )

        return apply_antialiasing(img) if self.antialiasing else img


@dataclass
class AmodalCompletionConfig(GeneratorConfig):
    """config for amodal completion dataset."""

    num_samples: int = field(
        default=5,
        metadata={"min": 1, "max": 50000, "step": 1, "label": "number of samples"},
    )
    circle_color: list = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "circle color (RGB)"},
    )
    square_color: list = field(
        default_factory=lambda: [0, 0, 0], metadata={"label": "square color (RGB)"}
    )
    background_color: list = field(
        default_factory=lambda: [100, 100, 100],
        metadata={"label": "background color (RGB)"},
    )
    output_folder: str = field(
        default="data/low_mid_vision/amodal_completion",
        metadata={"label": "output folder"},
    )


@register("amodal_completion", "low_mid_vision")
@generator(AmodalCompletionConfig)
def generate_all(config: AmodalCompletionConfig):
    """generate amodal completion dataset."""
    output_folder = Path(config.output_folder)
    ds = DrawCompletion(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    for cond in ["occlusion", "no_occlusion", "notched"]:
        (output_folder / cond).mkdir(exist_ok=True, parents=True)

    top_shapes = ["s", "c"]
    canvas_size = config.canvas_size

    check_square_fully_in_canvas = lambda cs: (
        cs[0] - side_square // 2 > 0
        and cs[0] + side_square // 2 < canvas_size[0]
        and cs[1] - side_square // 2 > 0
        and cs[1] + side_square // 2 < canvas_size[1]
    )
    get_center_square = lambda theta, ll: (
        np.cos(theta) * ll + center_circle[0],
        np.sin(theta) * ll + center_circle[1],
    )

    pbar = tqdm(total=config.num_samples, dynamic_ncols=True)
    completed_samples = 0

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "BackgroundColor",
                "TopShape",
                "CenterCircleLocation",
                "CenterSquareLocation",
                "RadiusCircle",
                "SideSquare",
                "SampleId",
                "CircleColor",
                "SquareColor",
            ]
        )

        while completed_samples < config.num_samples:
            radius_circle = random.randint(20, 40)
            side_square = radius_circle * 1.5
            diagonal_square = side_square * np.sqrt(2)
            center_circle = np.array(canvas_size) // 2
            theta = np.random.uniform(0, np.pi * 2)

            ll = np.random.uniform(
                diagonal_square / 2 + radius_circle,
                np.sqrt(canvas_size[1] ** 2 + canvas_size[0] ** 2),
            )
            center_square = get_center_square(theta, ll)
            if not check_square_fully_in_canvas(center_square):
                continue
            pbar.update(1)

            for top_shape in top_shapes:
                circle_col = (
                    generate_random_color()
                    if config.circle_color == "random"
                    else config.circle_color
                )
                square_col = (
                    generate_random_color()
                    if config.square_color == "random"
                    else config.square_color
                )
                img = ds.draw(
                    center_circle,
                    center_square,
                    circle_col,
                    square_col,
                    radius_circle,
                    side_square,
                    center_notched=None,
                    top=top_shape,
                )
                unique_hex = uuid.uuid4().hex[:8]
                path = Path("no_occlusion") / f"{top_shape}_{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        "no_occlusion",
                        ds.background,
                        top_shape,
                        center_circle,
                        center_square,
                        radius_circle,
                        side_square,
                        completed_samples,
                        circle_col,
                        square_col,
                    ]
                )

            max_dist_occluded = vector_length(side_square, theta) + radius_circle * 0.8
            ll = np.random.uniform(radius_circle // 1.2, max_dist_occluded)

            for notched in [True, False]:
                for top_shape in top_shapes:
                    center_circle = np.array(canvas_size) // 2
                    center_square = get_center_square(theta, ll)
                    if notched:
                        if top_shape == "c":
                            center_notched = center_circle
                            center_circle = get_center_square(
                                theta, -radius_circle * 0.4
                            )
                        else:
                            center_notched = center_square
                            center_square = get_center_square(theta, side_square * 1.4)
                    else:
                        center_notched = None
                    img = ds.draw(
                        center_circle,
                        center_square,
                        circle_col,
                        square_col,
                        radius_circle,
                        side_square,
                        center_notched=center_notched,
                        top=top_shape,
                    )
                    unique_hex = uuid.uuid4().hex[:8]
                    path = (
                        Path("notched" if notched else "occlusion")
                        / f"{completed_samples}_{top_shape}_{unique_hex}.png"
                    )
                    img.save(output_folder / path)
                    writer.writerow(
                        [
                            path,
                            "notched" if notched else "occlusion",
                            ds.background,
                            top_shape,
                            center_circle,
                            center_square,
                            radius_circle,
                            side_square,
                            completed_samples,
                            circle_col,
                            square_col,
                        ]
                    )

            completed_samples += 1

    return str(output_folder)
