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

    canvas_size = config.canvas_size
    output_folder = Path(config.output_folder)

    ds = DrawCompletion(
        background=config.background_color,  # type: ignore
        canvas_size=canvas_size,  # type: ignore
        antialiasing=config.antialiasing,
    )

    for cond in ["occluded", "non_occluded", "notched"]:
        (output_folder / cond).mkdir(exist_ok=True, parents=True)


    check_square_fully_in_canvas = lambda cs: (
        cs[0] - side_square // 2 > 0
        and cs[0] + side_square // 2 < canvas_size[0]
        and cs[1] - side_square // 2 > 0
        and cs[1] + side_square // 2 < canvas_size[1]
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "SampleID",
                "Condition",
                "ManipulatedShape",
                "Path",
                "BackgroundColor",
                "CenterCircleLocation",
                "CenterSquareLocation",
                "RadiusCircle",
                "SideSquare",
                "SampleId",
                "CircleColor",
                "SquareColor",
            ]
        )

        pbar = tqdm(total=config.num_samples, dynamic_ncols=True)
        completed_samples = 0

        while completed_samples < config.num_samples:
            radius_circle = random.randint(20, 40)
            side_square = np.ceil(radius_circle * 1.8).astype(int)
            diagonal_square = side_square * np.sqrt(2)

            base_center_dist = (radius_circle + diagonal_square / 2)  # minimum distance between centers so that there is no overalap

            while True:
                # determine circle position
                base_center_circle = np.random.uniform(0 + radius_circle, 224 - radius_circle, size=2).round().astype(int)

                # trace position of the non occluding square along a random direction
                theta = np.random.uniform(0, np.pi * 2)
                center_square_dir = np.array([np.sin(theta), np.cos(theta)])

                factor = np.random.uniform(1.05, 1.2) * base_center_dist # scale the distance between centers by a random value
                base_center_square = np.ceil(base_center_circle + factor * center_square_dir).astype(int)

                # if the non-occluding square is in the canvas, accept the image
                if check_square_fully_in_canvas(base_center_square):
                    break

            pbar.update(1)

            # generate colors
            circle_col = (
                generate_random_color() if config.circle_color == "random" else config.circle_color
            )
            square_col = (
                generate_random_color() if config.square_color == "random" else config.square_color
            )

            def write_row(path, condition, manipulated_shape, center_circle, center_square):
                writer.writerow(
                    [
                        completed_samples,
                        condition,
                        manipulated_shape,
                        path,
                        ds.background,
                        [i.item() for i in center_circle],
                        [i.item() for i in center_square],
                        radius_circle,
                        side_square,
                        completed_samples,
                        circle_col,
                        square_col,
                    ]
                )

            def save_image(img, condition, top_shape):
                unique_hex = uuid.uuid4().hex[:8]
                name = (top_shape + '_') if condition != 'non_occluded' else ''
                path = Path(condition) / f"{completed_samples}_{name}{unique_hex}.png"
                img.save(output_folder / path)
                return path

            # Generate base image
            base_img = ds.draw(
                base_center_circle,
                base_center_square,
                circle_col,
                square_col,
                radius_circle,
                side_square,
                center_notched=None,
                top='s',
            )
            path = save_image(base_img, 'non_occluded', None)
            write_row(path, 'non_occluded', None, base_center_circle, base_center_square)

            # Generate conditions for the square-on-top-variant
            # now determine the position of an occluded circle variant and generate it
            factor = np.random.uniform(0.25, 0.75) * base_center_dist
            center_occluding_square = np.ceil(
                base_center_circle + factor * center_square_dir
            ).astype(int)

            circle_occluded_img = ds.draw(
                base_center_circle,
                center_occluding_square,
                circle_col,
                square_col,
                radius_circle,
                side_square,
                center_notched=None,
                top='s',
            )
            path = save_image(circle_occluded_img, 'occluded', 'circle')
            write_row(path, 'occluded', 'square', base_center_circle, center_occluding_square)

            # with the same information, we can generate the notched variant
            circle_notched_img = ds.draw(
                base_center_circle,
                base_center_square,
                circle_col,
                square_col,
                radius_circle,
                side_square,
                center_notched=center_occluding_square,
                top='s',
            )
            path = save_image(circle_notched_img, 'notched', 'circle')
            write_row(path, 'notched', 'square', base_center_circle, base_center_square)


            # Generate conditions for the circle-on-top-variant
            # determine the position and generate an occluded square variant
            factor = np.random.uniform(0.25, 0.75) * base_center_dist
            center_occluding_circle = np.ceil(
                base_center_square - factor * center_square_dir
            ).astype(int)

            square_occluded_img = ds.draw(
                center_occluding_circle,
                base_center_square,
                circle_col,
                square_col,
                radius_circle,
                side_square,
                center_notched=None,
                top='c',
            )
            path = save_image(square_occluded_img, 'occluded', 'square')
            write_row(path, 'occluded', 'square', center_occluding_circle, base_center_square)

            square_notched_img = ds.draw(
                base_center_circle,
                base_center_square,
                circle_col,
                square_col,
                radius_circle,
                side_square,
                center_notched=center_occluding_circle,
                top='c',
            )
            path = save_image(square_notched_img, 'notched', 'square')
            write_row(path, 'notched', 'square', base_center_circle, base_center_square)

            # base_img.show()
            # square_occluded_img.show()
            # square_notched_img.show()

            completed_samples += 1

    return str(output_folder)
