"""muller-lyer illusion dataset generator."""

import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


def draw_arrow(draw, pos, theta, angle_arrow, arrow_length, width, color):
    """draw a muller-lyer arrow at a given position."""
    x, y = pos
    arrow_theta1 = theta - angle_arrow
    arrow_theta2 = theta + angle_arrow

    arrow_end_x1 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta1)))
    arrow_end_y1 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta1)))
    arrow_end_x2 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta2)))
    arrow_end_y2 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta2)))

    draw.line([(x, y), (arrow_end_x1, arrow_end_y1)], fill=color, width=width)
    draw.line([(x, y), (arrow_end_x2, arrow_end_y2)], fill=color, width=width)


class DrawMullerLyer(DrawStimuli):
    """draws muller-lyer illusion stimuli."""

    def generate_illusion(
        self,
        line_position_rel,
        line_length,
        arrow_angle,
        arrow_cap_angle,
        arrow_length,
        type,
    ):
        """generate a muller-lyer illusion image."""
        get_arrow_rnd_pos = lambda: (
            random.randint(arrow_length, self.canvas_size[0] - arrow_length),
            random.randint(arrow_length, self.canvas_size[1] - arrow_length),
        )

        img = self.create_canvas()
        d = ImageDraw.Draw(img)
        line_position = tuple(
            (np.array(line_position_rel) * self.canvas_size).astype(int)
        )
        if type == "scrambled":
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle + 180,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            d.line(
                (
                    np.round(line_position[0] - line_length // 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length // 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )
        else:
            d.line(
                (
                    np.round(line_position[0] - line_length / 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length / 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )
            draw_arrow(
                d,
                (line_position[0] - line_length // 2, line_position[1]),
                theta=(180 if type == "outward" else 0),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                (line_position[0] + line_length // 2, line_position[1]),
                theta=(0 if type == "outward" else 180),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
        return apply_antialiasing(img) if self.antialiasing else img


@dataclass
class MullerLyerConfig(GeneratorConfig):
    """config for muller-lyer illusion dataset."""

    num_samples_scrambled: int = field(
        default=5000,
        metadata={"min": 1, "max": 50000, "step": 10, "label": "scrambled samples"},
    )
    num_samples_illusory: int = field(
        default=500,
        metadata={"min": 1, "max": 5000, "step": 10, "label": "illusory samples"},
    )
    output_folder: str = field(
        default="data/visual_illusions/muller_lyer", metadata={"label": "output folder"}
    )


@register("muller_lyer", "visual_illusions")
@generator(MullerLyerConfig)
def generate_all(config: MullerLyerConfig):
    """generate muller-lyer illusion dataset."""
    output_folder = Path(config.output_folder)
    canvas_size = config.canvas_size
    conditions = ["scrambled", "inward", "outward"]
    for d in conditions:
        (output_folder / d).mkdir(exist_ok=True, parents=True)

    ds = DrawMullerLyer(
        background=config.background_color,
        canvas_size=canvas_size,
        width=1,
    )

    def get_random_params():
        """generate random parameters for one stimulus."""
        line_length = random.randint(
            int(canvas_size[0] * 0.25), int(canvas_size[0] * 0.67)
        )
        arrow_length = random.randint(
            int(canvas_size[0] * 0.07), int(canvas_size[0] * 0.134)
        )
        line_position = tuple(
            np.array(
                [
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[0] - arrow_length - line_length // 2,
                    ),
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[1] - arrow_length - line_length // 2,
                    ),
                ]
            )
            / canvas_size
        )
        cap_arrows_angle = random.randint(
            int(canvas_size[0] * 0.045), int(canvas_size[1] * 0.2)
        )
        angle_arrow = random.randint(0, 360)
        return line_length, line_position, arrow_length, cap_arrows_angle, angle_arrow

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "BackgroundColor",
                "LineLength",
                "LinePosition",
                "ArrowLength",
                "CapArrowAngle",
                "ArrowAngle",
                "IterNum",
            ]
        )

        for c in conditions:
            n = (
                config.num_samples_scrambled
                if c == "scrambled"
                else config.num_samples_illusory
            )
            for i in tqdm(range(n), desc=c):
                (
                    line_length,
                    line_position,
                    arrow_length,
                    cap_arrow_angle,
                    arrow_angle,
                ) = get_random_params()
                img = ds.generate_illusion(
                    line_position_rel=line_position,
                    line_length=line_length,
                    arrow_angle=arrow_angle,
                    arrow_cap_angle=cap_arrow_angle,
                    arrow_length=arrow_length,
                    type=c,
                )
                if config.antialiasing:
                    img = apply_antialiasing(img)
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        c,
                        ds.background,
                        line_length,
                        line_position,
                        arrow_length,
                        cap_arrow_angle,
                        arrow_angle,
                        i,
                    ]
                )

    return str(output_folder)
