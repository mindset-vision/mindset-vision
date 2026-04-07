import math
import random

import numpy as np
from PIL import ImageDraw

from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.misc import apply_antialiasing


def draw_arrow(draw, pos, theta, angle_arrow, arrow_length, width, color):
    x, y = pos
    arrow_theta1 = theta - angle_arrow
    arrow_theta2 = theta + angle_arrow

    arrow_end_x1 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta1)))
    arrow_end_y1 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta1)))
    arrow_end_x2 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta2)))
    arrow_end_y2 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta2)))

    # Draw the arrow lines
    draw.line([(x, y), (arrow_end_x1, arrow_end_y1)], fill=color, width=width)
    draw.line([(x, y), (arrow_end_x2, arrow_end_y2)], fill=color, width=width)


class DrawMullerLyer(DrawStimuli):
    def generate_illusion(
        self,
        line_position_rel,
        line_length,
        arrow_angle,
        arrow_cap_angle,
        arrow_length,
        type,
    ):
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
