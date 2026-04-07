import random

import numpy as np
from PIL import ImageDraw

from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.misc import apply_antialiasing


polys = [
    [
        (57.24191632273875, 56.91324204660534),
        (67.83138439623637, 82.164285681352),
        (32.38511628951794, 100.0),
        (30.804578657842352, 73.57800097038611),
        (39.790659001314935, 49.086636164923235),
        (0.0, 13.7323319073141),
        (94.72413166442718, 12.682935522851094),
        (57.24191632273875, 56.91324204660534),
    ],
    [
        (81.48562751057788, 66.09981880428393),
        (37.92965499595768, 77.0744200152708),
        (0.0, 68.97209639097005),
        (44.9389941502286, 34.540151429883004),
        (53.987122366845064, 49.564278276404),
        (84.56537465354238, 18.919456992657267),
        (100.0, 18.699487501673413),
        (81.48562751057788, 66.09981880428393),
    ],
    [
        (35.97327747005214, 45.791246316377155),
        (11.785178465160033, 100.0),
        (27.099520693118553, 42.4229653007916),
        (14.996849174804217, 38.15001140882252),
        (0.0, 14.776880790143396),
        (72.18476581016824, 0.4054647678679052),
        (88.90514143051871, 30.360919479254665),
        (35.97327747005214, 45.791246316377155),
    ],
    [
        (71.10439849304493, 58.11478725863726),
        (61.9937434893466, 60.73142522572432),
        (92.85669179071442, 92.82864165606136),
        (88.72099545300055, 99.99999999999999),
        (0.0, 39.73040200323956),
        (21.971533124959556, 37.83178086979938),
        (24.413419434667112, 10.00396765284736),
        (71.10439849304493, 58.11478725863726),
    ],
    [
        (99.99999999999999, 68.62298822991318),
        (1.0462848992740792, 74.25606525894862),
        (39.38222527207329, 41.822553930272285),
        (28.55589809787002, 40.16660427664721),
        (13.075399744528019, 0.0),
        (52.12530258921337, 38.638493749406464),
        (79.2845572364489, 40.26425593476456),
        (99.99999999999999, 68.62298822991318),
    ],
]


def calculate_centroid(points):
    """calculate centroid of a set of points."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = (max(x_coords) + min(x_coords)) / 2
    centroid_y = (max(y_coords) + min(y_coords)) / 2
    return centroid_x, centroid_y


def extend_line(line, factor):
    """extend a line segment by a given factor."""
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    x2_new, y2_new = x1 + dx * factor, y1 + dy * factor
    x1_new, y1_new = x1 - dx * (factor - 1), y1 - dy * (factor - 1)
    return (x1_new, y1_new, x2_new, y2_new)


def center_and_scale(points, canvas_size, shape_size):
    """center and scale polygon points to fit the canvas."""
    scaled_points = [
        (
            p[0]
            * shape_size
            / (max([p[0] for p in points]) - min([p[0] for p in points])),
            p[1]
            * shape_size
            / (max([p[1] for p in points]) - min([p[1] for p in points])),
        )
        for p in points
    ]
    centroid = calculate_centroid(scaled_points)

    translated_points = [
        (
            p[0] + canvas_size[0] / 2 - centroid[0],
            p[1] + canvas_size[1] / 2 - centroid[1],
        )
        for p in scaled_points
    ]

    return translated_points


def shift_line(line, width, height, max_shift):
    """shift a line by a random amount within bounds."""
    x1, y1, x2, y2 = line

    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)

    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)

    return x1, y1, x2, y2


def draw_number_exclude_range(min, max, not_min, not_max):
    """draw a random int in [min, max] excluding [not_min, not_max]."""
    while True:
        num = random.randint(min, max)
        if num < not_min or num > not_max:
            return num


class DrawEmbeddedFigures(DrawStimuli):
    """draws embedded figures with camouflaging lines."""

    def __init__(self, shape_size, *args, **kwargs):
        self.shape_size = shape_size
        super().__init__(*args, **kwargs)

    def draw_shape(
        self,
        original_points,
        extend_lines=False,
        num_shift_lines=5,
        num_rnd_lines=0,
    ):
        """draw a polygon with optional camouflage lines."""
        original_canvas_size = self.canvas_size
        self.canvas_size = tuple(np.array(self.canvas_size))
        canvas = self.create_canvas()
        draw = ImageDraw.Draw(canvas)
        points = center_and_scale(original_points, self.canvas_size, self.shape_size)
        points = [tuple(np.round(np.array(i)).astype(int)) for i in points]
        width, height = canvas.size

        for i in range(len(points) - 1):
            line = points[i] + points[i + 1]
            line = extend_line(line, self.canvas_size[0]) if extend_lines else line
            draw.line(line, **self.line_args)

        for i in range(num_shift_lines):
            i = random.choice(range(len(points) - 1))
            line = shift_line(
                points[i] + points[i + 1],
                width,
                height,
                max_shift=max((width // 2, height // 2)),
            )
            draw.line(
                extend_line(line, 100) if extend_lines else line, **self.line_args
            )

        for _ in range(num_rnd_lines):
            if random.random() < 0.5:
                x1, y1 = random.random() * width, 0
                x2, y2 = random.random() * width, height
            else:
                x1, y1 = 0, random.random() * height
                x2, y2 = width, random.random() * height
            draw.line((x1, y1, x2, y2), **self.line_args)

        self.canvas_size = original_canvas_size
        return apply_antialiasing(canvas) if self.antialiasing else canvas
