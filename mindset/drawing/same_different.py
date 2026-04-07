import random

import cv2
import numpy as np
from PIL import Image

from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.misc import apply_antialiasing

from mindset.drawing.same_different_utils import (
    get_line_points,
    sample_midpoints_lines,
    svrt_1_points,
)


class DrawSameDifferentStimuli(DrawStimuli):
    """draws same-different task stimuli with various shape types."""

    def svrt_1_img(
        self,
        category=1,
        size1=None,
        size2=None,
        regular=None,
        rotations=None,
        sides=None,
        irregularity=0.5,
        thickness=1,
        color_a=None,
        color_b=None,
        filled=False,
        closed=True,
    ):
        """return an image of a single svrt problem 1 instance."""
        img = np.array(self.create_canvas())
        color_a = self.fill if color_a is None else color_a

        if color_b is None:
            color_b = color_a

        points_a, points_b, _, _ = svrt_1_points(
            category=category,
            radii=(size1, size2),
            sides=sides,
            rotations=rotations,
            regular=regular,
            irregularity=irregularity,
            canvas_size=self.canvas_size,
        )

        poly_a = np.array(points_a, dtype=np.int32)
        poly_b = np.array(points_b, dtype=np.int32)

        poly_new_a = poly_a.reshape((-1, 1, 2))
        poly_new_b = poly_b.reshape((-1, 1, 2))

        if not filled:
            cv2.polylines(
                img, [poly_new_a], isClosed=closed, color=color_a, thickness=thickness
            )
            cv2.polylines(
                img,
                [poly_new_b],
                isClosed=closed,
                color=color_b,
                thickness=thickness,
            )
        else:
            cv2.fillPoly(img, [poly_new_a], color=color_a)
            cv2.fillPoly(img, [poly_new_b], color=color_b)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_straight_lines_sd_diffrot(self, category, size1, size2, line_thickness=1):
        """draw two straight lines with same or different rotations."""
        img = np.array(self.create_canvas())

        rotations = random.sample([0, 45, 90, 135], 2)
        rotation_1 = rotations[0]
        rotation_2 = rotations[1]

        if category == 1:
            rotation_2 = rotation_1

        midpoint_1, midpoint_2 = sample_midpoints_lines(
            sizes=(size1, size2), canvas_size=self.canvas_size
        )

        points_line_1 = get_line_points(
            size=size1, rotation=rotation_1, center=midpoint_1
        )
        points_line_2 = get_line_points(
            size=size2, rotation=rotation_2, center=midpoint_2
        )

        cv2.line(
            img, points_line_1[0], points_line_1[1], self.fill, thickness=line_thickness
        )
        cv2.line(
            img,
            points_line_2[0],
            points_line_2[1],
            self.fill,
            thickness=line_thickness,
        )

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_squares_sd(self, size1, size2, category):
        """draw two squares with same or different sizes."""
        img = np.array(self.create_canvas())

        if category == 1:
            size2 = size1

        x_1 = random.sample(list(range(2, self.canvas_size[0] - (size1 + 2))), 1)[0]
        y_1 = random.sample(list(range(2, self.canvas_size[1] - (size1 + 2))), 1)[0]
        x_2 = random.sample(list(range(2, self.canvas_size[0] - (size2 + 2))), 1)[0]
        y_2 = random.sample(list(range(2, self.canvas_size[1] - (size2 + 2))), 1)[0]
        start_point_1 = (x_1, y_1)
        start_point_2 = (x_2, y_2)
        end_point_1 = (x_1 + size1, y_1 + size1)
        end_point_2 = (x_2 + size2, y_2 + size2)

        img = cv2.rectangle(img, start_point_1, end_point_1, self.fill, 1)
        img = cv2.rectangle(img, start_point_2, end_point_2, self.fill, 1)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_rectangles_sd(self, size1, category):
        """draw two rectangles with same or different proportions."""
        img = np.array(self.create_canvas())
        const_dim = "x" if random.random() > 0.5 else "y"

        if const_dim == "y":
            size_x_1 = size1
            size_x_2 = (
                random.sample([size1 - size1 // 2, size1 + size1 // 2], 1)[0]
                if category == 0
                else size_x_1
            )
            size_y_1 = size1
            size_y_2 = size_y_1
        else:
            size_y_1 = size1
            size_y_2 = (
                random.sample([size1 - size1 // 2, size1 + size1 // 2], 1)[0]
                if category == 0
                else size_y_1
            )
            size_x_1 = size1
            size_x_2 = size_x_1

        x_1 = random.sample(list(range(2, self.canvas_size[0] - (size_x_1 + 2))), 1)[0]
        y_1 = random.sample(list(range(2, self.canvas_size[1] - (size_y_1 + 2))), 1)[0]
        x_2 = random.sample(list(range(2, self.canvas_size[0] - (size_x_2 + 2))), 1)[0]
        y_2 = random.sample(list(range(2, self.canvas_size[1] - (size_y_2 + 2))), 1)[0]
        start_point_1 = (x_1, y_1)
        start_point_2 = (x_2, y_2)
        end_point_1 = (x_1 + size_x_1, y_1 + size_y_1)
        end_point_2 = (x_2 + size_x_2, y_2 + size_y_2)

        img = cv2.rectangle(img, start_point_1, end_point_1, self.fill, 1)
        img = cv2.rectangle(img, start_point_2, end_point_2, self.fill, 1)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_connected_open_squares(
        self, size1, category, line_width=1, is_closed=False
    ):
        """draw two connected open-square shapes."""
        img = np.array(self.create_canvas())

        size = size1
        points_a = [
            [0, size], [0, 0], [size, 0], [size, size],
            [size, 2 * size], [2 * size, 2 * size], [2 * size, size],
        ]
        points_b = [
            [0, size], [0, 2 * size], [size, 2 * size], [size, size],
            [size, 0], [2 * size, 0], [2 * size, size],
        ]
        if category == 1:
            points_b = points_a

        translation_a = [
            np.random.randint(1, self.canvas_size[0] - size * 2),
            np.random.randint(1, self.canvas_size[0] - size * 2),
        ]
        translation_b = [
            np.random.randint(1, self.canvas_size[0] - size * 2),
            np.random.randint(1, self.canvas_size[0] - size * 2),
        ]
        points_a = [
            [sum(pair) for pair in zip(point, translation_a)] for point in points_a
        ]
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in points_b
        ]

        poly_a = np.array(points_a, dtype=np.int32)
        poly_b = np.array(points_b, dtype=np.int32)

        poly_new_a = poly_a.reshape((-1, 1, 2))
        poly_new_b = poly_b.reshape((-1, 1, 2))

        cv2.polylines(
            img, [poly_new_a], isClosed=is_closed, color=self.fill, thickness=line_width
        )
        cv2.polylines(
            img,
            [poly_new_b],
            isClosed=is_closed,
            color=self.fill,
            thickness=line_width,
        )

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img
