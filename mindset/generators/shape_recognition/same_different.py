"""same different task dataset generator."""
import copy
import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.misc import apply_antialiasing


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def svrt_1_points(
    canvas_size,
    category=1,
    radii=None,
    sides=None,
    rotations=None,
    regular=False,
    irregularity=0.25,
):
    """returns polygon points for a single instance of a SVRT problem 1."""
    radius_1, radius_2 = radii

    if sides is None:
        possible_sides = random.sample(list(range(3, 8)), 2)
        sides_1 = possible_sides[0]
        sides_2 = possible_sides[1]

    if rotations is None:
        rotation_1 = math.radians(random.randint(0, 360))
        rotation_2 = math.radians(random.randint(0, 360))

    if not regular and irregularity is None:
        max_dev_factor = np.random.choice([0.3, 0.4, 0.5, 0.6])
    else:
        max_dev_factor = irregularity
    max_dev_1 = int(radius_1 * max_dev_factor)
    min_dev_1 = radius_1 + max_dev_1
    max_dev_2 = int(radius_2 * max_dev_factor)
    min_dev_2 = radius_2 + max_dev_2

    translation_a = [
        np.random.randint(min_dev_1, canvas_size[0] - min_dev_1),
        np.random.randint(min_dev_1, canvas_size[1] - min_dev_1),
    ]

    translation_b = [
        np.random.randint(min_dev_2, canvas_size[0] - min_dev_2),
        np.random.randint(min_dev_2, canvas_size[1] - min_dev_2),
    ]

    if category == 0 and regular:
        points_a, _ = regular_polygon(
            sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a,
        )
        points_b, _ = regular_polygon(
            sides=sides_2, radius=radius_2, rotation=rotation_2, translation=translation_b,
        )

    elif category == 1 and regular:
        points_a, original_a = regular_polygon(
            sides=sides_1, radius=radius_1, rotation=rotation_1, translation=translation_a,
        )
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in original_a
        ]

    elif category == 0 and not regular:
        points_a, _ = irregular_polygon_from_regular(
            sides=sides_1, radius=radius_1, rotation=rotation_1,
            translation=translation_a, max_dev=max_dev_1,
        )
        points_b, _ = irregular_polygon_from_regular(
            sides=sides_2, radius=radius_2, rotation=rotation_2,
            translation=translation_b, max_dev=max_dev_2,
        )

    elif category == 1 and not regular:
        points_a, original_a = irregular_polygon_from_regular(
            sides=sides_1, radius=radius_1, rotation=rotation_1,
            translation=translation_a, max_dev=max_dev_1,
        )
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in original_a
        ]

    else:
        raise ValueError("wrong category or regular args!")

    return points_a, points_b, tuple(translation_b), radius_1


def regular_polygon(sides, radius=10, rotation=0, translation=None):
    """calculate the vertices of a regular polygon."""
    one_segment = math.pi * 2 / sides
    points = [
        (
            int(math.sin(one_segment * i + rotation) * radius),
            int(math.cos(one_segment * i + rotation) * radius),
        )
        for i in range(sides)
    ]

    original_points = copy.copy(points)
    if translation:
        points = [[sum(pair) for pair in zip(point, translation)] for point in points]
    return points, original_points


def rotate(origin, point, angle):
    """rotate a point counterclockwise by a given angle around a given origin."""
    oy, ox = origin
    py, px = point

    qx = ox + int(math.cos(angle) * (px - ox)) - int(math.sin(angle) * (py - oy))
    qy = oy + int(math.sin(angle) * (px - ox)) + int(math.cos(angle) * (py - oy))
    return int(qy), int(qx)


def sample_midpoints_lines(sizes, canvas_size):
    """sample random midpoints for two lines given their sizes."""
    size_1, size_2 = sizes
    x_1 = random.sample(
        list(range(int(size_1 / 2) + 2, canvas_size[0] - int(size_1 / 2 + 2))), 1
    )[0]
    y_1 = random.sample(
        list(range(int(size_1 / 2) + 2, canvas_size[1] - int(size_1 / 2 + 2))), 1
    )[0]
    x_2 = random.sample(
        list(range(int(size_2 / 2) + 2, canvas_size[0] - int(size_2 / 2 + 2))), 1
    )[0]
    y_2 = random.sample(
        list(range(int(size_2 / 2) + 2, canvas_size[1] - int(size_2 / 2 + 2))), 1
    )[0]
    point_1 = (x_1, y_1)
    point_2 = (x_2, y_2)

    return point_1, point_2


def get_line_points(size, rotation, center):
    """calculate endpoints of a line given size, rotation, and center."""
    radius = size / 2
    angle_1 = math.radians(rotation)
    angle_2 = math.radians(rotation + 180)

    x_1 = int(center[0] + int(radius * math.cos(angle_1)))
    y_1 = int(center[1] + int(radius * math.sin(angle_1)))

    x_2 = int(center[0] + int(radius * math.cos(angle_2)))
    y_2 = int(center[1] + int(radius * math.sin(angle_2)))

    return [(x_1, y_1), (x_2, y_2)]


def ccw_sort(polygon_points):
    """sort the points counter clockwise around the mean of all points."""
    polygon_points = np.array(polygon_points)
    mean = np.mean(polygon_points, axis=0)
    d = polygon_points - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return polygon_points[np.argsort(s), :]


def irregular_polygon_from_regular(
    sides, radius=1, rotation=0, translation=None, max_dev=0
):
    """build an irregular polygon by adding noise to a regular one."""
    points, original_points = regular_polygon(
        sides=sides, radius=radius, rotation=rotation, translation=translation
    )

    noise = [
        [
            np.random.randint(-max_dev, max_dev + 1),
            np.random.randint(-max_dev, max_dev + 1),
        ]
        for x in points
    ]
    points = [[x[0] + y[0], x[1] + y[0]] for x, y in zip(points, noise)]
    original_points = [
        [x[0] + y[0], x[1] + y[0]] for x, y in zip(original_points, noise)
    ]

    return ccw_sort(points), ccw_sort(original_points)


# ---------------------------------------------------------------------------
# drawing class
# ---------------------------------------------------------------------------

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
                img, [poly_new_b], isClosed=closed, color=color_b, thickness=thickness,
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
            img, points_line_2[0], points_line_2[1], self.fill, thickness=line_thickness,
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
            img, [poly_new_b], isClosed=is_closed, color=self.fill, thickness=line_width,
        )

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img


# ---------------------------------------------------------------------------
# overlap detection and dataset factory lambdas
# ---------------------------------------------------------------------------

def is_overlapping(img: np.array, background_color: tuple, threshold: int = 2):
    """detect whether two shapes overlap in an image."""
    img_c = img.copy()
    img_c[img == background_color] = 0

    gray = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    b_rects = []
    for c in cnts:
        b_rects.append(cv2.boundingRect(c))

    return len(b_rects) != 2


get_irregular_polygon = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, **kwargs
)

get_regular = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=True, sides=None, thickness=1, **kwargs
)

get_open = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, closed=False, **kwargs,
)

get_wider_line = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=2, **kwargs
)


def get_rnd_color(ds, label, **kwargs):
    """generate a shape with random color."""
    color = tuple(np.random.randint(1, high=256, size=3))
    color = (int(color[0]), int(color[1]), int(color[2]))
    return ds.svrt_1_img(
        category=label, regular=False, color_a=color, sides=None, thickness=1, **kwargs
    )


get_filled = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, filled=True, **kwargs,
)

get_straight_lines = lambda ds, label, **kwargs: ds.make_straight_lines_sd_diffrot(
    category=label, line_thickness=1, **kwargs
)

get_rectangles = lambda ds, label, **kwargs: ds.make_rectangles_sd(
    category=label, **kwargs
)

get_open_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, **kwargs
)

get_closed_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, is_closed=True, **kwargs
)


def is_integer(n):
    """check whether a value can be parsed as an integer."""
    try:
        int(n)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------

@dataclass
class SameDifferentConfig(GeneratorConfig):
    """config for same different task dataset."""
    num_samples: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "samples per type per condition"})
    size_shapes: str = field(default="20", metadata={"label": "shape size (int, rnd1, or rnd2)"})
    type_dataset: str = field(default="all", metadata={"label": "dataset type (all or specific name)"})
    output_folder: str = field(default="data/shape_and_object_recognition/same_different_task", metadata={"label": "output folder"})


@register("same_different", "shape_recognition")
@generator(SameDifferentConfig)
def generate_all(config: SameDifferentConfig):
    """generate same-different task dataset with various shape types."""
    output_folder = Path(config.output_folder)

    all_datasets = {
        "regular": get_regular,
        "irregular": get_irregular_polygon,
        "open": get_open,
        "wider_line": get_wider_line,
        "rnd_color": get_rnd_color,
        "filled": get_filled,
        "open_squares": get_open_squares,
        "rectangles": get_rectangles,
        "straight_lines": get_straight_lines,
        "closed_squares": get_closed_squares,
    }

    datasets = all_datasets if config.type_dataset == "all" else {config.type_dataset: all_datasets[config.type_dataset]}

    ds = DrawSameDifferentStimuli(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    labels = ["same", "diff"]
    for label in labels:
        for ds_name in datasets:
            (output_folder / ds_name / label).mkdir(exist_ok=True, parents=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "BackgroundColor", "TypeDataset", "SizeShape1", "SizeShape2", "SameDiff", "SampleNum"])

        for ds_name, dataset_fun in tqdm(datasets.items()):
            use_size2 = ds_name not in ["open_squares", "closed_squares", "rectangles"]
            for n in tqdm(range(config.num_samples), leave=False):
                for label in labels:
                    while True:
                        if is_integer(config.size_shapes):
                            size1, size2 = int(config.size_shapes), int(config.size_shapes)
                        elif config.size_shapes == "rnd1":
                            size1 = np.random.randint(ds.canvas_size[0] // 15, ds.canvas_size[0] // 4)
                            size2 = size1
                        else:
                            size1, size2 = np.random.randint(ds.canvas_size[0] // 15, ds.canvas_size[0] // 4, 2)

                        args = dict(label=1 if label == "same" else 0, size1=size1)
                        if use_size2:
                            args["size2"] = size2

                        img = dataset_fun(ds, **args)
                        if not is_overlapping(np.array(img), ds.background):
                            break

                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = Path(ds_name) / label / f"{unique_hex}.png"
                    img.save(output_folder / img_path)
                    writer.writerow([img_path, config.background_color, ds_name, size1, size2 if use_size2 else None, label, n])

    return str(output_folder)
