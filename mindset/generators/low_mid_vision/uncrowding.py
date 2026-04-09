"""uncrowding dataset generator."""

import csv
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.drawing.geometry import circle_perimeter, line, polygon, polygon_perimeter
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing

# ---------------------------------------------------------------------------
# shape pattern generators
# ---------------------------------------------------------------------------


def all_test_shapes():
    """return all test shape configurations for uncrowding experiments."""
    return (
        seven_shapesgen(5)
        + shapesgen(5)
        + Lynns_patterns()
        + ten_random_patterns()
        + Lynns_patterns()
    )


def shapesgen(max, emptyvect=True):
    """generate shape pattern lists up to given max shape id."""
    if max > 7:
        return

    s = [[]] if emptyvect else []
    for i in range(1, max + 1):
        s += [[i], [i, i, i], [i, i, i, i, i]]
        for j in range(1, max + 1):
            if j != i:
                s += [[i, j, i, j, i]]
    return s


def seven_shapesgen(max, emptyvect=True):
    """generate seven-element shape pattern lists."""
    if max > 7:
        return

    s = [[]] if emptyvect else []
    for i in range(1, max + 1):
        s += [[i, i, i, i, i, i, i]]
        for j in range(1, max + 1):
            if j != i:
                s += [[j, i, j, i, j, i, j]]
    return s


def Lynns_patterns():
    """return Lynn's predefined uncrowding test patterns."""
    squares = [1, 1, 1, 1, 1, 1, 1]
    one_square = [0, 0, 0, 1, 0, 0, 0]
    S = [squares]
    for x in [6, 2]:
        line1 = [x, 1, x, 1, x, 1, x]
        line2 = [1, x, 1, x, 1, x, 1]
        line0 = [x, 1, x, 0, x, 1, x]

        columns = [line1, line1, line1]
        checker = [line2, line1, line2]
        if x == 6:
            special = [1, x, 2, x, 1, x, 1]
        else:
            special = [1, x, 1, x, 6, x, 1]

        checker_special = [line2, line1, special]
        irreg = [[1, x, 1, x, x, 1, 1], line1, [1, 1, x, x, 1, x, 1]]
        cross = [one_square, line1, one_square]
        pompom = [line0, line1, line0]

        S += [line1, columns, checker, irreg, pompom, cross, checker_special]
    return S


def ten_random_patterns(newone=False):
    """return ten fixed random patterns (or generate new ones)."""
    if newone:
        patterns = np.zeros((10, 3, 7), dtype=int)
        basis = [0, 1, 2, 6]
        for pat in range(10):
            for row in range(3):
                for col in range(7):
                    a = np.random.choice(basis)
                    patterns[pat][row][col] = a
        return patterns

    return [
        [[6, 1, 1, 0, 1, 6, 2], [0, 1, 0, 1, 2, 1, 1], [1, 0, 1, 6, 6, 2, 6]],
        [[1, 6, 1, 1, 2, 0, 2], [6, 2, 2, 6, 0, 1, 2], [1, 1, 0, 6, 1, 1, 1]],
        [[1, 6, 1, 2, 2, 0, 2], [1, 0, 6, 1, 2, 2, 6], [2, 2, 0, 1, 0, 2, 1]],
        [[6, 6, 0, 1, 1, 6, 6], [1, 1, 1, 2, 2, 6, 1], [6, 6, 2, 1, 6, 0, 6]],
        [[0, 6, 2, 2, 2, 6, 6], [2, 0, 1, 1, 6, 6, 6], [1, 0, 6, 0, 2, 6, 2]],
        [[2, 1, 1, 6, 2, 6, 2], [6, 1, 0, 6, 1, 2, 1], [1, 6, 0, 2, 1, 2, 6]],
        [[1, 1, 0, 6, 6, 6, 1], [1, 0, 0, 1, 2, 1, 1], [2, 1, 0, 2, 6, 1, 6]],
        [[0, 6, 6, 2, 2, 0, 2], [1, 6, 1, 6, 6, 2, 2], [2, 1, 6, 1, 0, 2, 2]],
        [[6, 1, 2, 6, 1, 0, 1], [0, 1, 6, 2, 0, 6, 2], [1, 0, 1, 2, 6, 6, 6]],
        [[1, 0, 1, 6, 2, 6, 2], [0, 6, 6, 2, 0, 1, 1], [6, 6, 1, 6, 0, 2, 1]],
    ]


# ---------------------------------------------------------------------------
# main drawing class
# ---------------------------------------------------------------------------


class DrawUncrowding(DrawStimuli):
    """draws uncrowding stimuli with verniers and flanking shapes."""

    def __init__(self, bar_width, *args, **kwargs):
        self.bar_width = bar_width
        self.offset_height = 1
        super().__init__(*args, **kwargs)
        self.shape_size = None

    def draw_square(self):
        """draw a square shape patch."""
        resize_factor = 1.2
        patch = np.array(
            self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        )

        first_row = int((self.shape_size - self.shape_size / resize_factor) / 2)
        first_col = first_row
        side_size = int(self.shape_size / resize_factor)

        patch[
            first_row : first_row + self.bar_width,
            first_col : first_col + side_size + self.bar_width,
        ] = self.fill[:3]
        patch[
            first_row + side_size : first_row + self.bar_width + side_size,
            first_col : first_col + side_size + self.bar_width,
        ] = self.fill[:3]
        patch[
            first_row : first_row + side_size + self.bar_width,
            first_col : first_col + self.bar_width,
        ] = self.fill[:3]
        patch[
            first_row : first_row + side_size + self.bar_width,
            first_row + side_size : first_row + self.bar_width + side_size,
        ] = self.fill[:3]

        return patch

    def draw_circle(self):
        """draw a circle shape patch."""
        resizeFactor = 1.01
        radius = self.shape_size / (2 * resizeFactor) - 1
        patch = np.array(
            self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        )
        center = (int(self.shape_size / 2), int(self.shape_size / 2))
        rr, cc = circle_perimeter(*center, int(radius))
        patch[rr, cc] = self.fill[:3]
        return patch

    def draw_diamond(self):
        """draw a diamond shape patch."""
        S = self.shape_size
        mid = int(S / 2)
        patch = np.array(self.create_canvas(size=(S, S), background=self.background))
        for i in range(S):
            for j in range(S):
                if i == mid + j or i == mid - j or j == mid + i or j == 3 * mid - i - 1:
                    patch[i, j] = self.fill[:3]
        return patch

    def draw_noise(self):
        """draw a noise patch."""
        S = self.shape_size
        return np.random.normal(0, 0.1, size=(S // 3, S // 3))

    def draw_polygon(self, nSides, phi):
        """draw a regular polygon shape patch."""
        patch = np.array(
            self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        )
        center = (self.shape_size // 2, self.shape_size // 2)
        radius = self.shape_size / (2 * 1.0) - 1

        rowExtVertices = []
        colExtVertices = []
        for n in range(nSides):
            rowExtVertices.append(
                radius * np.sin(2 * np.pi * n / nSides + phi) + center[0]
            )
            colExtVertices.append(
                radius * np.cos(2 * np.pi * n / nSides + phi) + center[1]
            )

        RR, CC = polygon_perimeter(rowExtVertices, colExtVertices)
        patch[RR, CC] = self.fill[:3]
        return patch

    def draw_star(self, nTips, ratio, phi):
        """draw a star shape patch."""
        resizeFactor = 0.8
        patch = np.array(
            self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        )
        center = (int(self.shape_size / 2), int(self.shape_size / 2))
        radius = self.shape_size / (2 * resizeFactor)

        row_ext_vertices = []
        col_ext_vertices = []
        row_int_vertices = []
        col_int_vertices = []
        for n in range(2 * nTips):
            this_radius = radius
            if not n % 2:
                this_radius = radius / ratio

            row_ext_vertices.append(
                max(
                    min(
                        this_radius * np.sin(2 * np.pi * n / (2 * nTips) + phi)
                        + center[0],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            col_ext_vertices.append(
                max(
                    min(
                        this_radius * np.cos(2 * np.pi * n / (2 * nTips) + phi)
                        + center[1],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            row_int_vertices.append(
                max(
                    min(
                        (this_radius - self.bar_width)
                        * np.sin(2 * np.pi * n / (2 * nTips) + phi)
                        + center[0],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            col_int_vertices.append(
                max(
                    min(
                        (this_radius - self.bar_width)
                        * np.cos(2 * np.pi * n / (2 * nTips) + phi)
                        + center[1],
                        self.shape_size,
                    ),
                    0.0,
                )
            )

        RR, CC = polygon(row_ext_vertices, col_ext_vertices)
        rr, cc = polygon(row_int_vertices, col_int_vertices)
        patch[RR, CC] = self.fill[:3]
        patch[rr, cc] = self.background

        return patch

    def draw_irreg(self, n_sides_rough, repeat_shape):
        """draw an irregular polygon shape patch."""
        if repeat_shape:
            random.seed(1)

        patch = np.array(
            self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        )
        center = (int(self.shape_size / 2), int(self.shape_size / 2))
        angle = 0

        row_ext_vertices = []
        col_ext_vertices = []
        row_int_vertices = []
        col_int_vertices = []
        while angle < 2 * np.pi:
            if (
                np.pi / 4 < angle < 3 * np.pi / 4
                or 5 * np.pi / 4 < angle < 7 * np.pi / 4
            ):
                radius = (random.random() + 2.0) / 3.0 * self.shape_size / 2
            else:
                radius = (random.random() + 1.0) / 2.0 * self.shape_size / 2

            row_ext_vertices.append(radius * np.sin(angle) + center[0])
            col_ext_vertices.append(radius * np.cos(angle) + center[1])
            row_int_vertices.append(
                (radius - self.bar_width) * np.sin(angle) + center[0]
            )
            col_int_vertices.append(
                (radius - self.bar_width) * np.cos(angle) + center[1]
            )

            angle += (random.random() + 0.5) * (2 * np.pi / n_sides_rough)

        RR, CC = polygon(row_ext_vertices, col_ext_vertices)
        rr, cc = polygon(row_int_vertices, col_int_vertices)
        patch[RR, CC] = self.fill[:3]
        patch[rr, cc] = self.background

        if repeat_shape:
            random.seed(datetime.now())

        return patch

    def draw_stuff(self, nLines):
        """draw random lines on a patch."""
        patch = np.array(self.create_canvas(size=(self.shape_size, self.shape_size)))

        for n in range(nLines):
            r1, c1, r2, c2 = np.random.randint(self.shape_size, size=4)
            rr, cc = line(r1, c1, r2, c2)
            patch[rr, cc] = self.fill[:3]

        return patch

    def draw_vernier_in_patch(
        self, full_patch: np.array = None, offset=None, offset_size=None
    ):
        """draw a vernier stimulus within a shape patch."""
        if offset_size is None:
            offset_size = random.randint(1, int(self.bar_height / 2.0))

        patch = np.array(
            self.create_canvas(
                size=(
                    2 * self.bar_width + offset_size,
                    2 * self.bar_height + self.offset_height,
                ),
                background=self.background,
            )
        )
        patch[0 : self.bar_height, 0 : self.bar_width, :] = self.fill[:3]
        patch[
            self.bar_height + self.offset_height :, self.bar_width + offset_size :, :
        ] = self.fill[:3]

        if offset is None:
            if random.randint(0, 1):
                patch = np.fliplr(patch)
        elif offset == 1:
            patch = np.fliplr(patch)
        if full_patch is None:
            full_patch = np.array(
                self.create_canvas(
                    size=(self.shape_size, self.shape_size), background=self.background
                )
            )
        first_row = int((self.shape_size - patch.shape[0]) / 2)
        first_col = int((self.shape_size - patch.shape[1]) / 2)
        full_patch[
            first_row : first_row + patch.shape[0],
            first_col : first_col + patch.shape[1],
        ] = patch

        return full_patch

    def draw_shape(self, shapeID, offset=None, offset_size=None):
        """draw a shape by id (0=empty, 1=square, 2=circle, etc.)."""
        if shapeID == 0:
            patch = self.create_canvas(
                size=(self.shape_size, self.shape_size), background=self.background
            )
        if shapeID == 1:
            patch = self.draw_square()
        if shapeID == 2:
            patch = self.draw_circle()
        if shapeID == 3:
            patch = self.draw_polygon(6, 0)
        if shapeID == 4:
            patch = self.draw_polygon(8, np.pi / 8)
        if shapeID == 5:
            patch = self.draw_diamond()
        if shapeID == 6:
            patch = self.draw_star(7, 1.7, -np.pi / 14)
        if shapeID == 7:
            patch = self.draw_irreg(15, False)
        if shapeID == 8:
            patch = self.draw_irreg(15, True)
        if shapeID == 9:
            patch = self.draw_stuff(5)
        if shapeID == 10:
            patch = self.draw_noise()

        return patch

    def draw_stim(
        self,
        vernier_ext,
        shape_matrix,
        shape_size,
        vernier_in=False,
        offset=None,
        offset_size=None,
        fixed_position=None,
        noise_patch=None,
    ):
        """draw a complete uncrowding stimulus with vernier and flankers."""
        self.shape_size = shape_size
        self.bar_height = int(shape_size / 4 - self.bar_width / 4)

        if shape_matrix is None:
            ID = np.random.randint(1, 7)
            siz = np.random.randint(4) * 2 + 1
            h = np.random.randint(2) * 2 + 1
            shape_matrix = np.zeros((h, siz)) + ID

        image = np.array(self.create_canvas())
        padDist = int(self.shape_size / 6)
        shape_matrix = np.array(shape_matrix)

        if len(shape_matrix.shape) < 2:
            shape_matrix = np.expand_dims(shape_matrix, axis=0)

        if shape_matrix.size == 0:
            patch = np.array(
                self.create_canvas(
                    size=(self.shape_size, self.shape_size), background=self.background
                )
            )
        else:
            patch = np.array(
                self.create_canvas(
                    size=(
                        shape_matrix.shape[1] * self.shape_size
                        + (shape_matrix.shape[1] - 1) * 0
                        + 1,
                        shape_matrix.shape[0] * self.shape_size
                        + (shape_matrix.shape[0] - 1) * 0
                        + 1,
                    ),
                    background=self.background,
                )
            )

            for row in range(shape_matrix.shape[0]):
                for col in range(shape_matrix.shape[1]):
                    first_row = row * self.shape_size
                    first_col = col * self.shape_size
                    patch[
                        first_row : first_row + self.shape_size,
                        first_col : first_col + self.shape_size,
                    ] = self.draw_shape(shape_matrix[row, col], offset, offset_size)

        if vernier_in:
            first_row = int((patch.shape[0] - self.shape_size) / 2)
            first_col = int((patch.shape[1] - self.shape_size) / 2)
            target_patch = patch[
                first_row : (first_row + self.shape_size),
                first_col : first_col + self.shape_size,
            ]
            patch[
                first_row : (first_row + self.shape_size),
                first_col : first_col + self.shape_size,
            ] = self.draw_vernier_in_patch(target_patch, offset, offset_size)

        if fixed_position is None:
            first_row = random.randint(
                padDist, self.canvas_size[0] - (patch.shape[0] + padDist)
            )
            first_col = random.randint(
                padDist, self.canvas_size[1] - (patch.shape[1] + padDist)
            )
        else:
            n_elements = [
                max(shape_matrix.shape[0], 1),
                max(shape_matrix.shape[1], 1),
            ]
            first_row = fixed_position[0] - int(
                self.shape_size * (n_elements[0] - 1) / 2
            )
            first_col = fixed_position[1] - int(
                self.shape_size * (n_elements[1] - 1) / 2
            )

        image[
            first_row : first_row + patch.shape[0],
            first_col : first_col + patch.shape[1],
        ] = patch

        min_distance = 0

        if vernier_ext:
            ver_size = self.shape_size
            ver_patch = self.draw_vernier_in_patch(None, offset, offset_size)
            x = first_row
            y = first_col

            flag = 0
            while (
                x + ver_size + min_distance >= first_row
                and x <= min_distance + first_row + patch.shape[0]
                and y + ver_size >= first_col
                and y <= first_col + patch.shape[1]
            ):
                x = np.random.randint(
                    padDist, self.canvas_size[0] - (ver_size + padDist)
                )
                y = np.random.randint(
                    padDist, self.canvas_size[1] - (ver_size + padDist)
                )
                flag += 1
                if flag > 15:
                    print("problem in finding space for the extra vernier")

            image[x : x + ver_size, y : y + ver_size] = ver_patch

        if noise_patch is not None:
            image[
                noise_patch[0] : noise_patch[0] + self.shape_size // 2,
                noise_patch[1] : noise_patch[1] + self.shape_size // 2,
            ] = self.draw_noise()
        img = Image.fromarray(image).convert("RGBA")
        img = apply_antialiasing(img) if self.antialiasing else img
        return img


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class UncrowdingConfig(GeneratorConfig):
    """config for uncrowding dataset."""

    num_samples_vernier_inside: int = field(
        default=100,
        metadata={
            "min": 1,
            "max": 10000,
            "step": 10,
            "label": "vernier inside samples",
        },
    )
    num_samples_vernier_outside: int = field(
        default=100,
        metadata={
            "min": 1,
            "max": 10000,
            "step": 10,
            "label": "vernier outside samples",
        },
    )
    random_size: bool = field(default=True, metadata={"label": "random shape size"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/low_mid_level_vision/un_crowding",
        metadata={"label": "output folder"},
    )


@register("uncrowding", "low_mid_vision")
@generator(UncrowdingConfig)
def generate_all(config: UncrowdingConfig):
    """generate uncrowding dataset."""
    output_folder = Path(config.output_folder)

    vernier_in_out = ["outside", "inside"]
    vernier_type = [0, 1]

    for v in vernier_in_out:
        for c in vernier_type:
            (output_folder / v / str(c)).mkdir(exist_ok=True, parents=True)

    ds = DrawUncrowding(
        canvas_size=config.canvas_size,
        background=config.background_color,
        antialiasing=config.antialiasing,
        bar_width=1,
    )

    t = all_test_shapes()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "VernierInOut",
                "VernierType",
                "BackgroundColor",
                "ShapeCode",
                "ShapeSize",
                "IterNum",
            ]
        )

        for v_in_out in tqdm(vernier_in_out):
            num_requested = (
                config.num_samples_vernier_outside
                if v_in_out == "outside"
                else config.num_samples_vernier_inside
            )
            samples_per_cond = num_requested // len(t)
            if samples_per_cond == 0:
                print(
                    f"in order to have at least one sample per condition, the total number of sample has been increased to {len(t)}"
                )
                samples_per_cond = 1
            if samples_per_cond * len(t) != num_requested:
                print(
                    f"you specified {num_requested} for {v_in_out} but to keep the number of sample per subcategory equal, {samples_per_cond * len(t)} samples will be generated ({len(t)} categories, {samples_per_cond} samples per category)"
                )

            for v in vernier_type:
                for s in tqdm(t, leave=False):
                    for n in range(samples_per_cond):
                        shape_size = (
                            random.randint(
                                int(config.canvas_size[0] * 0.1),
                                config.canvas_size[0] // 7,
                            )
                            if config.random_size
                            else config.canvas_size[0] * 0.08
                        )
                        shape_size -= int(shape_size / 6)

                        img = ds.draw_stim(
                            vernier_ext=v_in_out == "outside",
                            shape_matrix=s,
                            shape_size=shape_size,
                            vernier_in=v_in_out == "inside",
                            fixed_position=None,
                            offset=v,
                            offset_size=None,
                            noise_patch=None,
                        )
                        strs = str(s).replace("], ", "nl")
                        shape_code = "".join(
                            i for i in strs if i not in [",", "[", "]", " "]
                        )
                        shape_code = shape_code if shape_code != "" else "none"

                        unique_hex = uuid.uuid4().hex[:8]
                        path = (
                            Path(v_in_out)
                            / str(v)
                            / f"{shape_code}_{n}_{unique_hex}.png"
                        )
                        img.save(output_folder / path)
                        writer.writerow(
                            [
                                path,
                                v_in_out,
                                v,
                                ds.background,
                                shape_code,
                                shape_size,
                                n,
                            ]
                        )

    return str(output_folder)
