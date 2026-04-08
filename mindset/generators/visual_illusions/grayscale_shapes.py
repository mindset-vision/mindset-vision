"""grayscale shapes dataset generator."""
import csv
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from random import choice

import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from tqdm import tqdm

from mindset.drawing.primitives import add_arrow
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


# ---------------------------------------------------------------------------
# shape configs
# ---------------------------------------------------------------------------

class ShapeConfigs:
    """contains the configs for the shapes."""

    def __init__(self):
        self.range_area_coords = (0.05, 1)
        self.range_area_circle = (0.05, 1)
        self.threshold_angle_polygon = 45
        self.frequency_ratio = {
            "chords": 1,
            "circles": 1,
            "ellipses": 1,
            "pie_slices": 1,
            "polygons": 0.3,
            "rectangles": 1,
            "regular_polygons": 1,
            "rounded_rectangles": 1,
        }

        sum_values = sum(self.frequency_ratio.values())

        for key in self.frequency_ratio:
            self.frequency_ratio[key] /= sum_values

        self.shape = np.random.choice(
            list(self.frequency_ratio.keys()), p=list(self.frequency_ratio.values())
        )

    def _refresh(self):
        """re-sample a random shape type."""
        self.shape = np.random.choice(
            list(self.frequency_ratio.keys()), p=list(self.frequency_ratio.values())
        )

    def _return_shape_config(self):
        """return shape name and its random parameters."""
        return {"shape": self.shape, "parameters": getattr(self, self.shape)()}

    def _calculate_area_coords(self, coord_1, coord_2):
        """calculate area from two bounding coordinates."""
        width = np.abs(coord_1[0] - coord_2[0])
        height = np.abs(coord_1[1] - coord_2[1])
        return width * height

    def _calculate_area_circle(self, radius):
        """calculate area of a circle."""
        return math.pi * radius**2

    def _calculate_angle(self, c1, c2, c3):
        """calculate the angle between three coordinates."""
        diff1 = c2 - c1
        diff2 = c3 - c2
        angle = math.atan2(diff2[1], diff2[0]) - math.atan2(diff1[1], diff1[0])
        angle = math.degrees(angle)
        angle += 360 if angle < 0 else 0
        return angle

    def circles(self):
        """random circle parameters."""
        x = np.random.rand()
        y = np.random.rand()
        min_radius = (min(self.range_area_circle) / math.pi) ** 0.5
        max_radius = (max(self.range_area_circle) / math.pi) ** 0.5
        r = np.random.uniform(min_radius, max_radius)
        fill = np.random.randint(1, 254)
        return {"x": x, "y": y, "r": r, "fill": fill}

    def chords(self):
        """random chord parameters."""
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        starting_angle, ending_angle = np.random.rand(2) * 360
        while abs(starting_angle - ending_angle) < 30:
            starting_angle, ending_angle = np.random.rand(2) * 360

        fill = np.random.randint(1, 254)
        return {
            "coord_1": coord_1,
            "coord_2": coord_2,
            "starting_angle": starting_angle,
            "ending_angle": ending_angle,
            "fill": fill,
        }

    def ellipses(self):
        """random ellipse parameters."""
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        width = np.random.uniform(0.3, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        fill = np.random.randint(1, 254)
        width = np.random.rand() / 2
        while width < 0.01:
            width = np.random.rand() / 2
        return {"coord_1": coord_1, "coord_2": coord_2, "fill": fill, "width": width}

    def pie_slices(self):
        """random pie slice parameters."""
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        starting_angle, ending_angle = np.random.rand(2) * 360
        while abs(starting_angle - ending_angle) < 30:
            starting_angle, ending_angle = np.random.rand(2) * 360

        fill = np.random.randint(1, 254)
        return {
            "coord_1": coord_1,
            "coord_2": coord_2,
            "starting_angle": starting_angle,
            "ending_angle": ending_angle,
            "fill": fill,
        }

    def polygons(self):
        """random polygon parameters."""
        n_points = np.random.randint(3, 10)

        coordinates = np.random.rand(n_points, 2)
        angles = [
            self._calculate_angle(c1, c2, c3)
            for c1, c2, c3 in zip(coordinates[:-2], coordinates[1:-1], coordinates[2:])
        ]
        while any(angle < self.threshold_angle_polygon for angle in angles):
            coordinates = np.random.rand(n_points, 2)
            angles = [
                self._calculate_angle(c1, c2, c3)
                for c1, c2, c3 in zip(
                    coordinates[:-2], coordinates[1:-1], coordinates[2:]
                )
            ]

        coordinates = coordinates.tolist()
        coordinates = [tuple(coord) for coord in coordinates]

        fill = np.random.randint(1, 254)
        return {"coordinates": coordinates, "fill": fill}

    def regular_polygons(self):
        """random regular polygon parameters."""
        x = np.random.rand()
        y = np.random.rand()
        min_radius = (min(self.range_area_circle) / math.pi) ** 0.5
        max_radius = (max(self.range_area_circle) / math.pi) ** 0.5
        r = np.random.uniform(min_radius, max_radius)

        bounding_circle_xyr = (x, y, r)

        n_sides = np.random.randint(3, 10)
        rotation = np.random.rand() * 360
        fill = np.random.randint(1, 254)
        return {
            "bounding_circle_xyr": bounding_circle_xyr,
            "n_sides": n_sides,
            "rotation": rotation,
            "fill": fill,
        }

    def rectangles(self):
        """random rectangle parameters."""
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        fill = np.random.randint(1, 254)
        return {"coord_1": coord_1, "coord_2": coord_2, "fill": fill}

    def rounded_rectangles(self):
        """random rounded rectangle parameters."""
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        radius = np.random.rand() / 8
        fill = np.random.randint(1, 254)
        return {"coord_1": coord_1, "coord_2": coord_2, "radius": radius, "fill": fill}


# ---------------------------------------------------------------------------
# color picker stimuli
# ---------------------------------------------------------------------------

class ColorPickerStimuli:
    """generates colour picker task stimuli on a grayscale canvas."""

    def __init__(self, canvas_size: tuple = (224, 224)):
        assert (
            canvas_size[0] == canvas_size[1]
        ), "the color picker train images has to be squares"
        self.target_image_width = canvas_size[0]
        self.initial_expansion = 1

        self.arrow_line_length = 45
        self.triangle_height = 30
        self.triangle_width = 20

        self.indicator_circle_radius = 20
        self.initial_image_width = self.target_image_width * self.initial_expansion

        self.canvas = new(
            "L", (self.initial_image_width, self.initial_image_width), color=0
        )
        self.draw = Draw(self.canvas)

    def add_circles(self, x, y, r, fill):
        """add a circle to the canvas."""
        x *= self.initial_image_width
        y *= self.initial_image_width
        r *= self.initial_image_width
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def add_chords(self, coord_1, coord_2, starting_angle, ending_angle, fill):
        """add a chord to the canvas."""
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.chord((coord_1, coord_2), starting_angle, ending_angle, fill=fill)

    def add_ellipses(self, coord_1, coord_2, fill, width):
        """add an ellipse to the canvas."""
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)
            coord_2 = (x1, y0)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.ellipse((coord_1, coord_2), fill=fill, width=width)

    def add_pie_slices(self, coord_1, coord_2, starting_angle, ending_angle, fill):
        """add a pie slice to the canvas."""
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.pieslice((coord_1, coord_2), starting_angle, ending_angle, fill=fill)

    def add_polygons(self, coordinates, fill):
        """add a polygon to the canvas."""
        coordinates = [
            tuple(map(self._multiply_by_canvas_size, coord)) for coord in coordinates
        ]
        self.draw.polygon(coordinates, fill=fill)

    def add_regular_polygons(self, bounding_circle_xyr, n_sides, rotation, fill):
        """add a regular polygon to the canvas."""
        bounding_circle_xyr = [
            i * self.initial_image_width for i in bounding_circle_xyr
        ]
        self.draw.regular_polygon(bounding_circle_xyr, n_sides, rotation, fill=fill)

    def add_rectangles(self, coord_1, coord_2, fill):
        """add a rectangle to the canvas."""
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.rectangle((coord_1, coord_2), fill=fill)

    def add_rounded_rectangles(self, coord_1, coord_2, radius, fill):
        """add a rounded rectangle to the canvas."""
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        coord_1 = tuple(map(int, coord_1))
        coord_2 = tuple(map(int, coord_2))
        radius = self._multiply_by_canvas_size(radius)
        self.draw.rounded_rectangle(xy=(coord_1, coord_2), radius=radius, fill=fill)

    def _add_indicator_circle(self, coord: tuple[float, float]):
        """add an indicator circle at the given coordinate."""
        fill = None
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        coord = (coord[0], coord[1] - 1)

        self.draw.ellipse(
            (
                coord[0] - self.indicator_circle_radius,
                coord[1] - self.indicator_circle_radius,
                coord[0] + self.indicator_circle_radius,
                coord[1] + self.indicator_circle_radius,
            ),
            fill=fill,
            outline=255,
            width=2,
        )

    def _get_pixel_color(self, coord: tuple[float, float]) -> int:
        """get the color of a pixel in the canvas."""
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        return self.canvas.getpixel(coord)

    def _mutate_pixel_color(self, coord: tuple[float, float], target_color: int):
        """mutate all pixels of a given color in the canvas."""
        pixel_color = self._get_pixel_color(coord)
        self.canvas.putdata(
            [
                target_color if pixel_color == color else color
                for color in self.canvas.getdata()
            ]
        )

    def _count_colors_withing_circle(self, coord: tuple[float, float]) -> int:
        """get the pixel colors within a circle given its center and radius."""
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        coord_1 = (
            coord[0] - self.indicator_circle_radius,
            coord[1] - self.indicator_circle_radius,
        )
        coord_2 = (
            coord[0] + self.indicator_circle_radius,
            coord[1] + self.indicator_circle_radius,
        )
        return len(self.canvas.crop((*coord_1, *coord_2)).getcolors())

    def _propose_arrow_coord(self) -> tuple[float, float]:
        """propose a coordinate for the arrow within the canvas."""
        coord = np.random.rand(2) * 0.8 + 0.1
        while self._get_pixel_color(coord) == 255:
            coord = np.random.rand(2) * 0.8 + 0.1
        return coord

    def _shrink_and_save(self, save_as: Path):
        """shrink the canvas to target size and save."""
        self.canvas = self.canvas.resize(
            (self.target_image_width, self.target_image_width),
            resample=Image.Resampling.LANCZOS,
        )
        self.canvas.save(save_as)

    def _check_undrawn_pixels(self):
        """check the number of undrawn pixels in the canvas."""
        return np.sum(np.array(self.canvas) == 0)

    def _check_proportion_dominant_color(self):
        """check the proportion of the dominant color in the canvas."""
        dominant = np.bincount(np.array(self.canvas).flatten()).argmax()
        return np.sum(np.array(self.canvas) == dominant) / self.initial_image_width**2

    def _check_proportion_smallest_segment(self):
        """check the size of the smallest segment in the canvas (in pixels)."""
        segments = self.canvas.getcolors(self.initial_image_width**2)
        segments.sort(key=lambda x: x[0])
        return segments[0][0] / self.initial_image_width**2

    def _check_num_shades(self):
        """check the number of shades of gray in the canvas."""
        return len(np.unique(np.array(self.canvas)))

    def _multiply_by_canvas_size(self, x):
        """scale a normalized coordinate to canvas pixel space."""
        return x * self.initial_image_width


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------

@dataclass
class GrayscaleShapesConfig(GeneratorConfig):
    """config for grayscale shapes dataset."""
    num_samples: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "number of samples"})
    arrow_size: float = field(default=1, metadata={"min": 0.1, "max": 10, "step": 0.1, "label": "arrow size"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/visual_illusions/grayscale_shapes", metadata={"label": "output folder"})


@register("grayscale_shapes", "visual_illusions")
@generator(GrayscaleShapesConfig)
def generate_all(config: GrayscaleShapesConfig):
    """generate grayscale shapes dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "all_images").mkdir(exist_ok=True, parents=True)
    shape_configs = ShapeConfigs()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Target Pixel Color", "Target Pixel Location"])

        for i in tqdm(range(config.num_samples)):
            img = ColorPickerStimuli(config.canvas_size)

            for step in range(20):
                shape_configs._refresh()
                random_shape_config = shape_configs._return_shape_config()
                getattr(img, "add_" + random_shape_config["shape"])(
                    **random_shape_config["parameters"]
                )

            propose_coordinates = img._propose_arrow_coord()
            counter = 0
            while img._count_colors_withing_circle(propose_coordinates) > 1 and counter < 100:
                propose_coordinates = img._propose_arrow_coord()
                counter += 1

            if counter == 100:
                continue

            pixel_color = img._get_pixel_color(propose_coordinates)
            coord = tuple(map(lambda x: int(x * img.canvas.size[0]), propose_coordinates))
            img.canvas = add_arrow(img.canvas, coord, arrow_size=config.arrow_size)
            img.canvas = apply_antialiasing(img) if config.antialiasing else img.canvas
            image_name = str(uuid.uuid4().hex[:8]) + ".png"
            img._shrink_and_save(save_as=output_folder / "all_images" / image_name)
            writer.writerow([str(Path("all_images") / image_name), pixel_color, coord])

    return str(output_folder)
