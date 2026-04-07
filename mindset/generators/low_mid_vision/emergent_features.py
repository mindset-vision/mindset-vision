"""emergent features dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.drawing.base import DrawStimuli
from mindset.utils.misc import apply_antialiasing


# ---------------------------------------------------------------------------
# drawing classes
# ---------------------------------------------------------------------------

class ConstrainedError(Exception):
    """raised when constraint-based point generation fails."""
    pass


class DrawEmergentFeaturesdots(DrawStimuli):
    """draws emergent feature dot patterns (proximity, orientation, linearity)."""

    def __init__(self, min_dist_borders=0, min_dist_bw_points=0, *args, **kwargs):
        self.min_dist_borders = min_dist_borders
        self.min_dist_bw_points = min_dist_bw_points
        super().__init__(*args, **kwargs)

    def get_empty_single(self):
        """generate a single random point within canvas bounds."""
        r = self.line_args["width"]
        x0 = np.random.randint(
            0 + r + self.min_dist_borders + self.borders_width,
            self.canvas_size[0] - r - self.borders_width - self.min_dist_borders,
        )
        y0 = np.random.randint(
            0 + r + self.min_dist_borders + self.borders_width,
            self.canvas_size[1] - r - self.borders_width - self.min_dist_borders,
        )
        return (((x0, y0),), ())

    def get_all_sets(self):
        """generate all emergent feature point sets."""
        while True:
            try:
                pp_empty = (), ()
                pp_empty_single = self.get_empty_single()
                pp_single = self.get_pair_points(pp_empty_single)
                pp_proximity = self.get_proximity_points(pp_single)
                pp_orientation = self.get_orientation_points(pp_single)
                pp_linearity = self.get_linearity_o_points(pp_orientation)
                names = [
                    "empty",
                    "empty-single",
                    "single",
                    "proximity",
                    "orientation",
                    "linearity",
                ]
                a = self.draw_all_dots(pp_single[0])
                pps = [
                    pp_empty,
                    pp_empty_single,
                    pp_single,
                    pp_proximity,
                    pp_orientation,
                    pp_linearity,
                ]
                ppsdict = {k: v for k, v in zip(names, pps)}
                sets = {
                    n: (self.draw_all_dots(pp[0]), self.draw_all_dots(pp[1]))
                    for n, pp in zip(names, pps)
                }
                return sets, ppsdict
            except ConstrainedError:
                continue

    def from_radians_get_line(self, radius, r):
        """convert radians to line endpoint coordinates."""
        return [
            (radius * np.cos(r), radius * np.sin(r)),
            (-radius * np.cos(r), -radius * np.sin(r)),
        ]

    def loc_to_int(self, loc):
        """convert location coordinates to integers."""
        return [tuple([int(i) for i in locc]) for locc in loc]

    def center_at_cavas(self, loc):
        """center coordinates at canvas midpoint."""
        return self.loc_to_int(np.array(loc) + np.array(self.canvas_size) / 2)

    def add_circles_to_loc(self, l, draw):
        """add circles at both endpoints of a location pair."""
        self.circle(draw, l[0], self.line_args["width"] // 2)
        self.circle(draw, l[1], self.line_args["width"] // 2)

    def get_pair_points(self, one_point=None):
        """generate a pair of random points with minimum distance constraint."""
        repeat = True
        while repeat:
            r = self.line_args["width"]
            if not one_point:
                x0 = np.random.randint(
                    0 + r + self.borders_width + self.min_dist_borders,
                    self.canvas_size[0] - r - self.borders_width - self.min_dist_borders,
                )
                y0 = np.random.randint(
                    0 + r + self.borders_width + self.min_dist_borders,
                    self.canvas_size[1] - r - self.borders_width - self.min_dist_borders,
                )
            else:
                (((x0, y0),), _) = one_point

            x1 = np.random.randint(
                0 + r + self.min_dist_borders + self.borders_width,
                self.canvas_size[0] - r - self.borders_width - self.min_dist_borders,
            )
            y1 = np.random.randint(
                0 + r + self.min_dist_borders + self.borders_width,
                self.canvas_size[1] - r - self.borders_width - self.min_dist_borders,
            )

            if (
                np.linalg.norm([np.array([x0, y0]) - np.array([x1, y1])])
                > self.min_dist_bw_points
            ):
                repeat = False

        return ((x0, y0),), ((x1, y1),)

    def get_proximity_points(self, pp=None, dist=None, **kwargs):
        """generate proximity grouping points between a pair."""
        r = self.line_args["width"]
        stop = False
        count = 0
        while not stop:
            if pp is None:
                ((x0, y0),), ((x1, y1),) = self.get_pair_points(**kwargs)
            else:
                ((x0, y0),), ((x1, y1),) = pp

            A = np.array([x0, y0])
            B = np.array([x1, y1])
            if dist is None:
                dist = np.random.uniform((r * 2) / (np.linalg.norm(A - B)), 0.4)
            else:
                stop = True
            L = A + dist * (B - A)
            xx_lin, yy_lin = int(L[0]), int(L[1])
            if (
                np.linalg.norm([xx_lin, yy_lin] - A) > r * 2
                and np.linalg.norm([xx_lin, yy_lin] - B) > r * 2
                and self.canvas_size[0] - r - self.borders_width - self.min_dist_borders
                > xx_lin
                > 0 + r + self.borders_width + self.min_dist_borders
                < yy_lin
                < self.canvas_size[1] - r - self.borders_width - self.min_dist_borders
            ):
                stop = True
            count += 1
            if count > 100:
                raise ConstrainedError("Can't generate proximity points")

        return (
            ((x0, y0), (xx_lin, yy_lin)),
            ((x1, y1), (xx_lin, yy_lin)),
        )

    def get_orientation_points(self, pp=None, **kwargs):
        """generate equidistant orientation points using circle intersection."""
        count = 0
        stop = False
        while not stop:
            r = self.line_args["width"]
            if pp is None:
                ((x0, y0),), ((x1, y1),) = self.get_pair_points(**kwargs)
            else:
                ((x0, y0),), ((x1, y1),) = pp

            diagonal = np.sqrt(self.canvas_size[0] ** 2 + self.canvas_size[1] ** 2)
            distance = np.linalg.norm(np.array([x0, y0]) - np.array([x1, y1]))
            radius_circles = np.random.uniform(
                np.max([r, distance / 2]), diagonal * 0.7
            )

            xor0, yor0, xor1, yor1 = self.intersections(
                x0, y0, radius_circles, x1, y1, radius_circles
            )

            if np.random.randint(1) == 0:
                xx_equi, yy_equi = xor0, yor0
            else:
                xx_equi, yy_equi = xor1, yor1

            if (
                0 + r + self.borders_width + self.min_dist_borders
                < xx_equi
                < self.canvas_size[0] - r - self.borders_width - self.min_dist_borders
                and 0 + r + self.borders_width + self.min_dist_borders
                < yy_equi
                < self.canvas_size[1] - r - self.borders_width - self.min_dist_borders
            ):
                stop = True
            count += 1
            if count > 20:
                raise ConstrainedError("Can't generate orientation points")

        return ((x0, y0), (xx_equi, yy_equi)), ((x1, y1), (xx_equi, yy_equi))

    def plot_all_points(self, pps):
        """plot all point sets as dot images."""
        ims = [self.draw_all_dots(pp) for idx, pp in enumerate(pps)]
        self.plot_all_imgs(ims)

    def plot_all_imgs(self, im):
        """plot all images in a grid."""
        fig, ax = plt.subplots(2, int(np.ceil(len(im) / 2)))
        ax = np.array([ax]) if len(im) == 1 else ax.flatten()
        [i.axis("off") for i in ax.flatten()]
        for idx, i in enumerate(im):
            ax[idx].imshow(i)

    def get_linearity_o_points(self, pp=None, **kwargs):
        """generate collinear points from orientation point set."""
        r = self.line_args["width"]
        count = 0
        stop = False
        while not stop:
            r = self.line_args["width"]
            if pp is None:
                ((x0, y0), (xor, yor)), (
                    (x1, y1),
                    (xor, yor),
                ) = self.get_orientation_points(**kwargs)
            else:
                ((x0, y0), (xor, yor)), ((x1, y1), (xor, yor)) = pp

            A = np.array([x0, y0])
            B = np.array([xor, yor])

            dd = np.random.uniform((r * 2) / (np.linalg.norm(A - B)), 0.4)
            L = B + dd * (A - B)
            xx_lin, yy_lin = int(L[0]), int(L[1])

            if (
                np.linalg.norm([xx_lin, yy_lin] - A) > r * 2
                and np.linalg.norm([xx_lin, yy_lin] - B) > r * 2
                and self.canvas_size[0] - r - self.borders_width + self.min_dist_borders
                > xx_lin
                > 0 + r + self.borders_width + self.min_dist_borders
                < yy_lin
                < self.canvas_size[1] - r - self.borders_width - self.min_dist_borders
            ):
                stop = True
            count += 1
            if count > 100:
                raise ConstrainedError("Can't generate linearity/o points")

        return ((x0, y0), (xor, yor), (xx_lin, yy_lin)), (
            (x1, y1),
            (xor, yor),
            (xx_lin, yy_lin),
        )

    def draw_set(self, pps):
        """draw incremental sets of dots from point configurations."""
        r = self.line_args["width"]
        images_set = []
        for idx, s in enumerate(pps[0]):
            images = []
            for im_pp in pps:
                im = self.create_canvas()
                draw = ImageDraw.Draw(im)
                for p in im_pp[0 : idx + 1]:
                    self.circle(draw, (p[0], p[1]), radius=r)
                images.append(im)
            images_set.append(images)
        return images_set

    def draw_all_dots(self, pps):
        """draw all dots on a canvas and optionally apply antialiasing."""
        r = self.line_args["width"]
        im = self.create_canvas()
        draw = ImageDraw.Draw(im)
        for p in pps:
            self.circle(draw, (p[0], p[1]), radius=r)
        return apply_antialiasing(im) if self.antialiasing else im

    def intersections(self, x0, y0, r0, x1, y1, r1):
        """find intersection points of two circles."""
        import math

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if d > r0 + r1:
            return {}
        if d < abs(r0 - r1):
            return {}
        if d == 0 and r0 == r1:
            return {}
        else:
            a = (r0**2 - r1**2 + d**2) / (2 * d)
            h = math.sqrt(r0**2 - a**2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d
            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d
            return int(x3), int(y3), int(x4), int(y4)


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------

@dataclass
class EmergentFeaturesConfig(GeneratorConfig):
    """config for emergent features dataset."""
    num_samples: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "number of samples"})
    output_folder: str = field(default="data/low_mid_level_vision/emergent_features", metadata={"label": "output folder"})


@register("emergent_features", "low_mid_vision")
@generator(EmergentFeaturesConfig)
def generate_all(config: EmergentFeaturesConfig):
    """generate emergent features dataset."""
    output_folder = Path(config.output_folder)
    all_types = ["single", "proximity", "orientation", "linearity"]

    for t in all_types:
        for pair in ["a", "b"]:
            (output_folder / t / pair).mkdir(exist_ok=True, parents=True)

    ds = DrawEmergentFeaturesdots(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        width=10,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "PairA/B", "SampleId"])
        for i in tqdm(range(config.num_samples)):
            all_sets = ds.get_all_sets()[0]
            for t in tqdm(all_types, leave=False):
                for ip, pair in enumerate(["a", "b"]):
                    path = Path(t) / pair / f"{i}.png"
                    all_sets[t][ip].save(output_folder / path)
                    writer.writerow([path, t, ds.background, pair, i])

    return str(output_folder)
