"""jastrow illusion dataset generator."""
import csv
import math
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.shape_based_image_generation.modules.parent import ParentStimuli
from mindset.utils.shape_based_image_generation.modules.shapes import Shapes


def get_random_params():
    """return random arc curvature, width, and sizes for jastrow stimuli."""
    arc_curvature = np.random.uniform(45, 60)
    width = np.random.uniform(3e-2, 2e-1)
    size_top = np.random.uniform(0.01, 0.06)
    size_bottom = np.random.uniform(0.01, 0.06)
    return arc_curvature, width, size_top, size_bottom


class JastrowParent(ParentStimuli):
    """parent stimuli with jastrow factor computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_jastrow_factor(self):
        """compute a jastrow factor between 0 and 1 indicating illusion strength."""
        assert (
            len(self.contained_shapes) == 2
        ), "Make sure there are only two arc shapes"
        assert all(
            [shape.shape == "arc" for shape in self.contained_shapes]
        ), "Shapes must be arc"

        rotations = [shape.rotation for shape in self.contained_shapes]
        positions = [shape.position for shape in self.contained_shapes]

        rotation_similarity = self.get_rotation_similarity(*rotations)

        self.position_similarity = np.linalg.norm(
            np.array(positions[0]) - np.array(positions[1])
        ) / math.sqrt(2)
        self.position_similarity = 1 - self.position_similarity

        jastrow_factor = rotation_similarity * self.position_similarity
        return jastrow_factor

    def get_rotation_similarity(self, rotation1, rotation2):
        """compute rotation similarity via cosine."""
        cos_sim = math.cos(math.radians(rotation1 - rotation2))
        return (cos_sim + 1) / 2


class DrawJastrow(DrawStimuli):
    """draws jastrow illusion stimuli."""

    def generate_jastrow_illusion(
        self, arc, width, size_red, size_blue, top_color, type_stimulus
    ):
        """generate a jastrow illusion image with two arcs."""
        correct_stimulus = False
        while not correct_stimulus:
            position_fun = lambda: (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            rotation_fun = lambda: random.uniform(0, 360)
            parent = JastrowParent(
                target_image_size=self.canvas_size,
                initial_expansion=4 if self.antialiasing else 1,
            )
            arcs_sizes = (
                [size_red, size_blue] if top_color == "red" else [size_blue, size_red]
            )
            if type_stimulus == "random_same_size":
                arcs_sizes = [np.mean(arcs_sizes)] * 2
            arc_1 = Shapes(parent=parent)
            arc_1.add_arc(size=arcs_sizes[0], arc=arc, width=width)
            (
                arc_1.move_to(position_fun()).rotate(rotation_fun())
                if "random" in type_stimulus
                else None
            )

            arc_2 = Shapes(parent=parent)
            arc_2.add_arc(size=arcs_sizes[1], arc=arc, width=width)
            (
                arc_2.move_to(position_fun()).rotate(rotation_fun())
                if "random" in type_stimulus
                else None
            )
            parent.center_shapes()

            if type_stimulus == "aligned" or type_stimulus == "illusory":
                arc_1.move_next_to(arc_2, "UP")
                arc_1.set_color(top_color).register()
                arc_2.set_color({"red": "blue", "blue": "red"}[top_color]).register()
            elif "random" in type_stimulus:
                while (
                    arc_1.is_touching(arc_2)
                    or parent.compute_jastrow_factor().round(3) > 0.7
                ):
                    arc_1 = arc_1.move_to(position_fun()).rotate(rotation_fun())
                    arc_2 = arc_2.move_to(position_fun()).rotate(rotation_fun())
                arc_1.set_color("red").register()
                arc_2.set_color("blue").register()

            self.create_canvas()
            parent.add_background(self.background)
            parent.shrink() if self.antialiasing else None

            if parent.count_pixels("red") > 0 and parent.count_pixels("blue") > 0:
                correct_stimulus = True
        return parent.canvas


@dataclass
class JastrowConfig(GeneratorConfig):
    """config for jastrow illusion dataset."""
    num_samples_illusory: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "illusory samples"})
    num_samples_random: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "random samples"})
    num_samples_aligned: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "aligned samples"})
    num_samples_random_same_size: int = field(default=50, metadata={"min": 1, "max": 5000, "step": 1, "label": "random same size samples"})
    output_folder: str = field(default="data/visual_illusions/jastrow", metadata={"label": "output folder"})


@register("jastrow", "visual_illusions")
@generator(JastrowConfig)
def generate_all(config: JastrowConfig):
    """generate jastrow illusion dataset."""
    output_folder = Path(config.output_folder)

    types = ["illusory", "random_same_size", "random", "aligned"]
    on_top_cols = ["red", "blue"]

    for type_name in types:
        for top in on_top_cols:
            subfolder = output_folder / type_name / ("" if "random" in type_name else f"{top}_on_top")
            subfolder.mkdir(exist_ok=True, parents=True)

    ds = DrawJastrow(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    num_samples_map = {
        "illusory": config.num_samples_illusory,
        "aligned": config.num_samples_aligned,
        "random": config.num_samples_random,
        "random_same_size": config.num_samples_random_same_size,
    }

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "BackgroundColor", "Type", "ArcSize", "ArcWidth", "SizeTop", "SizeBottom", "OnTop", "SampleNum", "SizeRed", "SizeBlue", "SizeRedMinusBlue"])

        for type_name in tqdm(types):
            num_samples = num_samples_map[type_name]
            for idx in tqdm(range(num_samples), leave=False):
                for top_color in on_top_cols:
                    arc_curvature, width, size_red, size_blue = get_random_params()
                    size_blue = size_red if type_name == "illusory" else size_blue
                    img = ds.generate_jastrow_illusion(
                        arc_curvature, width, size_red, size_blue, top_color, type_stimulus=type_name,
                    )
                    unique_hex = uuid.uuid4().hex[:8]
                    path = (
                        Path(type_name)
                        / ("" if type_name in ["random", "random_same_size"] else f"{top_color}_on_top")
                        / f"red{size_red:.2f}_blue{size_blue:.2f}_{idx}_{unique_hex}.png"
                    )
                    img.save(output_folder / path)
                    writer.writerow([
                        path, config.background_color, type_name, arc_curvature, width,
                        size_red, size_blue,
                        "none" if type_name in ["random", "random_same_size"] else top_color,
                        idx, size_red, size_blue, size_red - size_blue,
                    ])

    return str(output_folder)
