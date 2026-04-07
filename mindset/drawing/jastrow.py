import math
import random

import numpy as np

from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.shape_based_image_generation.modules.parent import ParentStimuli
from mindset.utils.shape_based_image_generation.modules.shapes import Shapes


def get_random_params():
    arc_curvature = np.random.uniform(45, 60)
    width = np.random.uniform(3e-2, 2e-1)
    size_top = np.random.uniform(0.01, 0.06)
    size_bottom = np.random.uniform(0.01, 0.06)
    return arc_curvature, width, size_top, size_bottom


class JastrowParent(ParentStimuli):
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

        # calculate the rotation similarity using the cosine similarity
        rotation_similarity = self.get_rotation_similarity(*rotations)

        # calculate the spatial similarity by calculating the distance between the two shapes and dividing by the max distance
        self.position_similarity = np.linalg.norm(
            np.array(positions[0]) - np.array(positions[1])
        ) / math.sqrt(2)
        self.position_similarity = 1 - self.position_similarity

        # calculate the Jastrow factor as the product of rotation similarity and spatial similarity
        jastrow_factor = rotation_similarity * self.position_similarity
        return jastrow_factor

    def get_rotation_similarity(self, rotation1, rotation2):
        cos_sim = math.cos(math.radians(rotation1 - rotation2))
        return (cos_sim + 1) / 2


class DrawJastrow(DrawStimuli):
    def generate_jastrow_illusion(
        self, arc, width, size_red, size_blue, top_color, type_stimulus
    ):
        correct_stimulus = False
        while not correct_stimulus:
            position_fun = lambda: (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            rotation_fun = lambda: random.uniform(0, 360)
            parent = JastrowParent(
                target_image_size=self.canvas_size,
                initial_expansion=4 if self.antialiasing else 1,
            )
            # With illusory or aligned, arc_1 is always the top 1. Then "on_top" decides it's color.
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

            self.create_canvas()  # dummy call to update the background for rnd-uniform mode
            parent.add_background(self.background)
            parent.shrink() if self.antialiasing else None

            if parent.count_pixels("red") > 0 and parent.count_pixels("blue") > 0:
                correct_stimulus = True
        return parent.canvas
