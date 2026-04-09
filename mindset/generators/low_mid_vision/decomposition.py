"""decomposition dataset generator."""
import csv
import random
import uuid
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.drawing.shapes.parent import ParentStimuli
from mindset.drawing.shapes.shapes import Shapes
from mindset.generators._base import GeneratorConfig, generator, register


class DrawDecomposition(DrawStimuli):
    """draw decomposition stimuli with paired shapes."""

    def __init__(self, shape_size, shape_color, moving_distance, *args, **kwargs):
        """initialise with shape geometry params."""
        self.shape_size = shape_size
        self.moving_distance = moving_distance
        self.shape_color = shape_color
        super().__init__(*args, **kwargs)

    def generate_canvas(self, shape_1_name, shape_2_name, split_type, cut_rotation, image_rotation=0, image_position=(0.5, 0.5)):
        """render a single decomposition image."""
        parent = ParentStimuli(target_image_size=self.canvas_size, initial_expansion=4 if self.antialiasing else 1)
        shape_1 = Shapes(parent)
        shape_2 = Shapes(parent)

        if shape_1_name.split("_")[0] == "puddle":
            shape_1.add_puddle(size=self.shape_size, seed=shape_1_name.split("_")[1])
            shape_2.add_puddle(size=self.shape_size, seed=shape_2_name.split("_")[1])
        else:
            getattr(shape_1, f"add_{shape_1_name}")(**{"size": self.shape_size})
            getattr(shape_2, f"add_{shape_2_name}")(**{"size": self.shape_size})

        shape_1.rotate(30)
        shape_2.rotate(30)
        shape_2.move_next_to(shape_1, "LEFT")

        if split_type == "no_split":
            shape_1.register()
            shape_2.register()
        elif split_type == "unnatural":
            piece_1, piece_2 = shape_2.cut(reference_point=(0.5, 0.5), angle_degrees=cut_rotation)
            index = np.argmax([piece_1.get_distance_from(shape_1), piece_2.get_distance_from(shape_1)])
            further_piece = [piece_1, piece_2][index]
            closer_piece = [piece_1, piece_2][1 - index]
            further_piece.move_apart_from(closer_piece, self.moving_distance)
            shape_1.register()
            piece_1.register()
            piece_2.register()
        else:
            shape_2.move_apart_from(shape_1, self.moving_distance)
            shape_1.register()
            shape_2.register()

        parent.binary_filter()
        parent.convert_color_to_color((255, 255, 255), self.shape_color)
        parent.move_to(image_position).rotate(image_rotation)
        self.create_canvas()
        parent.add_background(self.background)
        parent.shrink() if self.antialiasing else None
        return parent.canvas


@dataclass
class DecompositionConfig(GeneratorConfig):
    """config for decomposition dataset."""
    moving_distance: int = field(default=60, metadata={"min": 1, "max": 200, "step": 1, "label": "moving distance"})
    shape_color: list = field(default_factory=lambda: [255, 255, 255], metadata={"label": "shape color (RGB)"})
    number_unfamiliar_shapes: int = field(default=5, metadata={"min": 1, "max": 50, "step": 1, "label": "number of unfamiliar shapes"})
    output_folder: str = field(default="data/low_mid_vision/decomposition", metadata={"label": "output folder"})


@register("decomposition", "low_mid_vision")
@generator(DecompositionConfig)
def generate_all(config: DecompositionConfig):
    """generate decomposition dataset."""
    output_folder = Path(config.output_folder)

    familiar_shapes = ["arc", "circle", "square", "rectangle", "polygon", "triangle"]
    unfamiliar_shapes = [f"puddle_{i}" for i in range(config.number_unfamiliar_shapes)]

    combinations_familiar = [{"shape_1_name": s1, "shape_2_name": s2} for s1, s2 in product(familiar_shapes, familiar_shapes)]
    combinations_unfamiliar = [{"shape_1_name": s1, "shape_2_name": s2} for s1, s2 in product(unfamiliar_shapes, unfamiliar_shapes)]

    shapes_types = {"familiar": combinations_familiar, "unfamiliar": combinations_unfamiliar}
    split_types = ["no_split", "unnatural", "natural"]

    ds = DrawDecomposition(
        0.05, config.shape_color, config.moving_distance,
        background=config.background_color, canvas_size=config.canvas_size, antialiasing=config.antialiasing,
    )

    for name_comb in shapes_types:
        for split_type in split_types:
            (output_folder / name_comb / split_type).mkdir(exist_ok=True, parents=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "BackgroundColor", "ShapeType", "SplitType", "LeftShape", "RightShape", "CutRotation", "PairShapeId", "IterNum"])

        for name_comb, combs in tqdm(shapes_types.items()):
            for idx, c in enumerate(tqdm(combs, leave=False)):
                cut_rotation = random.uniform(0, 360)
                for split_type in split_types:
                    img = ds.generate_canvas(c["shape_1_name"], c["shape_2_name"], split_type=split_type, cut_rotation=cut_rotation)
                    unique_hex = uuid.uuid4().hex[:8]
                    path = Path(name_comb) / split_type / f"{unique_hex}.png"
                    img.save(output_folder / path)
                    writer.writerow([path, ds.background, name_comb, split_type, c["shape_1_name"], c["shape_2_name"], cut_rotation, idx])

    return str(output_folder)
