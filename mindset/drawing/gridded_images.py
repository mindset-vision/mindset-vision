import cv2
import numpy as np
from PIL import Image, ImageOps

from mindset.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawGriddedImages(DrawStimuli):
    """draws linedrawings with grid-based segment deletions."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def apply_grid_mask(
        self,
        image_path,
        grid_size,
        grid_thickness=1,
        grid_shift=0,
        rotation_degrees=0,
        complement=False,
    ):
        """apply a rotated grid mask to delete segments of a linedrawing."""
        opencv_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        img = ImageOps.invert(Image.fromarray(img).convert("L"))

        img = np.array(
            paste_linedrawing_onto_canvas(
                img, self.create_canvas(), self.line_args["fill"]
            )
        )

        height, width, _ = img.shape

        mask = np.full((height * 2, width * 2), False)

        for i in range(grid_shift, mask.shape[0], grid_size):
            mask[i : i + grid_thickness, :] = 1

        rotated_mask = np.array(
            Image.fromarray(mask).rotate(rotation_degrees, expand=True, fillcolor=(0))
        )
        rotated_mask = rotated_mask[
            rotated_mask.shape[0] // 2
            - height // 2 : rotated_mask.shape[0] // 2
            - height // 2
            + height,
            rotated_mask.shape[1] // 2
            - width // 2 : rotated_mask.shape[1] // 2
            - width // 2
            + width,
        ]

        if complement:
            img[~rotated_mask] = self.background
        else:
            img[rotated_mask] = self.background
        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img
