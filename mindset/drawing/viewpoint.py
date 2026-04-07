import os

import numpy as np
from PIL import Image

from mindset.utils.drawing_utils import DrawStimuli, resize_image_keep_aspect_ratio
from mindset.utils.misc import apply_antialiasing


class DrawETH(DrawStimuli):
    """draws ETH-80 dataset images with background removal."""

    def __init__(self, obj_longest_side, map_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.map_path = map_path

    def create_ETH(self, img_path):
        """load an ETH-80 image, crop to object bounds, and paste on canvas."""
        path_parts = img_path.split(os.sep)
        desired_path = os.path.join(*path_parts[-3:]).rstrip(".png")
        map_path = f"{self.map_path}/{desired_path}-map.png"

        map_pil = Image.open(map_path).convert("L")

        mask_array = np.array(map_pil)

        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        cropped_image = Image.open(img_path).crop((xmin, ymin, xmax, ymax))
        cropped_map_pil = map_pil.crop((xmin, ymin, xmax, ymax))

        canvas_only_obj = self.create_canvas(size=cropped_image.size)
        canvas_only_obj.paste(cropped_image, mask=cropped_map_pil)
        canvas_only_obj = Image.fromarray(
            resize_image_keep_aspect_ratio(
                np.array(canvas_only_obj), self.obj_longest_side
            )
        )

        canvas = self.create_canvas()
        paste_position = (
            (canvas.size[0] - canvas_only_obj.size[0]) // 2,
            (canvas.size[1] - canvas_only_obj.size[1]) // 2,
        )
        canvas.paste(canvas_only_obj, paste_position)

        return apply_antialiasing(canvas) if self.antialiasing else canvas
