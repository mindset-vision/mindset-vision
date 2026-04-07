import cv2
import numpy as np
from PIL import Image, ImageOps

from mindset.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawDottedImage(DrawStimuli):
    """draws dotted versions of linedrawing images."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def dotted_image(self, image_path, dot_distance, dot_size):
        """convert a linedrawing to a dotted contour image."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, b = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dotted_img = np.ones_like(img) * 255

        def draw_dot(image, x, y, size, color):
            half_size = size // 2
            cv2.rectangle(
                image,
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                color,
                -1,
            )

        for contour in contours:
            for i, point in enumerate(contour):
                if i % dot_distance == 0:
                    x, y = point[0]
                    draw_dot(dotted_img, x, y, dot_size, color=0)

        dotted_img = Image.fromarray(dotted_img)
        dotted_img = ImageOps.invert(dotted_img.convert("L"))

        canvas = paste_linedrawing_onto_canvas(
            dotted_img, self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas
