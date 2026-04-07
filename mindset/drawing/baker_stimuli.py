import cv2
from PIL import Image, ImageOps

from mindset.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawBakerStimuli(DrawStimuli):
    """draws baker 2022 linedrawing stimuli."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        """load, threshold and paste a linedrawing onto a canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.bitwise_not(binary_img)
        mask = ImageOps.invert(Image.fromarray(mask).convert("L"))

        canvas = paste_linedrawing_onto_canvas(mask, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas
