"""drawing class for 2d transformations dataset."""
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode

from mindset.utils.drawing_utils import DrawStimuli, resize_image_keep_aspect_ratio
from mindset.utils.misc import apply_antialiasing, my_affine


class DrawTransform(DrawStimuli):
    """applies affine transformations to linedrawing images."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_image_transformed(self, image_path, tr, rt, sc, sh):
        """load, resize, paste on canvas, apply affine transform."""
        img = Image.fromarray(
            resize_image_keep_aspect_ratio(np.array(Image.open(image_path)), self.obj_longest_side)
        )
        canvas = self.create_canvas()
        canvas.paste(img, (canvas.size[0] // 2 - img.size[0] // 2, canvas.size[1] // 2 - img.size[1] // 2))
        canvas = my_affine(canvas, translate=tr, angle=rt, scale=sc, shear=sh, interpolation=InterpolationMode.NEAREST, fill=self.background)
        canvas = ImageOps.invert(canvas.convert("L"))
        return apply_antialiasing(canvas) if self.antialiasing else canvas
