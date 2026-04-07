import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode, transforms

from mindset.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import (
    apply_antialiasing,
    get_affine_rnd_fun,
    my_affine,
)


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized linedrawings using line-segment patterns."""

    def add_line_pattern(self, canvas, line_length=10, slope_rad=0, density=1):
        """tile a canvas with short line segments at a given slope."""
        width, height = canvas.size

        draw = ImageDraw.Draw(canvas)

        line_segment = [
            (0, 0),
            (line_length * math.cos(slope_rad), line_length * math.sin(slope_rad)),
        ]
        horizontal_spacing = int(np.round(line_length / density))
        vertical_spacing = int(np.round(line_length / density))
        noise = 1
        for y in range(
            0,
            height,
            np.round(
                line_length * np.abs(math.sin(slope_rad)) + vertical_spacing
            ).astype(int),
        ):
            for x in range(
                0,
                width,
                np.round(line_length * math.cos(slope_rad) + horizontal_spacing).astype(
                    int
                ),
            ):
                x_offset, y_offset = random.randint(-noise, noise), random.randint(
                    -noise, noise
                )
                shifted_line_segment = [
                    (point[0] + x + x_offset, point[1] + y + y_offset)
                    for point in line_segment
                ]
                draw.line(shifted_line_segment, **self.line_args)
        return canvas

    def __init__(
        self,
        texturize_background,
        texturize_foreground,
        obj_longest_side,
        transform_code,
        density,
        *args,
        **kwargs,
    ):
        self.obj_longest_side = obj_longest_side
        self.density = density
        self.transform_code = transform_code
        self.texturize_foreground = texturize_foreground
        self.texturize_background = texturize_background
        super().__init__(*args, **kwargs)

    def draw_pattern(self, img_path, slope_rad, line_length):
        """draw a texturized linedrawing with line-segment patterns."""
        expand_factor = 1.5
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = get_mask_from_linedrawing(
            resize_image_keep_aspect_ratio(img, self.obj_longest_side), fill=True
        )

        bg_canvas = self.create_canvas(
            size=tuple((np.array(self.canvas_size) * expand_factor).astype(int))
        )
        if self.texturize_background:
            bg_canvas = self.add_line_pattern(
                canvas=bg_canvas,
                line_length=line_length,
                slope_rad=slope_rad,
                density=self.density,
            )
        perpendicular_radian = (
            (slope_rad + math.pi / 2)
            if abs(slope_rad + math.pi / 2) <= math.pi / 2
            else (slope_rad - math.pi / 2)
        )

        fg_canvas = self.create_canvas(size=mask.size, background=self.background)
        if self.texturize_foreground:
            canvas_foreg_text = self.add_line_pattern(
                fg_canvas,
                line_length,
                perpendicular_radian,
                self.density,
            )

        bg_canvas.paste(
            canvas_foreg_text,
            (
                bg_canvas.size[0] // 2 - mask.size[0] // 2,
                bg_canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )

        af = get_affine_rnd_fun(self.transform_code)()
        img = my_affine(
            bg_canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return apply_antialiasing(img) if self.antialiasing else img
