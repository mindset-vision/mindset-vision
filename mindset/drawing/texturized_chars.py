import math
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms

from mindset.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing

characters = string.ascii_letters + string.digits + string.punctuation


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized linedrawings using character patterns."""

    def __init__(self, obj_longest_side, transform_code, *args, **kwargs):
        self.transform_code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_canvas_char_pattered(
        self,
        size,
        tile_char,
        font_size,
        spacing=0,
        rotation_angle=45,
        font_path="mindset/assets/arial.ttf",
        background=None,
    ):
        """create a canvas tiled with a repeated character pattern."""
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple(
                [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
            ),
            background=background,
        )

        bbox = font.getbbox(tile_char + " " * spacing)
        char_width = bbox[2]
        char_heights = bbox[3]

        width, height = img.size
        num_x = width // char_width + 2
        draw = ImageDraw.Draw(img)
        tile_string = (tile_char + " " * spacing) * num_x

        for y in range(-20, height, char_heights):
            draw.text(
                (1, y),
                tile_string,
                fill=self.line_args["fill"],
                font=font,
            )
        if rotation_angle > 0:
            img = img.rotate(rotation_angle, resample=Image.Resampling.NEAREST)

        img = transforms.CenterCrop((size[1], size[0]))(img)
        return img

    def draw_pattern(
        self,
        img_path,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        """draw a texturized linedrawing with foreground and background char patterns."""
        opencv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        mask = get_mask_from_linedrawing(opencv_img, fill=True)
        canvas = self.get_canvas_char_pattered(
            size=tuple(np.array(self.canvas_size)),
            tile_char=background_char,
            font_size=background_font_size,
            rotation_angle=np.rad2deg(rotation_angle_rad),
            spacing=0,
        )

        perpendicular_radian = (
            (rotation_angle_rad + math.pi / 2)
            if abs(rotation_angle_rad + math.pi / 2) <= math.pi / 2
            else (rotation_angle_rad - math.pi / 2)
        )
        canvas_foreg_text = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
            background=self.background,
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas) if self.antialiasing else canvas
