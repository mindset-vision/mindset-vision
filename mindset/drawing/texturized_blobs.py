import math
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms

from mindset.utils.drawing_utils import DrawStimuli, get_mask_from_linedrawing
from mindset.utils.misc import apply_antialiasing
from mindset.utils.shape_based_image_generation.modules.parent import ParentStimuli
from mindset.utils.shape_based_image_generation.modules.shapes import Shapes

characters = string.ascii_letters + string.digits + string.punctuation


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized blob shapes using character patterns."""

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
    ):
        """create a canvas tiled with a repeated character pattern."""
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple([np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2)
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

    def draw_blob(self, blob_id):
        """draw a blob shape on a canvas."""
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color(self.fill)

        blob.add_puddle(size=0.2, seed=blob_id)

        blob.register()
        self.create_canvas()
        parent.add_background(self.background)

        return apply_antialiasing(parent.canvas) if self.antialiasing else parent.canvas

    def draw_pattern(
        self,
        blob_id,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        """draw a texturized blob with foreground and background character patterns."""
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color((0, 0, 0, 255))
        blob.add_puddle(size=0.2, seed=blob_id)
        blob.register()
        self.create_canvas()
        parent.add_background((255, 255, 255))

        mask = get_mask_from_linedrawing(
            np.array(parent.canvas.convert("L")), fill=True
        )
        canvas_bg = self.get_canvas_char_pattered(
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
        canvas_fg = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
        )

        canvas_bg.paste(
            canvas_fg,
            (
                canvas_bg.size[0] // 2 - mask.size[0] // 2,
                canvas_bg.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas_bg) if self.antialiasing else canvas_bg
