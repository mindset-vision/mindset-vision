"""shared drawing classes for linedrawings, global change, and silhouettes generators."""
import cv2
import numpy as np
from PIL import Image, ImageOps

from mindset.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawLinedrawingsSimple(DrawStimuli):
    """draws simple linedrawings (white stroke on canvas)."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        """load and paste a linedrawing onto a canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        img = ImageOps.invert(Image.fromarray(img).convert("L"))

        canvas = paste_linedrawing_onto_canvas(img, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


class DrawLinedrawingsGlobalChange(DrawStimuli):
    """draws whole/fragmented/frankenstein linedrawings (baker & elder 2022 style)."""

    def __init__(self, obj_longest_side, convert_to_silhouettes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.convert_to_silhouettes = convert_to_silhouettes

    def get_linedrawings(self, image_path, type):
        """produce whole, fragmented, or frankenstein version of a linedrawing."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.convert_to_silhouettes:
            contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            mask = np.ones_like(img) * 255

            cv2.drawContours(mask, contours, -1, (0), thickness=cv2.FILLED)

            [
                cv2.drawContours(mask, [c], -1, (0), thickness=cv2.FILLED)
                for c in contours
            ]
        else:
            mask = cv2.bitwise_not(binary_img)

        silhouette = Image.fromarray(mask)
        width, height = silhouette.size
        top_half = silhouette.crop((0, 0, width, height // 2))
        bottom_half = silhouette.crop((0, height // 2, width, height))

        if type in ["frankenstein", "fragmented"]:
            top_half = top_half.transpose(Image.FLIP_LEFT_RIGHT)

        top_half_np = np.array(top_half)
        bottom_half_np = np.array(bottom_half)
        if type == "frankenstein":
            top = np.min(np.where(top_half_np[-1] == 0))
            bottom = np.min(np.where(bottom_half_np[0] == 0))
        elif type == "fragmented":
            top = np.min(np.where(top_half_np[-1] == 0))
            bottom = np.max(np.where(bottom_half_np[0] == 0))
        else:  # type == "whole":
            bottom = 0
            top = 0
        top_offset = max(0, bottom - top)
        bottom_offset = max(0, top - bottom)

        new_canvas = Image.fromarray(
            np.ones(
                (
                    self.canvas_size[0],
                    max(top_half.size[0], bottom_half.size[0])
                    + max(top_offset, bottom_offset),
                )
            )
            * 255
        ).convert("RGB")

        new_canvas.paste(
            top_half,
            (
                top_offset,
                new_canvas.size[1] // 2 - top_half.size[1],
            ),
        )
        new_canvas.paste(
            bottom_half,
            (
                bottom_offset,
                new_canvas.size[1] // 2,
            ),
        )

        cs = tuple(
            np.array(self.canvas_size)
            * max(np.array(new_canvas.size) / self.canvas_size)
        )
        canvas = self.create_canvas(size=[int(i) for i in cs])
        canvas.paste(
            ImageOps.invert(new_canvas.convert("L")),
            (
                canvas.size[0] // 2 - new_canvas.size[0] // 2,
                canvas.size[1] // 2 - new_canvas.size[1] // 2,
            ),
        )
        canvas = canvas.resize(self.canvas_size)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


class DrawLinedrawingsSilhouettes(DrawStimuli):
    """draws silhouettes from linedrawings or silhouette inputs."""

    def __init__(self, obj_longest_side, input_image_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.input_image_type = input_image_type

    def get_linedrawings(self, image_path):
        """load image and convert to silhouette on canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.input_image_type == "linedrawings":
            contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            mask = np.ones_like(img) * 255

            cv2.drawContours(mask, contours, -1, (0), thickness=cv2.FILLED)

            [
                cv2.drawContours(mask, [c], -1, (0), thickness=cv2.FILLED)
                for c in contours
            ]
        else:
            mask = cv2.bitwise_not(binary_img)
        mask = ImageOps.invert(Image.fromarray(mask).convert("L"))

        canvas = paste_linedrawing_onto_canvas(mask, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas
