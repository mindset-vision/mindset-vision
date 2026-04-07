"""affine transformation helpers and random range sampling."""

import random

from torchvision.transforms import functional as F


def draw_random_from_ranges(ranges):
    """draw a random number uniformly distributed within a set of ranges."""
    total_span = sum(high - low for low, high in ranges)
    random_point = random.uniform(0, total_span)
    running_total = 0
    for low, high in ranges:
        running_total += high - low
        if random_point <= running_total:
            return random_point - (running_total - (high - low)) + low


def get_affine_rnd_fun(transf_values):
    """build a callable that returns random affine parameters from given ranges."""
    transf_ranges = {
        k: [v] if not isinstance(v[0], list) else v for k, v in transf_values.items()
    }

    tr = lambda: [
        (
            draw_random_from_ranges(transf_ranges["translation_X"])
            if "translation_X" in transf_values and transf_values["translation_X"]
            else 0
        ),
        (
            draw_random_from_ranges(transf_ranges["translation_Y"])
            if "translation_Y" in transf_values and transf_values["translation_Y"]
            else 0
        ),
    ]

    scale = (
        (lambda: draw_random_from_ranges(transf_ranges["scale"]))
        if "scale" in transf_values and transf_values["scale"]
        else lambda: 1.0
    )
    rot = (
        (lambda: draw_random_from_ranges(transf_ranges["rotation"]))
        if "rotation" in transf_values and transf_values["rotation"]
        else lambda: 0
    )
    return lambda: {"rt": rot(), "tr": tr(), "sc": scale(), "sh": 0.0}


def my_affine(img, translate, **kwargs):
    """apply an affine transform with translate as fractions of image size."""
    return F.affine(
        img,
        translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])],
        **kwargs,
    )
