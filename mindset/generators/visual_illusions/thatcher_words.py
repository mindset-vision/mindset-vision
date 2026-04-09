"""thatcher illusion words dataset generator."""

import csv
import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import new
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from scipy import ndimage
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register

# ---------------------------------------------------------------------------
# corpus reader
# ---------------------------------------------------------------------------


def read_corpus(path: Path):
    """read a word corpus from a text file."""
    corpus = open(path, "r").read()
    corpus: list[str] = [w for w in corpus.split("\n") if w != ""]
    corpus: list[str] = [w for w in corpus if len(w)]
    return corpus


# ---------------------------------------------------------------------------
# drawing class
# ---------------------------------------------------------------------------


class CreateData(DrawStimuli):
    """creates thatcherized word images."""

    def __init__(
        self,
        variance_font,
        coefficient_space,
        coefficient_translation,
        word_folder,
        font_folder,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.font_folder = font_folder
        self.word_folder = word_folder
        self.variance_font = variance_font
        self.coefficient_space = coefficient_space
        self.coefficient_translate = coefficient_translation

    def find_letter_bbox(self, image):
        """find the bounding box of the letter in the image based on the letter's color."""
        pixels = image.load()
        width, height = image.size

        min_x, min_y = width, height
        max_x, max_y = 0, 0

        for x in range(width):
            for y in range(height):
                if not pixels[x, y][:3] == self.background:
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    max_x, max_y = max(max_x, x), max(max_y, y)

        return min_x, min_y, max_x + 1, max_y + 1

    def textsize_for_drawing(self, text, font):
        """compute text bounding box size for drawing."""
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def real_bbox_text(self, text, font):
        """compute real bounding box of rendered text."""
        shape_letter = self.textsize_for_drawing(text, font=font)
        canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0))
        Draw(canvas_letter).text((0, 0), text, fill=self.fill, font=font)
        bbox = self.find_letter_bbox(canvas_letter)
        return bbox

    def create_images(self, word, name_font, size_font, idx_letters_to_rotate):
        """create a word image with optionally rotated letters."""
        self.word = word.upper()
        self.name_font = name_font
        self.size_font = size_font

        canvas = self.create_canvas()
        self.width, self.height = canvas.size[0], canvas.size[1]

        self.center: tuple = (self.width / 2, self.height / 2)

        self.letters_size_font_shift: list = [
            random.randint(-self.variance_font, self.variance_font)
            for _ in range(0, len(self.word))
        ]
        self.width_letters_cumulative = self.get_width_letters(Draw(canvas))
        self.shape_half_word: tuple = (
            self.width_letters_cumulative[-1] / 2,
            self.get_max_height(Draw(canvas)) / 2,
        )
        self.average_diagonal_length = self.get_average_diagonal_length(Draw(canvas))
        self.radius_translate = self.get_radius_translate()

        self.initial_h_pos = (
            self.width * 0.5 + (0.03 * self.width) - self.shape_half_word[0]
        )
        self.v_pos_base = self.height * 0.5
        for i in range(0, len(self.word)):
            letter_font = self.get_zoomed_font(self.letters_size_font_shift[i])

            shape_letter = self.textsize_for_drawing(self.word[i], font=letter_font)
            canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0))
            Draw(canvas_letter).text(
                (0, 0), self.word[i], fill=self.fill, font=letter_font
            )
            bbox = self.real_bbox_text(self.word[i], letter_font)
            width_l, height_l = bbox[2] - bbox[0], bbox[3] - bbox[1]
            canvas_letter = canvas_letter.crop(bbox)
            if i in idx_letters_to_rotate:
                canvas_letter = self.rotate(canvas_letter, angle=180)

            canvas_letter.putdata(self.set_background_transparent(canvas_letter))
            canvas.paste(
                im=canvas_letter,
                box=self.get_final_position_letter(i, height_l),
                mask=canvas_letter,
            )

        return canvas

    def rotate(self, image, angle: int):
        """rotate an image by a given angle."""
        image_array = np.array(image)
        rotated_array = ndimage.rotate(
            image_array, angle, cval=0.0, reshape=True, mode="constant", prefilter=True
        )
        return Image.fromarray(rotated_array)

    def get_radius_translate(self):
        """compute translation radius from coefficient and diagonal."""
        return self.coefficient_translate * self.average_diagonal_length

    def size_for_drawing(self, text, font):
        """compute text size for drawing purposes."""
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def get_w_h_letters(self, text, font):
        """get width and height of rendered text from real bbox."""
        bbox = self.real_bbox_text(text, font=font)
        width_l, height_l = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return width_l, height_l

    def get_width_letters(self, draw: Draw) -> list:
        """compute cumulative letter widths for positioning."""
        shapes_letters: list = [
            self.size_for_drawing(
                self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i])
            )
            for i in range(0, len(self.word))
        ]
        width_letters: list = [i[0] * self.coefficient_space for i in shapes_letters]
        width_letters.insert(0, 0)
        return np.cumsum(width_letters)

    def get_average_diagonal_length(self, draw: Draw) -> float:
        """compute average diagonal length of letters."""
        shapes_letters: list = [
            self.get_w_h_letters(
                self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i])
            )
            for i in range(0, len(self.word))
        ]
        return sum([math.sqrt(s[0] ** 2 + s[1] ** 2) for s in shapes_letters]) / len(
            shapes_letters
        )

    def get_max_height(self, draw: Draw) -> float:
        """compute max height across all letters in word."""
        shapes_letters: list = [
            self.get_w_h_letters(
                self.word[i],
                font=truetype(
                    str(Path("mindset", "assets", "words", "fonts") / self.name_font),
                    self.size_font + self.letters_size_font_shift[i],
                ),
            )
            for i in range(0, len(self.word))
        ]
        return max([s[1] for s in shapes_letters])

    def get_zoomed_font(self, zoom: int):
        """get font with size adjusted by zoom offset."""
        return truetype(
            str(Path("mindset", "assets", "words", "fonts") / self.name_font),
            self.size_font + zoom,
        )

    def set_background_transparent(self, image) -> list:
        """set background-colored pixels to transparent."""
        return [
            (lambda i: (*self.background, 0) if i[:3] == self.background else i)(i)
            for i in image.getdata()
        ]

    def get_translation_vector(self, radius: float) -> tuple:
        """get a random translation vector within given radius."""
        r = radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        return (r * math.cos(theta), r * math.sin(theta))

    def get_final_position_letter(self, instance, h_l) -> tuple:
        """compute final pixel position for a letter instance."""
        position_letter: tuple = (
            self.initial_h_pos + self.width_letters_cumulative[instance],
            self.height * 0.5 - h_l // 2,
        )
        position_letter: tuple = tuple(
            map(
                sum,
                zip(
                    self.get_translation_vector(self.radius_translate), position_letter
                ),
            )
        )
        return (int(position_letter[0]), int(position_letter[1]))


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------

LETTERS_NOT_TO_ROTATE = ["O", "W", "M", "N"]


@dataclass
class ThatcherWordsConfig(GeneratorConfig):
    """config for thatcher illusion words dataset."""

    jittery: float = field(
        default=0.04, metadata={"min": 0.0, "max": 1.0, "step": 0.01, "label": "jitter"}
    )
    num_words: int = field(
        default=100,
        metadata={"min": 1, "max": 1000, "step": 1, "label": "number of words"},
    )
    num_samples_per_word: int = field(
        default=5,
        metadata={"min": 1, "max": 100, "step": 1, "label": "samples per word"},
    )
    num_letters_per_word: list = field(
        default_factory=lambda: [5, 9], metadata={"label": "letters per word range"}
    )
    num_letters_to_rotate: int = field(
        default=2,
        metadata={"min": 1, "max": 10, "step": 1, "label": "letters to rotate"},
    )
    size_fonts: list = field(
        default_factory=lambda: [18, 35], metadata={"label": "font size range"}
    )
    use_random_words: bool = field(
        default=False, metadata={"label": "use random words"}
    )
    output_folder: str = field(
        default="data/visual_illusions/thatcher_words",
        metadata={"label": "output folder"},
    )


@register("thatcher_words", "visual_illusions")
@generator(ThatcherWordsConfig)
def generate_all(config: ThatcherWordsConfig):
    """generate thatcher illusion words dataset."""
    output_folder = Path(config.output_folder)
    word_folder = Path("mindset", "assets", "words")
    font_folder = word_folder / "fonts"

    conditions = [
        "straight",
        "inverted",
        "thatcherized_straight",
        "thatcherized_inverted",
    ]
    for cond in conditions:
        (output_folder / cond).mkdir(parents=True, exist_ok=True)

    if config.use_random_words:
        corpus = json.load(open(Path(word_folder, "random_strings.json"), "r"))
    else:
        corpus: list[str] = read_corpus(Path(word_folder, "1000-corpus.txt"))

    num_letters_per_word = config.num_letters_per_word
    if isinstance(num_letters_per_word, list):
        corpus = [
            i
            for i in corpus
            if num_letters_per_word[0] <= len(i) <= num_letters_per_word[1]
        ]
    else:
        corpus = [i for i in corpus if len(i) == num_letters_per_word]

    fonts: list = [f for f in os.listdir(font_folder) if f.endswith(".ttf")]

    num_letters_to_rotate = config.num_letters_to_rotate
    min_rotatable = (
        min(num_letters_to_rotate)
        if isinstance(num_letters_to_rotate, list)
        else num_letters_to_rotate
    )
    corpus = [
        w
        for w in corpus
        if len([ll for ll in w if ll not in LETTERS_NOT_TO_ROTATE]) >= min_rotatable
    ]
    corpus = random.sample(corpus, min(len(corpus), config.num_words))

    create = CreateData(
        canvas_size=config.canvas_size,
        background=config.background_color,
        antialiasing=config.antialiasing,
        variance_font=0,
        coefficient_translation=config.jittery,
        coefficient_space=1.5,
        word_folder=word_folder,
        font_folder=font_folder,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Condition",
                "Word",
                "LettersRotated",
                "IdxLetterRotated",
                "NameFont",
                "SizeFont",
                "IterNum",
            ]
        )

        for w in tqdm(corpus):
            w = w.upper()
            for count in tqdm(range(config.num_samples_per_word), leave=False):
                name_font = random.sample(fonts, 1)[0]
                size_font = (
                    random.randint(config.size_fonts[0], config.size_fonts[1])
                    if isinstance(config.size_fonts, list)
                    else config.size_fonts
                )

                nl = (
                    random.sample(
                        range(num_letters_to_rotate[0], num_letters_to_rotate[1] + 1), 1
                    )[0]
                    if isinstance(num_letters_to_rotate, list)
                    else num_letters_to_rotate
                )
                nl = min(len(w), nl)
                idx_letter_rotateable = [
                    idx for idx, ww in enumerate(w) if ww not in LETTERS_NOT_TO_ROTATE
                ]
                idx_letters_to_rotate = random.sample(
                    idx_letter_rotateable, min(len(idx_letter_rotateable), nl)
                )

                for cond in conditions:
                    ltr = [] if "thatcherized" not in cond else idx_letters_to_rotate
                    canvas = create.create_images(w, name_font, size_font, ltr)
                    if "inverted" in cond:
                        canvas = canvas.rotate(180)
                    uui = str(uuid.uuid4().hex[:8])
                    img_path = f"{cond}/{w}_{count}_{uui}.png"
                    canvas.save(output_folder / img_path)
                    writer.writerow(
                        [
                            img_path,
                            cond,
                            w,
                            [w[i] for i in idx_letters_to_rotate],
                            idx_letters_to_rotate,
                            name_font,
                            size_font,
                            count,
                        ]
                    )

    return str(output_folder)
