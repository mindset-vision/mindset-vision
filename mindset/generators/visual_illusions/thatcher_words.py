"""thatcher illusion words dataset generator."""
import csv
import json
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.generators.visual_illusions._thatcher_words_drawing import (
    CreateData,
    read_corpus,
)

LETTERS_NOT_TO_ROTATE = ["O", "W", "M", "N"]


@dataclass
class ThatcherWordsConfig(GeneratorConfig):
    """config for thatcher illusion words dataset."""
    jittery: float = field(default=0.04, metadata={"min": 0.0, "max": 1.0, "step": 0.01, "label": "jitter"})
    num_words: int = field(default=100, metadata={"min": 1, "max": 1000, "step": 1, "label": "number of words"})
    num_samples_per_word: int = field(default=5, metadata={"min": 1, "max": 100, "step": 1, "label": "samples per word"})
    num_letters_per_word: list = field(default_factory=lambda: [5, 9], metadata={"label": "letters per word range"})
    num_letters_to_rotate: int = field(default=2, metadata={"min": 1, "max": 10, "step": 1, "label": "letters to rotate"})
    size_fonts: list = field(default_factory=lambda: [18, 35], metadata={"label": "font size range"})
    use_random_words: bool = field(default=False, metadata={"label": "use random words"})
    output_folder: str = field(default="data/visual_illusions/thatcher_words", metadata={"label": "output folder"})


@register("thatcher_words", "visual_illusions")
@generator(ThatcherWordsConfig)
def generate_all(config: ThatcherWordsConfig):
    """generate thatcher illusion words dataset."""
    output_folder = Path(config.output_folder)
    word_folder = Path("assets", "words")
    font_folder = word_folder / "fonts"

    conditions = ["straight", "inverted", "thatcherized_straight", "thatcherized_inverted"]
    for cond in conditions:
        (output_folder / cond).mkdir(parents=True, exist_ok=True)

    if config.use_random_words:
        corpus = json.load(open(Path(word_folder, "random_strings.json"), "r"))
    else:
        corpus: list[str] = read_corpus(Path(word_folder, "1000-corpus.txt"))

    num_letters_per_word = config.num_letters_per_word
    if isinstance(num_letters_per_word, list):
        corpus = [i for i in corpus if num_letters_per_word[0] <= len(i) <= num_letters_per_word[1]]
    else:
        corpus = [i for i in corpus if len(i) == num_letters_per_word]

    fonts: list = [f for f in os.listdir(font_folder) if f.endswith(".ttf")]

    num_letters_to_rotate = config.num_letters_to_rotate
    min_rotatable = min(num_letters_to_rotate) if isinstance(num_letters_to_rotate, list) else num_letters_to_rotate
    corpus = [w for w in corpus if len([ll for ll in w if ll not in LETTERS_NOT_TO_ROTATE]) >= min_rotatable]
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
        writer.writerow(["Path", "Condition", "Word", "LettersRotated", "IdxLetterRotated", "NameFont", "SizeFont", "IterNum"])

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
                    random.sample(range(num_letters_to_rotate[0], num_letters_to_rotate[1] + 1), 1)[0]
                    if isinstance(num_letters_to_rotate, list)
                    else num_letters_to_rotate
                )
                nl = min(len(w), nl)
                idx_letter_rotateable = [idx for idx, ww in enumerate(w) if ww not in LETTERS_NOT_TO_ROTATE]
                idx_letters_to_rotate = random.sample(idx_letter_rotateable, min(len(idx_letter_rotateable), nl))

                for cond in conditions:
                    ltr = [] if "thatcherized" not in cond else idx_letters_to_rotate
                    canvas = create.create_images(w, name_font, size_font, ltr)
                    if "inverted" in cond:
                        canvas = canvas.rotate(180)
                    uui = str(uuid.uuid4().hex[:8])
                    img_path = f"{cond}/{w}_{count}_{uui}.png"
                    canvas.save(output_folder / img_path)
                    writer.writerow([
                        img_path, cond, w,
                        [w[i] for i in idx_letters_to_rotate],
                        idx_letters_to_rotate, name_font, size_font, count,
                    ])

    return str(output_folder)
