"""emergent features dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.generators.low_mid_vision._emergent_features_drawing import DrawEmergentFeaturesdots


@dataclass
class EmergentFeaturesConfig(GeneratorConfig):
    """config for emergent features dataset."""
    num_samples: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "number of samples"})
    output_folder: str = field(default="data/low_mid_level_vision/emergent_features", metadata={"label": "output folder"})


@register("emergent_features", "low_mid_vision")
@generator(EmergentFeaturesConfig)
def generate_all(config: EmergentFeaturesConfig):
    """generate emergent features dataset."""
    output_folder = Path(config.output_folder)
    all_types = ["single", "proximity", "orientation", "linearity"]

    for t in all_types:
        for pair in ["a", "b"]:
            (output_folder / t / pair).mkdir(exist_ok=True, parents=True)

    ds = DrawEmergentFeaturesdots(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        width=10,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "PairA/B", "SampleId"])
        for i in tqdm(range(config.num_samples)):
            all_sets = ds.get_all_sets()[0]
            for t in tqdm(all_types, leave=False):
                for ip, pair in enumerate(["a", "b"]):
                    path = Path(t) / pair / f"{i}.png"
                    all_sets[t][ip].save(output_folder / path)
                    writer.writerow([path, t, ds.background, pair, i])

    return str(output_folder)
