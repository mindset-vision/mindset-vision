# MindSet: Vision
![](https://i.ibb.co/pvTVHKw/0-05254-67.png)     ![](https://i.ibb.co/4SvMvCt/28.png)![](https://i.ibb.co/9N4YVxF/c-0.png)


### TL;DR: Just gimme the datasets!
**[MindSet Large on Kaggle](https://www.kaggle.com/datasets/mindsetvision/mindset)** (~0.5 GB)

**[MindSet Lite on Kaggle](https://www.kaggle.com/datasets/mindsetvision/mindset-lite)**  (~150 MB)


## Overview
The `MindSet: Vision` datasets are designed to facilitate the testing of DNNs against controlled experiments in psychology. `MindSet: Vision` datasets focus on a range of low-, middle-, and high-level visual findings that provide important constraints for computational theories. It also provides materials for DNN testing and demonstrates how to evaluate a DNN for each experiment using DNNs pretrained on ImageNet.

Paper: [arXiv:2404.05290](https://arxiv.org/abs/2404.05290)


## Install

```bash
pip install -e .              # core package
pip install -e ".[notebook]"  # + Jupyter notebook explorer
```

Requires Python >= 3.10.


## Quick start

### Browse generators

```bash
mindset list
```

### Interactive notebook

```bash
jupyter lab examples/explorer.ipynb
```

The notebook has an interactive widget with sliders for every parameter, plus a catalog of all 33 generators with sample images.

### Generate a dataset

```bash
# with defaults
mindset generate ebbinghaus -o data/ebbinghaus

# override specific parameters
mindset generate ebbinghaus --num-samples-scrambled 5000 --num-samples-illusory 50 -o data/ebbinghaus

# dump defaults to yaml, edit, then generate from config
mindset generate ebbinghaus --save-config
# edit ebbinghaus.yaml
mindset generate ebbinghaus --config ebbinghaus.yaml -o data/ebbinghaus

# generate all 33 datasets
mindset generate all
```

### Python API

```python
from mindset.cli import _load_registry
from mindset.generators import get_generator

registry = _load_registry()
get_generator("ebbinghaus")["func"](
    output_folder="data/ebbinghaus",
    num_samples_scrambled=5000,
    num_samples_illusory=50,
)
```


## Datasets

`MindSet: Vision` datasets are divided into three categories:

| Category | Generators |
|----------|-----------|
| Low and mid-level vision (9) | amodal_completion, decomposition, depth_drawings, emergent_features, nap_vs_mp_2d, nap_vs_mp_3d, relational_vs_coordinate, uncrowding, weber_law |
| Visual illusions (10) | adelson_checkerboard, ebbinghaus, grayscale_shapes, jastrow, lightness_contrast, muller_lyer, ponzo, thatcher_face, thatcher_words, tilt |
| Shape and object recognition (14) | dotted_linedrawings, embedded_figures, global_change, global_change_baker2022, leuven_embedded, linedrawings, same_different, segmented_images, silhouettes, texturized_blobs, texturized_chars, texturized_lines, transformations_2d, viewpoint_invariance |

**A detailed description of each dataset can be found in the related paper [here](https://openreview.net/forum?id=bAaM8cKoMl#discussion)**: refer to Section 2 for an overview, or to Appendix C for more detailed information, including the psychological significance of each dataset, references to relevant papers, and details on the structure of each dataset.

The datasets are structured into subfolders (conditions), which are organized based on the dataset's specific characteristics. At the root of each dataset, there's an `annotation.csv` file. This file lists the paths to individual images (starting from the dataset folder) along with their associated parameters. Such organization enables users to use the datasets either exploting their folder structure (e.g. through PyTorch's  ImageFolder) or by directly referencing the annotation file.


### Ready-To-Download Version

`MindSet: Vision` is model-agnostic and offers flexibility in the way each dataset is employed. Depending on the testing method, you may need a few samples or several thousand images. To cater to these needs, we provide two variants of the dataset on Kaggle:

- [Large Version](https://www.kaggle.com/datasets/valerio1988/mindset) with ~5000 samples for each condition.
- [Lite Version](https://www.kaggle.com/datasets/valerio1988/mindset-lite) with ~100 samples for each condition.

Both versions of the `MindSet: Vision` dataset are structured into folders, each containing a specific dataset. Due to Kaggle's current limitations, it's not possible to download these folders individually. Hence, if you need access to a specific dataset, you'll have to download the entire collection of datasets. Alternatively, you can generate the desired dataset on your own using the CLI or notebook.


## Project structure

```
mindset/
  generators/        # 33 stimulus generators (decorator + config dataclass pattern)
  drawing/           # shared drawing infrastructure (base classes, geometry, shapes)
  cli.py             # CLI entry point
  utils.py           # shared utilities
examples/
  explorer.ipynb     # interactive notebook with widget explorer + catalog
  data_generation/   # CLI usage examples
tests/
  test_smoke.py      # smoke tests (33 generators registered, generation works)
```


## Supported platforms

Tested on macOS, Ubuntu, and Windows with Python 3.10+.
