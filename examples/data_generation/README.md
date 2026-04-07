# data generation example

three ways to generate datasets with mindset-vision, illustrated with the ebbinghaus illusion.

## 1. command line only

generate directly with CLI flags:

```bash
mindset generate ebbinghaus --samples 10 --output /tmp/example_cli_only
```

use `mindset list` to see all 33 available generators.

## 2. config file only

generate from a yaml config file:

```bash
mindset generate ebbinghaus --config examples/data_generation/config.yaml
```

to create a config template from any generator's defaults:

```bash
mindset generate ebbinghaus --save-config
```

this dumps `ebbinghaus.yaml` with all parameters and their defaults.

## 3. config file + command line override

combine both - CLI flags take priority over config values:

```bash
mindset generate ebbinghaus --config examples/data_generation/config.yaml --samples 5 --output /tmp/example_override
```

here `--samples 5` overrides the config's `num_samples_scrambled: 20` and `num_samples_illusory: 10`, and `--output` overrides `output_folder`.

## output structure

each generated dataset contains:

```
output_folder/
  config.json       # parameters used for this generation
  annotation.csv    # image paths + stimulus parameters
  condition_1/      # images grouped by condition
    image_001.png
    ...
  condition_2/
    ...
```

## all generators

run `mindset list` to see all 33 generators grouped by category (visual illusions, low/mid-level vision, shape and object recognition).
