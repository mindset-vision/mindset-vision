#!/bin/bash
# example: three ways to generate datasets with mindset-vision

# 1. command line only
mindset generate ebbinghaus --samples 10 --output /tmp/example_cli_only

# 2. config file only
mindset generate ebbinghaus --config examples/data_generation/config.yaml

# 3. config file + command line override (CLI takes priority)
mindset generate ebbinghaus --config examples/data_generation/config.yaml --samples 5 --output /tmp/example_override
