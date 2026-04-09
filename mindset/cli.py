"""mindset CLI entry point."""

import argparse
import sys
import yaml
from dataclasses import asdict, fields


def _load_registry():
    """import all generators to populate the registry."""
    from importlib import import_module
    from pathlib import Path

    generators_dir = Path(__file__).parent / "generators"
    for category_dir in sorted(generators_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        for gen_file in sorted(category_dir.glob("*.py")):
            if gen_file.name.startswith("_"):
                continue
            module_path = f"mindset.generators.{category_dir.name}.{gen_file.stem}"
            import_module(module_path)

    from mindset.generators import REGISTRY

    return REGISTRY


def main():
    """main CLI entry point for mindset-vision."""
    parser = argparse.ArgumentParser(
        prog="mindset",
        description="mindset-vision: controlled visual datasets for testing DNNs against human vision",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="generate a dataset")
    gen.add_argument("dataset", help="dataset name or 'all'")
    gen.add_argument("--samples", type=int, help="override default sample count")
    gen.add_argument("--canvas-size", type=int, nargs=2, help="canvas width height")
    gen.add_argument("--output", "-o", help="output folder")
    gen.add_argument("--config", help="path to yaml config file")
    gen.add_argument(
        "--save-config", action="store_true", help="dump default config to yaml"
    )

    sub.add_parser("list", help="list available generators")

    ev = sub.add_parser("eval", help="run evaluation pipeline")
    ev.add_argument("pipeline", choices=["decoder", "similarity", "classify"])
    ev.add_argument("--config", help="path to config file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    registry = _load_registry()

    match args.command:
        case "list":
            from mindset.generators import list_generators

            for category, names in list_generators().items():
                print(f"\n  {category}:")
                for name in names:
                    config_cls = registry[name]["config_cls"]
                    print(f"    {name:30s} {config_cls.__doc__.strip()}")

        case "generate":
            if args.dataset == "all":
                for name, info in registry.items():
                    print(f"\ngenerating {name}...")
                    info["func"]()
            else:
                from mindset.generators import get_generator

                info = get_generator(args.dataset)
                config_cls = info["config_cls"]

                if args.save_config:
                    defaults = asdict(config_cls())
                    out_path = f"{args.dataset}.yaml"
                    with open(out_path, "w") as fh:
                        yaml.dump(
                            defaults, fh, default_flow_style=False, sort_keys=False
                        )
                    print(f"config saved to {out_path}")
                    return

                kwargs = {}
                if args.config:
                    with open(args.config) as fh:
                        kwargs = yaml.safe_load(fh)
                if args.output:
                    kwargs["output_folder"] = args.output
                if args.canvas_size:
                    kwargs["canvas_size"] = args.canvas_size
                if args.samples:
                    for fld in fields(config_cls):
                        if "samples" in fld.name:
                            kwargs[fld.name] = args.samples

                info["func"](**kwargs)

        case "eval":
            print(f"eval: {args.pipeline}")
            print("  (eval dispatch will be implemented in PR 6)")


if __name__ == "__main__":
    main()
