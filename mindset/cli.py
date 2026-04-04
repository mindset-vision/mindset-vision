"""mindset CLI entry point."""
import argparse
import sys


def main():
    """main CLI entry point for mindset-vision."""
    parser = argparse.ArgumentParser(
        prog="mindset",
        description="mindset-vision: controlled visual datasets for testing DNNs against human vision",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="generate a dataset")
    gen.add_argument("dataset", help="dataset name or 'all'")
    gen.add_argument("--samples", type=int, help="override sample count")
    gen.add_argument("--canvas-size", type=int, nargs=2, help="canvas width height")
    gen.add_argument("--output", "-o", help="output folder")
    gen.add_argument("--lite", action="store_true", help="use lite sample counts")
    gen.add_argument("--config", help="path to yaml config file")
    gen.add_argument("--save-config", action="store_true", help="dump config to yaml")

    sub.add_parser("list", help="list available generators")

    ev = sub.add_parser("eval", help="run evaluation pipeline")
    ev.add_argument("pipeline", choices=["decoder", "similarity", "classify"])
    ev.add_argument("--config", help="path to config file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        print("available generators:")
        print("  (registry will be populated in PR 3)")

    if args.command == "generate":
        print(f"generate: {args.dataset}")
        print("  (generator dispatch will be implemented in PR 3)")

    if args.command == "eval":
        print(f"eval: {args.pipeline}")
        print("  (eval dispatch will be implemented in PR 6)")


if __name__ == "__main__":
    main()
