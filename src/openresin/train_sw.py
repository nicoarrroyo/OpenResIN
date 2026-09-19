"""Command-line preparation for the V1 surface-water random forest."""

import argparse
import os
import sys

from . import config as c
def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-train-sw",
        description=(
            "Prepare the fixed April 2026 surface-water baseline datasets. "
            "Fit and evaluate actions will be added in the next checkpoint."))
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="validate completed labels and freeze train/test datasets")
    prepare.add_argument(
        "--input-dir",
        default=os.path.join(c.OUTPUTS_DIR, "label-water"),
        help="directory containing features, areas and completed labels "
             "(default: %(default)s)")
    prepare.add_argument(
        "--run-dir", required=True,
        help="new empty destination for the frozen run")
    prepare.add_argument(
        "--source-image-root",
        default=os.path.join(c.DATA_DIR, "sat-images"),
        help="directory containing the four recorded .SAFE scenes "
             "(default: %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import modelling_sw

    try:
        manifest = modelling_sw.prepare_datasets(
            args.input_dir, args.run_dir, args.source_image_root)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    train_rows = manifest["datasets"]["train"]["rows"]
    test_rows = manifest["datasets"]["test"]["rows"]
    print(f"prepared {train_rows} training rows and {test_rows} test rows")
    print(f"frozen run: {os.path.abspath(args.run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
