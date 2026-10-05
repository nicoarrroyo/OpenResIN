"""Command-line baseline for the V1 surface-water random forest."""

import argparse
import os
import sys

from . import config as c


def _add_run_arguments(parser):
    parser.add_argument(
        "--input-dir",
        default=os.path.join(c.OUTPUTS_DIR, "label-water"),
        help="directory containing features, areas and completed labels "
             "(default: %(default)s)")
    parser.add_argument(
        "--run-dir", required=True,
        help="run directory holding the frozen preparation")
    parser.add_argument(
        "--source-image-root",
        default=os.path.join(c.DATA_DIR, "sat-images"),
        help="directory containing the four recorded .SAFE scenes "
             "(default: %(default)s)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="openresin-train-sw",
        description=(
            "Prepare, fit and evaluate the fixed April 2026 surface-water "
            "baseline. Numerical settings stay fixed; use a new run "
            "directory when inputs or code change."))
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="validate completed labels and freeze train/test datasets")
    _add_run_arguments(prepare)
    fit = subparsers.add_parser(
        "fit",
        help="fit the fixed 100-tree forest from the training dataset")
    _add_run_arguments(fit)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="score the held-out areas and export water GeoTIFFs")
    _add_run_arguments(evaluate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import modelling_sw

    try:
        if args.action == "prepare":
            manifest = modelling_sw.prepare_datasets(
                args.input_dir, args.run_dir, args.source_image_root)
            train_rows = manifest["datasets"]["train"]["rows"]
            test_rows = manifest["datasets"]["test"]["rows"]
            print(f"prepared {train_rows} training rows "
                  f"and {test_rows} test rows")
            print(f"frozen run: {os.path.abspath(args.run_dir)}")
            return 0
        if args.action == "fit":
            modelling_sw.fit_run(
                args.input_dir, args.run_dir, args.source_image_root)
            print(f"fitted 100-tree forest in "
                  f"{os.path.abspath(args.run_dir)}")
            return 0
        if args.action == "evaluate":
            metrics = modelling_sw.evaluate_run(
                args.input_dir, args.run_dir, args.source_image_root)
            pooled = metrics["pooled"]
            print(f"evaluated {metrics['tile']} {metrics['month']}: "
                  f"pooled water F1 {pooled['f1']} "
                  f"from {pooled['support']['water']} true water pixels")
            print(f"metrics: {os.path.abspath(args.run_dir)}/v1-metrics.json")
            return 0
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
