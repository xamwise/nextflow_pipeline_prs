#!/usr/bin/env python
from drevalpy.datasets.curvecurator import preprocess
from pathlib import Path
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess CurveCurator viability data.")
    parser.add_argument("--path_data", type=str, default="", help="Path to base folder containing datasets.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--cores", type=int, default=0, help="The number of cores used for CurveCurator fitting.")
    return parser


def main(args):
    input_file = Path(args.path_data) / args.dataset_name / f"{args.dataset_name}_raw.csv"
    preprocess(
        input_file=input_file,
        output_dir=args.dataset_name,
        dataset_name=args.dataset_name,
        cores=args.cores
    )

if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    main(args)
