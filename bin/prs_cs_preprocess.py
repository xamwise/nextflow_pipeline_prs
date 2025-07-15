import pandas as pd

import argparse




def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess GWAS data for PRS-CS.")
    parser.add_argument("--input", type=str, default="data", help="Data directory path")
    parser.add_argument("--out", type=str, help="output file name")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load the input file (assumes tab-separated, adjust if needed)
    df = pd.read_csv(args.input, sep='\t', dtype=str)

    # Identify effect size column
    effect_col = None
    if 'BETA' in df.columns:
        effect_col = 'BETA'
    elif 'OR' in df.columns:
        effect_col = 'OR'
    else:
        raise ValueError("No BETA or OR column found in input file.")

    # Filter for required columns
    required_cols = ['SNP', 'A1', 'A2', effect_col, 'SE']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df_filtered = df[required_cols]

    # Save the filtered file
    df_filtered.to_csv(args.out, sep='\t', index=False)

if __name__ == "__main__":
    main()