import pandas as pd
from sklearn.model_selection import train_test_split
import argparse




def get_parser():
    parser = argparse.ArgumentParser(description="Train and predict using a drug response prediction model.")
    parser.add_argument("--path_data", type=str, default="data", help="Data directory path")
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LDO)")
    parser.add_argument("--test", type=float, help="test dataset size")
    parser.add_argument("--validate", type=float, help="validation dataset size")
    parser.add_argument("--out", type=str, help="output file name")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    fam = pd.read_csv(args.path_data, delim_whitespace=True, header=None)
    fam.columns = ["FID", "IID", "PID", "MID", "SEX", "PHENOTYPE"]

    # Split: 60% train, 20% validation, 20% test
    train, temp = train_test_split(fam[["FID", "IID"]], test_size=args.test, random_state=42)
    val, test = train_test_split(temp, test_size=args.validate, random_state=42)

    # Save to files
    train.to_csv(f"{args.out}_train.txt", sep="\t", header=False, index=False)
    val.to_csv(f"{args.out}_val.txt", sep="\t", header=False, index=False)
    test.to_csv(f"{args.out}_test.txt", sep="\t", header=False, index=False)
    
    
if __name__ == "__main__":
    main()