import argparse
import sys


def is_header(fields):
    # 'FID IID ...' header of .pheno / .cov / .covariate files, or a '#FID IID ...' plink2 header
    return fields[0].startswith('#') or (len(fields) > 1 and fields[1].upper() == 'IID')


def subset_by_ids(input_file, fam_file, out_file):
    """
    Subset a per-sample file (.pheno, .cov, .eigenvec, .covariate) to the samples of a .fam file.

    Rows are matched on IID (second column) and written in .fam order, so the output lines up
    row by row with the genotypes (SCT.R uses the phenotype vector positionally). Lines are
    copied verbatim, which keeps the header (if any), the delimiter and the number formatting.
    """
    with open(fam_file) as fh:
        fam_iids = [line.split()[1] for line in fh if line.strip()]

    header = None
    rows = {}
    with open(input_file) as fh:
        for i, line in enumerate(fh):
            fields = line.split()
            if not fields:
                continue
            if not line.endswith('\n'):
                line += '\n'
            if i == 0 and is_header(fields):
                header = line
            else:
                rows.setdefault(fields[1], line)

    missing = [iid for iid in fam_iids if iid not in rows]

    with open(out_file, 'w') as out:
        if header is not None:
            out.write(header)
        for iid in fam_iids:
            if iid in rows:
                out.write(rows[iid])

    print(f"{out_file}: {len(fam_iids) - len(missing)} of {len(fam_iids)} samples written.")
    if missing:
        print(f"WARNING: {len(missing)} samples of {fam_file} are missing in {input_file}", file=sys.stderr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Subset a per-sample file to the samples (and order) of a .fam file")
    parser.add_argument("--input", type=str, required=True, help="File to subset, with FID and IID as first two columns")
    parser.add_argument("--fam", type=str, required=True, help=".fam file with the samples to keep")
    parser.add_argument("--out", type=str, required=True, help="Output file")
    args = parser.parse_args()

    subset_by_ids(args.input, args.fam, args.out)
