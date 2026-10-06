import pandas as pd
import numpy as np
import argparse


def sumstats_to_gctb_ma(sumstats_file, snp_info_file, out_file):
    """
    Convert QC'd GWAS summary statistics to the .ma format of GCTB: SNP A1 A2 freq b se p N.

    - b is the effect of A1: BETA, or log(OR) for binary traits (se is on the log-odds scale).
    - freq is the frequency of A1. Depending on the GWAS, the MAF column of the QC'd summary
      statistics holds the frequency of A1, of A2 or of the minor allele. The interpretation that
      agrees best with the A1 frequencies of the LD reference (snp.info of the eigen-decomposed LD)
      is used; for the minor allele frequency, the minor allele of the LD reference is assumed.
    """
    ss = pd.read_csv(sumstats_file, sep=r'\s+')

    if 'BETA' in ss.columns:
        b = ss['BETA']
    elif 'OR' in ss.columns:
        b = np.log(ss['OR'])
    else:
        raise ValueError("No BETA or OR column found in input file.")

    missing = [col for col in ['SNP', 'A1', 'A2', 'MAF', 'SE', 'P', 'N'] if col not in ss.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    ref = pd.read_csv(snp_info_file, sep=r'\s+', usecols=['ID', 'A1', 'A2', 'A1Freq'])
    ref.columns = ['SNP', 'ref_A1', 'ref_A2', 'ref_A1Freq']
    ref = ref.drop_duplicates('SNP')
    merged = ss[['SNP', 'A1']].merge(ref, on='SNP', how='left')

    # Frequency of the summary statistics' A1 in the LD reference (NaN if not in it or alleles differ)
    ref_freq = np.where(merged['A1'] == merged['ref_A1'], merged['ref_A1Freq'],
                        np.where(merged['A1'] == merged['ref_A2'], 1 - merged['ref_A1Freq'], np.nan))
    in_ref = ~np.isnan(ref_freq)
    if not in_ref.any():
        raise ValueError(f"No SNPs of {sumstats_file} with matching alleles in {snp_info_file}.")

    candidates = {
        'A1 frequency': ss['MAF'].values,
        'A2 frequency': 1 - ss['MAF'].values,
        'minor allele frequency': np.where(ref_freq > 0.5, 1 - ss['MAF'], ss['MAF']),
    }
    mismatch = {name: np.mean(np.abs(f[in_ref] - ref_freq[in_ref]) > 0.2) for name, f in candidates.items()}
    used = min(mismatch, key=mismatch.get)
    freq = candidates[used]

    ma = pd.DataFrame({'SNP': ss['SNP'], 'A1': ss['A1'], 'A2': ss['A2'], 'freq': freq,
                       'b': b, 'se': ss['SE'], 'p': ss['P'], 'N': ss['N']})
    ma.to_csv(out_file, sep='\t', index=False)

    print(f"{out_file}: {len(ma)} SNPs, {in_ref.sum()} in the LD reference.")
    print(f"MAF column used as {used}: frequency differs by > 0.2 from the LD reference for "
          f"{mismatch[used]:.1%} of the SNPs (" +
          ", ".join(f"{name}: {m:.1%}" for name, m in mismatch.items()) + ")")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert QC'd GWAS summary statistics to the GCTB .ma format")
    parser.add_argument("--input", type=str, required=True, help="QC'd summary statistics (SNP, A1, A2, MAF, BETA or OR, SE, P, N)")
    parser.add_argument("--snp_info", type=str, required=True, help="snp.info of the eigen-decomposed LD reference")
    parser.add_argument("--out", type=str, required=True, help="Output .ma file")
    args = parser.parse_args()

    sumstats_to_gctb_ma(args.input, args.snp_info, args.out)
