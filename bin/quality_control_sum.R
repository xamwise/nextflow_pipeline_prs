#!/usr/bin/env Rscript

library(data.table)
library(optparse)


option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="sum stats file to perform qc on", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="out file name", metavar="character"),
  make_option(c("-i", "--info"), type="character", default=NULL,
              help="info score filter", metavar="numeric"),
  make_option(c("-m", "--maf"), type="character", default=NULL,
              help="mean allele frequency filter", metavar="numeric")
)  

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

# Read in file
dat <- fread(opt$file)

# Filter out SNPs with INFO and MAF thresholds
dat_qc <- dat[INFO > opt$info & MAF > opt$maf]

# Filter out duplicate SNPs
nodup_data <- dat_qc[!duplicated(SNP)]

# Filter out strand-ambiguous SNPs
qc_data <- nodup_data[!(
    (A1 == "A" & A2 == "T") |
    (A1 == "T" & A2 == "A") |
    (A1 == "G" & A2 == "C") |
    (A1 == "C" & A2 == "G")
)]

# Write the QC-filtered data to a gzipped file
fwrite(qc_data, opt$out, sep = "\t")
