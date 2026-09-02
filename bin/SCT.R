#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(bigstatsr)
  library(bigsnpr)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(optparse)
  library(xgboost)
})

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Any p-value <= 0 (float underflow in the sumstats file, e.g. 1e-320 parsed as
# 0) becomes Inf under -log10(). snp_grid_PRS() then evaluates its default
#   grid.lpS.thr = seq_log(0.1, 0.999 * max(lpS[unlist(all_keep)]), n_thr_lpS)
# and dies with "'to' must be a finite number". Floor p instead.
P_FLOOR <- 1e-300   # -log10 -> 300


option_list = list(
  make_option(c("-b", "--bed"), type="character", default=NULL,
              help="bedfile name without extension", metavar="character"),
  make_option(c("-f", "--sum_stats"), type="character", default=NULL,
              help="summary statistics file name", metavar="character"),
  make_option(c("-p", "--pheno"), type="character", default=NULL,
              help="phenotype file name (optional, uses fam file if not provided)", metavar="character"),
  make_option(c("--trait_type"), type="character", default="auto",
              help="trait type: 'binary', 'quantitative', or 'auto' [default: auto]", metavar="character"),
  make_option(c("-t", "--train_prop"), type="numeric", default=0.8,
              help="proportion of samples for training [default: 0.8]", metavar="numeric"),
  make_option(c("-n", "--n_train"), type="integer", default=NULL,
              help="number of training samples (overrides train_prop)", metavar="integer"),
  make_option(c("-k", "--n_folds"), type="integer", default=10,
              help="number of folds for cross-validation [default: 10]", metavar="integer"),
  make_option(c("-c", "--ncores"), type="integer", default=2,
              help="number of cores to use [default: all available]", metavar="integer"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="output prefix", metavar="character"),
  make_option(c("--out_dir"), type="character", default=".",
              help="output directory [default: current directory]", metavar="character"),
  make_option(c("--flip_effect_allele"), action="store_true", default=FALSE,
              help="only used when sumstats use allele1/allele2 naming: treat allele2 (not allele1) as the effect allele")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed) || is.null(opt$sum_stats) || is.null(opt$out)){
  print_help(opt_parser)
  stop("Required arguments: --bed, --sum_stats, and --out", call.=FALSE)
}

# Resolve output prefix / make sure the directory exists
if (!is.null(opt$out_dir) && nzchar(opt$out_dir) && opt$out_dir != "." &&
    !grepl("^(/|~)", opt$out) && !grepl("/", opt$out)) {
  opt$out <- file.path(opt$out_dir, opt$out)
}
dir.create(dirname(opt$out), recursive = TRUE, showWarnings = FALSE)

# Set number of cores
if (is.null(opt$ncores)) {
  NCORES <- nb_cores()
} else {
  NCORES <- opt$ncores
}
cat("Using", NCORES, "cores\n")

# Read or create bigSNP object
rds_file <- paste0(opt$bed, "_SCT.rds")
if (!file.exists(rds_file)) {
  cat("Converting BED file to bigSNP format...\n")
  snp_readBed(paste0(opt$bed, ".bed"), backingfile = paste0(opt$bed, "_SCT"))
}
obj.bigSNP <- snp_attach(rds_file)

# Get aliases for useful slots
G   <- obj.bigSNP$genotypes
CHR <- obj.bigSNP$map$chromosome
POS <- obj.bigSNP$map$physical.pos

# Get phenotype
if (!is.null(opt$pheno)) {
  cat("Reading phenotype file...\n")
  pheno_col <- "phenotype"  # Default name
  pheno_data <- fread(opt$pheno)
  # Merge with fam file

  y <- pheno_data$phenotype
  obj.bigSNP$fam$affection <- y

} else {
  # Use affection status from fam file
  cat("Using affection status from fam file\n")
  y <- obj.bigSNP$fam$affection
  pheno_col <- "affection"
}

if (length(y) != nrow(G)) {
  stop("Phenotype vector has length ", length(y),
       " but the genotype matrix has ", nrow(G), " rows. ",
       "The phenotype file must be in the same order as the .fam file.")
}

# Remove missing phenotypes
non_missing <- !is.na(y)
if (sum(non_missing) < nrow(G)) {
  cat("Removing", sum(!non_missing), "samples with missing phenotypes\n")
  y <- y[non_missing]
  G_indices <- which(non_missing)
} else {
  G_indices <- 1:nrow(G)
}

# Determine trait type
unique_vals <- unique(y[!is.na(y)])
n_unique <- length(unique_vals)

if (opt$trait_type == "auto") {
  if (all(unique_vals %in% c(0, 1))) {
    trait_type <- "binary"
    cat("Detected binary trait (0/1 coding)\n")
  } else if (all(unique_vals %in% c(1, 2))) {
    trait_type <- "binary"
    cat("Detected binary trait (1/2 coding), converting to 0/1\n")
    y <- y - 1
  } else if (n_unique == 2) {
    trait_type <- "binary"
    cat("Detected binary trait with values:", paste(unique_vals, collapse=", "), "\n")
    # Convert to 0/1
    y <- as.numeric(factor(y)) - 1
  } else if (n_unique > 10) {
    trait_type <- "quantitative"
    cat("Detected quantitative trait (", n_unique, "unique values)\n")
  } else {
    # Could be ordinal or quantitative with few values
    cat("Ambiguous trait type (", n_unique, "unique values). Treating as quantitative.\n")
    trait_type <- "quantitative"
  }
} else {
  trait_type <- opt$trait_type
  cat("Using specified trait type:", trait_type, "\n")

  if (trait_type == "binary") {
    if (!all(unique_vals %in% c(0, 1))) {
      if (all(unique_vals %in% c(1, 2))) {
        cat("Converting 1/2 coding to 0/1\n")
        y <- y - 1
      } else if (n_unique == 2) {
        cat("Converting to 0/1 coding\n")
        y <- as.numeric(factor(y)) - 1
      } else {
        stop("Binary trait specified but phenotype has ", n_unique, " unique values")
      }
    }
  }
}

# Check variance for quantitative traits
if (trait_type == "quantitative") {
  y_var <- var(y, na.rm = TRUE)
  if (y_var < 1e-10) {
    stop("Phenotype has essentially no variance. Cannot perform analysis.")
  }
  cat("Phenotype variance:", y_var, "\n")
  cat("Phenotype range:", min(y, na.rm = TRUE), "to", max(y, na.rm = TRUE), "\n")
}

# -----------------------------------------------------------------------------
# Read summary statistics
# -----------------------------------------------------------------------------
cat("Reading summary statistics...\n")
sumstats <- bigreadr::fread2(opt$sum_stats)

# Standardise column names.
# NOTE: the previous version tested every rule against a *snapshot* of the
# original names (`col_names`), so rules could never see the effect of earlier
# renames and some rules could never fire at all. Each rule below re-reads the
# current names and is a no-op if the target column already exists.
names(sumstats) <- tolower(trimws(names(sumstats)))

rename_to <- function(df, target, candidates) {
  nm <- names(df)
  if (target %in% nm) return(df)             # already named correctly
  hit <- which(nm %in% candidates)
  if (length(hit) >= 1L) names(df)[hit[1L]] <- target
  df
}

sumstats <- rename_to(sumstats, "chr",
                      c("chromosome", "chrom", "#chrom", "chr_name", "hg19chr"))
sumstats <- rename_to(sumstats, "pos",
                      c("bp", "position", "physical.pos", "base_pair_location",
                        "pos_b37", "bp_hg19", "bp_hg38"))
sumstats <- rename_to(sumstats, "rsid",
                      c("snp", "marker.id", "markername", "variant_id", "id"))
sumstats <- rename_to(sumstats, "p",
                      c("pval", "pvalue", "p.value", "p_value", "p-value",
                        "p_bolt_lmm", "p_bolt_lmm_inf", "frequentist_add_pvalue"))
sumstats <- rename_to(sumstats, "beta",
                      c("effect", "b", "effect_size", "log_odds", "beta_hat"))
sumstats <- rename_to(sumstats, "or", c("odds_ratio", "oddsratio"))

# Alleles. bigsnpr convention: a1 is the EFFECT allele (beta refers to a1),
# a0 is the other allele -- matching allele1/allele2 of the bigSNP map below.
sumstats <- rename_to(sumstats, "a1", c("effect_allele", "ea", "alt", "tested_allele"))
sumstats <- rename_to(sumstats, "a0", c("a2", "other_allele", "non_effect_allele",
                                        "nea", "ref", "reference_allele"))

if (!all(c("a0", "a1") %in% names(sumstats)) &&
    all(c("allele1", "allele2") %in% names(sumstats))) {
  eff <- if (isTRUE(opt$flip_effect_allele)) "allele2" else "allele1"
  oth <- if (isTRUE(opt$flip_effect_allele)) "allele1" else "allele2"
  names(sumstats)[names(sumstats) == eff] <- "a1"
  names(sumstats)[names(sumstats) == oth] <- "a0"
  cat("NOTE: sumstats use allele1/allele2 naming. Assuming '", eff,
      "' is the EFFECT allele (i.e. beta refers to it).\n", sep = "")
  cat("      If that is wrong, re-run with --flip_effect_allele.\n")
}

# OR -> beta (only if beta is genuinely absent)
if (!("beta" %in% names(sumstats)) && "or" %in% names(sumstats)) {
  or_vals <- suppressWarnings(as.numeric(sumstats$or))
  n_bad_or <- sum(!is.finite(or_vals) | or_vals <= 0, na.rm = TRUE)
  if (n_bad_or > 0)
    cat("WARNING:", n_bad_or, "non-positive/non-finite OR values -> beta set to NA\n")
  or_vals[!is.finite(or_vals) | or_vals <= 0] <- NA_real_
  sumstats$beta <- log(or_vals)
  cat("Converted OR to beta (log scale)\n")
}

# Coerce chr/pos to numeric (strips a leading "chr" prefix if present)
if (is.character(sumstats$chr) || is.factor(sumstats$chr)) {
  sumstats$chr <- suppressWarnings(
    as.integer(sub("^chr", "", tolower(as.character(sumstats$chr)))))
}
sumstats$pos <- suppressWarnings(as.numeric(sumstats$pos))

# Check required columns
required_cols <- c("chr", "pos", "a0", "a1", "beta", "p")
missing_cols <- setdiff(required_cols, names(sumstats))
if (length(missing_cols) > 0) {
  stop("Missing required columns in summary statistics: ",
       paste(missing_cols, collapse = ", "),
       "\nColumns present: ", paste(names(sumstats), collapse = ", "))
}

# Drop rows that cannot be matched at all
bad_rows <- with(sumstats, is.na(chr) | is.na(pos) | is.na(a0) | is.na(a1))
if (any(bad_rows)) {
  cat("Dropping", sum(bad_rows), "sumstats rows with missing chr/pos/alleles",
      "(e.g. non-autosomal contigs)\n")
  sumstats <- sumstats[!bad_rows, , drop = FALSE]
}
if (nrow(sumstats) == 0) stop("No usable rows left in the summary statistics.")

# -----------------------------------------------------------------------------
# Train / test split
# -----------------------------------------------------------------------------
set.seed(1)
n_samples <- length(G_indices)

if (!is.null(opt$n_train)) {
  n_train <- min(opt$n_train, n_samples - 10)  # Ensure we have at least 10 test samples
} else {
  n_train <- floor(opt$train_prop * n_samples)
}

# Create indices relative to the non-missing samples
train_idx <- sample(n_samples, n_train)
test_idx <- setdiff(1:n_samples, train_idx)

# Map back to original G indices
ind.train <- G_indices[train_idx]
ind.test <- G_indices[test_idx]

cat("Training samples:", length(ind.train), "\n")
cat("Test samples:", length(ind.test), "\n")

# Check variance in training set
y_train <- y[train_idx]
y_train_var <- var(y_train, na.rm = TRUE)
y_train_unique <- length(unique(y_train[!is.na(y_train)]))

cat("Training set: ", y_train_unique, "unique values, variance =", y_train_var, "\n")

if (y_train_unique < 2) {
  stop("Training set has only one unique value. Cannot train model. Check your phenotype data.")
}

if (trait_type == "quantitative" && y_train_var < 1e-10) {
  stop("Training set phenotype has essentially no variance. Cannot train model.")
}

# -----------------------------------------------------------------------------
# Match variants between genotype data and summary statistics
# -----------------------------------------------------------------------------
cat("Matching variants...\n")
map <- obj.bigSNP$map[, c(1, 4, 5, 6)]
names(map) <- c("chr", "pos", "a1", "a0")

# Try matching with strand flipping first
info_snp <- snp_match(sumstats[, required_cols], map)

# If few variants match, try without strand flipping
if (nrow(info_snp) < nrow(sumstats) * 0.5) {
  cat("Few variants matched with strand flipping, trying without...\n")
  info_snp <- snp_match(sumstats[, required_cols], map, strand_flip = FALSE)
}

cat("Matched", nrow(info_snp), "variants out of", nrow(sumstats), "\n")
if (nrow(info_snp) == 0) {
  stop("No variants matched between the genotypes and the summary statistics. ",
       "Check genome build, chromosome coding and allele columns.")
}

# -----------------------------------------------------------------------------
# Prepare beta and -log10(p) for all SNPs
# -----------------------------------------------------------------------------
# p-values of exactly 0 (float underflow, common in large meta-analyses) give
# lpval = Inf, which propagates into snp_grid_PRS()'s default threshold grid and
# throws "'to' must be a finite number". Floor them here, BEFORE clumping, so
# that clumping and the PRS grid see exactly the same set of usable SNPs.
p_clean <- suppressWarnings(as.numeric(info_snp$p))
n_bad_p <- sum(!is.finite(p_clean) | p_clean <= 0, na.rm = TRUE) + sum(is.na(p_clean))
if (n_bad_p > 0) {
  cat("WARNING:", n_bad_p, "p-values were <= 0, NA or non-finite;",
      "flooring at", P_FLOOR, "\n")
}
p_clean[is.na(p_clean) | !is.finite(p_clean) | p_clean <= 0] <- P_FLOOR
p_clean[p_clean > 1] <- 1

beta <- rep(NA_real_, ncol(G))
beta[info_snp$`_NUM_ID_`] <- suppressWarnings(as.numeric(info_snp$beta))
beta[!is.finite(beta)] <- NA_real_

lpval <- rep(NA_real_, ncol(G))
lpval[info_snp$`_NUM_ID_`] <- -log10(p_clean)
lpval[!is.finite(lpval)] <- NA_real_

# Keep the two masks consistent: a SNP with no usable beta must not be clumped
lpval[is.na(beta)] <- NA_real_
beta[is.na(lpval)]  <- NA_real_

n_usable <- sum(!is.na(lpval))
cat("Usable SNPs (finite beta and p):", n_usable, "\n")
if (n_usable == 0) stop("No SNPs with both a finite beta and a finite p-value.")
cat("lpval range:", min(lpval, na.rm = TRUE), "to", max(lpval, na.rm = TRUE), "\n")

# -----------------------------------------------------------------------------
# Genotype missingness check
# -----------------------------------------------------------------------------
# NOTE: the previous `sum(is.na(as.list(G)))` did not inspect the genotype
# matrix at all -- as.list() on an FBM returns the object's slots, so it always
# printed 0. Count real missing calls (code 3) on a deterministic subsample.
cat("Checking missing values in genotype data...\n")
chk_cols <- unique(round(seq(1, ncol(G), length.out = min(1000L, ncol(G)))))
cnts <- big_counts(G, ind.col = chk_cols)          # rows: 0, 1, 2, NA
n_geno_na <- sum(cnts[4, ])
cat("Missing calls in", length(chk_cols), "sampled SNPs:", n_geno_na,
    sprintf("(%.4f%%)\n", 100 * n_geno_na / (length(chk_cols) * nrow(G))))
if (n_geno_na > 0) {
  cat("WARNING: genotypes contain missing values. bigsnpr requires imputed data",
      "-- run snp_fastImputeSimple() / snp_fastImpute() first, or results will be wrong.\n")
}

# -----------------------------------------------------------------------------
# Clumping
# -----------------------------------------------------------------------------
cat("Performing clumping...\n")
all_keep <- snp_grid_clumping(G, CHR, POS,
                              ind.row = ind.train,
                              lpS = lpval,
                              exclude = which(is.na(lpval)),
                              ncores = NCORES)

cat("Clumping completed with", nrow(attr(all_keep, "grid")), "parameter sets\n")

# -----------------------------------------------------------------------------
# PRS over the C+T grid
# -----------------------------------------------------------------------------
cat("Computing PRS for multiple thresholds...\n")
cat("G backing file:", G$backingfile, "\n")

kept_idx <- unlist(all_keep)
if (length(kept_idx) == 0)
  stop("Clumping retained no SNPs -- check that lpval/exclude are correct.")

lp_max <- max(lpval[kept_idx], na.rm = TRUE)
cat("Max -log10(p) among clumped SNPs:", lp_max, "\n")
if (!is.finite(lp_max) || lp_max <= 0.1)
  stop("No usable -log10(p) among clumped SNPs (max = ", lp_max, "). ",
       "Cannot build a p-value threshold grid.")

# Pass the grid explicitly rather than relying on the lazily-evaluated default,
# so a non-finite value can never reach seq_log() unnoticed.
lpS_grid <- seq_log(0.1, 0.999 * lp_max, 50)

multi_PRS <- snp_grid_PRS(G, all_keep, beta, lpval,
                          ind.row = ind.train,
                          n_thr_lpS = 50,
                          grid.lpS.thr = lpS_grid,
                          ncores = NCORES)

cat("Computed", ncol(multi_PRS), "PRS for", nrow(multi_PRS), "individuals\n")

# -----------------------------------------------------------------------------
# Stacking
# -----------------------------------------------------------------------------
cat("Performing stacking...\n")
cat("Trait type for stacking:", trait_type, "\n")

# For quantitative traits, we need to standardize
if (trait_type == "quantitative") {
  y_mean <- mean(y_train, na.rm = TRUE)
  y_sd <- sd(y_train, na.rm = TRUE)
  cat("Standardizing quantitative phenotype (mean =", y_mean, ", sd =", y_sd, ")\n")
  y_train_std <- (y_train - y_mean) / y_sd
} else {
  y_train_std <- y_train
}

# Stacking - no family parameter, auto-detected from y.train
final_mod <- tryCatch({
  snp_grid_stacking(multi_PRS, y_train_std, ncores = NCORES, K = opt$n_folds)
}, error = function(e) {
  cat("Error in stacking:", conditionMessage(e), "\n")
  cat("Trying with K=2...\n")
  snp_grid_stacking(multi_PRS, y_train_std, ncores = NCORES, K = 2)
})

# Extract new beta values
new_beta <- final_mod$beta.G
ind_keep <- which(new_beta != 0)

cat("Number of non-zero SNPs:", length(ind_keep), "\n")
if (length(ind_keep) == 0)
  stop("Stacking shrank every coefficient to zero -- no predictive signal was retained.")

# Calculate predictions on test set
y_test <- y[test_idx]
pred_test <- final_mod$intercept +
  big_prodVec(G, new_beta[ind_keep], ind.row = ind.test, ind.col = ind_keep)

# Calculate appropriate metric based on trait type
if (trait_type == "binary") {
  # Calculate AUC for binary trait
  auc_result <- AUCBoot(pred_test, y_test)
  cat("Test AUC:", round(auc_result[1], 4), "\n")

  # Save AUC results
  auc_df <- data.frame(
    Mean = auc_result[1],
    CI_2.5 = auc_result[2],
    CI_97.5 = auc_result[3],
    SD = auc_result[4]
  )
  fwrite(auc_df, paste0(opt$out, "_AUC.txt"), sep = "\t")
} else {
  # For quantitative traits, calculate correlation and R-squared
  # Unstandardize predictions if we standardized during training
  if (exists("y_mean") && exists("y_sd")) {
    pred_test_orig <- pred_test * y_sd + y_mean
  } else {
    pred_test_orig <- pred_test
  }

  cor_test <- cor(pred_test_orig, y_test, use = "complete.obs")
  r2_test <- cor_test^2
  rmse_test <- sqrt(mean((pred_test_orig - y_test)^2, na.rm = TRUE))

  cat("Test correlation:", round(cor_test, 4), "\n")
  cat("Test R-squared:", round(r2_test, 4), "\n")
  cat("Test RMSE:", round(rmse_test, 4), "\n")

  # Save quantitative metrics
  quant_metrics <- data.frame(
    Correlation = cor_test,
    R_squared = r2_test,
    RMSE = rmse_test,
    N_test = sum(!is.na(y_test))
  )
  fwrite(quant_metrics, paste0(opt$out, "_metrics.txt"), sep = "\t")
}

# Calculate PRS for all individuals
cat("Calculating PRS for all individuals...\n")
pred_all <- final_mod$intercept +
  big_prodVec(G, new_beta[ind_keep], ind.col = ind_keep)

# For quantitative traits, unstandardize if needed
if (trait_type == "quantitative" && exists("y_mean") && exists("y_sd")) {
  pred_all <- pred_all * y_sd + y_mean
}

# Save PRS
prs_output <- data.table(
  FID = obj.bigSNP$fam$family.ID,
  IID = obj.bigSNP$fam$sample.ID,
  PRS = pred_all,
  is_train = 1:nrow(G) %in% ind.train
)
fwrite(prs_output, paste0(opt$out, "_PRS.csv"))
cat("PRS saved to:", paste0(opt$out, "_PRS.csv"), "\n")

# Save beta coefficients
# Get SNP information for non-zero betas
snp_info <- obj.bigSNP$map[ind_keep, ]
beta_output <- data.table(
  chr = snp_info$chromosome,
  rsid = snp_info$marker.ID,
  pos = snp_info$physical.pos,
  a1 = snp_info$allele1,
  a0 = snp_info$allele2,
  beta = new_beta[ind_keep]
)
fwrite(beta_output, paste0(opt$out, "_betas.csv"))
cat("Beta coefficients saved to:", paste0(opt$out, "_betas.csv"), "\n")

# Save stacking model summary
# fwrite(final_mod$mod, paste0(opt$out, "_stacking_summary.txt"), sep = "\t")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
if (requireNamespace("ggplot2", quietly = TRUE)) {

  # Plot comparing GWAS betas to SCT betas
  if (length(ind_keep) > 0) {
    plot_data <- data.frame(
      gwas_beta = beta[ind_keep],
      sct_beta = new_beta[ind_keep]
    )

    p1 <- ggplot(plot_data, aes(x = gwas_beta, y = sct_beta)) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      geom_abline(slope = 0, intercept = 0, color = "blue", linetype = "dotted") +
      geom_point(size = 0.6, alpha = 0.5) +
      theme_minimal() +
      labs(x = "Effect sizes from GWAS",
           y = "Non-zero effect sizes from SCT",
           title = "Comparison of GWAS and SCT effect sizes")

    ggsave(paste0(opt$out, "_beta_comparison.pdf"), p1, width = 8, height = 6)
  }

  # Plot PRS distribution
  plot_data2 <- data.frame(
    Phenotype = y,
    PRS = pred_all[G_indices]  # Match PRS to samples with phenotypes
  )

  if (trait_type == "binary") {
    plot_data2$Phenotype <- factor(plot_data2$Phenotype,
                                   levels = 0:1,
                                   labels = c("Control", "Case"))

    p2 <- ggplot(plot_data2[!is.na(plot_data2$Phenotype), ],
                 aes(x = PRS, fill = Phenotype)) +
      geom_density(alpha = 0.5) +
      theme_minimal() +
      labs(x = "Polygenic Risk Score",
           y = "Density",
           title = "PRS Distribution by Phenotype")
  } else {
    # For quantitative traits, show correlation
    p2 <- ggplot(plot_data2[!is.na(plot_data2$Phenotype), ],
                 aes(x = PRS, y = Phenotype)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = TRUE, color = "blue") +
      theme_minimal() +
      labs(x = "Polygenic Risk Score",
           y = pheno_col,
           title = paste("PRS vs", pheno_col)) +
      annotate("text", x = Inf, y = Inf,
               label = paste("r =", round(cor(plot_data2$PRS, plot_data2$Phenotype,
                                             use = "complete.obs"), 3)),
               hjust = 1.1, vjust = 1.1)
  }

  ggsave(paste0(opt$out, "_PRS_distribution.pdf"), p2, width = 8, height = 6)
  cat("Plots saved\n")
}

# -----------------------------------------------------------------------------
# Best single C+T model, for comparison
# -----------------------------------------------------------------------------
# NOTE: this block was previously duplicated, with the first copy setting
# s <- nrow(attr(all_keep, "grid")) (= number of clumping sets only). The
# correct stride is the number of (clumping set x p-threshold) combinations,
# i.e. nrow(grid2) after unnest().
cat("\nFinding best single C+T model for comparison...\n")

grid2 <- attr(all_keep, "grid") %>%
  mutate(thr.lp = list(attr(multi_PRS, "grid.lpS.thr")), id = row_number()) %>%
  unnest(cols = "thr.lp")

s <- nrow(grid2)
n_chr <- length(unique(CHR))
cat("Grid size s:", s, " n_chr:", n_chr, " ncol(multi_PRS):", ncol(multi_PRS), "\n")

if (s * n_chr != ncol(multi_PRS)) {
  if (ncol(multi_PRS) %% s == 0) {
    n_chr <- ncol(multi_PRS) %/% s
    cat("Adjusted n_chr to", n_chr, "based on ncol(multi_PRS)\n")
  } else {
    stop("Column layout mismatch: s (", s, ") does not divide ncol(multi_PRS) (",
         ncol(multi_PRS), ").")
  }
}

grid2$metric <- big_apply(multi_PRS, a.FUN = function(X, ind, s, n_chr, y.train, trait_type) {
  single_PRS <- rowSums(X[, ind + s * (0:(n_chr - 1)), drop = FALSE])
  if (trait_type == "binary") {
    bigstatsr::AUC(single_PRS, y.train)
  } else {
    abs(cor(single_PRS, y.train, use = "complete.obs"))
  }
}, ind = 1:s, s = s, n_chr = n_chr, y.train = y_train, trait_type = trait_type,
a.combine = 'c', block.size = 1, ncores = NCORES)

# Find best model
best_ct <- grid2 %>%
  arrange(desc(metric)) %>%
  slice(1)

metric_name <- if(trait_type == "binary") "AUC" else "Correlation"
cat("Best single C+T model: size =", best_ct$size,
    ", thr.r2 =", best_ct$thr.r2,
    ", thr.lp =", round(best_ct$thr.lp, 3),
    ", ", metric_name, "=", round(best_ct$metric, 4), "\n")

# Save best C+T parameters
fwrite(best_ct, paste0(opt$out, "_best_CT_params.txt"), sep = "\t")

cat("\n=== SCT COMPLETED SUCCESSFULLY ===\n")
cat("Trait type:", trait_type, "\n")
cat("Output files created:\n")
cat("  PRS scores:", paste0(opt$out, "_PRS.csv"), "\n")
cat("  Beta coefficients:", paste0(opt$out, "_betas.csv"), "\n")
cat("  Stacking summary:", paste0(opt$out, "_stacking_summary.txt"), "\n")
if (trait_type == "binary") {
  cat("  AUC results:", paste0(opt$out, "_AUC.txt"), "\n")
} else {
  cat("  Performance metrics:", paste0(opt$out, "_metrics.txt"), "\n")
}
cat("  Best C+T params:", paste0(opt$out, "_best_CT_params.txt"), "\n")