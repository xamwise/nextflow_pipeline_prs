#!/usr/bin/env Rscript

library(bigsnpr)
library(data.table)
library(dplyr)
library(ggplot2)
library(optparse)
library(Matrix)

option_list = list(
  make_option(c("-b", "--bed"), type="character", default=NULL, 
              help="bedfile name without extension", metavar="character"),
  make_option(c("-p", "--pheno"), type="character", default=NULL, 
              help="phenotype file name", metavar="character"),
  make_option(c("-c", "--cov"), type="character", default=NULL,
              help="covariate name", metavar="character"),
  make_option(c("-z", "--pcs"), type="character", default=NULL,
              help="principal component analysis file name", metavar="character"),
  make_option(c("-f", "--sum_stats"), type="character", default=NULL, 
              help="summary statistics file name", metavar="character"),
  make_option(c("-t", "--trait"), type="character", default=NULL, 
              help="trait type: 'binary' or 'quant'", metavar="character"),
  make_option(c("-n", "--n_cases"), type="numeric", default=NULL,
              help="number of cases (for binary traits)", metavar="numeric"),
  make_option(c("-m", "--n_controls"), type="numeric", default=NULL,
              help="number of controls (for binary traits)", metavar="numeric"),
  make_option(c("-s", "--n_samples"), type="numeric", default=NULL,
              help="total sample size (for quant traits)", metavar="numeric"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="output prefix", metavar="character"),
  make_option(c("--nlambda"), type="integer", default=20,
              help="number of lambda values [default: 20]", metavar="integer"),
  make_option(c("--maxiter"), type="integer", default=500,
              help="maximum iterations [default: 500]", metavar="integer"),
  make_option(c("--info_thresh"), type="numeric", default=0.3,
              help="INFO score threshold [default: 0.3]", metavar="numeric"),
  make_option(c("--maf_thresh"), type="numeric", default=0.01,
              help="MAF threshold [default: 0.01]", metavar="numeric"),
  make_option(c("--af_diff_thresh"), type="numeric", default=0.1,
              help="Allele frequency difference threshold [default: 0.1]", metavar="numeric"),
  make_option(c("--relax_qc"), type="logical", default=FALSE,
              help="Use relaxed QC filters [default: FALSE]", metavar="logical")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed) || is.null(opt$sum_stats) || is.null(opt$out)){
  print_help(opt_parser)
  stop("Required arguments: --bed, --sum_stats, and --out", call.=FALSE)
}

# Set up
NCORES <- nb_cores()
cat("Using", NCORES, "cores\n")

# Check if .rds file exists, if not create from .bed
rds_file <- paste0(opt$bed, ".rds")
if (!file.exists(rds_file)) {
  cat("Converting BED file to bigSNP format...\n")
  snp_readBed(paste0(opt$bed, ".bed"))
}

# Attach the genetic data
obj.bigsnp <- snp_attach(rds_file)
G <- obj.bigsnp$genotypes
map <- dplyr::transmute(obj.bigsnp$map,
                        chr = chromosome, pos = physical.pos,
                        a0 = allele2, a1 = allele1)

# Read summary statistics
cat("Reading summary statistics...\n")
sumstats <- fread(opt$sum_stats)

# Standardize column names if needed
col_mapping <- c(
  "CHR" = "chr", "chromosome" = "chr",
  "BP" = "pos", "POS" = "pos", "position" = "pos",
  "A1" = "a1", "allele1" = "a1", "effect_allele" = "a1",
  "A2" = "a0", "A0" = "a0", "allele2" = "a0", "other_allele" = "a0",
  "BETA" = "beta", "beta" = "beta", "OR" = "OR", "or" = "OR",
  "SE" = "beta_se", "se" = "beta_se", "standard_error" = "beta_se",
  "MAF" = "freq", "FREQ" = "freq", "frequency" = "freq", "maf" = "freq",
  "INFO" = "info", "info" = "info",
  "P" = "p", "p" = "p", "pvalue" = "p", "p_value" = "p",
  "N" = "n_eff", "n" = "n_eff", "sample_size" = "n_eff"
)

# Rename columns to standard names
for (old_name in names(col_mapping)) {
  if (old_name %in% names(sumstats)) {
    setnames(sumstats, old_name, col_mapping[old_name])
  }
}

# Convert OR to beta if needed
if ("OR" %in% names(sumstats) && !"beta" %in% names(sumstats)) {
  sumstats$beta <- log(sumstats$OR)
}

# Calculate effective sample size based on trait type
if (opt$trait == "binary") {
  if (is.null(opt$n_cases) || is.null(opt$n_controls)) {
    stop("For binary traits, specify --n_cases and --n_controls", call.=FALSE)
  }
  sumstats$n_eff <- 4 / (1 / opt$n_cases + 1 / opt$n_controls)
  cat("Binary trait: n_cases =", opt$n_cases, ", n_controls =", opt$n_controls, "\n")
} else if (opt$trait == "quant") {
  if (is.null(opt$n_samples)) {
    stop("For quant traits, specify --n_samples", call.=FALSE)
  }
  sumstats$n_eff <- opt$n_samples
  cat("quant trait: n_samples =", opt$n_samples, "\n")
} else if ("n_eff" %in% names(sumstats)) {
  cat("Using n_eff from summary statistics\n")
} else {
  stop("Specify trait type (--trait binary/quant) or ensure n_eff column exists", call.=FALSE)
}

# Match SNPs
cat("Matching SNPs between summary stats and reference panel...\n")
info_snp <- snp_match(sumstats, map, return_flip_and_rev = TRUE) %>% 
  mutate(freq = ifelse(`_REV_`, 1 - freq, freq), 
         `_REV_` = NULL, `_FLIP_`= NULL)

cat("Matched", nrow(info_snp), "SNPs\n")

# Quality control
cat("Performing quality control...\n")

# Calculate reference allele frequencies
af_ref <- big_colstats(G, ind.col = info_snp$`_NUM_ID_`, ncores = NCORES)$sum / (2 * nrow(G))
sd_ref <- sqrt(2 * af_ref * (1 - af_ref))

# Standard deviations from summary stats
sd_ss <- with(info_snp, 2 / sqrt(n_eff * beta_se^2 + beta^2))

# Standard deviations from allele frequencies (handle missing info scores)
if ("info" %in% names(info_snp) && !all(is.na(info_snp$info))) {
  sd_af <- with(info_snp, sqrt(2 * freq * (1 - freq) * info))
} else {
  cat("Warning: INFO scores not available, using allele frequencies only\n")
  sd_af <- with(info_snp, sqrt(2 * freq * (1 - freq)))
  info_snp$info <- 1  # Set to 1 if missing
}

# Allele frequency differences
diff <- af_ref - info_snp$freq

# Debug QC filters
cat("\nQC Filter Statistics:\n")
cat("  SD_ss < 0.05:", sum(sd_ss < 0.05, na.rm = TRUE), "\n")
cat("  SD_ref < 0.05:", sum(sd_ref < 0.05, na.rm = TRUE), "\n")
cat("  SD_ss vs SD_ref outliers:", sum(sd_ss < (0.5 * sd_ref) | sd_ss > (sd_ref + 0.1), na.rm = TRUE), "\n")
if ("info" %in% names(info_snp) && !all(info_snp$info == 1, na.rm = TRUE)) {
  cat("  INFO < threshold:", sum(info_snp$info < opt$info_thresh, na.rm = TRUE), "\n")
}
cat("  AF difference > threshold:", sum(abs(diff) > opt$af_diff_thresh, na.rm = TRUE), "\n")
cat("  MAF filter:", sum(info_snp$freq < opt$maf_thresh | info_snp$freq > (1 - opt$maf_thresh), na.rm = TRUE), "\n")

# Apply QC filters based on mode
if (opt$relax_qc) {
  cat("Using relaxed QC filters\n")
  is_bad <- 
    sd_ss < 0.01 | sd_ref < 0.01 |
    abs(diff) > 0.2 |
    info_snp$freq < 0.001 | info_snp$freq > 0.999
} else {
  # Apply standard QC filters
  is_bad <- 
    sd_ss < (0.5 * sd_ref) | sd_ss > (sd_ref + 0.1) | 
    sd_ss < 0.05 | sd_ref < 0.05 |
    abs(diff) > opt$af_diff_thresh
  
  # Only apply INFO filter if INFO scores are available and not all 1
  if ("info" %in% names(info_snp) && !all(info_snp$info == 1, na.rm = TRUE)) {
    is_bad <- is_bad | info_snp$info < opt$info_thresh
  }
  
  # Apply MAF filter
  is_bad <- is_bad | info_snp$freq < opt$maf_thresh | info_snp$freq > (1 - opt$maf_thresh)
}

# Handle NAs in filtering
is_bad[is.na(is_bad)] <- TRUE

cat("\nRemoving", sum(is_bad), "SNPs based on QC (", 
    round(100 * sum(is_bad) / length(is_bad), 2), "%)\n")

df_beta <- info_snp[!is_bad, ]
df_beta <- dplyr::filter(df_beta, chr %in% 1:22)
cat("Final number of SNPs:", nrow(df_beta), "\n")

# Check if we have enough SNPs
if (nrow(df_beta) < 100) {
  warning("Very few SNPs remain after QC (", nrow(df_beta), "). Consider relaxing QC thresholds.\n")
  
  if (!opt$relax_qc) {
    cat("\nTrying with relaxed QC filters...\n")
    cat("Re-run with --relax_qc TRUE to use relaxed filters from the start\n")
    
    # Try with more relaxed filters
    is_bad2 <- 
      sd_ss < 0.01 | sd_ref < 0.01 |
      abs(diff) > 0.15 |
      info_snp$freq < 0.001 | info_snp$freq > 0.999
    
    is_bad2[is.na(is_bad2)] <- TRUE
    
    df_beta_relaxed <- info_snp[!is_bad2, ]
    df_beta_relaxed <- dplyr::filter(df_beta_relaxed, chr %in% 1:22)
    cat("With relaxed QC, would have", nrow(df_beta_relaxed), "SNPs\n")
    
    if (nrow(df_beta_relaxed) > nrow(df_beta) * 10) {
      cat("Using relaxed QC filters due to low SNP count\n")
      df_beta <- df_beta_relaxed
      cat("Final number of SNPs after auto-relaxation:", nrow(df_beta), "\n")
    }
  }
}

if (nrow(df_beta) < 10) {
  stop("Too few SNPs remain after QC (", nrow(df_beta), "). Please check your data or try --relax_qc TRUE", call.=FALSE)
}


# Create correlation matrix
cat("\nComputing LD matrix...\n")

# Initialize variables
ld <- NULL
corr <- NULL
first_chr <- TRUE

for (chr in 1:22) {
  
  cat("Processing chromosome", chr, "...\n")
  
  ## indices in 'df_beta'
  ind.chr <- which(df_beta$chr == chr)
  
  if (length(ind.chr) == 0) {
    cat("  No SNPs on chromosome", chr, "\n")
    next  # Skip to next chromosome
  }
  
  ## indices in 'G'
  ind.chr2 <- df_beta$`_NUM_ID_`[ind.chr]
  
  # Skip if no SNPs
  if (length(ind.chr2) == 0) {
    cat("  No valid SNP indices on chromosome", chr, "\n")
    next
  }
  
  # genetic positions (in cM)
  # Use current directory for genetic map downloads (Nextflow working directory)
  POS2 <- snp_asGeneticPos(map$chr[ind.chr2], map$pos[ind.chr2], dir = ".")
  
  # compute the banded correlation matrix in sparse matrix format
  corr0 <- snp_cor(G, ind.col = ind.chr2, size = 3 / 1000, infos.pos = POS2, 
                   ncores = NCORES)
  
  # transform to SFBM (on-disk format) on the fly
  if (first_chr) {
    ld <- Matrix::colSums(corr0^2)
    # Store SFBM in current directory
    corr <- as_SFBM(corr0, "./corr", compact = TRUE)
    first_chr <- FALSE
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

# Check if we have any correlation matrix
if (is.null(corr)) {
  stop("No correlation matrix could be computed. Please check your data.", call.=FALSE)
}

cat("LD matrix size:", length(ld), "SNPs\n")

# Run lassosum2
cat("\nRunning lassosum2...\n")
cat("Number of SNPs for lassosum2:", nrow(df_beta), "\n")

beta_lassosum2 <- snp_lassosum2(
  corr, df_beta, 
  ncores = NCORES,
  nlambda = opt$nlambda, 
  maxiter = opt$maxiter
)

# Calculate PRS for all models
cat("Calculating polygenic risk scores...\n")
pred_grid2 <- big_prodMat(G, beta_lassosum2, 
                          ind.col = df_beta[["_NUM_ID_"]],
                          ncores = NCORES)

# Get parameters - always extract them
params2 <- attr(beta_lassosum2, "grid_param")

# Read phenotype and covariates if provided
if (!is.null(opt$pheno)) {
  cat("Reading phenotype file...\n")
  pheno_data <- fread(opt$pheno)
  
  # Merge with fam file
  fam_data <- obj.bigsnp$fam
  setDT(fam_data)
  
  # Debug: show column names
  cat("Fam data columns:", paste(names(fam_data), collapse=", "), "\n")
  
  # The fam object from bigsnpr has columns like: family.ID, sample.ID, paternal.ID, maternal.ID, sex, affection
  # Rename to standard FID/IID format
  if ("family.ID" %in% names(fam_data) && "sample.ID" %in% names(fam_data)) {
    setnames(fam_data, c("family.ID", "sample.ID"), c("FID", "IID"))
  } else if (!("FID" %in% names(fam_data) && "IID" %in% names(fam_data))) {
    # If columns are different, use first two columns as FID and IID
    old_names <- names(fam_data)[1:2]
    cat("Renaming", paste(old_names, collapse=", "), "to FID, IID\n")
    setnames(fam_data, old_names, c("FID", "IID"))
  }
  
  # Merge phenotype
  fam_data <- merge(fam_data, pheno_data, by = c("FID", "IID"), all.x = TRUE)
  
  # Read covariates if provided
  if (!is.null(opt$cov)) {
    cov_data <- fread(opt$cov)
    cat("Covariate columns:", paste(names(cov_data), collapse=", "), "\n")
    fam_data <- merge(fam_data, cov_data, by = c("FID", "IID"), all.x = TRUE)
  }
  
  # Read PCs if provided
  if (!is.null(opt$pcs)) {
    pcs_data <- fread(opt$pcs)
    # Check if first two columns are IDs
    if (ncol(pcs_data) > 2) {
      # Assume first two columns are FID and IID
      pc_names <- c("FID", "IID", paste0("PC", 1:(ncol(pcs_data)-2)))
      setnames(pcs_data, names(pcs_data), pc_names)
      cat("PC columns:", paste(pc_names, collapse=", "), "\n")
    }
    fam_data <- merge(fam_data, pcs_data, by = c("FID", "IID"), all.x = TRUE)
  }
  
  # Evaluate models
  cat("Evaluating models...\n")
  
  # Get phenotype column name (assume third column after FID, IID)
  pheno_col <- names(pheno_data)[3]
  cat("Using phenotype column:", pheno_col, "\n")
  
  # Check if phenotype column exists in merged data
  if (!pheno_col %in% names(fam_data)) {
    stop("Phenotype column '", pheno_col, "' not found in merged data", call.=FALSE)
  }
  
  # Build formula
  # Exclude standard fam columns and IDs
  exclude_cols <- c("FID", "IID", pheno_col, "V3", "V4", "V5", "V6", 
                    "paternal.ID", "maternal.ID", "sex", "affection", "x")
  covariates <- setdiff(names(fam_data), exclude_cols)
  
  cat("Available covariates:", paste(covariates, collapse=", "), "\n")
  
  if (length(covariates) > 0) {
    formula_str <- paste(pheno_col, "~ x +", paste(covariates, collapse = " + "))
  } else {
    formula_str <- paste(pheno_col, "~ x")
  }
  cat("Formula:", formula_str, "\n")
  
  # Evaluate each model
  if (opt$trait == "binary") {
    params2$score <- apply(pred_grid2, 2, function(x) {
      if (all(is.na(x))) return(NA)
      fam_data$x <- x
      # Remove rows with missing data for the model
      model_data <- na.omit(fam_data[, c("x", pheno_col, covariates), with = FALSE])
      if (nrow(model_data) < 10) return(NA)
      tryCatch({
        model <- glm(as.formula(formula_str), data = model_data, family = "binomial")
        if ("x" %in% rownames(summary(model)$coef)) {
          summary(model)$coef["x", 3]
        } else {
          NA
        }
      }, error = function(e) {
        cat("Model error:", e$message, "\n")
        NA
      })
    })
  } else {
    params2$score <- apply(pred_grid2, 2, function(x) {
      if (all(is.na(x))) return(NA)
      fam_data$x <- x
      # Remove rows with missing data for the model
      model_data <- na.omit(fam_data[, c("x", pheno_col, covariates), with = FALSE])
      if (nrow(model_data) < 10) return(NA)
      tryCatch({
        model <- lm(as.formula(formula_str), data = model_data)
        if ("x" %in% rownames(summary(model)$coef)) {
          summary(model)$coef["x", 3]
        } else {
          NA
        }
      }, error = function(e) {
        cat("Model error:", e$message, "\n")
        NA
      })
    })
  }
  
  # Find best model
  if (all(is.na(params2$score))) {
    warning("All models failed evaluation, using middle model\n")
    best_idx <- ceiling(nrow(params2) / 2)
  } else {
    best_idx <- which.max(abs(params2$score))
  }
  cat("Best model: lambda =", params2$lambda[best_idx], 
      ", delta =", params2$delta[best_idx], 
      ", score =", params2$score[best_idx], "\n")
  
} else {
  # If no phenotype, just use the middle model
  best_idx <- ceiling(nrow(params2) / 2)
  cat("No phenotype provided, using model", best_idx, "\n")
}

# Extract best beta and PRS
best_beta <- beta_lassosum2[, best_idx]
best_prs <- pred_grid2[, best_idx]

# Create output dataframes
# PRS output
# Handle different column naming conventions
if ("family.ID" %in% names(obj.bigsnp$fam) && "sample.ID" %in% names(obj.bigsnp$fam)) {
  prs_output <- data.table(
    FID = obj.bigsnp$fam$family.ID,
    IID = obj.bigsnp$fam$sample.ID,
    PRS = best_prs
  )
} else {
  # Use first two columns as FID and IID
  prs_output <- data.table(
    FID = obj.bigsnp$fam[[1]],
    IID = obj.bigsnp$fam[[2]],
    PRS = best_prs
  )
}

# Beta output
beta_output <- data.table(
  rsid = df_beta$rsid,
  chr = df_beta$chr,
  pos = df_beta$pos,
  a1 = df_beta$a1,
  a0 = df_beta$a0,
  beta = best_beta
)

# Save results
cat("\nSaving results...\n")

# Save PRS
prs_file <- paste0(opt$out, "_PRS.csv")
fwrite(prs_output, prs_file)
cat("PRS saved to:", prs_file, "\n")

# Save betas
beta_file <- paste0(opt$out, "_betas.csv")
fwrite(beta_output, beta_file)
cat("Betas saved to:", beta_file, "\n")

# Save model parameters
params_file <- paste0(opt$out, "_params.txt")
fwrite(params2, params_file, sep = "\t")
cat("Model parameters saved to:", params_file, "\n")

# Save plot if ggplot2 is available
if (requireNamespace("ggplot2", quietly = TRUE)) {
  plot_file <- paste0(opt$out, "_scores.pdf")
  
  # Only create score plot if phenotype was provided and scores exist
  if (!is.null(opt$pheno) && "score" %in% names(params2)) {
    p <- ggplot(params2, aes(x = lambda, y = score, color = as.factor(delta))) +
      theme_minimal() +
      geom_point() +
      geom_line() +
      scale_x_log10(breaks = 10^(-5:0)) +
      labs(y = "Model Score", color = "delta", title = "Lassosum2 Model Selection") +
      geom_vline(xintercept = params2$lambda[best_idx], linetype = "dashed", color = "red")
  } else {
    # Create a simple parameter plot without scores
    params2$model_idx <- 1:nrow(params2)
    p <- ggplot(params2, aes(x = lambda, y = model_idx, color = as.factor(delta))) +
      theme_minimal() +
      geom_point() +
      scale_x_log10(breaks = 10^(-5:0)) +
      labs(y = "Model Index", color = "delta", title = "Lassosum2 Models") +
      geom_vline(xintercept = params2$lambda[best_idx], linetype = "dashed", color = "red")
  }
  
  ggsave(plot_file, p, width = 8, height = 6)
  cat("Plot saved to:", plot_file, "\n")
}

cat("\n=== LASSOSUM2 COMPLETED SUCCESSFULLY ===\n")
cat("Best model index:", best_idx, "\n")
cat("Number of individuals with PRS:", nrow(prs_output), "\n")
cat("Number of SNPs with betas:", nrow(beta_output), "\n")