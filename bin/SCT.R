#!/usr/bin/env Rscript

library(bigsnpr)
library(data.table)
library(dplyr)
library(ggplot2)
library(optparse)

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
  make_option(c("-c", "--ncores"), type="integer", default=NULL,
              help="number of cores to use [default: all available]", metavar="integer"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="output prefix", metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed) || is.null(opt$sum_stats) || is.null(opt$out)){
  print_help(opt_parser)
  stop("Required arguments: --bed, --sum_stats, and --out", call.=FALSE)
}

# Set number of cores
if (is.null(opt$ncores)) {
  NCORES <- nb_cores()
} else {
  NCORES <- opt$ncores
}
cat("Using", NCORES, "cores\n")

# Read or create bigSNP object
rds_file <- paste0(opt$bed, ".rds")
if (!file.exists(rds_file)) {
  cat("Converting BED file to bigSNP format...\n")
  snp_readBed(paste0(opt$bed, ".bed"))
}
obj.bigSNP <- snp_attach(rds_file)

# Get aliases for useful slots
G   <- obj.bigSNP$genotypes
CHR <- obj.bigSNP$map$chromosome
POS <- obj.bigSNP$map$physical.pos

# Get phenotype
pheno_col <- "phenotype"  # Default name
if (!is.null(opt$pheno)) {
  cat("Reading phenotype file...\n")
  pheno_data <- fread(opt$pheno)
  # Merge with fam file
  fam_data <- obj.bigSNP$fam
  setDT(fam_data)
  
  # Handle column names
  if ("family.ID" %in% names(fam_data) && "sample.ID" %in% names(fam_data)) {
    setnames(fam_data, c("family.ID", "sample.ID"), c("FID", "IID"))
  } else if (!("FID" %in% names(fam_data) && "IID" %in% names(fam_data))) {
    old_names <- names(fam_data)[1:2]
    setnames(fam_data, old_names, c("FID", "IID"))
  }
  
  fam_data <- merge(fam_data, pheno_data, by = c("FID", "IID"), all.x = TRUE)
  # Assume phenotype is in the third column of pheno_data
  pheno_col <- names(pheno_data)[3]
  y <- fam_data[[pheno_col]]
  
} else {
  # Use affection status from fam file
  cat("Using affection status from fam file\n")
  y <- obj.bigSNP$fam$affection
  pheno_col <- "affection"
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

# Read summary statistics
cat("Reading summary statistics...\n")
sumstats <- bigreadr::fread2(opt$sum_stats)

# Standardize column names
col_names <- tolower(names(sumstats))
names(sumstats) <- col_names

# Try to identify and rename columns
if ("chromosome" %in% col_names) names(sumstats)[which(col_names == "chromosome")] <- "chr"
if ("rsid" %in% col_names) names(sumstats)[which(col_names == "rsid")] <- "marker.id"
if ("marker.id" %in% col_names) names(sumstats)[which(col_names == "marker.id")] <- "rsid"
if ("bp" %in% col_names) names(sumstats)[which(col_names == "bp")] <- "pos"
if ("position" %in% col_names) names(sumstats)[which(col_names == "position")] <- "pos"
if ("physical.pos" %in% col_names) names(sumstats)[which(col_names == "physical.pos")] <- "pos"
if ("allele1" %in% col_names) names(sumstats)[which(col_names == "allele1")] <- "a0"
if ("allele2" %in% col_names) names(sumstats)[which(col_names == "allele2")] <- "a1"
if ("a2" %in% col_names) names(sumstats)[which(col_names == "a2")] <- "a0"
if ("effect" %in% col_names) names(sumstats)[which(col_names == "effect")] <- "beta"
if ("or" %in% col_names && !"beta" %in% col_names) {
  sumstats$beta <- log(sumstats$or)
}
if ("pvalue" %in% col_names) names(sumstats)[which(col_names == "pvalue")] <- "p"
if ("p.value" %in% col_names) names(sumstats)[which(col_names == "p.value")] <- "p"

# Check required columns
required_cols <- c("chr", "pos", "a0", "a1", "beta", "p")
missing_cols <- setdiff(required_cols, names(sumstats))
if (length(missing_cols) > 0) {
  stop("Missing required columns in summary statistics: ", paste(missing_cols, collapse = ", "))
}

# Split data into training and test sets
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

# Match variants between genotype data and summary statistics
cat("Matching variants...\n")
map <- obj.bigSNP$map[, c(1, 4, 5, 6)]
names(map) <- c("chr", "pos", "a0", "a1")

# Try matching with strand flipping first
info_snp <- snp_match(sumstats[, required_cols], map)

# If few variants match, try without strand flipping
if (nrow(info_snp) < nrow(sumstats) * 0.5) {
  cat("Few variants matched with strand flipping, trying without...\n")
  info_snp <- snp_match(sumstats[, required_cols], map, strand_flip = FALSE)
}

cat("Matched", nrow(info_snp), "variants out of", nrow(sumstats), "\n")

# Prepare beta and p-values for all SNPs
beta <- rep(NA, ncol(G))
beta[info_snp$`_NUM_ID_`] <- info_snp$beta
lpval <- rep(NA, ncol(G))
lpval[info_snp$`_NUM_ID_`] <- -log10(info_snp$p)

# Perform clumping
cat("Performing clumping...\n")
all_keep <- snp_grid_clumping(G, CHR, POS, 
                              ind.row = ind.train,
                              lpS = lpval, 
                              exclude = which(is.na(lpval)),
                              ncores = NCORES)

cat("Clumping completed with", nrow(attr(all_keep, "grid")), "parameter sets\n")

# Calculate PRS for different thresholds
cat("Computing PRS for multiple thresholds...\n")
# Create backing file in current directory for Nextflow compatibility
multi_PRS <- snp_grid_PRS(G, all_keep, beta, lpval, 
                          ind.row = ind.train,
                          backingfile = "./multi_PRS", 
                          n_thr_lpS = 50, 
                          ncores = NCORES)

cat("Computed", ncol(multi_PRS), "PRS for", nrow(multi_PRS), "individuals\n")

# Perform stacking
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

# Perform stacking with appropriate parameters
tryCatch({
  final_mod <- snp_grid_stacking(
    multi_PRS, 
    y_train_std, 
    ncores = NCORES, 
    K = min(opt$n_folds, length(unique(y_train_std)) - 1),  # Ensure K is valid
    family = if(trait_type == "binary") "binomial" else "gaussian"
  )
  
  # Get the best model
  best_mod_idx <- which.min(final_mod$mod$validation_loss)
  cat("Best model: alpha =", final_mod$mod$alpha[best_mod_idx], "\n")
  cat("Validation loss:", final_mod$mod$validation_loss[best_mod_idx], "\n")
  
}, error = function(e) {
  cat("Error in stacking:", e$message, "\n")
  cat("Trying with reduced parameters...\n")
  
  # Try with fewer folds
  final_mod <<- snp_grid_stacking(
    multi_PRS, 
    y_train_std, 
    ncores = NCORES, 
    K = 2,  # Minimum folds
    family = if(trait_type == "binary") "binomial" else "gaussian"
  )
  
  best_mod_idx <<- which.min(final_mod$mod$validation_loss)
  cat("Best model with K=2: alpha =", final_mod$mod$alpha[best_mod_idx], "\n")
})

# Extract new beta values
new_beta <- final_mod$beta.G
ind_keep <- which(new_beta != 0)

cat("Number of non-zero SNPs:", length(ind_keep), "\n")

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
fwrite(final_mod$mod, paste0(opt$out, "_stacking_summary.txt"), sep = "\t")

# Create plots if ggplot2 is available
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

# Also find best single C+T model for comparison
cat("\nFinding best single C+T model for comparison...\n")
library(tidyr)

grid2 <- attr(all_keep, "grid") %>%
  mutate(thr.lp = list(attr(multi_PRS, "grid.lpS.thr")), id = row_number()) %>%
  unnest(cols = "thr.lp")

s <- nrow(attr(all_keep, "grid"))
n_chr <- length(unique(CHR))

# Evaluate each C+T model
grid2$metric <- big_apply(multi_PRS, a.FUN = function(X, ind, s, y.train) {
  # Sum over all chromosomes
  single_PRS <- rowSums(X[, ind + s * (0:(n_chr-1))])
  
  if (trait_type == "binary") {
    # Use AUC for binary traits
    bigstatsr::AUC(single_PRS, y.train)
  } else {
    # Use correlation for quantitative traits
    abs(cor(single_PRS, y.train, use = "complete.obs"))
  }
}, ind = 1:s, s = s, y.train = y_train,
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