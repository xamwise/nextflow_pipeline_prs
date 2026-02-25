#!/usr/bin/env Rscript

library(bigsnpr)
library(data.table)
library(magrittr)
library(optparse)

options(bigstatsr.check.parallel.blas = FALSE)
options(default.nproc.blas = NULL)

# ── Command-line options ──────────────────────────────────────────────────────

option_list <- list(
  make_option(c("-b", "--bed"),       type = "character", default = NULL,
              help = "Bed file name without extension",           metavar = "character"),
  make_option(c("-p", "--pheno"),     type = "character", default = NULL,
              help = "Phenotype file name",                       metavar = "character"),
  make_option(c("-c", "--cov"),       type = "character", default = NULL,
              help = "Covariate file name",                       metavar = "character"),
  make_option(c("-z", "--pcs"),       type = "character", default = NULL,
              help = "Principal component analysis file name",    metavar = "character"),
  make_option(c("-l", "--ld"),        type = "character", default = NULL,
              help = "LD reference file name (.rds)",             metavar = "character"),
  make_option(c("-f", "--sum_stats"), type = "character", default = NULL,
              help = "Summary statistics file name",              metavar = "character"),
  make_option(c("-t", "--trait"),     type = "character", default = NULL,
              help = "Trait type: 'bin' or 'quant'",             metavar = "character"),
  make_option(c("-m", "--model"),     type = "character", default = NULL,
              help = "Model type: 'inf', 'grid', or 'auto'",     metavar = "character"),
  make_option(c("-o", "--out"),       type = "character", default = NULL,
              help = "Output file name",                          metavar = "character")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$bed)) {
  stop("At least one argument must be supplied (input file).", call. = FALSE)
}

# ── Load input data ───────────────────────────────────────────────────────────

phenotype <- fread(opt$pheno)
covariate <- fread(opt$cov)
pcs       <- fread(opt$pcs)

colnames(pcs) <- c("FID", "IID", paste0("PC", 1:6))

pheno <- merge(phenotype, covariate) %>% merge(., pcs)

info     <- readRDS(opt$ld)
sumstats <- bigreadr::fread2(opt$sum_stats)

# ── Format summary statistics ─────────────────────────────────────────────────

names(sumstats) <- c("chr", "pos", "rsid", "a1", "a0",
                     "n_eff", "beta_se", "p", "OR", "INFO", "MAF")

sumstats$beta <- log(sumstats$OR)
sumstats      <- sumstats[sumstats$rsid %in% info$rsid, ]

# ── Prepare genotype data ─────────────────────────────────────────────────────

NCORES   <- nb_cores()
file_dir <- unlist(strsplit(opt$bed, "[.]"))
tmp      <- tempfile(tmpdir = paste(file_dir[1], "tmp-data", sep = "/"))
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)

rds_file <- paste0(opt$bed, "_LD2.rds")
if (!file.exists(rds_file)) {
  cat("Converting BED file to bigSNP format...\n")
  snp_readBed(paste0(opt$bed, ".bed"), backingfile = paste0(opt$bed, "_LD2"))
}
obj.bigSNP <- snp_attach(rds_file)
genotype   <- obj.bigSNP$genotypes

map        <- obj.bigSNP$map[-3]
names(map) <- c("chr", "rsid", "pos", "a1", "a0")

# Ensure chromosome columns are compatible
sumstats$chr <- as.integer(gsub("chr", "", as.character(sumstats$chr), ignore.case = TRUE))
map$chr      <- as.integer(gsub("chr", "", as.character(map$chr),      ignore.case = TRUE))

info_snp <- snp_match(sumstats, map)

CHR  <- map$chr
POS  <- map$pos
POS2 <- snp_asGeneticPos(CHR, POS, dir = ".")

# ── Compute LD matrix ─────────────────────────────────────────────────────────

corr <- NULL
ld   <- NULL

for (chr in 1:22) {
  ind.chr  <- which(info_snp$chr == chr)
  ind.chr2 <- info_snp$`_NUM_ID_`[ind.chr]

  corr0 <- snp_cor(
    genotype,
    ind.col   = ind.chr2,
    ncores    = NCORES,
    infos.pos = POS2[ind.chr2],
    size      = 3 / 1000
  )

  if (chr == 1) {
    ld   <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp)
  } else {
    ld   <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

# ── LD score regression ───────────────────────────────────────────────────────

fam.order <- as.data.table(obj.bigSNP$fam)
setnames(fam.order, c("family.ID", "sample.ID"), c("FID", "IID"))

df_beta <- info_snp[, c("beta", "beta_se", "n_eff", "_NUM_ID_")]

ldsc <- snp_ldsc(
  ld,
  length(ld),
  chi2        = (df_beta$beta / df_beta$beta_se)^2,
  sample_size = df_beta$n_eff,
  blocks      = NULL
)

h2_est <- max(ldsc[["h2"]], 1e-4)

# ── Null model ────────────────────────────────────────────────────────────────

y <- pheno[fam.order, on = c("FID", "IID")]

null.formula <- paste("PC", 1:6, sep = "", collapse = "+") %>%
  paste0("phenotype~Sex+", .) %>%
  as.formula

if (opt$trait == "bin") {
  if (!requireNamespace("fmsb", quietly = TRUE)) {
    install.packages("fmsb", repos = "http://cran.us.r-project.org")
  }
  library(fmsb)

  null.model <- glm(null.formula, data = y, family = binomial)
  null.r2    <- fmsb::NagelkerkeR2(null.model)$R2

} else if (opt$trait == "quant") {
  null.model <- lm(null.formula, data = y)
  null.r2    <- summary(null.model)$r.squared

} else {
  stop("Trait type not recognised. Please specify either 'bin' or 'quant'.")
}

# ── Helper: extract R² appropriate to trait type ──────────────────────────────

get_r2 <- function(formula, data, trait) {
  if (trait == "bin") {
    fit <- glm(formula, data = data, family = binomial)
    return(fmsb::NagelkerkeR2(fit)$R2)
  } else {
    fit <- lm(formula, data = data)
    return(summary(fit)$r.squared)
  }
}

# ── PRS regression formula ────────────────────────────────────────────────────

reg.formula <- paste("PC", 1:6, sep = "", collapse = "+") %>%
  paste0("phenotype~PRS+Sex+", .) %>%
  as.formula

# ── Fit PRS model ─────────────────────────────────────────────────────────────

prs_output  <- NULL
beta_output <- NULL

if (opt$model == "inf") {

  beta_inf <- snp_ldpred2_inf(corr, df_beta, h2 = h2_est)

  pred_inf <- big_prodVec(
    genotype, beta_inf,
    ind.row = 1:nrow(genotype),
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat     <- y
  reg.dat$PRS <- pred_inf
  full.r2     <- get_r2(reg.formula, reg.dat, opt$trait)

  result <- data.table(infinitesimal = full.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_inf)
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = beta_inf)

} else if (opt$model == "grid") {

  p_seq  <- signif(seq_log(1e-4, 1, length.out = 17), 2)
  h2_seq <- round(h2_est * c(0.7, 1, 1.4), 4)

  grid.param <- expand.grid(p = p_seq, h2 = h2_seq, sparse = c(FALSE, TRUE))
  beta_grid  <- snp_ldpred2_grid(corr, df_beta, grid.param, ncores = NCORES)

  pred_grid <- big_prodMat(
    genotype, beta_grid,
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat <- y
  max.r2  <- 0
  best.i  <- 1

  for (i in 1:ncol(pred_grid)) {
    reg.dat$PRS <- pred_grid[, i]
    r2.i <- get_r2(reg.formula, reg.dat, opt$trait)
    if (r2.i > max.r2) {
      max.r2 <- r2.i
      best.i <- i
    }
  }

  result <- data.table(grid = max.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_grid[, best.i])
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = beta_grid[, best.i])

} else if (opt$model == "auto") {

  multi_auto <- snp_ldpred2_auto(
    corr, df_beta,
    h2_init    = h2_est,
    vec_p_init = seq_log(1e-4, 0.9, length.out = NCORES),
    ncores     = NCORES
  )

  beta_auto <- sapply(multi_auto, function(auto) auto$beta_est)

  pred_auto <- big_prodMat(
    genotype, beta_auto,
    ind.row = 1:nrow(genotype),
    ind.col = info_snp$`_NUM_ID_`
  )

  # Keep chains whose SD is within 3 MADs of the median
  pred_scaled     <- apply(pred_auto, 2, sd)
  final_beta_auto <- rowMeans(beta_auto[, abs(pred_scaled - median(pred_scaled)) < 3 * mad(pred_scaled)])

  pred_auto <- big_prodVec(
    genotype, final_beta_auto,
    ind.row = 1:nrow(genotype),
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat     <- y
  reg.dat$PRS <- pred_auto
  full.r2     <- get_r2(reg.formula, reg.dat, opt$trait)

  result <- data.table(auto = full.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_auto)
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = final_beta_auto)

} else {
  stop("Model type not recognised. Please specify either 'inf', 'grid', or 'auto'.")
}

# ── Save output ───────────────────────────────────────────────────────────────

if (is.null(opt$out)) stop("Output file name must be specified.")

out_base <- sub("\\.[^.]*$", "", opt$out)

fwrite(result, opt$out, sep = "\t")
cat("Main results saved to:          ", opt$out, "\n")

if (!is.null(prs_output)) {
  prs_file <- paste0(out_base, "_PRS.csv")
  fwrite(prs_output, prs_file, sep = ",")
  cat("PRS scores saved to:            ", prs_file, "\n")
}

if (!is.null(beta_output)) {
  beta_file <- paste0(out_base, "_betas.csv")
  fwrite(beta_output, beta_file, sep = ",")
  cat("Posterior effect sizes saved to:", beta_file, "\n")
}