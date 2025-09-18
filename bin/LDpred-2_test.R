library(bigsnpr)
options(bigstatsr.check.parallel.blas = FALSE)
options(default.nproc.blas = NULL)

library(data.table)
library(magrittr)


#!/usr/bin/env Rscript
library("optparse")
 
option_list = list(
  make_option(c("-b", "--bed"), type="character", default=NULL, 
              help="bedfile name without extension", metavar="character"),
  make_option(c("-p", "--pheno"), type="character", default=NULL, 
              help="phenotype file name", metavar="character"),
  make_option(c("-c", "--cov"), type="character", default=NULL,
              help="covariate name", metavar="character"),
  make_option(c("-z", "--pcs"), type="character", default=NULL,
              help="principal component analysis file name", metavar="character"), 
  make_option(c("-l", "--ld"), type="character", default=NULL, 
              help="LD file name", metavar="character"),
  make_option(c("-f", "--sum_stats"), type="character", default=NULL, 
              help="additional data file name", metavar="character"),
  make_option(c("-t", "--trait"), type="character", default=NULL, 
              help="binary or quantitative trait options are bin and quant", metavar="character"),
  make_option(c("-m", "--model"), type="character", default=NULL, 
              help="model options are inf, grid and auto", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="results data file name", metavar="character")
)            
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}



phenotype <- fread(opt$pheno)
covariate <- fread(opt$cov)
pcs <- fread(opt$pcs)
# rename columns
colnames(pcs) <- c("FID","IID", paste0("PC",1:6))
# generate required table
pheno <- merge(phenotype, covariate) %>%
  merge(., pcs)

# info <- readRDS(runonce::download_file(
#   "https://ndownloader.figshare.com/files/25503788",
#   fname = "map_hm3_ldpred2.rds"))
info <- readRDS(opt$ld)


# Read in the summary statistic file
sumstats <- bigreadr::fread2(opt$sum_stats) 

# LDpred 2 require the header to follow the exact naming
names(sumstats) <-
  c("chr",
    "pos",
    "rsid",
    "a1",
    "a0",
    "n_eff",
    "beta_se",
    "p",
    "OR",
    "INFO",
    "MAF")
# Transform the OR into log(OR)
sumstats$beta <- log(sumstats$OR)
# Filter out hapmap SNPs
sumstats <- sumstats[sumstats$rsid%in% info$rsid,]


#LD matrix

# Get maximum amount of cores
NCORES <- nb_cores()
# Open a temporary file
# split string to get the directory
# and the file name
opt$bed
file_dir <- unlist(strsplit(opt$bed, "[.]"))
#file_dir[1]

tmp <- tempfile(tmpdir = paste(file_dir[1], "tmp-data", sep = "/"))
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)
# Initialize variables for storing the LD score and LD matrix
corr <- NULL
ld <- NULL
# We want to know the ordering of samples in the bed file 
fam.order <- NULL

# FIX: Check if the .rds file already exists before preprocessing
rds_file <- paste(opt$bed, "rds", sep = ".")
if (!file.exists(rds_file)) {
  # preprocess the bed file (only need to do once for each data set)
  snp_readBed(paste(opt$bed, "bed", sep = "."))
}

# now attach the genotype object
obj.bigSNP <- snp_attach(rds_file)

# extract the SNP information from the genotype
map <- obj.bigSNP$map[-3]
names(map) <- c("chr", "rsid", "pos", "a1", "a0")
# perform SNP matching
info_snp <- snp_match(sumstats, map)
# Assign the genotype to a variable for easier downstream analysis
genotype <- obj.bigSNP$genotypes
# Rename the data structures
CHR <- map$chr
POS <- map$pos
# get the CM information from 1000 Genome
# will download the 1000G file to the current directory (".")
POS2 <- snp_asGeneticPos(CHR, POS, dir = ".")
# calculate LD
for (chr in 1:22) {
  # Extract SNPs that are included in the chromosome
  ind.chr <- which(info_snp$chr == chr)
  ind.chr2 <- info_snp$`_NUM_ID_`[ind.chr]
  # Calculate the LD
  corr0 <- snp_cor(
    genotype,
    ind.col = ind.chr2,
    ncores = NCORES,
    infos.pos = POS2[ind.chr2],
    size = 3 / 1000
  )
  if (chr == 1) {
    ld <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp)
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}
# We assume the fam order is the same across different chromosomes
fam.order <- as.data.table(obj.bigSNP$fam)
# Rename fam order
setnames(fam.order,
         c("family.ID", "sample.ID"),
         c("FID", "IID"))


# LD score regression

df_beta <- info_snp[,c("beta", "beta_se", "n_eff", "_NUM_ID_")]
ldsc <- snp_ldsc(   ld, 
                    length(ld), 
                    chi2 = (df_beta$beta / df_beta$beta_se)^2,
                    sample_size = df_beta$n_eff, 
                    blocks = NULL)
h2_est <- ldsc[["h2"]]

opt$trait

if (opt$trait == "bin") {
  library(fmsb)
  # Reformat the phenotype file such that y is of the same order as the 
  # sample ordering in the genotype file
  y <- pheno[fam.order, on = c("FID", "IID")]
  # Calculate the null R2
  # use glm for binary trait 
  # (will also need the fmsb package to calculate the pseudo R2)
  null.model <- paste("PC", 1:6, sep = "", collapse = "+") %>%
      paste0("Height~Sex+", .) %>%
      as.formula %>%
      glm(., data = y, family=binomial) %>%
      summary
  null.r2 <- fmsb::NagelkerkeR2(null.model)
} else if (opt$trait == "quant") {
  # Reformat the phenotype file such that y is of the same order as the 
  # sample ordering in the genotype file
  y <- pheno[fam.order, on = c("FID", "IID")]
  # Calculate the null R2
  # use glm for binary trait 
  # (will also need the fmsb package to calculate the pseudo R2)
  null.model <- paste("PC", 1:6, sep = "", collapse = "+") %>%
    paste0("Height~Sex+", .) %>%
    as.formula %>%
    lm(., data = y) %>%
    summary
  null.r2 <- null.model$r.squared
} else {
  stop("Trait type not recognized. Please specify either 'bin' or 'quant'.")
}

opt$model

# Initialize variables to store PRS and betas for export
prs_output <- NULL
beta_output <- NULL

if (opt$model == "inf"){
  beta_inf <- snp_ldpred2_inf(corr, df_beta, h2 = h2_est)

  if(is.null(obj.bigSNP)){
    obj.bigSNP <- snp_attach(rds_file)
  }
  genotype <- obj.bigSNP$genotypes
  # calculate PRS for all samples
  ind.test <- 1:nrow(genotype)
  pred_inf <- big_prodVec(    genotype,
                              beta_inf,
                              ind.row = ind.test,
                              ind.col = info_snp$`_NUM_ID_`)


  reg.formula <- paste("PC", 1:6, sep = "", collapse = "+") %>%
    paste0("Height~PRS+Sex+", .) %>%
    as.formula
  reg.dat <- y
  reg.dat$PRS <- pred_inf
  inf.model <- lm(reg.formula, dat=reg.dat) %>%
    summary
  (result <- data.table(
    infinitesimal = inf.model$r.squared - null.r2,
    null = null.r2
  ))
  
  # Store PRS for output
  prs_output <- data.table(
    FID = fam.order$FID,
    IID = fam.order$IID,
    PRS = pred_inf
  )
  
  # Store posterior effects for output
  beta_output <- data.table(
    rsid = info_snp$rsid,
    chr = info_snp$chr,
    pos = info_snp$pos,
    a1 = info_snp$a1,
    a0 = info_snp$a0,
    beta = beta_inf
  )

} else if (opt$model == "grid"){ 
  # Prepare data for grid model
  p_seq <- signif(seq_log(1e-4, 1, length.out = 17), 2)
  h2_seq <- round(h2_est * c(0.7, 1, 1.4), 4)
  grid.param <-
      expand.grid(p = p_seq,
              h2 = h2_seq,
              sparse = c(FALSE, TRUE))
  # Get adjusted beta from grid model
  beta_grid <-
      snp_ldpred2_grid(corr, df_beta, grid.param, ncores = NCORES)

  if(is.null(obj.bigSNP)){
      obj.bigSNP <- snp_attach(rds_file)
  }
  genotype <- obj.bigSNP$genotypes
  # calculate PRS for all samples
  ind.test <- 1:nrow(genotype)
  pred_grid <- big_prodMat(   genotype, 
                              beta_grid, 
                              ind.col = info_snp$`_NUM_ID_`)

  reg.formula <- paste("PC", 1:6, sep = "", collapse = "+") %>%
      paste0("Height~PRS+Sex+", .) %>%
      as.formula
  reg.dat <- y
  max.r2 <- 0
  best.i <- 1
  for(i in 1:ncol(pred_grid)){
      reg.dat$PRS <- pred_grid[,i]
      grid.model <- lm(reg.formula, dat=reg.dat) %>%
          summary  
      if(max.r2 < grid.model$r.squared){
          max.r2 <- grid.model$r.squared
          best.i <- i
      }
  }
  (result <- data.table(
      grid = max.r2 - null.r2,
      null = null.r2
  ))
  
  # Store best PRS for output
  prs_output <- data.table(
    FID = fam.order$FID,
    IID = fam.order$IID,
    PRS = pred_grid[, best.i]
  )
  
  # Store best posterior effects for output
  beta_output <- data.table(
    rsid = info_snp$rsid,
    chr = info_snp$chr,
    pos = info_snp$pos,
    a1 = info_snp$a1,
    a0 = info_snp$a0,
    beta = beta_grid[, best.i]
  )

} else if (opt$model == "auto"){
# Get adjusted beta from the auto model
  multi_auto <- snp_ldpred2_auto(
      corr,
      df_beta,
      h2_init = h2_est,
      vec_p_init = seq_log(1e-4, 0.9, length.out = NCORES),
      ncores = NCORES
  )
  beta_auto <- sapply(multi_auto, function(auto)
      auto$beta_est)

  if(is.null(obj.bigSNP)){
      obj.bigSNP <- snp_attach(rds_file)
  }
  genotype <- obj.bigSNP$genotypes
  # calculate PRS for all samples
  ind.test <- 1:nrow(genotype)
  pred_auto <-
      big_prodMat(genotype,
                  beta_auto,
                  ind.row = ind.test,
                  ind.col = info_snp$`_NUM_ID_`)
  # scale the PRS generated from AUTO
  pred_scaled <- apply(pred_auto, 2, sd)
  final_beta_auto <-
      rowMeans(beta_auto[,
                  abs(pred_scaled -
                      median(pred_scaled)) <
                      3 * mad(pred_scaled)])
  pred_auto <-
      big_prodVec(genotype,
          final_beta_auto,
          ind.row = ind.test,
          ind.col = info_snp$`_NUM_ID_`)
  
  reg.formula <- paste("PC", 1:6, sep = "", collapse = "+") %>%
      paste0("Height~PRS+Sex+", .) %>%
      as.formula
  reg.dat <- y
  reg.dat$PRS <- pred_auto
  auto.model <- lm(reg.formula, dat=reg.dat) %>%
      summary
  (result <- data.table(
      auto = auto.model$r.squared - null.r2,
      null = null.r2
  ))
  
  # Store PRS for output
  prs_output <- data.table(
    FID = fam.order$FID,
    IID = fam.order$IID,
    PRS = pred_auto
  )
  
  # Store posterior effects for output
  beta_output <- data.table(
    rsid = info_snp$rsid,
    chr = info_snp$chr,
    pos = info_snp$pos,
    a1 = info_snp$a1,
    a0 = info_snp$a0,
    beta = final_beta_auto
  )

} else {
  stop("Model type not recognized. Please specify either 'inf', 'grid' or 'auto'.")
}

# Save the results to a file
if (is.null(opt$out)) {
  stop("Output file name must be specified.")
} else {
  # Save main results
  fwrite(result, opt$out, sep="\t")
  cat("Main results saved to:", opt$out, "\n")
  
  # Create base filename without extension
  out_base <- sub("\\.[^.]*$", "", opt$out)
  
  # Save PRS scores for each individual
  if (!is.null(prs_output)) {
    prs_file <- paste0(out_base, "_PRS.csv")
    fwrite(prs_output, prs_file, sep=",")
    cat("PRS scores saved to:", prs_file, "\n")
  }
  
  # Save posterior effect sizes for each SNP
  if (!is.null(beta_output)) {
    beta_file <- paste0(out_base, "_betas.csv")
    fwrite(beta_output, beta_file, sep=",")
    cat("Posterior effect sizes saved to:", beta_file, "\n")
  }
}