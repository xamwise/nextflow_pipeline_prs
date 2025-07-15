library(bigsnpr)

obj.bigsnp <- snp_attach("tmp-data/GWAS_data_sorted_QC.rds")
G <- obj.bigsnp$genotypes
NCORES <- nb_cores()
map <- dplyr::transmute(obj.bigsnp$map,
                        chr = chromosome, pos = physical.pos,
                        a0 = allele2, a1 = allele1)

gz <- runonce::download_file(
  "https://figshare.com/ndownloader/files/38077323",
  dir = "tmp-data", fname = "sumstats_CAD_tuto.csv.gz")
readLines(gz, n = 3)

sumstats <- bigreadr::fread2(
  gz,
  select    = c("CHR", "BP", "A2", "A1", "OR", "SE", "MAF", "INFO"),
  col.names = c("chr", "pos", "a0", "a1", "beta", "beta_se", "freq", "info"))

# GWAS effective sample size for binary traits (4 / (1 / n_case + 1 / n_control))
# For quantitative traits, just use the total sample size for `n_eff`.
sumstats$n_eff <- 4 / (1 / 20791 + 1 / 323124) 

library(dplyr)
info_snp <- snp_match(sumstats, map, return_flip_and_rev = TRUE) %>% 
  mutate(freq = ifelse(`_REV_`, 1 - freq, freq), 
         `_REV_` = NULL, `_FLIP_`= NULL) %>% 
  print()

hist(info_snp$n_eff)    # all the same values, otherwise filter at 70% of max
hist(info_snp$info)     # very good imputation; filter e.g. at 0.7
summary(info_snp$freq)

af_ref <- big_colstats(G, ind.col = info_snp$`_NUM_ID_`, ncores = NCORES)$sum / (2 * nrow(G))
sd_ref <- sqrt(2 * af_ref * (1 - af_ref))
sd_ss <- with(info_snp, 2 / sqrt(n_eff * beta_se^2 + beta^2))
is_bad <-
  sd_ss < (0.5 * sd_ref) | sd_ss > (sd_ref + 0.1) | 
  sd_ss < 0.05 | sd_ref < 0.05  # basically filtering small MAF

library(ggplot2)
ggplot(slice_sample(data.frame(sd_ref, sd_ss, is_bad), n = 50e3)) +
  geom_point(aes(sd_ref, sd_ss, color = is_bad), alpha = 0.5) +
  theme_bigstatsr(0.9) +
  scale_color_viridis_d(direction = -1) +
  geom_abline(linetype = 2) +
  labs(x = "Standard deviations in the reference set",
       y = "Standard deviations derived from the summary statistics",
       color = "To remove?")

sd_af <- with(info_snp, sqrt(2 * freq * (1 - freq) * info))

ggplot(slice_sample(data.frame(sd_af, sd_ss), n = 50e3)) +
  geom_point(aes(sd_af, sd_ss), alpha = 0.5) +
  theme_bigstatsr(0.9) +
  geom_abline(linetype = 2, color = "red") +
  labs(x = "Standard deviations derived from allele frequencies",
       y = "Standard deviations derived from the summary statistics")

diff <- af_ref - info_snp$freq
hist(diff, "FD", xlim = c(-0.1, 0.1))

is_bad2 <-
  sd_ss < (0.7 * sd_af) | sd_ss > (sd_af + 0.1) | 
  sd_ss < 0.05 | sd_af < 0.05 |
  info_snp$info < 0.7 | abs(diff) > 0.07
mean(is_bad2)

df_beta <- info_snp[!is_bad2, ]

# Precomputed genetic positions (in cM) to avoid downloading large files in this tuto
gen_pos <- readRDS(runonce::download_file(
  "https://figshare.com/ndownloader/files/38247288",
  dir = "tmp-data", fname = "gen_pos_tuto.rds"))

df_beta <- dplyr::filter(df_beta, chr %in% 1:4)  # TO REMOVE (for speed here)

for (chr in 1:22) {  # REPLACE BY 1:22
  
  print(chr)
  
  corr0 <- runonce::save_run({
    
    ## indices in 'sumstats'
    ind.chr <- which(df_beta$chr == chr)
    ## indices in 'G'
    ind.chr2 <- df_beta$`_NUM_ID_`[ind.chr]
    
    # genetic positions (in cM)
    # POS2 <- snp_asGeneticPos(map$chr[ind.chr2], map$pos[ind.chr2], dir = "tmp-data")
    POS2 <- gen_pos[ind.chr2]  # USE snp_asGeneticPos() IN REAL CODE
    
    # compute the banded correlation matrix in sparse matrix format
    snp_cor(G, ind.col = ind.chr2, size = 3 / 1000, infos.pos = POS2, 
            ncores = NCORES)
    
  }, file = paste0("tmp-data/corr_chr", chr, ".rds"))
  
  # transform to SFBM (on-disk format) on the fly
  if (chr == 1) {
    ld <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, "tmp-data/corr", compact = TRUE)
  } else {
    ld <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

file.size(corr$sbk) / 1024^3  # file size in GB

beta_lassosum2 <- snp_lassosum2(
  corr, df_beta, ncores = NCORES,
  nlambda = 10, maxiter = 50)      # TO REMOVE, for speed here

pred_grid2 <- big_prodMat(G, beta_lassosum2, ind.col = df_beta[["_NUM_ID_"]],
                          ncores = NCORES)

params2 <- attr(beta_lassosum2, "grid_param")
params2$score <- apply(pred_grid2, 2, function(x) {
  if (all(is.na(x))) return(NA)  # models that diverged substantially
  summary(glm(
    CAD ~ x + sex + age, data = obj.bigsnp$fam, family = "binomial"
  ))$coef["x", 3]
})

ggplot(params2, aes(x = lambda, y = score, color = as.factor(delta))) +
  theme_bigstatsr() +
  geom_point() +
  geom_line() +
  scale_x_log10(breaks = 10^(-5:0)) +
  labs(y = "GLM Z-Score", color = "delta")

best_grid_lassosum2 <- params2 %>%
  mutate(id = row_number()) %>%
  arrange(desc(score)) %>%
  slice(1) %>%
  pull(id) %>% 
  beta_lassosum2[, .]



