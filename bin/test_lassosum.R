#install.packages(c("devtools","RcppArmadillo", "data.table", "Matrix"), dependencies=TRUE)
# install_github("tshmak/lassosum")

library(devtools)


library(lassosum)
# Prefer to work with data.table as it speeds up file reading
library(data.table)
library(methods)
library(magrittr)
# For multi-threading, you can use the parallel package and 
# invoke cl which is then passed to lassosum.pipeline
library(parallel)
library(optparse)

option_list = list(
  make_option(c("-b", "--bed"), type="character", default=NULL, 
              help="bedfile name without extension", metavar="character"),
  make_option(c("-p", "--pheno"), type="character", default=NULL, 
              help="phenotype file name", metavar="character"),
  make_option(c("-c", "--cov"), type="character", default=NULL,
              help="covariate name", metavar="character"),
  make_option(c("-z", "--pcs"), type="character", default=NULL,
              help="principal component analysis file name", metavar="character"), 
  # make_option(c("-", "--ld"), type="character", default=NULL, 
  #             help="ld file name", metavar="character"),
  make_option(c("-f", "--sum_stats"), type="character", default=NULL, 
              help="additional data file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
              help="results data file name", metavar="character")
)  

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input bed).n", call.=FALSE)
}


# This will invoke 2 threads. 
cl <- makeCluster(2)

sum.stat <- opt$sum_stats
bfile <- opt$bed
# Read in and process the covariates
covariate <- fread(opt$cov)
pcs <- fread(opt$pcs) %>%
  setnames(., colnames(.), c("FID","IID", paste0("PC",1:6)))
# Need as.data.frame here as lassosum doesn't handle data.table 
# covariates very well
cov <- merge(covariate, pcs)

# We will need the EUR.hg19 file provided by lassosum 
# which are LD regions defined in Berisa and Pickrell (2015) for the European population and the hg19 genome.
ld.file <- "EUR.hg19"
# output prefix
prefix <- opt$out
# Read in the target phenotype file
target.pheno <- fread(opt$pheno)[,c("FID", "IID", "Height")]
# Read in the summary statistics
ss <- fread(sum.stat)
# Remove P-value = 0, which causes problem in the transformation
ss <- ss[!P == 0]
# Transform the P-values into correlation
cor <- p2cor(p = ss$P,
             n = ss$N,
             sign = log(ss$OR)
)
fam <- fread(paste0(bfile, ".fam"))
fam[,ID:=do.call(paste, c(.SD, sep=":")),.SDcols=c(1:2)]


# Run the lassosum pipeline
# The cluster parameter is used for multi-threading
# You can ignore that if you do not wish to perform multi-threaded processing
out <- lassosum.pipeline(
  cor = cor,
  chr = ss$CHR,
  pos = ss$BP,
  A1 = ss$A1,
  A2 = ss$A2,
  ref.bfile = bfile,
  test.bfile = bfile,
  LDblocks = ld.file, 
  cluster=cl
)
# Store the R2 results
target.res <- validate(out, pheno = as.data.frame(target.pheno), covar=as.data.frame(cov))
# Get the maximum R2
r2 <- max(target.res$validation.table$value)^2

# writte out the results
write.table(target.res$validation.table, 
            file=paste0(prefix, "_lassosum.txt"), 
            sep="\t", 
            row.names=F, 
            quote=F)

# write maximum R2
write.table(r2, 
            file=paste0(prefix, "_lassosum_r2.txt"), 
            sep="\t", 
            row.names=F, 
            quote=F)
