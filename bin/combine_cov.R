library(data.table)
library(optparse)

option_list = list(
  make_option(c("-c", "--cov"), type="character", default=NULL,
              help="covariate name", metavar="character"),
  make_option(c("-z", "--pcs"), type="character", default=NULL,
              help="principal component analysis file name", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL,
              help="results data file name", metavar="character")
)  

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$cov)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input cov).n", call.=FALSE)
}


covariate <- fread(opt$cov)
pcs <- fread(opt$pcs, header=F)
colnames(pcs) <- c("FID","IID", paste0("PC",1:6))
cov <- merge(covariate, pcs)
fwrite(cov,opt$out, sep="\t")