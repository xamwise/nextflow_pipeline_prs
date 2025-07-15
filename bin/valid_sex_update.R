library(data.table)

#!/usr/bin/env Rscript
library("optparse")
 
option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
  make_option(c("-s", "--sex_check"), type="character", default=NULL,
              help="sex_check result file name", metavar="character"),
  make_option(c("-o", "--out"), type="character", default=NULL, 
              help="output file name", metavar="character")  
)
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}


# Read in file
valid <- fread(opt$file)
dat <- fread(opt$sex_check)[FID%in%valid$FID]
fwrite(dat[STATUS=="OK",c("FID","IID")], opt$out, sep="\t") 