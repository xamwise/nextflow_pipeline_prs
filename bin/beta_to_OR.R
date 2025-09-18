library(data.table)
library(optparse)


option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL, 
              help="bedfile name without extension", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
              help="results data file name", metavar="character")
)  

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}
dat <- fread(opt$file)
fwrite(dat[,OR:=exp(BETA)], opt$out, sep="\t")