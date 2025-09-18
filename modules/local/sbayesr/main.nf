process sbayesr {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sbayesr", mode: 'copy'

    input:
    val ma_file
    val ld_folder
    val out_prefix
    val annot
    val bed
    val out_dir
    

    output:
    val out_dir

    script:
    """
    mkdir -p ${out_dir}
    
    echo "Tidying summary statistics..."

    Rscript -e "SBayesRC::tidy(mafile='$ma_file', LDdir='$ld_folder', \
                  output='${out_dir}/${out_prefix}_tidy.ma', log2file=TRUE)"

    echo "Imputing missing summary statistics..."

    Rscript -e "SBayesRC::impute(mafile='${out_dir}/${out_prefix}_tidy.ma', LDdir='$ld_folder', \
                  output='${out_dir}/${out_prefix}_imp.ma', log2file=TRUE)"

    echo "Running SBayesRC..."

    Rscript -e "SBayesRC::sbayesrc(mafile='${out_dir}/${out_prefix}_imp.ma', LDdir='$ld_folder', \
                  outPrefix='${out_dir}/${out_prefix}_sbrc', annot='$annot', log2file=TRUE)"

    echo "Calculating PRS..."

    plink --bfile $bed --score '${out_dir}/${out_prefix}_sbrc.txt' --out $out_dir/prs_sbrc

    """
}