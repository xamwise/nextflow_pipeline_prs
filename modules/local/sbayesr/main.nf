process sbayesr {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sbayesr", mode: 'copy'

    input:
    val ma_file
    val ldfolder
    val out_prefix
    val annot
    


    out:
    val ${out_prefix}_sbrc

    script:
    """
    Rscript -e "SBayesRC::tidy(mafile='$ma_file', LDdir='$ld_folder', \
                  output='${out_prefix}_tidy.ma', log2file=TRUE)"

    Rscript -e "SBayesRC::impute(mafile='${out_prefix}_tidy.ma', LDdir='$ld_folder', \
                  output='${out_prefix}_imp.ma', log2file=TRUE)"

    Rscript -e "SBayesRC::sbayesrc(mafile='${out_prefix}_imp.ma', LDdir='$ld_folder', \
                  outPrefix='${out_prefix}_sbrc', annot='$annot', log2file=TRUE)"
    
     """
}