process quality_control_sum{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/quality_control_sum", mode: 'copy'

    input:
    val input
    val out
    val info
    val maf

    output:
    val out

    script:
    """
    Rscript ${params.base_dir}/bin/quality_control_sum.R   --file $input \\
                                    --out $out\\
                                    --info $info \\
                                    --maf $maf
    """
}


