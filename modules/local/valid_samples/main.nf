process valid_samples{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/valid_samples", mode: 'copy'

    input:
    val placeholder
    val input
    val out
  
    output:
    val out

    script:
    """
    Rscript ${params.base_dir}/bin/valid_samples.R   --file $input \\
                                    --out $out
    """
}