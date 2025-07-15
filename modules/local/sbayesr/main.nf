process sbayesr {
    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/sbayesr", mode: 'copy'

    input:
    
    output:
    path('sbayesr*.html'), emit: sbayesr, optional: true

    script:
    """
    /Users/max/Desktop/PRS_Models/nextflow-eval-pipeline/bin/gctb --sbayes R 
     --ldm ../ldm/sparse/chr22/1000G_eur_chr22.ldm.sparse
     --pi 0.95,0.02,0.02,0.01
     --gamma 0.0,0.01,0.1,1
     --gwas-summary ../ma/sim_1.ma
     --chain-length 10000
     --burn-in 2000
     --out-freq 10
     --out sim_1
     """
}