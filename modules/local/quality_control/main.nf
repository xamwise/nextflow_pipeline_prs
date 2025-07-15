process quality_control{

    tag "${name}"
    label 'process_single'
    publishDir "out/${params.run_id}/quality_control", mode: 'copy'

    input:
    val input
    val out
    val maf
    val mind
    val geno
    val hwe

    output:
    val out

    script:
//     """
//     plink --bfile $input \\
//         --geno $geno \\
//         --mind $mind \\
//         --maf $maf \\
//         --hwe $hwe \\
//         --write-snplist \\
//         --make-just-fam \\
//         --out $out
//     """
// }

    """
    plink \\
    --bfile $input \\
    --maf $maf \\
    --hwe $hwe \\
    --geno $geno \\
    --mind $mind \\
    --write-snplist \\
    --make-just-fam \\
    --out $out
    """
}