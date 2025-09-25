#!/bin/bash

# Nextflow Pipeline Runner Script
# Runs QC pipeline first, then three pipelines in parallel

set -e  # Exit on any error

echo "Starting Nextflow pipeline execution..."
echo "==========================================="

# Step 1: Run QC pipeline and wait for completion
echo "Step 1: Running QC pipeline..."
echo "Command: nextflow run workflows/qc_pipeline.nf -params-file workflows/config/params_qc.yaml"
echo ""

if nextflow run workflows/qc_pipeline.nf -params-file workflows/config/params_qc.yaml; then
    echo "✓ QC pipeline completed successfully"
    echo ""
else
    echo "✗ QC pipeline failed"
    exit 1
fi

# Step 2: Run the three pipelines in parallel
echo "Step 2: Running remaining pipelines in parallel..."
echo "==========================================="

# Start all three pipelines in parallel
echo "Starting PRS models pipeline..."
nextflow run workflows/prs_models_pipeline.nf -params-file workflows/config/params_prs.yaml &
PRS_PID=$!

echo "Starting sklearn pipeline..."
nextflow run workflows/sklearn_pipeline.nf -params-file workflows/config/sklearn_config.yaml &
SKLEARN_PID=$!

echo "Starting deep learning PRS pipeline..."
nextflow run workflows/dl_prs.nf --params-file workflows/config/dl_config.yaml &
DL_PID=$!

echo ""
echo "All parallel pipelines started. Waiting for completion..."
echo "PRS models pipeline PID: $PRS_PID"
echo "Sklearn pipeline PID: $SKLEARN_PID" 
echo "DL PRS pipeline PID: $DL_PID"
echo ""

# Wait for all parallel jobs to complete and capture their exit codes
wait $PRS_PID
PRS_EXIT=$?

wait $SKLEARN_PID
SKLEARN_EXIT=$?

wait $DL_PID
DL_EXIT=$?

# Check results
echo "Parallel execution completed. Results:"
echo "==========================================="

if [ $PRS_EXIT -eq 0 ]; then
    echo "PRS models pipeline completed successfully"
else
    echo "PRS models pipeline failed (exit code: $PRS_EXIT)"
fi

if [ $SKLEARN_EXIT -eq 0 ]; then
    echo "Sklearn pipeline completed successfully"
else
    echo "Sklearn pipeline failed (exit code: $SKLEARN_EXIT)"
fi

if [ $DL_EXIT -eq 0 ]; then
    echo "Deep learning PRS pipeline completed successfully"
else
    echo "Deep learning PRS pipeline failed (exit code: $DL_EXIT)"
fi

# Overall result
if [ $PRS_EXIT -eq 0 ] && [ $SKLEARN_EXIT -eq 0 ] && [ $DL_EXIT -eq 0 ]; then
    echo ""
    echo "All pipelines completed successfully!"
    exit 0
else
    echo ""
    echo "One or more pipelines failed"
    exit 1
fi