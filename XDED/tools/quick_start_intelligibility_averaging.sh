#!/bin/bash
# Quick start script for intelligibility-based weight averaging baseline

echo "================================================================================"
echo "INTELLIGIBILITY-BASED WEIGHT AVERAGING - QUICK START"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Create 4 averaged models (HIGH and VERY_LOW bands, excluded/included)"
echo "  2. Evaluate models on test speakers (M08 for HIGH, M01 for VERY_LOW)"
echo "  3. Display summary results"
echo ""
echo "Total models: 4"
echo "Total evaluations: 4"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

cd /home/zsim710/XDED/XDED

echo ""
echo "================================================================================"
echo "STEP 1: Creating Averaged Models"
echo "================================================================================"
echo ""

python3 tools/run_intelligibility_averaging.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error during model averaging. Please check the output above."
    exit 1
fi

echo ""
echo "✅ Averaged models created successfully!"
echo ""
echo "Models saved in:"
echo "  - results/intelligibility_averaging/excluded/"
echo "  - results/intelligibility_averaging/included/"
echo ""
read -p "Press Enter to continue with evaluation..."

echo ""
echo "================================================================================"
echo "STEP 2: Evaluating Models"
echo "================================================================================"
echo ""

./tools/evaluate_intelligibility_averaging.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error during evaluation. Please check the output above."
    exit 1
fi

echo ""
echo "✅ Evaluation completed successfully!"
echo ""
read -p "Press Enter to view summary..."

echo ""
echo "================================================================================"
echo "STEP 3: Results Summary"
echo "================================================================================"
echo ""

python3 tools/summarize_intelligibility_results.py

echo ""
echo "================================================================================"
echo "COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in: results/intelligibility_averaging/evaluation/"
echo ""
echo "Next steps:"
echo "  - Compare with knowledge distillation approach"
echo "  - Analyze generalization gap (WER_excluded - WER_included)"
echo "  - See INTELLIGIBILITY_AVERAGING_README.md for details"
echo ""
