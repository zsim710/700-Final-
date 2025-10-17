#!/bin/bash
# This script demonstrates the entire workflow for comparing speaker model
# averaging vs WRA averaging for cross-speaker generalization.

# Base directory for SA model checkpoints
# Replace with the path provided by your supervisor
SA_CHECKPOINT_DIR="/mnt/Research/qwan121/ICASSP_SA"

# Directory for saving averaged models and results
OUTPUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging"
mkdir -p $OUTPUT_DIR

# Test set path - replace with actual path to test data
TESTSET_PATH="/home/zsim710/XDED/XDED/datasets/uaspeech/test_data.json"

# List of all speakers to include in the experiment
SPEAKERS=("F03" "F04" "F05" "M01" "M04" "M05" "M07" "M08" "M09" "M10" "M11" "M12" "M14" "M16")

# Held-out speakers to evaluate
HELD_OUT_SPEAKERS=("M05" "M08" "F04")

# Step 1: Prepare speaker checkpoint mapping JSON file
echo "Preparing speaker checkpoint mapping..."
python3 /home/zsim710/XDED/XDED/tools/prepare_checkpoint_mapping.py \
  --base_dir $SA_CHECKPOINT_DIR \
  --output_path $OUTPUT_DIR/speaker_checkpoints.json \
  --speakers "${SPEAKERS[@]}"

# Step 2: Average models for each held-out speaker
for held_out in "${HELD_OUT_SPEAKERS[@]}"; do
  echo "Creating averaged model excluding $held_out..."
  python3 /home/zsim710/XDED/XDED/tools/average_sa_models.py \
    --checkpoint_paths_file $OUTPUT_DIR/speaker_checkpoints.json \
    --output_path $OUTPUT_DIR/${held_out}_held_out_averaged.pt \
    --held_out $held_out \
    --device cuda
done

# Step 3: Compare averaging methods for each held-out speaker
for held_out in "${HELD_OUT_SPEAKERS[@]}"; do
  echo "Comparing averaging methods for $held_out..."
  python3 /home/zsim710/XDED/XDED/tools/compare_sa_averaging.py \
    --checkpoint_paths_file $OUTPUT_DIR/speaker_checkpoints.json \
    --held_out $held_out \
    --testset_path $TESTSET_PATH \
    --output_dir $OUTPUT_DIR \
    --device cuda
done

# Step 4: Aggregate results across all held-out speakers
echo "Aggregating results..."
python3 - << EOF
import os
import json
import numpy as np

output_dir = "$OUTPUT_DIR"
held_out_speakers = ${HELD_OUT_SPEAKERS[@]/#/\"}
held_out_speakers = [s + "\"" for s in held_out_speakers]

# Load individual results
results = []
for speaker in held_out_speakers:
    result_path = os.path.join(output_dir, f"{speaker}_comparison_results.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            results.append(json.load(f))

# Compute averages
if results:
    # Extract key metrics
    wra_vs_best = [r['wra_vs_best'] for r in results]
    model_avg_vs_best = [r['model_avg_vs_best'] for r in results]
    model_avg_vs_wra = [r['model_avg_vs_wra'] for r in results]
    
    # Compute means
    summary = {
        "num_held_out_speakers": len(results),
        "avg_wra_vs_best": np.mean(wra_vs_best),
        "avg_model_avg_vs_best": np.mean(model_avg_vs_best),
        "avg_model_avg_vs_wra": np.mean(model_avg_vs_wra),
        "held_out_speakers": [r['held_out_speaker'] for r in results]
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "aggregated_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nAggregated Results Summary:")
    print(f"Number of held-out speakers: {summary['num_held_out_speakers']}")
    print(f"Avg WRA vs best individual: {summary['avg_wra_vs_best']:.2f}%")
    print(f"Avg model averaging vs best individual: {summary['avg_model_avg_vs_best']:.2f}%")
    print(f"Avg model averaging vs WRA: {summary['avg_model_avg_vs_wra']:.2f}%")
    print(f"Results saved to: {summary_path}")
else:
    print("No results found to aggregate.")
EOF

echo "Experiment complete! Results are in $OUTPUT_DIR"