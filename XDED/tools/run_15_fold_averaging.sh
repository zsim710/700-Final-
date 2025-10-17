#!/bin/bash
# Script to run 15-fold weight averaging experiment
# For each of the 15 speakers, leave that speaker out and average the other 14 models.

# Activate conda environment
source ~/.bashrc
conda activate FirstDeep

# Paths
CHECKPOINT_JSON="/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
OUTPUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models"
mkdir -p "$OUTPUT_DIR"

# List of all speakers
SPEAKERS=("F02" "F03" "F04" "F05" "M01" "M04" "M05" "M07" "M08" "M09" "M10" "M11" "M12" "M14" "M16")

echo "Running 15-fold averaging experiment"
echo "Total speakers: ${#SPEAKERS[@]}"
echo ""

# Run averaging for each held-out speaker
for i in "${!SPEAKERS[@]}"; do
    held_out="${SPEAKERS[$i]}"
    fold=$((i + 1))
    
    echo "================================================================================"
    echo "FOLD $fold/15: Averaging models with $held_out held out"
    echo "================================================================================"
    
    output_path="$OUTPUT_DIR/${held_out}_held_out_averaged.pt"
    
    python3 /home/zsim710/XDED/XDED/tools/average_sa_models.py \
        --checkpoint_paths_file "$CHECKPOINT_JSON" \
        --output_path "$output_path" \
        --held_out "$held_out" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully averaged models for $held_out held-out"
        echo ""
    else
        echo "✗ Error averaging models for $held_out"
        echo ""
    fi
done

echo "================================================================================"
echo "All 15 folds completed!"
echo "Averaged models saved in: $OUTPUT_DIR"
echo "================================================================================"
