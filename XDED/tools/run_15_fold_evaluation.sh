#!/bin/bash
# Run inference for all 15 folds of the weight averaging experiment
# For each fold:
#   1. Test the averaged model on the held-out speaker
#   2. Test each of the 14 individual SA models on the held-out speaker (for WRA comparison)

set -e  # Exit on error

# Activate conda environment
source ~/.bashrc
conda activate FirstDeep

# Paths
CHECKPOINT_JSON="/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
AVERAGED_CHECKPOINTS_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints"
TEST_DATA_DIR="/home/zsim710/partitions/uaspeech/by_speakers"
OUTPUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/evaluation"
INFERENCE_SCRIPT="/home/zsim710/XDED/XDED/tools/test_sa_model.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# All speakers
SPEAKERS=("F02" "F03" "F04" "F05" "M01" "M04" "M05" "M07" "M08" "M09" "M10" "M11" "M12" "M14" "M16")

echo "================================================================================"
echo "Running 15-fold Weight Averaging Evaluation"
echo "================================================================================"
echo ""

# Process each fold
for i in "${!SPEAKERS[@]}"; do
    held_out="${SPEAKERS[$i]}"
    fold=$((i + 1))
    
    echo "================================================================================"
    echo "FOLD $fold/15: Evaluating with $held_out held out"
    echo "================================================================================"
    echo ""
    
    # Create fold-specific output directory
    fold_output_dir="$OUTPUT_DIR/fold_${fold}_${held_out}"
    mkdir -p "$fold_output_dir"
    
    # Test data for held-out speaker
    test_csv="$TEST_DATA_DIR/${held_out}.csv"
    
    if [ ! -f "$test_csv" ]; then
        echo "✗ Test data not found: $test_csv"
        continue
    fi
    
    # 1. Test the averaged model on held-out speaker
    echo "--------------------------------------------------------------------------------"
    echo "1. Testing AVERAGED MODEL on $held_out"
    echo "--------------------------------------------------------------------------------"
    
    averaged_checkpoint="$AVERAGED_CHECKPOINTS_DIR/${held_out}_held_out"
    averaged_output="$fold_output_dir/averaged_model_results.json"
    
    if [ -d "$averaged_checkpoint" ]; then
        echo "Checkpoint: $averaged_checkpoint"
        echo "Output: $averaged_output"
        echo ""
        
        python3 "$INFERENCE_SCRIPT" \
            --checkpoint_dir "$averaged_checkpoint" \
            --test_csv "$test_csv" \
            --speaker_id "$held_out" \
            --output_file "$averaged_output" \
            --device cuda
        
        if [ $? -eq 0 ]; then
            echo "✓ Averaged model evaluation complete"
        else
            echo "✗ Error evaluating averaged model"
        fi
    else
        echo "✗ Averaged checkpoint not found: $averaged_checkpoint"
    fi
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "2. Testing INDIVIDUAL MODELS on $held_out (for WRA comparison)"
    echo "--------------------------------------------------------------------------------"
    echo ""
    
    # Test each individual SA model (excluding held-out speaker)
    individual_count=0
    for other_speaker in "${SPEAKERS[@]}"; do
        if [ "$other_speaker" == "$held_out" ]; then
            continue  # Skip the held-out speaker
        fi
        
        individual_count=$((individual_count + 1))
        echo "[$individual_count/14] Testing $other_speaker model on $held_out..."
        
        # Find the individual SA model checkpoint directory
        # The checkpoint paths are in the JSON, but we need to find the parent directory
        # that contains the full checkpoint setup
        individual_checkpoint_base="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_${other_speaker}_"*
        
        # Find the actual checkpoint directory
        individual_checkpoint=$(find /mnt/Research/qwan121/ICASSP_SA -type d -name "val_uncommon_${other_speaker}_*" 2>/dev/null | head -1)
        
        if [ -z "$individual_checkpoint" ]; then
            echo "  ✗ Checkpoint not found for $other_speaker"
            continue
        fi
        
        # Find the latest CKPT directory
        ckpt_dir=$(find "$individual_checkpoint" -type d -name "CKPT+*" 2>/dev/null | sort | tail -1)
        
        if [ -z "$ckpt_dir" ]; then
            echo "  ✗ CKPT directory not found in $individual_checkpoint"
            continue
        fi
        
        individual_output="$fold_output_dir/individual_${other_speaker}_results.json"
        
        # Note: Individual SA model checkpoints might not have the same structure as our averaged ones
        # They might need hyperparams.yaml, tokenizer, etc. to be copied/configured
        # For now, we'll try to use them directly
        
        echo "  Checkpoint: $ckpt_dir"
        echo "  Output: $individual_output"
        
        # Skip for now - individual SA models need proper setup
        # TODO: Set up individual SA models with proper hyperparams and tokenizer like we did for averaged models
        echo "  (Skipping - individual SA models need proper checkpoint setup)"
        
        # Uncomment when individual checkpoints are properly set up:
        # python3 "$INFERENCE_SCRIPT" \
        #     --checkpoint_dir "$ckpt_dir" \
        #     --test_csv "$test_csv" \
        #     --speaker_id "$held_out" \
        #     --output_file "$individual_output" \
        #     --device cuda
        
        echo ""
    done
    
    echo "================================================================================"
    echo "Fold $fold complete!"
    echo "================================================================================"
    echo ""
done

echo "================================================================================"
echo "ALL 15 FOLDS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Next step: Aggregate results and compute WRA metrics"
