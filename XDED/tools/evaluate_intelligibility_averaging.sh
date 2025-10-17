#!/bin/bash
# Evaluate intelligibility-based averaged models (HIGH and VERY_LOW bands only)

set -e

BASE_CHECKPOINT="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00"
RESULT_DIR="/home/zsim710/XDED/XDED/results/intelligibility_averaging"

# Test speakers (HIGH and VERY_LOW only)
declare -A TEST_SPEAKERS
TEST_SPEAKERS[HIGH]="M08"
TEST_SPEAKERS[VERY_LOW]="M01"

# Test CSVs (corrected paths)
declare -A TEST_CSVS
TEST_CSVS[M08]="/home/zsim710/partitions/uaspeech/by_speakers/M08.csv"
TEST_CSVS[M01]="/home/zsim710/partitions/uaspeech/by_speakers/M01.csv"

echo "========================================"
echo "EVALUATING INTELLIGIBILITY-BASED AVERAGING"
echo "Bands: HIGH and VERY_LOW"
echo "========================================"

# Evaluate both excluded and included scenarios
for scenario in excluded included; do
    echo ""
    echo "========================================"
    echo "Scenario: $scenario"
    echo "========================================"
    
    for band in HIGH VERY_LOW; do
        test_speaker="${TEST_SPEAKERS[$band]}"
        test_csv="${TEST_CSVS[$test_speaker]}"
        
        if [ "$scenario" == "excluded" ]; then
            model_path="$RESULT_DIR/excluded/${band}_averaged_exclude_${test_speaker}.pt"
        else
            model_path="$RESULT_DIR/included/${band}_averaged_include_${test_speaker}.pt"
        fi
        
        # Check if model exists
        if [ ! -f "$model_path" ]; then
            echo "⚠️  Model not found: $model_path"
            echo "   Skipping $band band..."
            continue
        fi
        
        # Setup checkpoint directory
        ckpt_dir="$RESULT_DIR/inference/${scenario}/${band}_${test_speaker}"
        mkdir -p "$ckpt_dir"
        
        echo "Setting up checkpoint directory: $ckpt_dir"
        
        # Copy supporting files with corrected paths
        cp "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/hyperparams.yaml" "$ckpt_dir/"
        cp "/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt" "$ckpt_dir/"
        cp -r "/home/zsim710/XDED/tokenizers/sa_official/tokenizer" "$ckpt_dir/" 2>/dev/null || true
        cp "$BASE_CHECKPOINT/normalizer.ckpt" "$ckpt_dir/"
        
        # Copy averaged model weights
        echo "Copying averaged model weights..."
        cp "$model_path" "$ckpt_dir/model.ckpt"
        
        # Run evaluation
        output_file="$RESULT_DIR/evaluation/${scenario}/${band}_${test_speaker}_results.json"
        mkdir -p "$(dirname "$output_file")"
        
        echo "Testing $band band on $test_speaker ($scenario)..."
        python3 /home/zsim710/XDED/XDED/tools/test_averaged_model_simple.py \
            --checkpoint_dir "$ckpt_dir" \
            --test_csv "$test_csv" \
            --speaker_id "$test_speaker" \
            --output_file "$output_file" \
            --device cuda \
            --simple_norm
        
        echo "✓ Completed $band band"
        echo ""
    done
done

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
echo "Results saved in: $RESULT_DIR/evaluation/"
