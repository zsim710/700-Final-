#!/bin/bash
# Create full SpeechBrain checkpoint directories for all 15 averaged models
# This copies supporting files from a reference SA model and includes our averaged weights

set -e  # Exit on error

# Paths
REFERENCE_SA="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775"
REFERENCE_CKPT="$REFERENCE_SA/save/CKPT+2024-07-11+20-46-30+00"
TOKENIZER_DIR="/home/zsim710/XDED/tokenizers/sa_official"
AVERAGED_MODELS="/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models"
OUTPUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints"

echo "Creating full checkpoint directories for averaged models..."
echo "Reference SA model: $REFERENCE_SA"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# All speakers
SPEAKERS=("F02" "F03" "F04" "F05" "M01" "M04" "M05" "M07" "M08" "M09" "M10" "M11" "M12" "M14" "M16")

# Process each speaker
for speaker in "${SPEAKERS[@]}"; do
    echo "================================================"
    echo "Setting up checkpoint for ${speaker} held-out"
    echo "================================================"
    
    CKPT_DIR="$OUTPUT_DIR/${speaker}_held_out"
    
    # Create directory structure
    mkdir -p "$CKPT_DIR"
    
    # Copy hyperparams.yaml from reference model
    echo "  Copying hyperparams.yaml..."
    cp "$TOKENIZER_DIR/hyperparams.yaml" "$CKPT_DIR/"
    
    # Copy tokenizer files
    echo "  Copying tokenizer files..."
    cp "$TOKENIZER_DIR/tokenizer.ckpt" "$CKPT_DIR/"
    cp -r "$TOKENIZER_DIR/tokenizer" "$CKPT_DIR/" 2>/dev/null || echo "    (tokenizer directory not found, skipping)"
    
    # Copy normalizer from reference checkpoint
    echo "  Copying normalizer.ckpt..."
    cp "$REFERENCE_CKPT/normalizer.ckpt" "$CKPT_DIR/"
    
    # Copy other supporting files that might be needed
    echo "  Copying other supporting files..."
    cp "$REFERENCE_CKPT/brain.ckpt" "$CKPT_DIR/" 2>/dev/null || echo "    (brain.ckpt not found, skipping)"
    cp "$REFERENCE_CKPT/noam_scheduler.ckpt" "$CKPT_DIR/" 2>/dev/null || echo "    (noam_scheduler.ckpt not found, skipping)"
    cp "$REFERENCE_CKPT/counter.ckpt" "$CKPT_DIR/" 2>/dev/null || echo "    (counter.ckpt not found, skipping)"
    cp "$REFERENCE_CKPT/CKPT.yaml" "$CKPT_DIR/" 2>/dev/null || echo "    (CKPT.yaml not found, skipping)"
    
    # Copy our averaged model weights as model.ckpt
    echo "  Copying averaged model weights..."
    cp "$AVERAGED_MODELS/${speaker}_held_out_averaged.pt" "$CKPT_DIR/model.ckpt"
    
    # Verify the checkpoint directory
    echo "  Verifying checkpoint directory..."
    if [ -f "$CKPT_DIR/model.ckpt" ] && [ -f "$CKPT_DIR/hyperparams.yaml" ] && [ -f "$CKPT_DIR/tokenizer.ckpt" ]; then
        echo "  ✓ Checkpoint directory created successfully"
        echo "  Files in $CKPT_DIR:"
        ls -lh "$CKPT_DIR"
    else
        echo "  ✗ Error: Missing required files in checkpoint directory"
        exit 1
    fi
    
    echo ""
done

echo "================================================"
echo "All 15 checkpoint directories created!"
echo "================================================"
echo ""
echo "Summary:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Total checkpoints: ${#SPEAKERS[@]}"
echo ""
echo "You can now use these checkpoints for inference with SpeechBrain."
echo ""
echo "Example usage:"
echo "  python inference_script.py \\"
echo "    --checkpoint_dir $OUTPUT_DIR/F02_held_out \\"
echo "    --test_csv /home/zsim710/partitions/uaspeech/by_speakers/F02.csv"
