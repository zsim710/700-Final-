#!/bin/bash
# DEBUG Training: Include M08 in training set (no held-out)
# 
# Purpose: Isolate whether poor performance is due to:
#   1. Domain shift (train speakers != test speaker)
#   2. Fundamental training/model issues
#
# If this works: Need better generalization (data augmentation, regularization)
# If this fails: Training process itself is broken

set -e

HELD_OUT="M08"  # Will be included in training (not actually held out)
SAVE_DIR="checkpoints_nemo_debug"

echo "=================================================="
echo "DEBUG: NeMo Student with M08 INCLUDED in Training"
echo "=================================================="
echo "Speaker: $HELD_OUT (INCLUDED in training, not held-out)"
echo "Teacher logits: decoder (not CTC)"
echo "Expected: Should learn to fit M08 if training works"
echo ""

# Stage 1: Freeze encoder (warm up decoder head only)
echo "Stage 1: Training decoder head (encoder frozen, 5 epochs)"
python train_student.py \
  --held_out $HELD_OUT \
  --teacher_logits_type decoder \
  --student_backbone nemo \
  --freeze_nemo_encoder \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 5 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir $SAVE_DIR \
  --include_held_out_in_training

echo ""
echo "Stage 1 complete. Starting Stage 2..."
echo ""

# Stage 2: Unfreeze encoder (full fine-tuning)
echo "Stage 2: Fine-tuning full model (35 epochs)"
python train_student.py \
  --held_out $HELD_OUT \
  --teacher_logits_type decoder \
  --student_backbone nemo \
  --batch_size 8 \
  --lr 2e-4 \
  --warmup_steps 2000 \
  --epochs 35 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir $SAVE_DIR \
  --resume ${SAVE_DIR}/student_${HELD_OUT}/latest.pt \
  --include_held_out_in_training

echo ""
echo "Training complete! Checkpoint saved to: ${SAVE_DIR}/student_${HELD_OUT}/best.pt"
echo ""
echo "To evaluate (M08 was in training, so this is training set performance):"
echo "python eval_student.py \\"
echo "  --checkpoint ${SAVE_DIR}/student_${HELD_OUT}/best.pt \\"
echo "  --held_out $HELD_OUT \\"
echo "  --decode_mode decoder \\"
echo "  --max_decode_len 10 \\"
echo "  --device cuda \\"
echo "  --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \\"
echo "  --output eval_${HELD_OUT}_nemo_debug_decoder.json"
echo ""
echo "Expected results:"
echo "  - If WRA > 60%: Training works, issue was domain shift"
echo "  - If WRA < 10%: Training broken, need to debug model/loss"
