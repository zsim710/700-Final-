#!/bin/bash
# Training script for NeMo student with CTC-based KD (CORRECT approach)
# 
# Background: Initial decoder-KD attempt failed (0% WRA) due to feature space mismatch
# between NeMo encoder (176-dim) and teacher decoder expectations (144-dim SpeechBrain features).
# CTC-KD works because it only requires vocab alignment, not feature compatibility.

set -e

HELD_OUT="M08"
SAVE_DIR="checkpoints_nemo_ctc"

echo "=================================================="
echo "NeMo Student Training with CTC-KD"
echo "=================================================="
echo "Held-out speaker: $HELD_OUT"
echo "Pretrained baseline: 41.68% WRA"
echo "Expected after training: 50-65% WRA"
echo ""

# Stage 1: Freeze encoder (warm up CTC head only)
echo "Stage 1: Training CTC head (encoder frozen, 5 epochs)"
python train_student.py \
  --held_out $HELD_OUT \
  --teacher_logits_type ctc \
  --student_backbone nemo \
  --freeze_nemo_encoder \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 5 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --blank_prob_threshold 0.95 \
  --device cuda \
  --save_dir $SAVE_DIR

echo ""
echo "Stage 1 complete. Starting Stage 2..."
echo ""

# Stage 2: Unfreeze encoder (full fine-tuning)
echo "Stage 2: Fine-tuning full model (40 epochs)"
python train_student.py \
  --held_out $HELD_OUT \
  --teacher_logits_type ctc \
  --student_backbone nemo \
  --batch_size 8 \
  --lr 2e-4 \
  --warmup_steps 2000 \
  --epochs 40 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --blank_prob_threshold 0.95 \
  --device cuda \
  --save_dir $SAVE_DIR \
  --resume ${SAVE_DIR}/student_${HELD_OUT}/latest.pt

echo ""
echo "Training complete! Checkpoint saved to: ${SAVE_DIR}/student_${HELD_OUT}/best.pt"
echo ""
echo "To evaluate:"
echo "python eval_student.py \\"
echo "  --checkpoint ${SAVE_DIR}/student_${HELD_OUT}/best.pt \\"
echo "  --held_out $HELD_OUT \\"
echo "  --decode_mode ctc \\"
echo "  --device cuda \\"
echo "  --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \\"
echo "  --output eval_${HELD_OUT}_nemo_ctc.json"
