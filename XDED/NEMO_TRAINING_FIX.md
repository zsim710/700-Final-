# NeMo Student Training Issue & Fix

## Problem Diagnosis

### Baseline Performance
- **Pretrained NeMo model**: 41.68% WRA on M08 test set
- **After decoder-KD training**: 0% WRA (complete failure)

### Root Cause
**Feature Space Mismatch in Decoder-based Knowledge Distillation**

1. **Teacher decoder logits** were extracted using:
   - SpeechBrain Conformer encoders (14 speaker-specific models)
   - Encoder outputs: [B, T, 144] (SpeechBrain feature space)
   - Decoder trained on these specific acoustic features

2. **Student model** uses:
   - NeMo Conformer encoder (pretrained on LibriSpeech)
   - Encoder outputs: [B, T, 176] (NeMo feature space)
   - Dimension mismatch: 144 vs 176
   - **Different feature representations** even if dimensions matched

3. **What went wrong**:
   - Student decoder receives NeMo encoder features [B, T, 176]
   - Teacher decoder logits expect SpeechBrain features [B, T, 144]
   - KL divergence loss tries to match distributions, but:
     - Input features are fundamentally incompatible
     - Decoder learns to minimize loss on wrong feature space
     - Results in nonsensical predictions (0% accuracy)

### Evidence
```
Decoder-KD Results (WRONG):
- Target: "COME" → Predicted: "AL" (token 113)
- Target: "SEVEN" → Predicted: "FIELD IT" (tokens 1255, 18)
- Target: "YALE" → Predicted: "DRAGG MUST" (tokens 2715, 129)
- WRA: 0.00%
- Model collapsed to ~143 unique predictions across 1785 samples

CTC Decoding (ALSO WRONG):
- Target: "COME" → Predicted: "GRAY DECAY DIMINISH WINTER..." (24 tokens)
- Massive over-generation (avg 23 words for single-word targets)
- WER: 2300%
```

## Solution: Use CTC-based Knowledge Distillation

### Why CTC-KD Works
1. **Frame-level alignment only**: CTC operates on encoder outputs directly, no decoder involved
2. **No feature space assumption**: KL divergence on CTC logits doesn't require matching encoder architectures
3. **Vocabulary alignment**: Both use same 5000-token vocab (teacher CTC head was retrained)

### Implementation Changes

#### 1. Auto-correct to CTC mode for NeMo backbone
```python
# In train_student.py
if args.student_backbone == 'nemo' and args.teacher_logits_type == 'decoder':
    print("⚠️  WARNING: Decoder-based KD is incompatible with NeMo backbone!")
    print("   Automatically switching to --teacher_logits_type ctc")
    args.teacher_logits_type = 'ctc'
```

#### 2. Updated training command
```bash
# CORRECT: CTC-based KD for NeMo
python train_student.py \
  --held_out M08 \
  --teacher_logits_type ctc \
  --student_backbone nemo \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 40 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir checkpoints_nemo_ctc

# WRONG (auto-corrected): Decoder-KD with NeMo
python train_student.py \
  --held_out M08 \
  --teacher_logits_type decoder \  # Will be auto-switched to 'ctc'
  --student_backbone nemo \
  ...
```

#### 3. Remove decoder components (optional optimization)
Since NeMo + decoder-KD doesn't work, we could simplify NeMoHybridStudent to only have CTC head:
- Remove `forward_decoder()` method
- Remove `decode_greedy()` method  
- Remove Transformer decoder layers
- Keep only `forward()` for CTC

## Expected Results with CTC-KD

### Baseline (pretrained NeMo, no training)
- WRA: 41.68% on M08

### After CTC-KD Training (expected)
- **Conservative estimate**: 45-55% WRA
  - Pretrained starts at 41.68%
  - CTC-KD from 14 teachers should provide regularization
  - Fine-tuning on UA-Speech should improve dysarthric adaptation
  
- **Optimistic estimate**: 55-65% WRA
  - If ensemble knowledge transfers well
  - If NeMo encoder adapts to dysarthric patterns
  
### Key Differences from Decoder-KD Failure
- ✅ No feature space mismatch
- ✅ Direct CTC head training on ensemble CTC distributions
- ✅ Preserves pretrained encoder knowledge
- ✅ Simple frame-level alignment

## Training Recipe

### Stage 1: Freeze encoder (5 epochs, LR 3e-4)
```bash
python train_student.py \
  --held_out M08 \
  --teacher_logits_type ctc \
  --student_backbone nemo \
  --freeze_nemo_encoder \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 5 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir checkpoints_nemo_ctc
```

### Stage 2: Unfreeze encoder (40 epochs, LR 2e-4)
```bash
python train_student.py \
  --held_out M08 \
  --teacher_logits_type ctc \
  --student_backbone nemo \
  --batch_size 8 \
  --lr 2e-4 \
  --warmup_steps 2000 \
  --epochs 40 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir checkpoints_nemo_ctc \
  --resume checkpoints_nemo_ctc/student_M08/latest.pt
```

### Evaluation
```bash
# CTC decoding (recommended)
python eval_student.py \
  --checkpoint checkpoints_nemo_ctc/student_M08/best.pt \
  --held_out M08 \
  --decode_mode ctc \
  --device cuda \
  --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
  --output eval_M08_nemo_ctc.json
```

## Lessons Learned

### ❌ Don't Mix Encoder Architectures in Decoder-KD
- Teacher decoder expects specific encoder features
- Can't swap encoders without retraining decoder
- Feature space compatibility is critical

### ✅ CTC-KD is Architecture-Agnostic
- Only requires vocab alignment
- Works across different encoder architectures
- Frame-level distillation is robust

### 🔍 Always Test Pretrained Baseline
- Pretrained NeMo: 41.68% WRA
- Identified that encoder works, training process was broken
- Saved weeks of debugging wrong component

## Next Steps

1. **Retrain M08 with CTC-KD** (expected ~50-60% WRA)
2. **Compare with SpeechBrain baseline** (from-scratch training)
3. **Train other folds** (M16, M04, M05)
4. **Ablation studies**:
   - With/without freezing encoder
   - Different temperatures (1.5, 2.0, 3.0)
   - Different teacher aggregation methods
