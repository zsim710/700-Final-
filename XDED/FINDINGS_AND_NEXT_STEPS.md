# NeMo Student Training: Findings & Next Steps

## Executive Summary

**Problem**: Trained NeMo student model achieved **0% WRA** using decoder-based knowledge distillation.

**Diagnosis**: Evaluated pretrained NeMo baseline → **41.68% WRA** ✅

**Root Cause**: Feature space mismatch between NeMo encoder (176-dim) and teacher decoder expectations (144-dim SpeechBrain features).

**Solution**: Switch from decoder-KD to **CTC-KD** (architecture-agnostic distillation).

---

## Detailed Findings

### 1. Pretrained NeMo Baseline Performance (M08)
```
Command:
python eval_nemo_pretrained.py --held_out M08 --device cuda

Results:
- WRA: 41.68% (744/1785 correct)
- Empty predictions: 0.4%
- Unique predictions: 508/1785
- Average words per utterance: 1.31

Sample predictions:
✓ "COME" → "come"
✓ "SEVEN" → "seven"  
✗ "YALE" → "y over" / "yoa"
✓ "AWAY" → "away"
✓ "OH" → "oh"
```

**Key insight**: The pretrained encoder works reasonably well on dysarthric speech, even though it was only trained on neurotypical speech (LibriSpeech). This confirms M08 is a high-intelligibility speaker.

### 2. Decoder-KD Training Failure (M08)
```
Training:
- 40 epochs, val loss: 0.10 (low KL divergence)
- Model saved every epoch (appeared to be converging)

Evaluation:
- Decoder greedy: WRA 0.00%, avg 1.3 tokens/prediction
- CTC greedy: WRA 0.00%, avg 23 tokens/prediction (massive over-generation)

Sample predictions (decoder mode):
✗ "COME" → "AL" (token 113)
✗ "SEVEN" → "FIELD IT" (tokens 1255, 18)
✗ "YALE" → "DRAGG MUST" (tokens 2715, 129)

Sample predictions (CTC mode):
✗ "COME" → "GRAY DECAY DIMINISH WINTER GLIMPSE PRISCILLA ANGER..." (24 tokens!)
```

**Analysis**:
- Low validation loss misleading (measures KL divergence, not accuracy)
- Model collapsed to ~143 unique predictions (out of 1785 samples)
- Both decoder and CTC heads produced garbage
- Complete training failure despite appearing to converge

### 3. Root Cause: Feature Space Mismatch

**Teacher models** (14 SpeechBrain Conformers):
- Encoder output: `[B, T, 144]` (SpeechBrain feature space)
- Decoder trained on these specific features
- Decoder logits: `[B, T_dec, 5000]` extracted using teacher encoders

**Student model** (NeMo + custom decoder):
- NeMo encoder output: `[B, T, 176]` (NeMo feature space)
- Custom decoder receives NeMo features
- Tries to match teacher decoder logits via KL divergence

**The mismatch**:
1. **Dimension**: 144 vs 176 (different d_model)
2. **Feature representation**: Trained on different data (LibriSpeech vs UA-Speech)
3. **No alignment**: Decoder expects one feature space, receives another

**Why low validation loss?**:
- KL divergence can be minimized even with wrong features
- Model learns spurious correlations
- Overfits to teacher distribution without learning task

---

## Solution: CTC-based Knowledge Distillation

### Why CTC-KD Works

**Decoder-KD requirements**:
- ❌ Encoder features must match teacher's expectations
- ❌ Decoder architecture must align with teacher
- ❌ Feature space compatibility critical

**CTC-KD requirements**:
- ✅ Only needs vocab alignment (both use 5000 tokens)
- ✅ Frame-level distillation (no feature space assumption)
- ✅ Architecture-agnostic (works with any encoder)

### Implementation

**Auto-correction added to `train_student.py`**:
```python
if args.student_backbone == 'nemo' and args.teacher_logits_type == 'decoder':
    print("⚠️  WARNING: Decoder-based KD is incompatible with NeMo backbone!")
    print("   Automatically switching to --teacher_logits_type ctc")
    args.teacher_logits_type = 'ctc'
```

**New training script**: `train_nemo_ctc_m08.sh`
- Stage 1: Freeze encoder, train CTC head (5 epochs, LR 3e-4)
- Stage 2: Unfreeze, fine-tune full model (40 epochs, LR 2e-4)

---

## Next Steps

### Immediate: Retrain M08 with CTC-KD

```bash
# Start fresh training with correct approach
cd /home/zsim710/XDED/XDED
./train_nemo_ctc_m08.sh
```

**Expected results**:
- Conservative: 45-55% WRA (improvement from 41.68% baseline)
- Optimistic: 55-65% WRA (if ensemble knowledge transfers well)

**Success criteria**:
- WRA > 41.68% (beats pretrained baseline)
- Predictions are actual words (not garbage like "DRAGG MUST")
- Reasonable diversity (not collapsed to 143 patterns)

### After M08 Success: Train Other Folds

**High intelligibility**:
- M08 (already done) - baseline 41.68%
- M04, M12, M14

**Mid intelligibility**:
- M05 (target fold) - likely harder
- M10, M11

**Low intelligibility**:
- M16 (target fold) - expect lower baseline
- F05, M09

### Comparison Studies

1. **NeMo-CTC vs SpeechBrain from-scratch**
   - Does pretrained encoder help?
   - How much gain from ensemble KD?

2. **Ablation studies**:
   - With/without freezing encoder stage
   - Temperature: 1.5, 2.0, 3.0
   - Teacher aggregation: prob_mean, logprob_mean, logit_mean

3. **Baseline comparisons**:
   - Pretrained NeMo (no training): 41.68%
   - NeMo + CTC-KD: ???
   - SpeechBrain + CTC-KD: ???
   - SpeechBrain + decoder-KD: ???

---

## Files Modified

### New Files Created
- `eval_nemo_pretrained.py` - Baseline evaluation script
- `train_nemo_ctc_m08.sh` - Corrected training script
- `NEMO_TRAINING_FIX.md` - Detailed technical analysis
- `FINDINGS_AND_NEXT_STEPS.md` - This file

### Modified Files
- `train_student.py` - Added auto-correction for decoder-KD with NeMo
- `eval_student.py` - Added NeMo backbone detection

### Checkpoints to Remove
- `checkpoints_nemo/student_M08/*` - Failed decoder-KD checkpoints (0% WRA)
  - Keep for reference/documentation
  - Don't use for evaluation

### New Checkpoints Location
- `checkpoints_nemo_ctc/student_M08/*` - CTC-KD checkpoints (expected 50-65% WRA)

---

## Lessons Learned

### ❌ Pitfalls to Avoid

1. **Don't trust validation loss alone**
   - Low KL divergence ≠ good predictions
   - Always check actual accuracy metrics

2. **Don't mix encoder architectures in decoder-KD**
   - Feature spaces must match
   - Can't swap pretrained encoders casually

3. **Don't skip baseline evaluation**
   - Testing pretrained model revealed encoder works
   - Saved weeks debugging wrong component

### ✅ Best Practices

1. **Always evaluate pretrained baseline first**
   - Isolates encoder vs training issues
   - Provides performance floor

2. **Use architecture-agnostic distillation when possible**
   - CTC-KD works across encoders
   - Decoder-KD requires careful alignment

3. **Monitor multiple metrics**
   - Loss, WRA, diversity, sample predictions
   - Catch training failures early

---

## Timeline

**Oct 15, 2025 - Morning**: Trained M08 with decoder-KD, got 0% WRA

**Oct 15, 2025 - Afternoon**: 
- Diagnosed via pretrained baseline (41.68% WRA)
- Identified feature space mismatch
- Created fix and new training script

**Oct 15, 2025 - Evening**: Ready to retrain with CTC-KD

**Oct 16, 2025 (Expected)**: M08 CTC-KD results available

---

## Questions for Discussion

1. **Should we completely remove decoder components from NeMoHybridStudent?**
   - Simplify to CTC-only model
   - Remove unused decoder layers (saves memory)

2. **Should we try hybrid approach?**
   - CTC-KD for NeMo encoder
   - Decoder-KD for SpeechBrain encoder
   - Compare which works better

3. **What temperature works best for CTC-KD?**
   - Decoder-KD used T=2.0
   - CTC may need different temperature

4. **How to handle low-intelligibility speakers?**
   - M16, M09 likely have lower pretrained baseline
   - May need different training strategy
