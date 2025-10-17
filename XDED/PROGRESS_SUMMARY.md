# Logit Distillation Progress Summary

## ✅ Completed

### 1. Removed Feature Extraction Components
- **OLD**: AudioCNN → mel-spectrograms → 256-dim features → domain heads
- **NEW**: Pure logit-level distillation (no audio processing needed)
- All redundant components (AudioCNN, audio transforms, audio loaders) can be removed/ignored

### 2. Verified Logit Dataset Structure
Successfully tested loading from `/home/zsim710/XDED/speechbrain/exp_results/logit_extraction/`:

| Speaker | Utterances | Intelligibility | Role |
|---------|-----------|-----------------|------|
| F02 | 1,785 | HIGH | Teacher |
| F03 | 1,616 | HIGH | Teacher |
| M04 | 1,275 | HIGH | Teacher |
| M07 | 1,785 | HIGH | Teacher |
| **M08** | **1,785** | **HIGH** | **Test (Held-out)** |
| M12 | 1,530 | HIGH | Teacher |
| M14 | 1,785 | HIGH | Teacher |
| F04 | 1,785 | MID | Teacher |
| **M05** | **1,785** | **MID** | **Test (Held-out)** |
| M10 | 1,785 | MID | Teacher |
| M11 | 1,530 | MID | Teacher |
| F05 | 1,785 | LOW | Teacher |
| M09 | 1,785 | LOW | Teacher |
| **M16** | **1,530** | **LOW** | **Test (Held-out)** |
| **M01** | **765** | **VERY_LOW** | **Test (Held-out)** |

**Logit Format**: `[seq_len, 5000]` - decoder logits with vocab_size=5000

### 3. Identified VERY_LOW Challenge
- M01 is the ONLY speaker in VERY_LOW intelligibility
- **Solution**: Use all 14 other speakers as cross-level teachers for M01 fold
- This tests whether cross-intelligibility distillation works

### 4. Created Configuration Files
- `configs/logit_distillation_config.yaml` - training configuration
- `LOGIT_DISTILLATION_PLAN.md` - complete strategy documentation
- `test_logit_loading.py` - verification script

## 🔄 In Progress

### 3. Knowledge Distillation Trainer
**Current Task**: Building the student model and training loop

**Components Needed**:
1. **Student Model Architecture**
   ```python
   Input: Teacher logits [num_teachers, seq_len, vocab_size]
   Process: Learn ensemble weights or transform
   Output: Student logits [seq_len, vocab_size]
   ```

2. **Teacher Ensemble**
   - Option A: Simple average of N teacher logits
   - Option B: Learned attention weights over teachers
   - Option C: Transformer-based fusion

3. **Loss Function**
   ```python
   # KL divergence with temperature
   loss_kd = KL_div(
       softmax(student_logits / T),
       softmax(teacher_ensemble / T)
   ) * T^2
   ```

4. **Training Loop**
   - Load teacher logits for current fold
   - Compute ensemble target
   - Train student to match ensemble
   - Validate on held-out speaker

## 📋 Next Steps

### Immediate (Next):
1. **Create simple student model** - Start with basic MLP or small Transformer
2. **Implement ensemble method** - Begin with simple averaging
3. **Write training script** - One fold first (e.g., HIGH → M08)
4. **Test on single batch** - Verify loss computation works

### Then:
5. **Full training on Fold 1** (HIGH → M08)
6. **Evaluate on M08** - Decode logits, compute WER
7. **Extend to all 4 folds**
8. **Compare results** across intelligibility levels

## 🎯 Key Decisions Made

1. **Pure Logit Distillation**: No feature extraction, work directly with SA model outputs
2. **Leave-One-Out**: 4 folds, one per intelligibility level
3. **VERY_LOW Strategy**: Cross-level distillation (all 14 speakers → M01)
4. **Temperature Scaling**: Use temperature=3.0 for soft targets
5. **Decoder Logits**: Use decoder logits (not CTC) as primary signal

## 📊 Expected Timeline

- [x] Setup & verification - **DONE**
- [ ] Student model implementation - **1-2 hours**
- [ ] Single fold training - **2-4 hours**
- [ ] All 4 folds + evaluation - **4-8 hours**
- [ ] Analysis & reporting - **2-4 hours**

**Total**: ~1-2 days of development work

## 🚀 Ready to Proceed

All prerequisites complete. Ready to implement the student model and training loop!
