# M08 Fold: Cross-Intelligibility Ensemble Strategy

## 🎯 Overview
Train a student model using knowledge distillation from **ALL 14 speaker-adaptive models** (excluding M08), then test on M08.

## 📊 Corrected Intelligibility Levels

Based on the provided table:

- **VERY_LOW**: M04, F03, M12, M01 (4 speakers)
- **LOW**: M07, F02, M16 (3 speakers)  
- **MID**: M05, M11, F04 (3 speakers)
- **HIGH**: M09, M14, M10, M08, F05 (5 speakers)

## 🔬 M08 Fold Configuration

### Teacher Models (14 speakers)
All speakers EXCEPT M08:
```
F02, F03, F04, F05, M01, M04, M05, M07, M09, M10, M11, M12, M14, M16
```

**Training Data**:
- Total utterances: 22,526
- From 14 different SA models
- Cross-intelligibility (all 4 levels represented)

### Test Set
- **Speaker**: M08 (HIGH intelligibility)
- **Utterances**: 1,785
- **Train/Test Ratio**: 12.62x

## 🧠 Ensemble Strategy

### Approach: Cross-Intelligibility Knowledge Distillation

Instead of training within one intelligibility level, we use **ALL available teachers** regardless of intelligibility level. This tests:
1. Can the student learn from diverse speaker characteristics?
2. Does cross-level knowledge transfer improve generalization?
3. How do different intelligibility levels contribute to the ensemble?

### Teacher Ensemble Composition

| Intelligibility | Teachers | Utterances | Percentage |
|----------------|----------|------------|-----------|
| VERY_LOW | F03, M01, M04, M12 | 5,186 | 23.0% |
| LOW | F02, M07, M16 | 5,100 | 22.6% |
| MID | F04, M05, M11 | 5,100 | 22.6% |
| HIGH | F05, M09, M10, M14 | 7,140 | 31.7% |
| **Total** | **14 teachers** | **22,526** | **100%** |

**Observation**: Roughly balanced across intelligibility levels (20-32% each)

## 🎓 Knowledge Distillation Architecture

### Phase 1: Simple Average Ensemble

```python
# For each utterance in training:
1. Load logits from all 14 teachers: [14, seq_len, vocab_size]
2. Compute average ensemble: mean(teacher_logits, dim=0) → [seq_len, vocab_size]
3. Apply temperature scaling: teacher_soft = softmax(ensemble / T)
4. Train student to match: KL_div(student_soft, teacher_soft)
```

### Student Model Options

**Option A: Identity/Pass-through (Baseline)**
- Just learn to reproduce the average ensemble
- No parameters to train
- Establishes performance ceiling

**Option B: Simple MLP Refinement**
```python
Input: Ensemble logits [seq_len, vocab_size]
→ Linear(vocab_size, hidden_dim)
→ ReLU + Dropout
→ Linear(hidden_dim, vocab_size)
Output: Refined logits [seq_len, vocab_size]
```

**Option C: Attention-Weighted Ensemble**
```python
Input: All teacher logits [14, seq_len, vocab_size]
→ Learn attention weights over 14 teachers
→ Weighted sum based on learned importance
Output: Smart ensemble [seq_len, vocab_size]
```

## 📊 Expected Outcomes

### Success Criteria

1. **Student WER ≤ Best Individual Teacher**
   - From table: M08's own DST baseline = 12% WER
   - From table: M08's conformer = 7.9% WER (common words)
   - Goal: Student should approach or beat these

2. **Student WER ≈ Oracle Ensemble WER**
   - Oracle = Best possible ensemble (voting/averaging)
   - Student should learn to approximate this

3. **Generalization to Held-Out Speaker**
   - M08 is HIGH intelligibility
   - Ensemble includes VERY_LOW, LOW, MID speakers
   - Tests cross-level knowledge transfer

### Evaluation Metrics

1. **WER (Word Error Rate)**: Primary metric
   - Decode student logits using greedy/beam search
   - Compare to ground truth transcriptions
   
2. **Logit-Level Metrics**:
   - MSE between student and ensemble logits
   - KL divergence (how well student matches ensemble distribution)
   - Correlation of probability distributions

3. **Per-Utterance Analysis**:
   - Which utterances benefit from ensemble?
   - Which teachers contribute most?
   - Error analysis by word type

## 🚀 Implementation Plan

### Step 1: Data Preparation ✅
- [x] Verify all 14 teacher logits load correctly
- [x] Confirm M08 test set accessible
- [x] Total: 22,526 train + 1,785 test utterances

### Step 2: Ensemble Creation
- [ ] Load all 14 teacher logits for each utterance
- [ ] Compute simple average ensemble
- [ ] Save ensemble targets to disk (optional, for faster loading)

### Step 3: Student Model
- [ ] Start with Option A (baseline: just use ensemble average)
- [ ] Implement Option B (simple MLP) if baseline works
- [ ] Implement Option C (attention) if needed

### Step 4: Training Loop
- [ ] KL divergence loss with temperature T=3.0
- [ ] Adam optimizer, lr=0.001
- [ ] Train for 50 epochs with early stopping
- [ ] Monitor validation loss on M08

### Step 5: Evaluation
- [ ] Greedy decoding of student logits
- [ ] Compute WER on M08 test set  
- [ ] Compare to individual teacher WER
- [ ] Compare to oracle ensemble WER

### Step 6: Analysis
- [ ] Visualize attention weights (if using Option C)
- [ ] Per-teacher contribution analysis
- [ ] Error analysis: which words fail?
- [ ] Intelligibility level impact

## 📝 Next Immediate Actions

1. **Create dataset loader that:**
   - Loads all 14 teacher logits for each utterance
   - Handles utterance matching across speakers
   - Returns ensemble average as target

2. **Implement baseline student:**
   - Just use ensemble average (no learnable params)
   - Decode and compute WER
   - This gives us the oracle performance

3. **If baseline works well:**
   - No need for complex student model
   - Ensemble average is sufficient
   - Move to next fold (M05, M16, M01)

4. **If baseline needs improvement:**
   - Add simple MLP (Option B)
   - Learn to refine ensemble predictions
   - Then try attention (Option C)

## ⚠️ Key Challenge

**Utterance Matching Problem:**
- Different speakers have different utterances
- F03 has 1,616 utterances, M01 has only 765
- Need to identify which utterances are SHARED across speakers
- Options:
  1. Use only utterances available in ALL 14 teachers (strict matching)
  2. Use utterance ID from metadata to match
  3. Use all utterances from each teacher separately (no matching)
  
**Recommendation**: Check metadata to see if utterance IDs indicate the same underlying text across speakers

---

**STATUS**: Ready to implement ensemble creation and baseline student model! 🚀
