# Logit-Level Knowledge Distillation Plan

## 🎯 Objective
Train a student model using knowledge distillation from 14 SA (Speaker-Adaptive) teacher models, evaluated in a leave-one-out fashion across 4 intelligibility levels.

## 📊 Dataset Structure

### Logit Files Location
```
/home/zsim710/XDED/speechbrain/exp_results/logit_extraction/
├── F02/F02/F02_decoder_logits.pt  (1785 utterances)
├── F03/F03/F03_decoder_logits.pt  (1616 utterances)
├── F04/F04/F04_decoder_logits.pt  (1785 utterances)
├── F05/F05/F05_decoder_logits.pt  (1785 utterances)
├── M01/M01/M01_decoder_logits.pt  (765 utterances)
├── M04/M04/M04_decoder_logits.pt  (1275 utterances)
├── M05/M05/M05_decoder_logits.pt  (1785 utterances)
├── M07/M07/M07_decoder_logits.pt  (1785 utterances)
├── M08/M08/M08_decoder_logits.pt  (1785 utterances)
├── M09/M09/M09_decoder_logits.pt  (1785 utterances)
├── M10/M10/M10_decoder_logits.pt  (1785 utterances)
├── M11/M11/M11_decoder_logits.pt  (1530 utterances)
├── M12/M12/M12_decoder_logits.pt  (1530 utterances)
├── M14/M14/M14_decoder_logits.pt  (1785 utterances)
└── M16/M16/M16_decoder_logits.pt  (1530 utterances)
```

### Logit Format
- **Shape**: `[seq_len, vocab_size]` where vocab_size = 5000
- **Type**: Decoder logits from transformer-based SA models
- **Metadata**: Each speaker has corresponding `*_metadata.json` with utterance IDs and targets

## 🎭 Intelligibility Levels

### HIGH Intelligibility (7 speakers)
- **Speakers**: F02, F03, M04, M07, M08, M12, M14
- **Held-out**: M08 (1,785 test utterances)
- **Teachers**: F02, F03, M04, M07, M12, M14 (6 speakers, 9,776 train utterances)

### MID Intelligibility (4 speakers)
- **Speakers**: F04, M05, M10, M11
- **Held-out**: M05 (1,785 test utterances)
- **Teachers**: F04, M10, M11 (3 speakers, 5,100 train utterances)

### LOW Intelligibility (3 speakers)
- **Speakers**: F05, M09, M16
- **Held-out**: M16 (1,530 test utterances)
- **Teachers**: F05, M09 (2 speakers, 3,570 train utterances)

### VERY_LOW Intelligibility (1 speaker) ⚠️
- **Speakers**: M01
- **Held-out**: M01 (765 test utterances)
- **Teachers**: **NONE** (0 speakers, 0 train utterances)
- **Problem**: Cannot do leave-one-out with only 1 speaker!

## 🔧 Solution for VERY_LOW

### Option 1: Cross-Intelligibility Ensemble (RECOMMENDED)
Use all 14 other speakers as teachers for M01:
- Teachers: F02, F03, F04, F05, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16
- Training samples: 24,091 utterances
- Rationale: Tests generalization across all intelligibility levels

### Option 2: Weighted Cross-Level Ensemble
Use all 14 speakers but weight LOW intelligibility speakers higher:
- F05, M09, M16 (LOW): weight = 2.0
- F04, M05, M10, M11 (MID): weight = 1.5  
- Others (HIGH): weight = 1.0

### Option 3: Skip VERY_LOW Fold
Only evaluate on 3 folds (HIGH, MID, LOW) and report results

**DECISION**: Use Option 1 for completeness

## 📝 Revised 4-Fold Strategy

### Fold 1: HIGH → M08
```
Teachers: F02, F03, M04, M07, M12, M14 (6 speakers from HIGH only)
Test: M08 (1,785 utterances)
Train: 9,776 utterances
```

### Fold 2: MID → M05
```
Teachers: F04, M10, M11 (3 speakers from MID only)
Test: M05 (1,785 utterances)
Train: 5,100 utterances
```

### Fold 3: LOW → M16
```
Teachers: F05, M09 (2 speakers from LOW only)
Test: M16 (1,530 utterances)
Train: 3,570 utterances
```

### Fold 4: VERY_LOW → M01 (Cross-Level)
```
Teachers: ALL 14 other speakers (cross-intelligibility)
Test: M01 (765 utterances)
Train: 24,091 utterances
```

## 🧠 Knowledge Distillation Architecture

### Teacher Models (SA Models)
- 14 pre-trained speaker-adaptive models
- Each produces logits: `[seq_len, 5000]`
- Already extracted and saved

### Student Model
```python
class StudentModel(nn.Module):
    # Option A: Simple logit transformer
    input: [batch, seq_len, vocab_size]
    → Transformer layers
    → output: [batch, seq_len, vocab_size]
    
    # Option B: Even simpler - direct ensemble learning
    input: [batch, num_teachers, seq_len, vocab_size]
    → Attention weights over teachers
    → output: [batch, seq_len, vocab_size]
```

### Loss Function
```python
# Soft target loss (KL divergence with temperature)
teacher_ensemble = average([teacher1_logits, ..., teacher14_logits])
teacher_soft = softmax(teacher_ensemble / temperature)
student_soft = softmax(student_logits / temperature)
loss_kd = KL_divergence(student_soft, teacher_soft) * (temperature^2)

# Optional: Hard target loss if ground truth available
loss_ce = CrossEntropy(student_logits, ground_truth)

# Combined loss
loss_total = alpha * loss_kd + (1 - alpha) * loss_ce
```

## 📊 Evaluation Metrics

For each fold:
1. **WER (Word Error Rate)**: Decode student logits and compare to ground truth
2. **CER (Character Error Rate)**: Character-level accuracy
3. **Logit MSE**: Mean squared error between student and teacher ensemble logits
4. **Logit Correlation**: Pearson correlation between distributions

## 🚀 Implementation Steps

### Step 1: Dataset Loader ✅
- [x] Load pre-extracted logits from all 15 speakers
- [x] Implement 4-fold leave-one-out split
- [x] Handle variable sequence lengths with padding

### Step 2: Student Model Architecture
- [ ] Design simple student model (MLP or small Transformer)
- [ ] Implement teacher ensemble (averaging or attention-based)
- [ ] Add temperature-scaled softmax

### Step 3: Training Loop
- [ ] KL divergence loss between student and teacher ensemble
- [ ] Optional: Cross-entropy with ground truth labels
- [ ] Training for each of 4 folds

### Step 4: Evaluation
- [ ] Decode student logits using greedy/beam search
- [ ] Calculate WER/CER against ground truth
- [ ] Compare to individual teacher performance

### Step 5: Analysis
- [ ] Performance by intelligibility level
- [ ] Student vs teacher ensemble comparison
- [ ] Visualization of learned attention weights (if using attention)

## 🎯 Expected Outcomes

**Goal**: Student model learns to ensemble 14 teacher logits and generalizes to held-out speaker

**Success Metrics**:
- Student WER < Average individual teacher WER
- Student WER approaches ensemble average WER
- Consistent performance across intelligibility levels

**Insights**:
- Does cross-level distillation (VERY_LOW fold) work?
- Which teachers contribute most to ensemble?
- Can we reduce from 14 teachers to fewer without losing performance?
