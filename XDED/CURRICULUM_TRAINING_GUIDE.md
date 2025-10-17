# Curriculum Learning Training Guide

## 🎓 Overview

This guide walks through training the NeMoHybridStudent model with curriculum learning, which progressively trains on easy-to-hard samples for better generalization to unseen speakers.

## 📋 Prerequisites

✅ **Completed**:
1. ✅ Modified `LogitEnsembleDataset` with curriculum support
2. ✅ Created `precompute_difficulty.py` with known speaker WERs
3. ✅ Integrated curriculum scheduler into `train_student.py`

⏳ **To Do**:
1. Run `precompute_difficulty.py` to generate difficulty scores
2. Train model with curriculum learning
3. Evaluate and compare with baseline

---

## 🚀 Step-by-Step Workflow

### Step 1: Generate Curriculum Difficulty Scores

Run the preprocessing script to compute per-utterance difficulty scores:

```bash
cd /home/zsim710/XDED/XDED
python precompute_difficulty.py
```

**Expected Output**:
- File: `curriculum_difficulty_scores.json`
- Contains: Per-utterance difficulty scores (0=easiest, 1=hardest)
- Based on: 60% speaker WER + 40% utterance entropy

**Verify Output**:
```bash
# Check if file was created
ls -lh curriculum_difficulty_scores.json

# Preview first few scores (optional)
python -c "import json; data=json.load(open('curriculum_difficulty_scores.json')); print('Total utterances:', len(data.get('utterance_scores', {}))); print('Example scores:', list(data.get('utterance_scores', {}).items())[:5])"
```

---

### Step 2: Train with Curriculum Learning (Square Root Schedule)

Train using the **sqrt schedule** (recommended - balances fast early learning with gradual difficulty increase):

```bash
CUDA_VISIBLE_DEVICES=1 python train_student.py \
    --student_backbone nemo \
    --nemo_model_name nvidia/stt_en_conformer_ctc_small \
    --teacher_logits_type decoder \
    --held_out M08 \
    --epochs 40 \
    --lr 0.0002 \
    --batch_size 8 \
    --temperature 2.0 \
    --teacher_agg logprob_mean \
    --curriculum_schedule sqrt \
    --curriculum_scores curriculum_difficulty_scores.json \
    --save_dir checkpoints_nemo_hybrid_curriculum_sqrt/student_M08/
```

**Key Arguments**:
- `--curriculum_schedule sqrt`: Square root competence growth (fast → slow)
- `--curriculum_scores curriculum_difficulty_scores.json`: Difficulty scores file
- `--save_dir checkpoints_nemo_hybrid_curriculum_sqrt/student_M08/`: Output directory

**Expected Behavior**:
- Epoch 1: ~30% of data (easiest utterances)
- Epoch 10: ~50% of data
- Epoch 20: ~71% of data
- Epoch 40: 100% of data (all utterances)

---

### Step 3: Train with Linear Schedule (Alternative)

For comparison, try **linear schedule** (constant growth rate):

```bash
CUDA_VISIBLE_DEVICES=1 python train_student.py \
    --student_backbone nemo \
    --nemo_model_name nvidia/stt_en_conformer_ctc_small \
    --teacher_logits_type decoder \
    --held_out M08 \
    --epochs 40 \
    --lr 0.0002 \
    --batch_size 8 \
    --temperature 2.0 \
    --teacher_agg logprob_mean \
    --curriculum_schedule linear \
    --curriculum_scores curriculum_difficulty_scores.json \
    --save_dir checkpoints_nemo_hybrid_curriculum_linear/student_M08/
```

**Expected Behavior**:
- Epoch 1: 2.5% of data
- Epoch 10: 25% of data
- Epoch 20: 50% of data
- Epoch 40: 100% of data

---

### Step 4: Train with Step Schedule (Alternative)

For **step-wise increase** (discrete jumps at milestones):

```bash
CUDA_VISIBLE_DEVICES=1 python train_student.py \
    --student_backbone nemo \
    --nemo_model_name nvidia/stt_en_conformer_ctc_small \
    --teacher_logits_type decoder \
    --held_out M08 \
    --epochs 40 \
    --lr 0.0002 \
    --batch_size 8 \
    --temperature 2.0 \
    --teacher_agg logprob_mean \
    --curriculum_schedule step \
    --curriculum_scores curriculum_difficulty_scores.json \
    --save_dir checkpoints_nemo_hybrid_curriculum_step/student_M08/
```

**Expected Behavior**:
- Epochs 1-10: 30% of data
- Epochs 11-20: 60% of data
- Epochs 21-30: 80% of data
- Epochs 31-40: 100% of data

---

### Step 5: Baseline Training (No Curriculum)

For comparison, train without curriculum:

```bash
CUDA_VISIBLE_DEVICES=1 python train_student.py \
    --student_backbone nemo \
    --nemo_model_name nvidia/stt_en_conformer_ctc_small \
    --teacher_logits_type decoder \
    --held_out M08 \
    --epochs 40 \
    --lr 0.0002 \
    --batch_size 8 \
    --temperature 2.0 \
    --teacher_agg logprob_mean \
    --curriculum_schedule none \
    --save_dir checkpoints_nemo_hybrid_baseline/student_M08/
```

**Note**: This uses all data from epoch 1 (no progressive difficulty).

---

### Step 6: Evaluate All Models

Evaluate each trained model on held-out M08 test set:

```bash
# Evaluate sqrt schedule
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_curriculum_sqrt/student_M08/best.pt \
    --held_out_speaker M08 \
    --output_file results_curriculum_sqrt_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer

# Evaluate linear schedule
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_curriculum_linear/student_M08/best.pt \
    --held_out_speaker M08 \
    --output_file results_curriculum_linear_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer

# Evaluate step schedule
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_curriculum_step/student_M08/best.pt \
    --held_out_speaker M08 \
    --output_file results_curriculum_step_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer

# Evaluate baseline (no curriculum)
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_baseline/student_M08/best.pt \
    --held_out_speaker M08 \
    --output_file results_baseline_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer
```

---

## 📊 Curriculum Schedules Explained

### Square Root Schedule (Recommended)
```
competence = sqrt(epoch / total_epochs)
```
- **Characteristics**: Fast initial growth, then slows down
- **Best for**: Quick adaptation to easy samples while gradually introducing harder ones
- **Example (40 epochs)**:
  - Epoch 1: 16% data
  - Epoch 4: 32% data
  - Epoch 9: 47% data
  - Epoch 16: 63% data
  - Epoch 25: 79% data
  - Epoch 40: 100% data

### Linear Schedule
```
competence = epoch / total_epochs
```
- **Characteristics**: Constant growth rate
- **Best for**: Uniform progression
- **Example (40 epochs)**:
  - Epoch 1: 2.5% data
  - Epoch 10: 25% data
  - Epoch 20: 50% data
  - Epoch 30: 75% data
  - Epoch 40: 100% data

### Step Schedule
```
if epoch < 25%: competence = 0.3
elif epoch < 50%: competence = 0.6
elif epoch < 75%: competence = 0.8
else: competence = 1.0
```
- **Characteristics**: Discrete jumps at milestones
- **Best for**: Stable training phases with sudden difficulty increases
- **Example (40 epochs)**:
  - Epochs 1-10: 30% data
  - Epochs 11-20: 60% data
  - Epochs 21-30: 80% data
  - Epochs 31-40: 100% data

---

## 📈 Expected Improvements

Based on curriculum learning research and UASpeech characteristics:

### 1. Better Generalization
- **Baseline WER on M08**: ~40-60% (from previous training)
- **Curriculum WER on M08**: Expected 5-15% relative improvement
- **Why**: Progressive learning prevents overfitting to hard speakers

### 2. Faster Convergence
- **Baseline**: Validation loss plateaus after ~20 epochs
- **Curriculum**: May reach similar loss in ~15 epochs
- **Why**: Early focus on learnable patterns

### 3. More Stable Training
- **Baseline**: May show loss spikes when encountering hard batches
- **Curriculum**: Smoother loss curves due to gradual difficulty increase

### 4. Better Feature Learning
- **Hypothesis**: Model learns more robust acoustic-phonetic mappings
- **Evidence**: Should see better performance on LOW/VERY_LOW speakers

---

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **Curriculum Competence** (printed each epoch):
   ```
   📚 Curriculum competence: 0.500 (765/1530 samples)
   ```
   - Shows what % of data is being used
   - Should increase over training

2. **Training Loss**:
   - Should decrease smoothly
   - Curriculum may show faster early decrease

3. **Validation Loss**:
   - Should decrease and stabilize
   - Curriculum may achieve lower final loss

4. **Valid-frame Ratio (VFR)**:
   - % of frames used in loss computation
   - Should remain stable (>80%)

### Example Training Output

```
================================================================================
Epoch 5/40
================================================================================
📚 Curriculum competence: 0.354 (542/1530 samples)
Learning rate (start of epoch): 2.000000e-04
Training: 100%|████████| 68/68 [01:23<00:00,  1.23s/it]
📊 Train Loss: 1.2345 | Valid-frame ratio: 87.3%
Learning rate (end of epoch): 2.000000e-04
Validation: 100%|████████| 13/13 [00:05<00:00,  2.45it/s]
📊 Val Loss: 1.1234 | Valid-frame ratio: 85.1%
✅ Saved best model (val_loss: 1.1234)
```

---

## 🧪 Ablation Studies (Optional)

To understand curriculum impact, try these experiments:

### 1. Speaker-Only Difficulty
Modify `precompute_difficulty.py`:
```python
combined_score = 1.0 * speaker_difficulty + 0.0 * utterance_difficulty  # Only speaker WER
```

### 2. Utterance-Only Difficulty
Modify `precompute_difficulty.py`:
```python
combined_score = 0.0 * speaker_difficulty + 1.0 * utterance_difficulty  # Only entropy
```

### 3. Different Weight Combinations
```python
# More weight on speaker difficulty
combined_score = 0.8 * speaker_difficulty + 0.2 * utterance_difficulty

# More weight on utterance difficulty
combined_score = 0.4 * speaker_difficulty + 0.6 * utterance_difficulty
```

---

## 📂 Output Files

After training and evaluation, you'll have:

```
XDED/
├── curriculum_difficulty_scores.json           # Difficulty scores
├── checkpoints_nemo_hybrid_curriculum_sqrt/    # Sqrt schedule model
│   └── student_M08/
│       ├── best.pt                             # Best checkpoint
│       ├── latest.pt                           # Latest checkpoint
│       ├── config.json                         # Training config
│       └── checkpoint_epoch_*.pt               # Periodic checkpoints
├── checkpoints_nemo_hybrid_curriculum_linear/  # Linear schedule model
├── checkpoints_nemo_hybrid_curriculum_step/    # Step schedule model
├── checkpoints_nemo_hybrid_baseline/           # Baseline (no curriculum)
├── results_curriculum_sqrt_M08.json            # Sqrt results
├── results_curriculum_linear_M08.json          # Linear results
├── results_curriculum_step_M08.json            # Step results
└── results_baseline_M08.json                   # Baseline results
```

---

## 🎯 Next Steps

1. **Run Step 1**: Generate difficulty scores
   ```bash
   python precompute_difficulty.py
   ```

2. **Run Step 2**: Train with sqrt schedule (recommended)
   ```bash
   # Use command from Step 2 above
   ```

3. **Monitor Training**: Watch for smooth loss decrease and competence growth

4. **Evaluate**: Compare curriculum vs baseline WER on M08

5. **Analyze**: Look for improvements in generalization

---

## ⚠️ Troubleshooting

### Issue: "FileNotFoundError: Curriculum scores file not found"
**Solution**: Run `precompute_difficulty.py` first to generate the scores file.

### Issue: "Curriculum competence: 1.000 (all samples)"
**Cause**: Using `--curriculum_schedule none` or missing `--curriculum_scores`
**Solution**: Specify a schedule (`linear`/`sqrt`/`step`) and provide scores file.

### Issue: Training loss not decreasing
**Check**:
1. Verify curriculum competence is increasing each epoch
2. Check learning rate is not too low
3. Try starting with easier schedule (sqrt instead of linear)

### Issue: Very slow training with curriculum
**Cause**: Recreating DataLoader each epoch adds overhead
**Impact**: Minor (~5% slower) - worth it for curriculum benefits

---

## 📚 References

1. **Curriculum Learning**: Bengio et al. (2009)
2. **Speech Recognition Curriculum**: Xu et al. (2020) - Curriculum Learning for Speech Recognition
3. **UASpeech Dataset**: Kim et al. (2008) - Dysarthric speech database for speaker identification
4. **Knowledge Distillation**: Hinton et al. (2015) - Distilling the Knowledge in a Neural Network

---

## ✅ Summary Checklist

- [ ] Generated `curriculum_difficulty_scores.json`
- [ ] Trained model with sqrt schedule
- [ ] Trained baseline model (no curriculum)
- [ ] Evaluated both models on M08 test set
- [ ] Compared WER/WRA metrics
- [ ] Analyzed loss curves and training dynamics
- [ ] Documented findings

**Good luck with your curriculum learning experiments! 🚀**
