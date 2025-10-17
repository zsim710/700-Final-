# ✅ Curriculum Learning: Ready to Train!

## 🎉 Setup Complete

All components for curriculum learning have been successfully implemented and tested:

### ✅ Completed Steps:

1. **Created `precompute_difficulty.py`**
   - Hardcoded speaker WERs for 14 speakers (F05: 2.05% to M04: 86.4%)
   - Computed entropy from decoder logits for 1785 utterances
   - Combined scores: 60% speaker WER + 40% utterance entropy
   
2. **Modified `LogitEnsembleDataset`**
   - Added `curriculum_scores_file` parameter
   - Implemented curriculum loading and ordering methods
   - Added `get_curriculum_subset_indices(competence)` method
   
3. **Integrated into `train_student.py`**
   - Added curriculum arguments: `--curriculum_schedule`, `--curriculum_scores`
   - Implemented `get_curriculum_competence()` scheduler function
   - Modified training loop to use `SubsetRandomSampler` for curriculum sampling
   
4. **Generated Difficulty Scores**
   - File: `/home/zsim710/XDED/XDED/curriculum_difficulty_scores.json`
   - Size: 490KB
   - Contains: 1785 utterances with difficulty scores (0.105 to 1.000)
   
5. **Validated Integration**
   - All tests passed (schedule functions, dataset integration, competence validity)

---

## 📊 Difficulty Score Statistics

### Overall Distribution:
- **Total utterances**: 1,785
- **Min difficulty**: 0.105 (easiest)
- **Max difficulty**: 1.000 (hardest)
- **Mean difficulty**: 0.656
- **Median difficulty**: 0.708

### By Speaker (from generated data):
| Speaker | WER (%) | Utterances | Avg Difficulty | Band |
|---------|---------|------------|----------------|------|
| F02 | 16.85 | 25 | 0.137 | HIGH |
| F03 | 44.15 | 461 | 0.408 | HIGH |
| M12 | 47.00 | 24 | 0.430 | MID |
| M04 | 86.40 | 1,275 | 0.760 | LOW |

### Example Utterances:
**Easiest 5:**
- `B3_C2_M2`: 0.105 (F02, WER: 16.9%, Entropy: 0.00)
- `B3_C8_M2`: 0.106 (F02, WER: 16.9%, Entropy: 0.00)
- `B3_C1_M2`: 0.107 (F02, WER: 16.9%, Entropy: 0.01)

**Hardest 5:**
- `B3_UW96_M6`: 0.966 (M04, WER: 86.4%, Entropy: 1.66)
- `B3_CW23_M3`: 0.968 (M04, WER: 86.4%, Entropy: 1.66)
- `B3_UW96_M3`: 1.000 (M04, WER: 86.4%, Entropy: 1.81)

---

## 🚀 How to Train with Curriculum Learning

### Option 1: Square Root Schedule (Recommended)

**Best for**: Balanced training with fast initial learning + gradual difficulty increase

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

**Data progression (40 epochs)**:
- Epoch 1: 0% data (starts from epoch 0, so ~16% after first update)
- Epoch 5: 35% data
- Epoch 10: 50% data
- Epoch 20: 71% data
- Epoch 40: 100% data

---

### Option 2: Linear Schedule

**Best for**: Uniform constant growth rate

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

**Data progression (40 epochs)**:
- Epoch 1: 2.5% data
- Epoch 10: 25% data
- Epoch 20: 50% data
- Epoch 30: 75% data
- Epoch 40: 100% data

---

### Option 3: Step Schedule

**Best for**: Discrete training phases with stable difficulty

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

**Data progression (40 epochs)**:
- Epochs 1-10: 30% data (easy phase)
- Epochs 11-20: 60% data (medium phase)
- Epochs 21-30: 80% data (hard phase)
- Epochs 31-40: 100% data (all data)

---

### Baseline (No Curriculum)

**For comparison** - trains with all data from start:

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

---

## 📈 Expected Training Output

When training with curriculum, you'll see this for each epoch:

```
================================================================================
Epoch 5/40
================================================================================
📚 Curriculum competence: 0.354 (632/1785 samples)
Learning rate (start of epoch): 2.000000e-04
Training: 100%|████████| 79/79 [01:45<00:00,  1.33s/it]
📊 Train Loss: 1.2345 | Valid-frame ratio: 87.3%
Learning rate (end of epoch): 2.000000e-04
Validation: 100%|████████| 13/13 [00:05<00:00,  2.45it/s]
📊 Val Loss: 1.1234 | Valid-frame ratio: 85.1%
✅ Saved best model (val_loss: 1.1234)
```

Key indicators:
- **Curriculum competence**: Shows % of data being used (increases each epoch)
- **Sample count**: Shows current samples / total samples
- **Train/Val loss**: Should decrease smoothly
- **Valid-frame ratio**: Should stay >80%

---

## 🎯 Evaluation

After training completes, evaluate on held-out M08 speaker:

```bash
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_curriculum_sqrt/student_M08/best.pt \
    --held_out_speaker M08 \
    --output_file results_curriculum_sqrt_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer
```

Compare results:
```bash
# Curriculum (sqrt)
cat results_curriculum_sqrt_M08.json | grep -E '"wer"|"wra"'

# Baseline (no curriculum) - previous training
cat results_nemo_M08.json | grep -E '"wer"|"wra"'
```

---

## 📊 Expected Improvements

Based on curriculum learning research:

### 1. Better Generalization
- **Metric**: Lower WER on held-out M08 speaker
- **Expected**: 5-15% relative improvement over baseline
- **Reason**: Progressive learning prevents overfitting to hard speakers

### 2. Faster Convergence  
- **Metric**: Lower validation loss in fewer epochs
- **Expected**: Reach similar performance 20-30% faster
- **Reason**: Early focus on learnable patterns accelerates learning

### 3. Smoother Training
- **Metric**: Lower variance in loss curves
- **Expected**: More stable training without loss spikes
- **Reason**: Gradual difficulty increase avoids sudden hard samples

### 4. Better Features
- **Metric**: Improved performance on LOW/VERY_LOW intelligibility speakers
- **Expected**: Better acoustic-phonetic representations
- **Reason**: Structured learning builds more robust features

---

## 🔍 Monitoring Tips

### Key Metrics to Track:

1. **Curriculum Competence**:
   - Should increase from 0 to 1.0 over training
   - Check that sample count grows appropriately
   
2. **Training Loss**:
   - Should decrease smoothly (no major spikes)
   - Curriculum may show faster early decrease
   
3. **Validation Loss**:
   - Best indicator of generalization
   - Curriculum should achieve lower final val loss
   
4. **Learning Rate**:
   - Should follow Noam schedule (warmup then decay)
   - Peak at ~5000 steps, then gradually decrease

### Warning Signs:

- ⚠️ **Competence stuck at 1.0**: Check if curriculum_scores file is loaded
- ⚠️ **Loss not decreasing**: May need to adjust learning rate or temperature
- ⚠️ **Val loss diverging**: Possible overfitting, try stronger curriculum (linear instead of sqrt)

---

## 📂 Output Files

After training, you'll have:

```
XDED/
├── curriculum_difficulty_scores.json          # ✅ Generated
├── checkpoints_nemo_hybrid_curriculum_sqrt/   # Training output
│   └── student_M08/
│       ├── best.pt                            # Best model checkpoint
│       ├── latest.pt                          # Latest checkpoint
│       ├── config.json                        # Training config
│       └── checkpoint_epoch_*.pt              # Periodic checkpoints
└── results_curriculum_sqrt_M08.json           # Evaluation results
```

---

## 🧪 Next Steps

1. **Start Training** (recommended: sqrt schedule):
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

2. **Monitor Training**:
   - Watch curriculum competence increase
   - Check train/val loss decrease
   - Verify sample count grows

3. **Evaluate**:
   - Run eval_student.py on trained model
   - Compare WER/WRA with baseline

4. **Experiment** (optional):
   - Try different schedules (linear, step)
   - Adjust alpha/beta weights in difficulty scores
   - Train on other held-out speakers (M05, M16, etc.)

---

## 📚 Reference Documents

- **Implementation Details**: `CURRICULUM_IMPLEMENTATION.md`
- **Training Guide**: `CURRICULUM_TRAINING_GUIDE.md`
- **Test Results**: Output from `test_curriculum_integration.py` (all tests passed ✅)

---

## ✅ Summary

**Status**: 🟢 **READY TO TRAIN**

All components are implemented, tested, and validated. The curriculum learning system is fully functional and ready for training experiments.

**Recommended first experiment**: Train with sqrt schedule and compare with baseline (no curriculum).

**Good luck with your curriculum learning experiments! 🚀**
