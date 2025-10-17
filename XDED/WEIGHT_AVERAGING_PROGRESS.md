# 15-Fold Weight Averaging Experiment - Progress Update

## ✅ COMPLETED TASKS

### 1. Model Weight Averaging (100% Complete)
- ✅ Located all 15 SA model checkpoints at `/mnt/Research/qwan121/ICASSP_SA/`
- ✅ Created mapping JSON: `/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json`
- ✅ Successfully averaged weights for all 15 folds
- ✅ Created 15 averaged model weight files (~53MB each)
  - Location: `/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models/`

### 2. Inference Checkpoint Preparation (100% Complete)
- ✅ Created full SpeechBrain checkpoint directories for all 15 averaged models
- ✅ Each checkpoint directory contains:
  - `model.ckpt` - Averaged model weights
  - `hyperparams.yaml` - Model configuration
  - `tokenizer.ckpt` + `tokenizer` - Tokenizer files
  - `normalizer.ckpt` - Feature normalization stats
  - `brain.ckpt`, `noam_scheduler.ckpt`, `counter.ckpt`, `CKPT.yaml` - Supporting files
- ✅ Location: `/home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints/`

### 3. Inference Scripts (100% Complete)
- ✅ Created `tools/test_sa_model.py` - Inference script for testing models
- ✅ Created `tools/run_15_fold_evaluation.sh` - Orchestrates all 15 fold evaluations
- ✅ Scripts use SpeechBrain's `EncoderDecoderASR` interface

## 🔄 NEXT STEPS

### Step 1: Test One Averaged Model (Quick Validation)

Run a single test to verify the inference pipeline works:

```bash
cd /home/zsim710/XDED
conda activate FirstDeep

python3 /home/zsim710/XDED/XDED/tools/test_sa_model.py \
  --checkpoint_dir /home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints/F02_held_out \
  --test_csv /home/zsim710/partitions/uaspeech/by_speakers/F02.csv \
  --speaker_id F02 \
  --output_file /home/zsim710/XDED/XDED/results/speaker_averaging/evaluation/test_F02_averaged.json \
  --device cuda
```

**Expected output:**
- WER percentage for F02 averaged model on F02 test data
- JSON file with detailed results

### Step 2: Run All 15 Averaged Model Tests

Once the single test works, run all 15 folds:

```bash
# Test just the averaged models (no individual models yet)
for speaker in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
    echo "Testing averaged model for $speaker..."
    python3 /home/zsim710/XDED/XDED/tools/test_sa_model.py \
        --checkpoint_dir /home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints/${speaker}_held_out \
        --test_csv /home/zsim710/partitions/uaspeech/by_speakers/${speaker}.csv \
        --speaker_id $speaker \
        --output_file /home/zsim710/XDED/XDED/results/speaker_averaging/evaluation/fold_${speaker}_averaged.json \
        --device cuda
done
```

### Step 3: Set Up Individual SA Model Checkpoints (For WRA Comparison)

To compute WRA, we need to test all 14 individual models on each held-out speaker. This requires setting up full checkpoint directories for the individual SA models (similar to what we did for averaged models).

**Option A: Copy setup for all individual models**
```bash
# For each of the 15 speakers, create a full checkpoint directory
# This would create 15 more checkpoint directories
```

**Option B: Use existing SA models directly** (if they already have proper structure)
```bash
# Check if individual SA model checkpoints already have the necessary files
ls -la /mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/
```

### Step 4: Test Individual Models for WRA

Once individual SA models are set up, test each one on all held-out speakers:

```bash
# For each held-out speaker
for held_out in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
    # Test each other speaker's model on this held-out speaker
    for other_speaker in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
        if [ "$other_speaker" != "$held_out" ]; then
            # Test other_speaker's model on held_out's data
            python3 test_sa_model.py \
                --checkpoint_dir /path/to/${other_speaker}_checkpoint \
                --test_csv /home/zsim710/partitions/uaspeech/by_speakers/${held_out}.csv \
                --speaker_id ${held_out} \
                --output_file results/${held_out}_from_${other_speaker}.json
        fi
    done
done
```

### Step 5: Compute WRA and Compare

Once we have results from both averaged models and individual models, we can compute:

1. **Averaged Model Performance**: WER for each averaged model on its held-out speaker
2. **WRA (Equal Weighting)**: Average accuracy across all 14 individual models
3. **WRA (Inverse WER Weighting)**: Weighted average favoring better models
4. **Comparison**: Determine if weight averaging or WRA generalizes better

## 📊 Expected Results Structure

```
/home/zsim710/XDED/XDED/results/speaker_averaging/
├── averaged_models/          # 15 weight-averaged model files ✅
├── inference_checkpoints/    # 15 full checkpoint directories ✅
├── evaluation/              # Evaluation results (in progress)
│   ├── fold_F02_averaged.json
│   ├── fold_F02_individual_F03.json
│   ├── fold_F02_individual_F04.json
│   └── ... (225 files total: 15 averaged + 15*14 individual)
└── aggregated_results.json  # Final summary comparing weight avg vs WRA
```

## 🎯 Research Questions to Answer

1. **Does weight averaging work for cross-speaker generalization?**
   - Compare averaged model WER to best individual model WER
   
2. **Weight averaging vs WRA: Which is better?**
   - Compare averaged model WER to WRA (ensemble of individual models)
   
3. **Is it practical?**
   - Weight averaging = 1 model at inference (fast)
   - WRA = 14 models at inference (slow)
   - If weight averaging is competitive, it's much more practical

## 📝 Tools Created

| Tool | Purpose | Status |
|------|---------|--------|
| `find_sa_checkpoints.py` | Find all SA model checkpoints | ✅ Complete |
| `average_sa_models.py` | Average model weights | ✅ Complete |
| `run_15_fold_averaging.py` | Run all 15 folds of averaging | ✅ Complete |
| `setup_averaged_checkpoints.sh` | Create full checkpoint dirs | ✅ Complete |
| `test_sa_model.py` | Inference script for testing | ✅ Complete |
| `run_15_fold_evaluation.sh` | Run all evaluations | ✅ Complete |
| `aggregate_results.py` | Compute WRA and final comparison | ⏳ TODO |

## 💡 Key Insight

This weight averaging approach is a **zero-cost baseline**:
- No training required (just tensor averaging)
- No knowledge distillation
- Single model at inference time (unlike WRA which needs 14 models)
- Takes ~1 second per fold to create averaged model

If it works, it's the simplest possible approach to cross-speaker generalization!

## ⚠️ Current Blocker

Need to test if the inference script works with the averaged model checkpoints. Once verified, we can proceed with the full 15-fold evaluation.

## 📋 Immediate Next Action

**Run this command** (with conda environment activated):
```bash
cd /home/zsim710/XDED
conda activate FirstDeep

python3 /home/zsim710/XDED/XDED/tools/test_sa_model.py \
  --checkpoint_dir /home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints/F02_held_out \
  --test_csv /home/zsim710/partitions/uaspeech/by_speakers/F02.csv \
  --speaker_id F02 \
  --output_file /home/zsim710/XDED/XDED/results/speaker_averaging/evaluation/test_F02_averaged.json \
  --device cuda
```

If this works, we can proceed with testing all 15 averaged models!
