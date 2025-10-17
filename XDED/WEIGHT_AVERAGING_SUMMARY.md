# 15-Fold Weight Averaging Experiment - Summary

## Completed Work ✅

### 1. Model Weight Averaging (DONE)

We have successfully created 15 averaged models using mathematical weight averaging:

**Location:** `/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models/`

**Files Created:**
```
F02_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16)
F03_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16)
F04_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16)
F05_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16)
M01_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M04, M05, M07, M08, M09, M10, M11, M12, M14, M16)
M04_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M05, M07, M08, M09, M10, M11, M12, M14, M16)
M05_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M07, M08, M09, M10, M11, M12, M14, M16)
M07_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M08, M09, M10, M11, M12, M14, M16)
M08_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M09, M10, M11, M12, M14, M16)
M09_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M10, M11, M12, M14, M16)
M10_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M11, M12, M14, M16)
M11_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M12, M14, M16)
M12_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M14, M16)
M14_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M16)
M16_held_out_averaged.pt  (53.1 MB) - Average of 14 models (F02, F03, F04, F05, M01, M04, M05, M07, M08, M09, M10, M11, M12, M14)
```

### 2. Averaging Method

**Simple Mathematical Average of Weights:**
```python
averaged_weight = (model_1.weight + model_2.weight + ... + model_14.weight) / 14
```

**Key Properties:**
- ✅ No training or fine-tuning required
- ✅ No knowledge distillation
- ✅ Just pure parameter averaging
- ✅ Computationally very cheap (one-time tensor averaging)
- ✅ Each averaged model has 477 parameter tensors

### 3. Tools Created

1. **`tools/find_sa_checkpoints.py`** - Automatically finds all SA model checkpoints
2. **`tools/average_sa_models.py`** - Averages model weights from multiple SA models
3. **`tools/run_15_fold_averaging.py`** - Orchestrates all 15 folds of averaging
4. **`tools/evaluate_15_fold_averaging.py`** - Framework for evaluation (needs inference integration)

### 4. Data Prepared

**Speaker Checkpoint Mapping:**
`/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json`

Contains paths to all 15 individual SA models:
- F02: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/save/CKPT+2024-07-11+17-18-17+00/model.ckpt`
- F03: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00/model.ckpt`
- ... (and 13 more)

**Test Data:**
`/home/zsim710/partitions/uaspeech/by_speakers/{speaker}.csv`
- Available for all 15 speakers
- Format: ID, duration, wav, spk_id, wrd

## What This Experiment Will Test 🎯

**Research Question:**
> For cross-speaker generalization in dysarthric speech recognition, is it better to:
> 1. **Weight Averaging**: Average the model parameters from multiple speaker models (cheap, one model)
> 2. **WRA Averaging**: Run all models separately and average their predictions (expensive, 14 models at inference)

**Hypothesis:**
If weight averaging works well, it would be a much simpler and more efficient alternative to:
- Knowledge distillation (requires training)
- Ensemble methods / WRA (requires running multiple models)

## Next Steps 🔄

To complete this experiment, you need to:

### Step 1: Set Up Inference Pipeline

The averaged models are just weights (`model.ckpt` format). To use them, you need to create full SpeechBrain checkpoint directories:

```bash
# For each averaged model, create a directory with:
# 1. model.ckpt (our averaged weights) ✅ Already have
# 2. hyperparams.yaml (from any SA model)
# 3. tokenizer.ckpt (from any SA model)
# 4. normalizer.ckpt (from any SA model)
```

**Script to automate this:**
```bash
#!/bin/bash
# Create full checkpoint directories for averaged models

BASE_SA="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00"
AVERAGED_MODELS="/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models"
OUTPUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints"

for speaker in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
    CKPT_DIR="$OUTPUT_DIR/${speaker}_held_out"
    mkdir -p "$CKPT_DIR"
    
    # Copy supporting files from a reference SA model
    cp "$BASE_SA/hyperparams.yaml" "$CKPT_DIR/"
    cp "$BASE_SA/tokenizer.ckpt" "$CKPT_DIR/"
    cp "$BASE_SA/normalizer.ckpt" "$CKPT_DIR/"
    cp "$BASE_SA/noam_scheduler.ckpt" "$CKPT_DIR/" 2>/dev/null || true
    cp "$BASE_SA/brain.ckpt" "$CKPT_DIR/" 2>/dev/null || true
    
    # Copy our averaged model weights
    cp "$AVERAGED_MODELS/${speaker}_held_out_averaged.pt" "$CKPT_DIR/model.ckpt"
    
    echo "Created checkpoint directory for ${speaker} held-out"
done
```

### Step 2: Run Inference

For each of the 15 folds:

**A. Test Averaged Model:**
```bash
# Test averaged model on held-out speaker
python test_model.py \
    --checkpoint_dir /path/to/F02_held_out \
    --test_csv /home/zsim710/partitions/uaspeech/by_speakers/F02.csv \
    --output results/F02_averaged_results.json
```

**B. Test Individual Models (for WRA):**
```bash
# Test each of the 14 individual models on F02
for other_speaker in F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
    python test_model.py \
        --checkpoint_dir /mnt/Research/qwan121/ICASSP_SA/val_uncommon_${other_speaker}_*/7775/save/CKPT+* \
        --test_csv /home/zsim710/partitions/uaspeech/by_speakers/F02.csv \
        --output results/F02_from_${other_speaker}.json
done
```

### Step 3: Compute Metrics

For each fold, compute:
1. **Averaged Model WER** - WER of the weight-averaged model on held-out speaker
2. **Individual Model WERs** - WER of each of the 14 individual models on held-out speaker
3. **WRA (equal weighting)** - Average accuracy of all individual models
4. **WRA (inverse WER weighting)** - Weighted average favoring better models

### Step 4: Compare

Across all 15 folds:
- Average WER of weight-averaged models
- Average WRA
- Determine which approach generalizes better

## Expected Outcomes 📊

**If Weight Averaging Wins:**
- Simpler baseline than knowledge distillation
- Much more efficient than ensemble/WRA (single model vs 14 models)
- Shows that speaker-specific features can be linearly combined

**If WRA Wins:**
- Ensemble methods are still better
- But weight averaging might still be competitive
- Consider it as a fast approximation

## Key Insight 💡

This weight averaging approach is:
- **Simpler than Knowledge Distillation:** No training required, just tensor averaging
- **Cheaper than WRA:** One averaged model vs 14 models at inference time
- **Zero-cost baseline:** Takes ~1 second per fold to create averaged model

It answers: "*Can we get cross-speaker generalization for free by just averaging parameters?*"

## Files and Directories

```
/home/zsim710/XDED/XDED/
├── results/speaker_averaging/
│   ├── speaker_checkpoints.json          # Mapping of speakers to SA model paths
│   ├── averaged_models/                  # 15 averaged model weights ✅
│   │   ├── F02_held_out_averaged.pt
│   │   ├── F03_held_out_averaged.pt
│   │   └── ... (13 more)
│   ├── inference_checkpoints/            # Full SpeechBrain checkpoints (TODO)
│   └── evaluation/                       # Evaluation results (TODO)
│       ├── fold_1_F02_results.json
│       ├── fold_2_F03_results.json
│       └── aggregated_15_fold_results.json
├── tools/
│   ├── find_sa_checkpoints.py           # Find all SA checkpoints ✅
│   ├── average_sa_models.py             # Average model weights ✅
│   ├── run_15_fold_averaging.py         # Run all 15 folds ✅
│   └── evaluate_15_fold_averaging.py    # Evaluation framework (needs inference)
└── WEIGHT_AVERAGING_NEXT_STEPS.md       # Detailed next steps
```

## Questions for Your Supervisor

1. **Do you have existing inference code** for the SA models that we can use?
2. **Should we proceed** with creating the full checkpoint directories?
3. **What format** do you want the final results in?
4. **Any specific metrics** beyond WER you want to track?

---

**Status:** Model averaging complete ✅ | Evaluation setup ready | Awaiting inference integration
