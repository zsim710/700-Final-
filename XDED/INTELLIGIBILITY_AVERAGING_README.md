# Intelligibility-Based Weight Averaging Baseline

## Overview

This baseline tests whether **simple weight averaging** of SA models within the same intelligibility band can improve generalization to unseen speakers.

## Experiment Design

### **Intelligibility Bands (2 bands, 4 models total):**

| Band | Speakers | Test Speaker | Models |
|------|----------|--------------|--------|
| **HIGH** | M09, M14, M10, M08, F05 | M08 | 2 models (excluded/included) |
| **VERY_LOW** | M12, M01, F03, M04 | M01 | 2 models (excluded/included) |

### **Scenarios:**

1. **EXCLUDED**: Test speaker NOT included in averaging (true generalization)
2. **INCLUDED**: Test speaker included in averaging (oracle/upper bound)

### **Total Models: 4**
- HIGH_excluded (4 speakers: M09, M14, M10, F05)
- HIGH_included (5 speakers: M09, M14, M10, M08, F05)
- VERY_LOW_excluded (3 speakers: M12, F03, M04)
- VERY_LOW_included (4 speakers: M12, M01, F03, M04)

---

## File Structure

```
XDED/tools/
├── average_sa_models_intelligibility.py    # Core averaging logic
├── run_intelligibility_averaging.py        # Orchestration script
├── evaluate_intelligibility_averaging.sh   # Evaluation runner
├── summarize_intelligibility_results.py    # Results summary
└── test_averaged_model_simple.py           # Updated inference script

XDED/results/
├── speaker_averaging/
│   └── speaker_checkpoints.json            # SA model paths
└── intelligibility_averaging/
    ├── excluded/                           # Models WITHOUT test speaker
    │   ├── HIGH_averaged_exclude_M08.pt
    │   └── VERY_LOW_averaged_exclude_M01.pt
    ├── included/                           # Models WITH test speaker
    │   ├── HIGH_averaged_include_M08.pt
    │   └── VERY_LOW_averaged_include_M01.pt
    ├── inference/                          # Full checkpoint dirs
    │   ├── excluded/
    │   │   ├── HIGH_M08/
    │   │   └── VERY_LOW_M01/
    │   └── included/
    │       ├── HIGH_M08/
    │       └── VERY_LOW_M01/
    └── evaluation/                         # Results JSON
        ├── excluded/
        │   ├── HIGH_M08_results.json
        │   └── VERY_LOW_M01_results.json
        └── included/
            ├── HIGH_M08_results.json
            └── VERY_LOW_M01_results.json
```

---

## Usage

### **Step 1: Create Averaged Models**

Run weight averaging for both scenarios (excluded and included):

```bash
cd /home/zsim710/XDED/XDED
python3 tools/run_intelligibility_averaging.py
```

This creates **4 averaged models total**:
- `HIGH_averaged_exclude_M08.pt` (4 speakers: M09, M14, M10, F05)
- `HIGH_averaged_include_M08.pt` (5 speakers: M09, M14, M10, M08, F05)
- `VERY_LOW_averaged_exclude_M01.pt` (3 speakers: M12, F03, M04)
- `VERY_LOW_averaged_include_M01.pt` (4 speakers: M12, M01, F03, M04)

**Optional**: Run only one scenario:
```bash
# Only excluded
python3 tools/run_intelligibility_averaging.py --exclude_held_out

# Only included
python3 tools/run_intelligibility_averaging.py --include_held_out
```

---

### **Step 2: Evaluate All Models**

```bash
chmod +x tools/evaluate_intelligibility_averaging.sh
./tools/evaluate_intelligibility_averaging.sh
```

This will:
1. Set up checkpoint directories with supporting files
2. Test each averaged model on its corresponding test speaker
3. Save results to JSON files

---

### **Step 3: View Results**

```bash
python3 tools/summarize_intelligibility_results.py
```

**Expected Output:**
```
================================================================================
INTELLIGIBILITY-BASED AVERAGING RESULTS SUMMARY
Bands: HIGH and VERY_LOW
================================================================================

EXCLUDED SCENARIO (Test speaker excluded from averaging):
--------------------------------------------------------------------------------
Band            Test Speaker    WER (%)      Accuracy (%)   
--------------------------------------------------------------------------------
HIGH            M08             12.34        87.66          
VERY_LOW        M01             45.67        54.33          

INCLUDED SCENARIO (Test speaker included from averaging):
--------------------------------------------------------------------------------
Band            Test Speaker    WER (%)      Accuracy (%)   
--------------------------------------------------------------------------------
HIGH            M08             8.91         91.09          
VERY_LOW        M01             38.45        61.55          

================================================================================
INTERPRETATION:
================================================================================
EXCLUDED: Test speaker NOT included in weight averaging (true generalization)
INCLUDED: Test speaker included in weight averaging (should perform better)

Expected: INCLUDED scenario should show lower WER than EXCLUDED
================================================================================
```

---

## Test Data

### **CSV Paths (Updated):**
```python
TEST_CSV_PATHS = {
    'M08': '/home/zsim710/partitions/uaspeech/by_speakers/M08.csv',
    'M01': '/home/zsim710/partitions/uaspeech/by_speakers/M01.csv',
}
```

### **Supporting Files (Corrected Paths):**
```python
HYPERPARAMS = '/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/hyperparams.yaml'
TOKENIZER = '/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt'
NORMALIZER = '/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/normalizer.ckpt'
```

### **CSV Format:**
```
ID,duration,wav,spk_id,wrd
M08_B3_C10_M5,6.202375,/mnt/DataSets/UASpeech/audio/M08/M08_B3_C10_M5.wav,M08,LINE
```

---

## Key Differences from Previous 15-Fold Approach

| Previous (15-fold) | Current (Intelligibility-based) |
|--------------------|----------------------------------|
| 15 averaged models (one per speaker) | 4 averaged models (2 bands × 2 scenarios) |
| Test on all 15 speakers | Test on 2 speakers (M08, M01) |
| No intelligibility grouping | Group by HIGH/VERY_LOW bands |
| No exclusion toggle | EXCLUDED vs INCLUDED scenarios |

---

## Expected Results

### **Hypothesis:**
- **INCLUDED** scenario should show **better performance** (lower WER) because the test speaker's knowledge is in the averaged weights
- **EXCLUDED** scenario tests **true generalization** to unseen speakers

### **Comparison:**
- **HIGH band** (M08): Expected WER ~10-20% (good speakers)
- **VERY_LOW band** (M01): Expected WER ~40-60% (hard speakers)

### **Key Metric:**
```
Generalization Gap = WER_excluded - WER_included
```

Larger gap indicates the averaged model benefits significantly from including the test speaker's weights.

---

## Troubleshooting

### Issue: "No speakers found for band"
**Fix**: Check `speaker_checkpoints.json` has all required speakers

### Issue: "Model not found"
**Fix**: Run Step 1 first to create averaged models

### Issue: "CSV file not found"
**Fix**: Verify paths in `/home/zsim710/partitions/uaspeech/by_speakers/`

---

## Next Steps After Baseline

Once baseline results are obtained:
1. Compare with **knowledge distillation** approach
2. Compare with **ensemble** approach (voting/averaging predictions)
3. Analyze which band benefits more from averaging
4. Test on additional speakers (M05, M16 for MID/LOW bands - if needed)

---

## References

- Base checkpoint: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/`
- Speaker mappings: `/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json`
