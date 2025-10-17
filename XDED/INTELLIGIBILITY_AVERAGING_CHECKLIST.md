# ✅ Pre-Execution Checklist

## Files Created/Updated with Corrected Paths:

- [x] `tools/average_sa_models_intelligibility.py` - Core averaging (NEW)
- [x] `tools/run_intelligibility_averaging.py` - Orchestration (NEW)
- [x] `tools/evaluate_intelligibility_averaging.sh` - Evaluation with CORRECTED paths
- [x] `tools/test_averaged_model_simple.py` - Updated CSV paths
- [x] `tools/summarize_intelligibility_results.py` - Results summary (NEW)
- [x] `tools/quick_start_intelligibility_averaging.sh` - One-click runner (NEW)
- [x] `tools/verify_setup.py` - Updated with CORRECTED file paths
- [x] `INTELLIGIBILITY_AVERAGING_README.md` - Complete documentation
- [x] `INTELLIGIBILITY_AVERAGING_SUMMARY.md` - Quick reference

## Corrected File Paths:

### ✅ Supporting Files (Fixed):
```bash
# OLD (incorrect):
HYPERPARAMS="/mnt/Research/.../val_uncommon_M08_E0D2/.../hyperparams.yaml"
TOKENIZER="/mnt/Research/.../val_uncommon_M08_E0D2/.../tokenizer.ckpt"

# NEW (correct):
HYPERPARAMS="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/hyperparams.yaml"
TOKENIZER="/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt"
NORMALIZER="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/normalizer.ckpt"
```

### ✅ Test CSV Files (Fixed):
```bash
# OLD (incorrect):
/home/spt853/results/uaspeech/M08/7775/save/M08.csv
/home/spt853/results/uaspeech/M01/7775/save/M01.csv

# NEW (correct):
/home/zsim710/partitions/uaspeech/by_speakers/M08.csv
/home/zsim710/partitions/uaspeech/by_speakers/M01.csv
```

## Verification Status:

```bash
$ python3 tools/verify_setup.py
✅ All checks PASSED
```

## Ready to Execute:

### Quick Start (Recommended):
```bash
cd /home/zsim710/XDED/XDED
./tools/quick_start_intelligibility_averaging.sh
```

### Or Step-by-Step:
```bash
# Step 1: Create averaged models
python3 tools/run_intelligibility_averaging.py

# Step 2: Evaluate
./tools/evaluate_intelligibility_averaging.sh

# Step 3: View results
python3 tools/summarize_intelligibility_results.py
```

## Expected Output Files:

```
results/intelligibility_averaging/
├── excluded/
│   ├── HIGH_averaged_exclude_M08.pt
│   └── VERY_LOW_averaged_exclude_M01.pt
├── included/
│   ├── HIGH_averaged_include_M08.pt
│   └── VERY_LOW_averaged_include_M01.pt
└── evaluation/
    ├── excluded/
    │   ├── HIGH_M08_results.json
    │   └── VERY_LOW_M01_results.json
    └── included/
        ├── HIGH_M08_results.json
        └── VERY_LOW_M01_results.json
```

## Key Changes from Previous Attempt:

1. **Only 2 bands** (HIGH and VERY_LOW) instead of 4
2. **Corrected hyperparams.yaml** path (from F02 model)
3. **Corrected tokenizer** path (from `/home/zsim710/XDED/tokenizers/`)
4. **Corrected test CSV** paths (from `/home/zsim710/partitions/`)
5. **Test speakers**: M08 (HIGH) and M01 (VERY_LOW) only

## Status: 🟢 READY TO RUN

All files verified, all paths corrected, all dependencies installed.

Execute when ready! 🚀
