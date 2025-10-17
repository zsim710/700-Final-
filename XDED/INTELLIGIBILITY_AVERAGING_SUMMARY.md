# ✅ Intelligibility-Based Weight Averaging - Ready to Run!

## 🎯 Objective

Create a **generalization baseline** by averaging SA model weights within intelligibility bands, then test on held-out speakers to measure cross-speaker generalization.

---

## 📊 Experiment Design

### **Intelligibility Bands:**
- **HIGH**: M09, M14, M10, M08, F05 (5 speakers)
- **VERY_LOW**: M12, M01, F03, M04 (4 speakers)

### **Test Speakers:**
- **HIGH band**: M08
- **VERY_LOW band**: M01

### **Two Scenarios:**
1. **EXCLUDED**: Test speaker NOT in averaging (true generalization)
2. **INCLUDED**: Test speaker included in averaging (should perform better)

### **Total Models to Create:**
- `HIGH_averaged_exclude_M08.pt` (4 speakers: M09, M14, M10, F05)
- `HIGH_averaged_include_M08.pt` (5 speakers: M09, M14, M10, M08, F05)
- `VERY_LOW_averaged_exclude_M01.pt` (3 speakers: M12, F03, M04)
- `VERY_LOW_averaged_include_M01.pt` (4 speakers: M12, M01, F03, M04)

---

## ✅ Pre-Flight Check: PASSED

All required files and dependencies verified:

### **Scripts:**
- ✅ `tools/average_sa_models_intelligibility.py` - Core averaging logic
- ✅ `tools/run_intelligibility_averaging.py` - Orchestration
- ✅ `tools/evaluate_intelligibility_averaging.sh` - Evaluation runner
- ✅ `tools/test_averaged_model_simple.py` - Inference engine
- ✅ `tools/summarize_intelligibility_results.py` - Results analysis
- ✅ `tools/quick_start_intelligibility_averaging.sh` - One-click execution

### **Data:**
- ✅ Speaker checkpoint mapping: `results/speaker_averaging/speaker_checkpoints.json`
- ✅ M08 test CSV: `/home/zsim710/partitions/uaspeech/by_speakers/M08.csv`
- ✅ M01 test CSV: `/home/zsim710/partitions/uaspeech/by_speakers/M01.csv`

### **Supporting Files (Corrected Paths):**
- ✅ Hyperparams: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/hyperparams.yaml`
- ✅ Tokenizer: `/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt`
- ✅ Normalizer: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/normalizer.ckpt`

### **Dependencies:**
- ✅ PyTorch: 2.5.1+cu124
- ✅ SpeechBrain
- ✅ Pandas: 2.3.3
- ✅ jiwer

---

## 🚀 How to Run

### **Option 1: One-Click Execution** (Recommended)

```bash
cd /home/zsim710/XDED/XDED
./tools/quick_start_intelligibility_averaging.sh
```

This will:
1. Create 4 averaged models (both scenarios, both bands)
2. Evaluate all 4 models on their test speakers
3. Display summary results

---

### **Option 2: Step-by-Step Execution**

#### **Step 1: Create Averaged Models**
```bash
cd /home/zsim710/XDED/XDED
python3 tools/run_intelligibility_averaging.py
```

**Output:** 4 `.pt` files in:
- `results/intelligibility_averaging/excluded/`
- `results/intelligibility_averaging/included/`

**Optional**: Run only one scenario:
```bash
# Only excluded scenario
python3 tools/run_intelligibility_averaging.py --exclude_held_out

# Only included scenario
python3 tools/run_intelligibility_averaging.py --include_held_out
```

---

#### **Step 2: Evaluate Models**
```bash
chmod +x tools/evaluate_intelligibility_averaging.sh
./tools/evaluate_intelligibility_averaging.sh
```

**Output:** 4 JSON result files in:
- `results/intelligibility_averaging/evaluation/excluded/`
- `results/intelligibility_averaging/evaluation/included/`

---

#### **Step 3: View Summary**
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
Generalization Gap = WER_excluded - WER_included
================================================================================
```

---

## 📈 Expected Results

### **Hypothesis:**
- **INCLUDED** should have **lower WER** (test speaker's knowledge in weights)
- **EXCLUDED** tests **true generalization** to unseen speakers

### **Performance Ranges:**
- **HIGH band (M08)**: Expected WER ~10-20% (easier speakers)
- **VERY_LOW band (M01)**: Expected WER ~40-60% (harder speakers)

### **Key Metric:**
```
Generalization Gap = WER_excluded - WER_included
```

**Interpretation:**
- **Large gap** (>5% WER): Averaging benefits greatly from test speaker
- **Small gap** (<2% WER): Good generalization even without test speaker
- **Negative gap**: Unexpected, may indicate overfitting in included scenario

---

## 📁 Output Structure

```
results/intelligibility_averaging/
├── excluded/                              # Models WITHOUT test speaker
│   ├── HIGH_averaged_exclude_M08.pt
│   └── VERY_LOW_averaged_exclude_M01.pt
├── included/                              # Models WITH test speaker
│   ├── HIGH_averaged_include_M08.pt
│   └── VERY_LOW_averaged_include_M01.pt
├── inference/                             # Full checkpoint directories
│   ├── excluded/
│   │   ├── HIGH_M08/
│   │   │   ├── hyperparams.yaml
│   │   │   ├── tokenizer.ckpt
│   │   │   ├── normalizer.ckpt
│   │   │   └── model.ckpt (averaged weights)
│   │   └── VERY_LOW_M01/
│   └── included/
│       └── ... (same structure)
└── evaluation/                            # Results
    ├── excluded/
    │   ├── HIGH_M08_results.json
    │   └── VERY_LOW_M01_results.json
    └── included/
        ├── HIGH_M08_results.json
        └── VERY_LOW_M01_results.json
```

---

## 🔍 Troubleshooting

### Issue: "No speakers found for band"
**Solution:** Verify `speaker_checkpoints.json` has all required speakers

### Issue: "Model not found"
**Solution:** Run Step 1 first (create averaged models)

### Issue: "CSV file not found"
**Solution:** Check paths in `/home/zsim710/partitions/uaspeech/by_speakers/`

### Issue: "Import error: speechbrain"
**Solution:** Ensure you're in the FirstDeep conda environment

### Issue: "CUDA out of memory"
**Solution:** Use `--device cpu` in evaluation script (slower but works)

---

## 📊 Next Steps After Baseline

1. **Compare with Knowledge Distillation:**
   - Train student model using ensemble teacher logits
   - Compare WER with averaged weights baseline
   
2. **Compare with Ensemble Inference:**
   - Run ensemble with voting/averaging at prediction time
   - Measure latency vs accuracy tradeoff

3. **Analyze Band Differences:**
   - Does HIGH band benefit more from averaging than VERY_LOW?
   - Are generalization gaps consistent across bands?

4. **Extend to MID/LOW Bands (Optional):**
   - Test on M05 (MID) and M16 (LOW)
   - Full 4-band analysis

---

## 📚 Key Files Reference

| File | Purpose | Location |
|------|---------|----------|
| Averaging Script | Core weight averaging logic | `tools/average_sa_models_intelligibility.py` |
| Orchestration | Create all 4 models | `tools/run_intelligibility_averaging.py` |
| Evaluation | Test models on speakers | `tools/evaluate_intelligibility_averaging.sh` |
| Inference | SpeechBrain model loading | `tools/test_averaged_model_simple.py` |
| Summary | Display results | `tools/summarize_intelligibility_results.py` |
| Quick Start | One-click execution | `tools/quick_start_intelligibility_averaging.sh` |
| Checkpoint Map | Speaker → model path | `results/speaker_averaging/speaker_checkpoints.json` |

---

## 🎓 Research Context

**Goal:** Establish a **simple baseline** for cross-speaker generalization using weight averaging before comparing with more complex approaches (knowledge distillation, ensemble methods).

**Why This Matters:**
- Weight averaging is **computationally cheap** (one forward pass per model)
- Provides insight into **parameter-level** vs **prediction-level** ensembling
- Tests if dysarthric speech models can share knowledge via **linear interpolation**

**Expected Outcome:**
- Baseline WER for comparison with distillation-based student models
- Understanding of which intelligibility bands benefit most from averaging
- Evidence for whether weight space is convex for dysarthric ASR

---

## ✅ Status: Ready to Execute

All checks passed! You can now run the experiment.

**Recommended Command:**
```bash
cd /home/zsim710/XDED/XDED
./tools/quick_start_intelligibility_averaging.sh
```

**Estimated Runtime:**
- Averaging: ~5-10 minutes (loading 9 checkpoints, averaging weights)
- Evaluation: ~20-30 minutes per model (decoding test sets)
- Total: ~2 hours for all 4 models

---

**Good luck with your baseline experiment! 🚀**
