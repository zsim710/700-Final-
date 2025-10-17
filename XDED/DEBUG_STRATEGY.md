# Debug Strategy: Include Held-Out Speaker in Training

## Supervisor's Suggestion

**Problem**: Student model keeps failing evaluation (0% WRA) regardless of approach (CTC-KD, Decoder-KD)

**Hypothesis**: Two possible root causes:
1. **Domain shift**: Train speakers (13 others) ≠ Test speaker (M08) → generalization failure
2. **Training broken**: Model fundamentally can't learn the task

**Debug approach**: Include M08 in training set → isolates which hypothesis is correct

---

## Experimental Setup

### Normal Training (Fails)
```
Training speakers: F02, F03, F04, F05, M04, M05, M07, M09, M10, M11, M12, M14, M16 (13 speakers)
Test speaker: M08 (held-out, never seen)
Result: 0% WRA

Problem: Can't distinguish domain shift vs broken training
```

### Debug Training (New)
```
Training speakers: ALL 14 speakers including M08
Test speaker: M08 (same as training)
Expected results:
  - If WRA > 60%: Training works! Issue is domain shift/generalization
  - If WRA < 10%: Training is broken, fundamental issue with model/loss
```

---

## Implementation Changes

### 1. Added `--include_held_out_in_training` Flag

**File**: `train_student.py`

```python
parser.add_argument('--include_held_out_in_training', action='store_true',
    help='DEBUG: Include held-out speaker in training')
```

**Behavior**:
- When set: M08 becomes a teacher (included in ensemble), not held-out
- Dataset loads all 14 speakers for training
- Test set still uses M08, but now it's "training set performance"
- **Disables auto-CTC-switch**: Allows decoder-KD in debug mode

### 2. Updated Dataset Loading Logic

```python
if args.include_held_out_in_training:
    train_dataset = LogitEnsembleDataset(
        held_out_speaker=None,  # No speaker held out, use all
        split="train",
        use_decoder_logits=(args.teacher_logits_type == 'decoder'),
        ...
    )
else:
    train_dataset = LogitEnsembleDataset(
        held_out_speaker=args.held_out,  # Normal: exclude M08
        split="train",
        ...
    )
```

### 3. Created Debug Training Script

**File**: `train_nemo_debug_m08.sh`

- Uses decoder-KD (not CTC) as requested by supervisor
- Includes M08 in training via `--include_held_out_in_training` flag
- Two-stage training: freeze encoder (5 epochs) → unfreeze (35 epochs)
- Saves to `checkpoints_nemo_debug/` to avoid overwriting

---

## Why Decoder-KD for Debug (Reversing Previous Decision)

### Previous Analysis (Normal Training)
- ❌ Decoder-KD failed: 0% WRA
- ✅ Pretrained NeMo baseline: 41.68% WRA
- 🔍 Root cause: Feature space mismatch (NeMo 176-dim vs SpeechBrain 144-dim)
- 💡 Solution: Use CTC-KD instead

### Supervisor's Rationale for Debug
1. **Isolate the variable**: If we switch both training data AND distillation method, we won't know which fixed it
2. **Test worst case**: If decoder-KD works when M08 is in training, then:
   - Decoder-KD itself is viable
   - Original failure was domain shift, not method incompatibility
3. **Potentially contradicts our analysis**: Maybe feature mismatch isn't the issue?

### Expected Outcomes

**Scenario A: Decoder-KD works with M08 in training (WRA > 60%)**
```
Conclusion: 
  - Training process is fine
  - Original failure was domain shift (13 speakers → M08)
  - Feature space mismatch might not be the real issue
  
Next steps:
  - Improve generalization (data augmentation, stronger regularization)
  - Try decoder-KD with all speakers
  - Investigate why CTC-KD also failed
```

**Scenario B: Decoder-KD still fails (WRA < 10%)**
```
Conclusion:
  - Training is fundamentally broken
  - Feature space mismatch confirmed
  - Decoder-KD incompatible with NeMo encoder
  
Next steps:
  - Use CTC-KD instead (as we planned)
  - Or use SpeechBrain encoder (not NeMo)
  - Debug loss function / training loop
```

**Scenario C: Moderate success (WRA 20-50%)**
```
Conclusion:
  - Both issues contribute (domain shift + method)
  - Partial learning possible
  
Next steps:
  - Try CTC-KD with M08 in training (control experiment)
  - Compare CTC vs decoder with same train set
```

---

## Running the Debug Experiment

### Step 1: Start Training
```bash
cd /home/zsim710/XDED/XDED
./train_nemo_debug_m08.sh
```

**Training time**: ~2-3 hours for 40 epochs (Stage 1: 5 epochs, Stage 2: 35 epochs)

### Step 2: Monitor Training
```bash
# Check latest checkpoint info
ls -lht checkpoints_nemo_debug/student_M08/

# Watch training progress (if running in background)
tail -f nohup.out  # or whatever log file
```

**What to look for**:
- Validation loss should decrease (not just stay flat)
- Valid-frame ratio should be reasonable (~50-60%)
- Training should converge (not oscillate)

### Step 3: Evaluate
```bash
python eval_student.py \
  --checkpoint checkpoints_nemo_debug/student_M08/best.pt \
  --held_out M08 \
  --decode_mode decoder \
  --max_decode_len 10 \
  --device cuda \
  --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
  --output eval_M08_nemo_debug_decoder.json \
  --num_samples 50
```

**Metrics to check**:
- **WRA** (Word Recognition Accuracy): Primary metric for single-word task
- **Prediction diversity**: Should have many unique predictions (not collapsed)
- **Sample predictions**: Should be actual words, not garbage

### Step 4: Interpret Results

| WRA Range | Interpretation | Next Action |
|-----------|---------------|-------------|
| > 70% | ✅ Training works! Domain shift was the issue | Test generalization strategies |
| 50-70% | ⚠️ Partial success, both issues contribute | Compare CTC vs Decoder on same data |
| 20-50% | 🤔 Model learning something, but struggling | Debug loss function, try CTC |
| < 20% | ❌ Training broken, fundamental issue | Switch to CTC-KD or SpeechBrain encoder |

---

## Comparison Matrix

| Experiment | Train Data | Method | Expected WRA | Purpose |
|------------|-----------|--------|--------------|---------|
| **Original** | 13 speakers (no M08) | Decoder-KD | 0% ✗ | Failed baseline |
| **CTC attempt** | 13 speakers (no M08) | CTC-KD | Not tested | Planned fix |
| **Debug (this)** | 14 speakers (with M08) | Decoder-KD | ??? | Isolate root cause |
| **Control** | 14 speakers (with M08) | CTC-KD | ??? | Compare methods |

---

## Follow-Up Experiments (Based on Results)

### If Debug Works (WRA > 60%)
1. **Test CTC-KD with M08 in training** (control)
   - Compare decoder vs CTC on same train set
   - Isolate method effect vs data effect

2. **Gradual held-out testing**
   - Train with 13 speakers, test on M08 (original setup)
   - Use both CTC and decoder
   - Measure generalization gap

3. **Improve generalization**
   - Data augmentation (speed, pitch, noise)
   - Stronger regularization (dropout, weight decay)
   - Domain adaptation techniques

### If Debug Fails (WRA < 10%)
1. **Switch to CTC-KD immediately**
   - Run `train_nemo_ctc_m08.sh` (already created)
   - Compare CTC results

2. **Try SpeechBrain encoder** (not NeMo)
   - Train from scratch with decoder-KD
   - Ensures feature compatibility

3. **Debug training loop**
   - Check loss values (should decrease)
   - Visualize attention patterns
   - Inspect decoder predictions during training

---

## Files Created/Modified

### New Files
- ✅ `train_nemo_debug_m08.sh` - Debug training script
- ✅ `DEBUG_STRATEGY.md` - This documentation

### Modified Files
- ✅ `train_student.py` - Added `--include_held_out_in_training` flag

### Existing Files (Reference)
- `train_nemo_ctc_m08.sh` - CTC-KD training (fallback if debug fails)
- `eval_student.py` - Evaluation script (supports both backbones)
- `FINDINGS_AND_NEXT_STEPS.md` - Previous analysis

---

## Timeline

**Oct 15, 2025 - Morning**: Initial decoder-KD training failed (0% WRA)

**Oct 15, 2025 - Afternoon**: 
- Diagnosed feature space mismatch
- Created CTC-KD solution
- Evaluated pretrained baseline (41.68% WRA)

**Oct 15, 2025 - Evening**: 
- Supervisor suggested debug approach
- Implemented include-held-out-in-training mode
- Ready to test decoder-KD with M08 in training

**Oct 16, 2025 (Planned)**: Debug results available

---

## Key Questions This Answers

1. **Can the model learn AT ALL?**
   - If WRA > 60% with M08 in training → Yes
   - If WRA < 10% → No, fundamental issue

2. **Is decoder-KD viable with NeMo?**
   - If works with M08 in training → Yes, just needs better generalization
   - If fails even with M08 → No, use CTC-KD instead

3. **Was our feature-mismatch analysis correct?**
   - If debug succeeds → Maybe not, domain shift was the issue
   - If debug fails → Confirmed, feature spaces incompatible

4. **What's the next best approach?**
   - See "Follow-Up Experiments" section above
   - Decision tree based on debug results
