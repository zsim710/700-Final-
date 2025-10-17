# Debug Training Session - M08 Included

## Status: RUNNING ✅

**Started**: Oct 15, 2025 @ 20:09 UTC  
**Script**: `train_nemo_debug_m08.sh`  
**Mode**: DEBUG (M08 included in training)  
**Method**: Decoder-based Knowledge Distillation

---

## Configuration

### Training Data
- **Teachers**: 14 speakers (F02, F03, F04, F05, M04, M05, M07, **M08**, M09, M10, M11, M12, M14, M16)
- **M08 status**: INCLUDED as teacher (not held-out)
- **Training samples**: 1,760 utterances
- **Validation samples**: 100 utterances
- **Average teachers per utterance**: 13.3

### Model Architecture
- **Backbone**: NeMo Conformer (nvidia/stt_en_conformer_ctc_small)
- **Total parameters**: 18.3M
  - Pretrained encoder: ~13M (frozen in Stage 1)
  - Custom decoder + heads: 5.3M (trainable)
- **Decoder**: 4-layer Transformer, d_model=176

### Training Parameters
- **Method**: Decoder-KD (not CTC)
- **Temperature**: 2.0
- **Teacher aggregation**: logprob_mean
- **Blank prob threshold**: 0.95

#### Stage 1 (Epochs 1-5): Frozen Encoder
- Learning rate: 3e-4
- Batch size: 8
- Encoder: FROZEN
- Decoder: TRAINABLE

#### Stage 2 (Epochs 6-40): Full Fine-tuning
- Learning rate: 2e-4
- Warmup steps: 2000
- Batch size: 8
- Encoder: TRAINABLE
- Decoder: TRAINABLE

---

## Progress Monitor

### Epoch 1/5 (Stage 1)
```
Status: Running (58% complete)
Loss: 20.23
Valid-frame ratio: 58.3%
Learning rate: ~0.0406 (Noam warmup)
Speed: ~15.85 it/s
```

### Expected Timeline
- Stage 1 (5 epochs): ~10-15 minutes
- Stage 2 (35 epochs): ~60-90 minutes
- **Total**: ~70-105 minutes (1.2-1.7 hours)

---

## What This Tests

### Hypothesis A: Domain Shift Issue
```
IF training works (WRA > 60% on M08):
  → Original failure was because M08 wasn't in training
  → Decoder-KD is viable, just needs better generalization
  → Feature space mismatch might not be the real blocker
```

### Hypothesis B: Training Broken
```
IF training fails (WRA < 10% on M08):
  → Something fundamentally wrong with training/model
  → Confirms feature space mismatch theory
  → Need to use CTC-KD or SpeechBrain encoder
```

---

## Monitoring Commands

### Check Training Progress
```bash
# Watch log output
tail -f /home/zsim710/XDED/XDED/nohup.out  # if running in background

# Check latest checkpoint
ls -lht /home/zsim710/XDED/XDED/checkpoints_nemo_debug/student_M08/

# View checkpoint info
python -c "import torch; ckpt=torch.load('checkpoints_nemo_debug/student_M08/latest.pt', map_location='cpu'); print(f'Epoch: {ckpt[\"epoch\"]}, Train Loss: {ckpt[\"train_loss\"]:.4f}, Val Loss: {ckpt[\"val_loss\"]:.4f}')"
```

### Kill Training (if needed)
```bash
pkill -f "train_student.py"
```

---

## Evaluation Plan

### After Training Completes

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

### Key Metrics to Check
1. **WRA** (Word Recognition Accuracy) - Primary metric
2. **Prediction diversity** - Should have many unique predictions
3. **Sample quality** - Predictions should be real words
4. **WER** - Word Error Rate (for multi-word if applicable)

### Interpretation Guidelines

| WRA | Verdict | Interpretation |
|-----|---------|---------------|
| > 70% | ✅ **Training Works!** | Domain shift was the issue. Decoder-KD viable. |
| 50-70% | ⚠️ **Partial Success** | Both domain shift AND method contribute. |
| 20-50% | 🤔 **Mixed** | Model learning something, but struggling. |
| < 20% | ❌ **Training Broken** | Fundamental issue. Try CTC-KD instead. |

---

## Next Steps (Based on Results)

### If WRA > 60% (Training Works)
1. **Compare with CTC-KD** on same data (M08 included)
   - Isolate method effect
   - See if decoder or CTC is better

2. **Test generalization** with normal held-out setup
   - Train on 13 speakers (exclude M08)
   - Test on M08
   - Measure domain shift gap

3. **Improve generalization**
   - Data augmentation (speed, pitch, noise)
   - Stronger regularization
   - Domain adaptation techniques

### If WRA < 20% (Training Broken)
1. **Switch to CTC-KD immediately**
   - Run `train_nemo_ctc_m08.sh`
   - Confirms feature mismatch theory

2. **Try SpeechBrain encoder** (not NeMo)
   - Ensures feature compatibility
   - Decoder-KD with matching features

3. **Debug training internals**
   - Visualize attention patterns
   - Check gradient flow
   - Inspect loss components

---

## Checkpoints

### Location
```
/home/zsim710/XDED/XDED/checkpoints_nemo_debug/student_M08/
```

### Files
- `best.pt` - Best validation loss checkpoint
- `latest.pt` - Most recent checkpoint (for resuming)
- `checkpoint_epoch_*.pt` - Saved every 10 epochs
- `config.json` - Training configuration

### Loading for Analysis
```python
import torch

# Load checkpoint
ckpt = torch.load('checkpoints_nemo_debug/student_M08/best.pt')

# Check metadata
print(f"Epoch: {ckpt['epoch']}")
print(f"Train Loss: {ckpt['train_loss']:.4f}")
print(f"Val Loss: {ckpt['val_loss']:.4f}")

# Load model
from models.nemo_hybrid_student import NeMoHybridStudent
model = NeMoHybridStudent(vocab_size=5000)
model.load_state_dict(ckpt['model_state_dict'])
```

---

## Comparison with Previous Attempts

| Attempt | Train Data | Method | Result | Notes |
|---------|-----------|--------|--------|-------|
| **Original** | 13 speakers (no M08) | Decoder-KD | 0% WRA ❌ | Complete failure |
| **CTC Plan** | 13 speakers (no M08) | CTC-KD | Not tested | Planned fix |
| **Debug (This)** | 14 speakers (with M08) | Decoder-KD | Running... | Tests if training works |

---

## Documentation

- `DEBUG_STRATEGY.md` - Full strategy explanation
- `FINDINGS_AND_NEXT_STEPS.md` - Previous failure analysis
- `NEMO_TRAINING_FIX.md` - Feature mismatch diagnosis
- `THIS_FILE.md` - Current training session status

---

**Last Updated**: Oct 15, 2025 @ 20:10 UTC  
**Expected Completion**: Oct 15, 2025 @ 21:30 UTC  
**Terminal ID**: `d339a8fc-d4af-4610-9832-261d263a55b9`
