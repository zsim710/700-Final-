# NeMo-backed Student Model Integration

## Summary

Successfully integrated NVIDIA's pretrained Conformer-CTC encoder (nvidia/stt_en_conformer_ctc_small) as an alternative student backbone for decoder-based knowledge distillation.

## What Changed

### 1. New Model: `NeMoHybridStudent`
- **File**: `models/nemo_hybrid_student.py`
- Wraps NeMo's pretrained Conformer-CTC encoder (~13M params)
- Attaches custom decoder + CTC heads for 5000-token vocab (matching teacher logits)
- Interface-compatible with existing `StudentConformer`
  - `forward(wavs, wav_lens)` → encoder_out, ctc_logits
  - `forward_decoder(encoder_out, targets, wav_lens)` → decoder_logits
  - `decode_greedy(encoder_out, max_len)` → predictions

### 2. Training Script Updates
- **File**: `train_student.py`
- Added flags:
  - `--student_backbone {sb, nemo}` - choose SpeechBrain or NeMo encoder
  - `--nemo_model_name` (default: nvidia/stt_en_conformer_ctc_small)
  - `--freeze_nemo_preprocessor`, `--freeze_nemo_encoder` - optional freezing
- Auto-detects model d_model for scheduler config

### 3. Environment Setup
- Installed PyTorch 2.5.1 + NeMo 2.5.0 (compatible versions)
- Resolved torch/torchaudio version conflicts

## Model Specifications

### NeMoHybridStudent (with nvidia/stt_en_conformer_ctc_small)
- **Encoder**: Conformer (16 layers, d_model=176, 4 heads, ~13M params)
  - Pretrained on several thousand hours of English speech
  - Subsampling factor: 4 (audio 16kHz → features ~100Hz → encoder ~25Hz)
  - Output: [B, T_enc, 176]
- **Decoder**: Custom Transformer (4 layers, 176-dim, auto-adjusted heads, ~3M params)
- **CTC Head**: Linear 176→5000 (~0.9M params)
- **Total**: ~18.3M parameters (13M frozen optional; 5M trainable decoder/heads)

### Key Architectural Details
- NeMo encoder returns [B, D, T] - transposed to [B, T, D] for compatibility
- NeMo d_model=176 requires nhead divisors (auto-adjusted to 8 or 11)
- Vocab size: 5000 (matches teacher logits; NeMo's internal BPE tokenizer not used)

## Usage

### Installation
```bash
# Install compatible PyTorch + NeMo
pip uninstall -y torch torchvision torchaudio
pip install 'torch==2.5.1' 'torchvision==0.20.1' 'torchaudio==2.5.1'
pip install --no-deps 'nemo_toolkit==2.5.0' 'nemo_run==0.6.0' 'nemo_text_processing==1.1.0' 'megatron_core==0.14.0' 'nvidia-modelopt==0.35.0'
```

### Training with NeMo Backbone

#### Two-stage approach (recommended):

**Stage A: Warm-start with frozen encoder (5 epochs)**
```bash
python /home/zsim710/XDED/XDED/train_student.py \
  --held_out M08 \
  --teacher_logits_type decoder \
  --student_backbone nemo \
  --nemo_model_name nvidia/stt_en_conformer_ctc_small \
  --freeze_nemo_encoder \
  --batch_size 8 \
  --lr 3e-4 \
  --warmup_steps 2000 \
  --epochs 5 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir /home/zsim710/XDED/XDED/checkpoints_nemo
```

**Stage B: Unfreeze and continue (40 epochs)**
```bash
python /home/zsim710/XDED/XDED/train_student.py \
  --held_out M08 \
  --teacher_logits_type decoder \
  --student_backbone nemo \
  --nemo_model_name nvidia/stt_en_conformer_ctc_small \
  --batch_size 8 \
  --lr 2e-4 \
  --warmup_steps 2000 \
  --epochs 40 \
  --temperature 2.0 \
  --teacher_agg logprob_mean \
  --device cuda \
  --save_dir /home/zsim710/XDED/XDED/checkpoints_nemo \
  --resume /home/zsim710/XDED/XDED/checkpoints_nemo/student_M08/latest.pt
```

#### Repeat for other folds:
Replace `--held_out M08` with `M16`, `M04`, `M05`

#### Keep M01 in training (override default exclusion):
Add `--exclude_speakers ""` (empty list)

### Evaluation
Use existing `eval_student.py` (supports both backbones automatically):
```bash
python /home/zsim710/XDED/XDED/eval_student.py \
  --checkpoint /home/zsim710/XDED/XDED/checkpoints_nemo/student_M08/best.pt \
  --test_csv /mnt/Research/qwan121/UASpeech/test/M08.csv \
  --mode decoder_greedy \
  --max_decode_len 10 \
  --vocab_size 5000
```

## Testing

Smoke test (`test_nemo_hybrid.py`) validates:
- ✅ NeMo model loads (downloads checkpoint on first run)
- ✅ Encoder forward: [B, T_audio] → [B, T_enc, 176]
- ✅ CTC head: [B, T_enc, 176] → [B, T_enc, 5000]
- ✅ Decoder forward (teacher-forced): [B, T_enc, 176] + [B, T_tgt] → [B, T_tgt, 5000]
- ✅ Greedy decoding: [B, T_enc, 176] → [B, ≤max_len] predictions

Run test:
```bash
cd /home/zsim710/XDED/XDED && python test_nemo_hybrid.py
```

## Assumptions & Constraints

- **Input audio**: 16kHz mono WAV (NeMo preprocessor expects this; no auto-resampling)
- **Teacher logits**: Must be 5000-dim decoder logits (same as before)
- **BOS/EOS/blank**: Indices 0, 1, 2 (unchanged from original setup)
- **KD compatibility**: All existing KD features work (teacher aggregation, first-token suppression, PL CE, etc.)

## Next Steps

1. **Train all folds** with NeMo backbone and compare WRA/WER vs. SpeechBrain-based student
2. **Ablation studies**:
   - Frozen encoder vs. unfrozen
   - Different LR schedules
   - Vary decoder depth (2/4/6 layers)
3. **Optional enhancements**:
   - Add automatic unfreezing after N epochs within single run
   - Add torchaudio resampler if data aren't 16kHz
   - Support other NeMo models (medium/large variants)

## Files Modified/Created

- **Created**:
  - `models/nemo_hybrid_student.py` (new hybrid model class)
  - `test_nemo_hybrid.py` (smoke test)
  - `debug_nemo_encoder.py` (debug util)
  - `NEMO_INTEGRATION.md` (this file)

- **Modified**:
  - `train_student.py` (added `--student_backbone` and NeMo instantiation logic)

## Quality Gates

- ✅ **Build**: All files compile without syntax errors
- ✅ **Lint**: VSCode reports only benign import warnings (resolved at runtime)
- ✅ **Smoke test**: `test_nemo_hybrid.py` passes all forward/decoder/greedy tests
- ✅ **Environment**: PyTorch 2.5.1, NeMo 2.5.0, CUDA available

## Known Issues & Warnings

- **pynvml deprecation warning**: Harmless; NeMo transitioning to nvidia-ml-py
- **NeMo data loader warnings**: Expected; we're using NeMo encoder only, not its data pipeline
- **Version conflicts**: If you see torch 2.8.0 after install, run force-reinstall of torch 2.5.1 (see installation section)

---
**Date**: 2025-10-15  
**Author**: GitHub Copilot (Agent)  
**Status**: ✅ Ready for training
