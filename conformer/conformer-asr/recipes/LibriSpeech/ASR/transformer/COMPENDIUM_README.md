# Research Compendium: Speaker-Adaptive ASR Models and Logit Extraction

## Directory Overview

This directory contains the implementation for training speaker-adaptive (SA) ASR models on the UASpeech corpus and extracting their predictions (logits) for downstream knowledge distillation. The work leverages SpeechBrain's Transformer-based ASR architecture adapted for dysarthric speech recognition.

**Location:** `/home/zsim710/XDED/conformer/conformer-asr/recipes/LibriSpeech/ASR/transformer`

---

## Table of Contents

1. [Purpose and Context](#purpose-and-context)
2. [Key Components](#key-components)
3. [Experimental Setup](#experimental-setup)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Logit Extraction](#logit-extraction)
7. [Verification Procedures](#verification-procedures)
8. [Cross-Testing Experiments](#cross-testing-experiments)
9. [Replication Instructions](#replication-instructions)
10. [File Descriptions](#file-descriptions)
11. [Future Development](#future-development)

---

## Purpose and Context

This directory implements the **first stage** of the XDED (Cross-Domain Ensemble Distillation) pipeline:

1. Train individual speaker-adaptive models (one per speaker) on UASpeech data
2. Extract decoder and CTC logits from each trained model
3. Verify logit extraction quality through token-level comparison
4. Cross-test SA models on other speakers to understand domain transfer

These extracted logits serve as the teacher signals for the multi-teacher knowledge distillation implemented in the main XDED directory.

---

## Key Components

### Core Scripts

- **`train.py`**: Main training script for speaker-adaptive ASR models
- **`extract_sa_logits.py`**: Extract CTC and decoder logits from trained SA models
- **`verify_logits.py`**: Core verification functions for logit quality checking
- **`run_token_verification.py`**: Orchestration script for running verification across all models
- **`run_all_sa_models.py`**: Automated cross-testing of SA models on all speakers
- **`uaspeech_prepare.py`**: Data preparation script for UASpeech corpus

### Configuration Files

Located in `hparams/exp/uaspeech/`:

- **`ua_SA_val_uncommon_WRA.yaml`**: Configuration with uncommon validation words
- **Multiple speaker-specific YAML files**: e.g., `ua_SA_val_uncommon_M08_E0D2.yaml`

---

## Experimental Setup

### Dataset: UASpeech Corpus

- **Speakers:** 15 total (11 male, 4 female)
- **Intelligibility bands:**
  - HIGH: M08, M09, M10, M14, F05
  - MID: M05, M11, F04
  - LOW: M16, F02, M07
  - VERY_LOW: M01, M12, F03, M04
- **Vocabulary:** 5000 tokens (SentencePiece tokenization)
- **Audio format:** 16kHz WAV files
- **Partition location:** `/home/zsim710/partitions/uaspeech/by_speakers/`

### Model Architecture

- **Base:** SpeechBrain Transformer ASR
- **Encoder:** Conformer/Transformer with CNN frontend
- **Decoder:** Autoregressive Transformer decoder
- **CTC head:** Frame-level classification head
- **Output vocabulary:** 5000 tokens

### Training Configuration

- **Epochs:** 200 (with early stopping)
- **Batch size:** 8 (configurable)
- **Optimizer:** AdamW
- **Learning rate:** Peak 0.0008 with Noam scheduling
- **Gradient clipping:** Applied
- **Mixed precision:** Enabled for efficiency
- **Checkpoint selection:** Best validation loss

---

## Data Preparation

### Step 1: Prepare UASpeech Data

```bash
python uaspeech_prepare.py
```

**What it does:**
- Creates per-speaker CSV files with columns: `ID,duration,wav,wrd,spk_id`
- Splits data into train/validation sets
- Generates tokenizer from training transcripts
- Saves partition files to `/home/zsim710/partitions/uaspeech/by_speakers/`

**Output structure:**
```
/home/zsim710/partitions/uaspeech/by_speakers/
├── M01.csv
├── M04.csv
├── M05.csv
...
└── F05.csv
```

---

## Model Training

### Speaker-Adaptive Model Training

For each speaker, train an individual SA model:

```bash
python train.py hparams/exp/uaspeech/ua_SA_val_uncommon_<SPEAKER>.yaml \
    --sa_model <SPEAKER> \
    --data_folder /path/to/uaspeech/audio
```

**Example for speaker M08:**
```bash
python train.py hparams/exp/uaspeech/ua_SA_val_uncommon_M08_E0D2.yaml \
    --sa_model M08 \
    --data_folder /mnt/data/uaspeech
```

### Batch Training Script

To train all speakers systematically, use:

```bash
bash scripts/run_sa_WRA_freezeE.sh
```

**What the script does:**
- Iterates through all 15 speakers
- Launches training with appropriate YAML configuration
- Saves checkpoints to `results/<experiment_id>/save/`
- Logs training metrics and validation WRA

### Checkpoint Locations

Trained SA model checkpoints are stored at:
```
/mnt/Research/qwan121/ICASSP_SA/val_uncommon_<SPEAKER>_<ID>/save/
```

Example:
```
/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/
```

---

## Logit Extraction

### Purpose

Extract pre-softmax logits (both CTC frame-level and decoder token-level) from each trained SA model for use as teacher signals in knowledge distillation.

### Extraction Script

```bash
python extract_sa_logits.py \
    --speaker <SPEAKER_ID> \
    --checkpoint_dir <PATH_TO_CHECKPOINT> \
    --hparams_file <PATH_TO_YAML> \
    --output_dir <OUTPUT_PATH>
```

**Example:**
```bash
python extract_sa_logits.py \
    --speaker M08 \
    --checkpoint_dir /mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/ \
    --hparams_file /mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/hyperparams.yaml \
    --output_dir /home/zsim710/speechbrain/exp_results/logit_extraction
```

### What Gets Extracted

For each utterance:

1. **CTC logits** (frame-level):
   - Shape: `[num_frames, vocab_size]`
   - Pre-softmax scores over vocabulary at each time frame
   - Saved as: `<speaker>_ctc_logits.pt` (list of tensors)

2. **Decoder logits** (token-level):
   - Shape: `[num_tokens, vocab_size]`
   - Teacher-forced autoregressive predictions
   - Pre-softmax scores for each output token position
   - Saved as: `<speaker>_decoder_logits.pt` (list of tensors)

3. **Metadata**:
   - Utterance IDs
   - Ground-truth transcripts
   - Tokenized targets
   - Saved as: `<speaker>_metadata.json`

### Output Structure

```
/home/zsim710/speechbrain/exp_results/logit_extraction/
├── F02/
│   ├── F02_ctc_logits.pt
│   ├── F02_decoder_logits.pt
│   └── F02_metadata.json
├── M08/
│   ├── M08_ctc_logits.pt
│   ├── M08_decoder_logits.pt
│   └── M08_metadata.json
...
└── M16/
    ├── M16_ctc_logits.pt
    ├── M16_decoder_logits.pt
    └── M16_metadata.json
```

---

## Verification Procedures

### Purpose

Ensure extracted logits are valid by decoding them and comparing to original model predictions and ground truth.

### Token-Level Verification

**Run verification for all models:**
```bash
python run_token_verification.py
```

**What it checks:**

1. **Token sequence matching:**
   - Decode logits using argmax
   - Compare predicted tokens to ground truth tokens
   - Report exact sequence matches

2. **Text-level matching:**
   - Decode tokens to text using tokenizer
   - Compare decoded text to ground truth transcripts
   - Report text match percentage

3. **Statistics reported:**
   - Exact token sequence matches
   - Token-level accuracy (% correct tokens)
   - Sequence length matches
   - Non-empty prediction rate
   - Quality assessment (Excellent ≥95%, Good ≥80%)

### Verification Output Example

```
🎯 TOKEN-LEVEL LOGIT VERIFICATION
================================================================================

📚 Loading tokenizer...
✅ Tokenizer loaded successfully

🔍 Verifying model: F03
────────────────────────────────────────────────────────────────────────────────
Loaded 1785 utterances

Sample Comparison (first 5 utterances):
----------------------------------------
Utterance: F03_B1_UW1_M4
  Ground truth: THIS
  CTC decoded:  THIS
  Decoder:      THIS

[... more samples ...]

Statistics:
  Decoder exact matches: 1234/1785 (69.1%)
  CTC non-empty:        1750/1785 (98.0%)
  Token accuracy:       94.3%

✅ Quality: Good
================================================================================
```

---

## Cross-Testing Experiments

### Purpose

Evaluate how well each SA model generalizes to other speakers by testing on all speaker combinations.

### Automated Cross-Testing

```bash
python run_all_sa_models.py
```

**What it does:**

1. For each SA model checkpoint:
   - Backs up original configuration YAML
   - Iterates through all 15 speakers as test targets
   - Dynamically edits YAML with:
     - `sa_model`: Source speaker ID
     - `speaker`: Test speaker ID
     - `load_ckpt`: Path to SA checkpoint
     - `test_csv`: Path to test speaker's CSV
   - Runs: `python train.py <YAML> --test_only`
   - Logs results and WRA metrics

2. Tracks successes and failures across all combinations
3. Generates cross-testing matrix

### Expected Output

```
================================================================================
CROSS-TESTING SA MODELS
================================================================================

Testing M10 on all speakers...
  ✓ M10 → F02: WRA = 45.2%
  ✓ M10 → M08: WRA = 67.8%
  ✓ M10 → M01: WRA = 12.3%
  [... more results ...]

Testing M11 on all speakers...
  [... results ...]

Summary:
  Total tests: 210 (14 models × 15 speakers)
  Successful: 198
  Failed: 12
  Average cross-speaker WRA: 38.4%
```

---

## Replication Instructions

### Prerequisites

1. **Environment setup:**
   ```bash
   conda env create -f environment.yaml
   conda activate xded
   ```

2. **Install SpeechBrain:**
   ```bash
   cd /home/zsim710/XDED/speechbrain
   pip install -e .
   ```

3. **Verify UASpeech data location:**
   - Audio files in `/mnt/data/uaspeech/`
   - Partition CSVs in `/home/zsim710/partitions/uaspeech/by_speakers/`

### Full Pipeline Replication

#### Step 1: Prepare Data (if not already done)
```bash
python uaspeech_prepare.py
```

#### Step 2: Extract Logits

For each trained model:
```bash
python extract_sa_logits.py \
    --speaker <SPEAKER> \
    --checkpoint_dir <CHECKPOINT_PATH> \
    --hparams_file <YAML_PATH> \
    --output_dir /home/zsim710/speechbrain/exp_results/logit_extraction
```


#### Step 3: Verify Logits

```bash
python run_token_verification.py
```

#### Step 4: Cross-Testing

```bash
python run_all_sa_models.py
```

---

## File Descriptions

### Training and Preparation

| File | Purpose | Key Functions |
|------|---------|---------------|
| `train.py` | Main ASR training script | Model instantiation, training loop, evaluation |
| `uaspeech_prepare.py` | Data preparation for UASpeech | CSV generation, train/test splits, tokenizer creation |
| `librispeech_prepare.py` | Data preparation for LibriSpeech | (Reference implementation) |

### Logit Extraction

| File | Purpose | Key Functions |
|------|---------|---------------|
| `extract_sa_logits.py` | Extract CTC and decoder logits | `extract_logits()`, saves `.pt` and `.json` files |
| `test_extract_logits.py` | Unit tests for extraction | Verify extraction correctness |
| `test_logit_storage.py` | Test storage mechanisms | Check file I/O integrity |

### Verification

| File | Purpose | Key Functions |
|------|---------|---------------|
| `verify_logits.py` | Core verification functions | `verify_model_tokens()`, `load_tokenizer()` |
| `run_token_verification.py` | Orchestration for verification | Batch verification across models |

### Cross-Testing

| File | Purpose | Key Functions |
|------|---------|---------------|
| `run_all_sa_models.py` | Automated cross-testing | YAML editing, batch testing, result logging |
| `run_speaker_tests.py` | Individual speaker testing | Single-speaker test runs |

### Configuration

| Directory/File | Purpose |
|----------------|---------|
| `hparams/exp/uaspeech/` | UASpeech-specific configurations |
| `hparams/exp/uaspeech/ua_SA_template.yaml` | Base template for SA training |
| `hparams/exp/uaspeech/ua_SA_val_uncommon_<SPEAKER>.yaml` | Per-speaker configurations |

### Batch Scripts

Located in `scripts/`:

| Script | Purpose |
|--------|---------|
| `run_sa_WRA_freezeE.sh` | Train all SA models with frozen encoder |
| `partial_run_sa_WRA_freezeE.sh` | Train subset of SA models |
| `run_ua_control_WRA.sh` | Control experiments (baseline) |

---

## Results and Checkpoints

### Checkpoint Storage

All trained SA model checkpoints are stored at:
```
/mnt/Research/qwan121/ICASSP_SA/val_uncommon_<SPEAKER>_<ID>/
```

Each checkpoint directory contains:
- `save/CKPT+<timestamp>/`: Model weights
- `hyperparams.yaml`: Full configuration
- `tokenizer.ckpt`: SentencePiece model
- `normalizer.ckpt`: Audio normalizer
- `train_log.txt`: Training logs

### Extracted Logits

All extracted logits are stored at:
```
/home/zsim710/speechbrain/exp_results/logit_extraction/
```

### Cross-Testing Results

Cross-testing matrices and logs are stored at:
```
Results P4/cross testing/
```

---

## Key Experimental Findings

### Speaker WER Distribution

From `XDED/precompute_difficulty.py`:

| Speaker | Intelligibility | WER (%) |
|---------|----------------|---------|
| F05 | HIGH | 2.05 |
| M14 | HIGH | 5.41 |
| M10 | HIGH | 6.84 |
| M09 | HIGH | 8.36 |
| M08 | HIGH | 9.56 |
| F04 | MID | 15.29 |
| M11 | MID | 25.82 |
| M05 | MID | 30.77 |
| M16 | LOW | 38.96 |
| F02 | LOW | 39.99 |
| M07 | LOW | 46.79 |
| F03 | VERY_LOW | 60.23 |
| M12 | VERY_LOW | 69.11 |
| M01 | VERY_LOW | 83.06 |
| M04 | VERY_LOW | 86.40 |

### Verification Results

- **Token accuracy:** 92-98% across all models (Excellent/Good quality)
- **Exact sequence matches:** 65-85% depending on speaker intelligibility
- **Non-empty predictions:** >98% for all models

### Cross-Testing Patterns

- **Within-band transfer:** Generally better (e.g., HIGH→HIGH performs well)
- **Cross-band transfer:** Significant degradation (e.g., HIGH→VERY_LOW drops dramatically)
- **VERY_LOW speakers:** Consistently difficult for all SA models

---

## Future Development

### Recommended Enhancements

1. **Beam search decoding:**
   - Currently only greedy decoding is implemented
   - Beam search could improve verification accuracy

2. **Multi-GPU training:**
   - Parallelize SA model training across speakers
   - Reduce total training time from ~120 hours to ~8 hours

3. **Automatic hyperparameter tuning:**
   - Grid search over learning rate, batch size, architecture
   - Per-speaker optimization

4. **Data augmentation:**
   - SpecAugment for robustness
   - Speed/pitch perturbation
   - Could improve generalization

5. **Alternative architectures:**
   - Conformer variants (different kernel sizes)
   - Whisper-based models (see `train_with_whisper.py`)
   - Comparison study

6. **Entropy analysis:**
   - Correlate entropy with speaker intelligibility
   - Use as curriculum signal

---

## Contact and Maintenance

**Primary maintainers:**
- Zoe Sim (zsim710)
- Project advisors

**Last updated:** October 2025

**Related directories:**
- Main XDED implementation: `/home/zsim710/XDED/XDED/`
- SpeechBrain fork: `/home/zsim710/XDED/speechbrain/`
- Tokenizers: `/home/zsim710/XDED/tokenizers/`

---

## References

1. SpeechBrain: A General-Purpose Speech Toolkit (Ravanelli et al., 2021)
2. UASpeech corpus: Speech database for dysarthric speech research
3. Knowledge Distillation (Hinton et al., 2015)
4. Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)

---

## Appendix: Common Issues and Solutions

### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size in YAML configuration or use gradient accumulation

### Issue 2: Tokenizer Not Found
**Solution:** Verify tokenizer.ckpt path in hyperparams.yaml points to `/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt`

### Issue 3: CSV Files Missing
**Solution:** Run `uaspeech_prepare.py` to regenerate partition files

### Issue 4: Checkpoint Loading Fails
**Solution:** Ensure checkpoint path includes full directory: `save/CKPT+<timestamp>/`

### Issue 5: Verification Shows Low Accuracy
**Solution:** Check if correct tokenizer is being used; verify vocab_size matches (5000)

---

**End of Compendium README for Transformer Directory**
