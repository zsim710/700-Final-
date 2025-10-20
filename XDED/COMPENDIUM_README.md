# Research Compendium: XDED Main Implementation

## Directory Overview

This directory contains the core implementation of XDED (Cross-Domain Ensemble Distillation), including all training scripts, model architectures, evaluation pipelines, curriculum learning, and experimental utilities. This is the central hub where multi-teacher knowledge distillation is performed to train student models that generalize across dysarthric speakers.

**Location:** `/home/zsim710/XDED/XDED`

---

## Table of Contents

1. [Purpose and Architecture](#purpose-and-architecture)
2. [Directory Structure](#directory-structure)
3. [Core Training Pipeline](#core-training-pipeline)
4. [Model Architectures](#model-architectures)
5. [Evaluation Framework](#evaluation-framework)
6. [Curriculum Learning](#curriculum-learning)
7. [Data Loading and Ensemble](#data-loading-and-ensemble)
8. [Experimental Configurations](#experimental-configurations)
9. [Testing and Validation](#testing-and-validation)
10. [Checkpoint Management](#checkpoint-management)
11. [Replication Guide](#replication-guide)
12. [Development History](#development-history)

---

## Purpose and Architecture

### XDED Pipeline Overview

XDED implements a multi-teacher knowledge distillation framework for cross-speaker generalization in dysarthric ASR:

```
┌─────────────────────────────────────────────────────────────┐
│                    Teacher Models (SA)                      │
│  ┌────────┐  ┌────────┐  ┌────────┐       ┌────────┐        │
│  │  F02   │  │  F03   │  │  M08   │  ...  │  M16   │        │
│  │ logits │  │ logits │  │ logits │       │ logits │        │
│  └────────┘  └────────┘  └────────┘       └────────┘        │
│      ↓            ↓           ↓                 ↓           │
└─────────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │   Ensemble Aggregation (τ=2.0)      │
         │   • Probability averaging           │
         │   • Temperature scaling             │
         │   • Blank-aware normalization       │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │   Student Model Training            │
         │   • NeMo Hybrid or SB Conformer     │
         │   • KL divergence loss              │
         │   • Optional curriculum learning    │
         └─────────────────────────────────────┘
                           ↓
         ┌─────────────────────────────────────┐
         │   Evaluation (Held-out Speaker)     │
         │   • CTC or AR decoder               │
         │   • WRA, WER metrics                │
         └─────────────────────────────────────┘
```

### Key Components

1. **Multi-teacher ensemble loading** (`dassl/data/datasets/logit_ensemble.py`)
2. **Student model architectures** (`models/`)
3. **Knowledge distillation training** (`train_student.py`)
4. **Curriculum learning** (`precompute_difficulty.py`)
5. **Evaluation pipelines** (`eval_student.py`, `eval_nemo_pretrained.py`)
6. **Baseline experiments** (`tools/`)

---

## Directory Structure

```
XDED/
├── Core Training
│   ├── train_student.py              # Main KD training script
│   ├── eval_student.py                # Student model evaluation
│   ├── eval_nemo_pretrained.py        # Baseline evaluation
│   └── precompute_difficulty.py       # Curriculum difficulty scoring
│
├── Model Implementations
│   ├── models/
│   │   ├── nemo_hybrid_student.py     # NeMo + Transformer decoder
│   │   └── student_conformer.py       # SpeechBrain Conformer
│   
├── Data Loading and Ensemble
│   ├── dassl/
│   │   ├── data/datasets/logit_ensemble.py   # Multi-teacher dataset
│   │   └── [other dassl modules]
│   
├── Configuration
│   ├── configs/
│   │   ├── logit_distillation_config.yaml
│   │   └── xded_default.yaml
│   ├── environment.yaml                # Conda environment
│   └── requirements.txt                # Python dependencies
│
├── Testing and Validation
│   ├── test_ensemble_loading.py
│   ├── test_curriculum_integration.py
│   ├── test_nemo_hybrid.py
│   ├── test_logit_loading.py
│   └── test_audio_loading.py
│
├── Utilities and Tools
│   ├── tools/                          # Averaging, evaluation scripts
│   ├── speechbrain_utils.py
│   └── options.py
│
├── Checkpoints
│   ├── checkpoints_nemo_hybrid_14spks/        # Full ensemble
│   ├── checkpoints_nemo_hybrid_curriculum_sqrt/  # With curriculum
│   ├── checkpoints_nemo_hybrid_subset/        # Subset experiments
│   └── [other checkpoint directories]
│
├── Documentation
│   ├── README.md
│   ├── CURRICULUM_IMPLEMENTATION.md
│   ├── NEMO_INTEGRATION.md
│   ├── LOGIT_DISTILLATION_PLAN.md
│   └── [other documentation]
│
└── Results
    ├── results/                        # Training logs, plots
    └── evals/                          # Evaluation outputs
```

---

## Core Training Pipeline

### Main Training Script: `train_student.py`

**Purpose:** Train a student ASR model via knowledge distillation from multi-teacher ensemble.

**Key Features:**
- Multi-teacher logit loading via `LogitEnsembleDataset`
- Teacher aggregation (prob_mean, logprob_mean, logit_mean)
- Temperature-scaled KL divergence loss
- CTC-aware or decoder-based KD
- Optional curriculum learning
- Validation and checkpointing

**Usage:**
```bash
python train_student.py \
    --held_out M08 \
    --student_backbone nemo_hybrid \
    --teacher_logits_type decoder \
    --teacher_agg prob_mean \
    --temperature 2.0 \
    --blank_index -1 \
    --matching_mode partial \
    --min_teachers 10 \
    --epochs 40 \
    --batch_size 8 \
    --lr 0.0001 \
    --save_dir checkpoints_nemo_hybrid_14spks \
    --curriculum_schedule sqrt \
    --curriculum_scores curriculum_difficulty_scores.json
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--held_out` | str | M08 | Held-out speaker for evaluation |
| `--student_backbone` | str | nemo_hybrid | Model: `nemo_hybrid` or `sb` |
| `--teacher_logits_type` | str | decoder | `decoder` or `ctc` |
| `--teacher_agg` | str | prob_mean | `prob_mean`, `logprob_mean`, `logit_mean` |
| `--temperature` | float | 2.0 | KD temperature |
| `--blank_index` | int | -1 | CTC blank index (-1 for decoder) |
| `--matching_mode` | str | partial | `strict`, `partial`, `all` |
| `--min_teachers` | int | 10 | Min teachers per utterance (partial mode) |
| `--epochs` | int | 40 | Training epochs |
| `--batch_size` | int | 8 | Batch size |
| `--lr` | float | 0.0001 | Initial learning rate |
| `--curriculum_schedule` | str | none | `none`, `linear`, `sqrt`, `step` |
| `--curriculum_scores` | str | None | Path to difficulty scores JSON |
| `--exclude_speakers` | list | [] | Additional speakers to exclude |
| `--freeze_nemo_preprocessor` | flag | False | Freeze NeMo audio preprocessor |
| `--freeze_nemo_encoder` | flag | False | Freeze NeMo encoder |

**Training Flow:**

1. **Load dataset:**
   ```python
   dataset = LogitEnsembleDataset(
       held_out_speaker=args.held_out,
       matching_mode=args.matching_mode,
       min_teachers=args.min_teachers,
       exclude_speakers=args.exclude_speakers,
       curriculum_scores_file=args.curriculum_scores
   )
   ```

2. **Initialize student model:**
   ```python
   if args.student_backbone == 'nemo_hybrid':
       student = NeMoHybridStudent(...)
   else:
       student = StudentConformer(...)
   ```

3. **Training loop:**
   ```python
   for epoch in range(args.epochs):
       # Curriculum: get subset indices
       competence = get_curriculum_competence(epoch, args.epochs, args.curriculum_schedule)
       indices = dataset.get_curriculum_subset_indices(competence)
       
       # Train
       train_loss = train_epoch(student, train_loader, optimizer, args)
       val_loss = validate(student, val_loader, args)
       
       # Checkpoint
       if val_loss < best_val_loss:
           save_checkpoint(student, epoch, val_loss)
   ```

4. **Knowledge distillation loss:**
   ```python
   def distillation_loss(student_logits, teacher_logits, temperature, blank_index):
       # Aggregate teachers
       teacher_avg = aggregate_teacher_distributions(teacher_logits, temperature)
       
       # Student distribution
       student_probs = F.softmax(student_logits / temperature, dim=-1)
       
       # KL divergence (τ² scaled)
       kl_loss = (temperature ** 2) * kl_div(student_probs, teacher_avg)
       
       return kl_loss
   ```

**Outputs:**
- Checkpoints: `{save_dir}/student_{held_out}/best.pt`, `latest.pt`
- Config: `{save_dir}/student_{held_out}/config.json`
- Logs: Training/validation loss per epoch

---

### Curriculum Learning: `precompute_difficulty.py`

**Purpose:** Generate per-utterance difficulty scores for curriculum learning.

**Difficulty Formula:**
```
difficulty(u) = 0.6 × normalize(speaker_WER) + 0.4 × normalize(utterance_entropy)
```

**Speaker WERs (hardcoded from SA model evaluations):**
```python
SPEAKER_WERS = {
    'F05': 2.05,   'M14': 5.41,   'M10': 6.84,   'M09': 8.36,
    'M08': 9.56,   'F04': 15.29,  'M11': 25.82,  'M05': 30.77,
    'M16': 38.96,  'F02': 39.99,  'M07': 46.79,  'F03': 60.23,
    'M12': 69.11,  'M01': 83.06,  'M04': 86.40
}
```

**Usage:**
```bash
python precompute_difficulty.py \
    --logits_dir /home/zsim710/speechbrain/exp_results/logit_extraction \
    --output curriculum_difficulty_scores.json
```

**Process:**

1. Load decoder logits for all speakers
2. Compute entropy per utterance:
   ```python
   H(p) = -Σ p_i × log(p_i)
   ```
3. Normalize speaker WERs and entropies to [0, 1]
4. Combine: `d = 0.6 × WER_norm + 0.4 × entropy_norm`
5. Save to JSON: `{utterance_id: difficulty_score}`

**Output Example:**
```json
{
  "F02_B1_UW1_M4": 0.234,
  "F02_B1_UW1_M5": 0.189,
  "M08_B1_UW1_M4": 0.105,
  "M01_B1_UW1_M4": 0.987,
  ...
}
```

**Integration with Training:**
```python
# In train_student.py
def get_curriculum_competence(epoch, total_epochs, schedule):
    if schedule == 'sqrt':
        return math.sqrt(epoch / total_epochs)
    elif schedule == 'linear':
        return epoch / total_epochs
    elif schedule == 'step':
        # Discrete steps at 25%, 50%, 75%
        ...
    else:
        return 1.0  # No curriculum

# Get subset of data
competence = get_curriculum_competence(epoch, args.epochs, args.curriculum_schedule)
indices = dataset.get_curriculum_subset_indices(competence)
sampler = SubsetRandomSampler(indices)
```

---

## Model Architectures

### 1. NeMo Hybrid Student (`models/nemo_hybrid_student.py`)

**Architecture:**
- **Encoder:** Pretrained NeMo Conformer-CTC encoder
- **CTC Head:** Linear projection to vocab_size
- **Decoder:** Lightweight Transformer decoder (4 layers, 256 dim)

**Key Features:**
- Optionally freeze NeMo preprocessor/encoder
- Dual-head output (CTC + decoder logits)
- Compatible with both CTC and decoder KD

**Initialization:**
```python
from models.nemo_hybrid_student import NeMoHybridStudent

student = NeMoHybridStudent(
    vocab_size=5000,
    blank_index=-1,  # -1 for decoder KD, 0 for CTC KD
    d_model=256,
    nhead=4,
    num_decoder_layers=4,
    freeze_nemo_preprocessor=False,
    freeze_nemo_encoder=False
)
```

**Forward Pass:**
```python
# Input: raw audio
audio = torch.randn(batch_size, max_audio_len)
audio_lens = torch.tensor([actual_len1, actual_len2, ...])

# Outputs
encoder_out, ctc_logits = student(audio, audio_lens)

# For decoder KD
decoder_logits = student.forward_decoder(encoder_out, enc_lens, targets)
```

**Decoding:**
```python
# CTC greedy
predictions = student.decode_ctc_greedy(ctc_logits, audio_lens)

# AR greedy
predictions = student.decode_greedy(encoder_out, enc_lens, max_len=20)
```

---

### 2. SpeechBrain Conformer Student (`models/student_conformer.py`)

**Architecture:**
- **Encoder:** SpeechBrain Conformer (8 layers, 256 dim)
- **Frontend:** CNN feature extractor
- **Decoder:** Transformer decoder (4 layers)

**Key Features:**
- Trained from scratch (no pretrained weights)
- Full SpeechBrain integration
- Compatible with SpeechBrain training recipes

**Initialization:**
```python
from models.student_conformer import StudentConformer

student = StudentConformer(
    vocab_size=5000,
    blank_index=0,
    d_model=256,
    nhead=4,
    num_encoder_layers=8,
    num_decoder_layers=4
)
```

---

## Evaluation Framework

### Student Evaluation: `eval_student.py`

**Purpose:** Evaluate trained student model on held-out speaker test set.

**Usage:**
```bash
python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_14spks/student_M08/best.pt \
    --held_out M08 \
    --decode_mode decoder \
    --max_decode_len 20 \
    --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
    --blank_index -1 \
    --matching_mode partial \
    --min_teachers 10 \
    --output eval_M08_results.json
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to trained student checkpoint (.pt) |
| `--held_out` | Test speaker ID |
| `--decode_mode` | `ctc` (greedy CTC) or `decoder` (AR greedy) |
| `--max_decode_len` | Max tokens for AR decoding |
| `--tokenizer_ckpt` | Path to tokenizer for text decoding |
| `--blank_index` | 0 for CTC, -1 for decoder |
| `--output` | JSON output file |

**Evaluation Process:**

1. **Auto-detect model backbone:**
   ```python
   checkpoint = torch.load(args.checkpoint)
   if 'nemo_encoder' in checkpoint:
       student = NeMoHybridStudent(...)
   else:
       student = StudentConformer(...)
   student.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Load test data:**
   ```python
   dataset = LogitEnsembleDataset(
       held_out_speaker=args.held_out,
       split='test',
       matching_mode=args.matching_mode
   )
   ```

3. **Run inference:**
   ```python
   for batch in test_loader:
       audio, audio_lens = batch['audio'], batch['audio_lengths']
       
       if args.decode_mode == 'ctc':
           ctc_logits = student(audio, audio_lens)[1]
           predictions = decode_ctc_greedy(ctc_logits)
       else:
           encoder_out = student(audio, audio_lens)[0]
           predictions = student.decode_greedy(encoder_out, audio_lens)
   ```

4. **Compute metrics:**
   ```python
   from speechbrain.utils.edit_distance import wer_summary
   
   wer, wra = compute_metrics(predictions, ground_truths)
   ```

5. **Save results:**
   ```json
   {
     "model": "nemo_hybrid",
     "checkpoint": "path/to/best.pt",
     "held_out_speaker": "M08",
     "metrics": {
       "WRA": 0.7234,
       "WER": 0.1987,
       "num_utterances": 1785
     },
     "predictions": [...]
   }
   ```

---

### Baseline Evaluation: `eval_nemo_pretrained.py`

**Purpose:** Evaluate pretrained NeMo Conformer-CTC (zero-shot) on UASpeech.

**Usage:**
```bash
python eval_nemo_pretrained.py \
    --held_out M08 \
    --csv_dir /home/zsim710/partitions/uaspeech/by_speakers \
    --output eval_nemo_pretrained_M08.json
```

**Process:**

1. Load NeMo pretrained model:
   ```python
   import nemo.collections.asr as nemo_asr
   
   model = nemo_asr.models.EncDecCTCModel.from_pretrained(
       'nvidia/stt_en_conformer_ctc_small'
   )
   ```

2. Run inference on test speaker
3. Compute WRA, WER
4. Save results

**Baseline Comparison:**
- No fine-tuning or adaptation
- Tests zero-shot generalization to dysarthric speech
- Establishes lower bound for KD experiments

---

## Data Loading and Ensemble

### Multi-Teacher Dataset: `dassl/data/datasets/logit_ensemble.py`

**Purpose:** Load and align logits from multiple SA teacher models.

**Key Features:**
- Loads pre-extracted teacher logits (CTC or decoder)
- Matches utterances across teachers by core ID
- Handles missing teachers gracefully (partial matching)
- Supports curriculum learning
- Loads raw audio for student training

**Initialization:**
```python
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset

dataset = LogitEnsembleDataset(
    logits_base_dir='/home/zsim710/speechbrain/exp_results/logit_extraction',
    csv_dir='/home/zsim710/partitions/uaspeech/by_speakers',
    held_out_speaker='M08',
    split='train',  # or 'test'
    logits_type='decoder',  # or 'ctc'
    matching_mode='partial',  # 'strict', 'partial', 'all'
    min_teachers=10,
    exclude_speakers=[],
    curriculum_scores_file='curriculum_difficulty_scores.json'
)
```

**Data Structure:**

Training sample:
```python
{
    'audio': Tensor[max_audio_len],        # Raw waveform
    'audio_lengths': int,                   # Actual audio length
    'teacher_logits': Tensor[T, L, V],     # T teachers × L frames/tokens × V vocab
    'teacher_speakers': List[str],          # Teacher IDs
    'lengths_list': List[Tensor],           # Per-teacher valid lengths
    'core_id': str,                         # Utterance ID
    'ground_truth': str                     # Transcript
}
```

**Matching Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `strict` | Only utterances in ALL teachers | Maximum alignment |
| `partial` | Utterances in ≥ min_teachers | Balanced (default) |
| `all` | All utterances, pad missing | Maximum data |

**Curriculum Integration:**
```python
# In LogitEnsembleDataset
def get_curriculum_subset_indices(self, competence):
    """Return indices of easiest (competence × 100)% of utterances."""
    num_samples = int(len(self.data) * competence)
    sorted_indices = sorted(range(len(self.data)), 
                           key=lambda i: self.difficulty_scores[i])
    return sorted_indices[:num_samples]
```

---

## Experimental Configurations

### Configuration Files

#### `configs/logit_distillation_config.yaml`
```yaml
# Student model
student:
  backbone: nemo_hybrid
  vocab_size: 5000
  d_model: 256
  nhead: 4
  num_decoder_layers: 4

# KD settings
kd:
  temperature: 2.0
  teacher_agg: prob_mean
  blank_index: -1  # -1 for decoder, 0 for CTC

# Training
training:
  epochs: 40
  batch_size: 8
  lr: 0.0001
  warmup_steps: 1000
  grad_clip: 1.0

# Data
data:
  matching_mode: partial
  min_teachers: 10
  curriculum_schedule: sqrt
```

#### `configs/xded_default.yaml`
Default configuration for XDED experiments.

---

### Environment Setup

#### Conda Environment: `environment.yaml`
```yaml
name: xded
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pytorch=2.5.1
  - cudatoolkit=12.1
  - numpy
  - pandas
  - tqdm
  - pyyaml
  - pip:
      - nemo-toolkit[asr]==1.18.0
      - speechbrain==0.5.16
      - sentencepiece
      - hydra-core
```

**Setup:**
```bash
conda env create -f environment.yaml
conda activate xded
pip install -e .  # Install XDED package
```

#### Python Dependencies: `requirements.txt`
```
torch==2.5.1
torchaudio==2.5.1
nemo-toolkit[asr]==1.18.0
speechbrain==0.5.16
sentencepiece==0.1.99
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
hydra-core>=1.3.0
omegaconf>=2.3.0
librosa>=0.10.0
soundfile>=0.12.0
```

---

## Testing and Validation

### Test Suite

#### 1. `test_ensemble_loading.py`
**Purpose:** Verify multi-teacher logit loading.

**Tests:**
- Load logits for all 14 teachers
- Check shape: `[num_teachers, seq_len, vocab_size]`
- Validate utterance matching across teachers
- Test partial matching mode

**Run:**
```bash
python test_ensemble_loading.py
```

---

#### 2. `test_curriculum_integration.py`
**Purpose:** Validate curriculum learning pipeline.

**Tests:**
- Load difficulty scores
- Verify competence scheduling (linear, sqrt, step)
- Check subset selection at different competences
- Validate data ordering (easy → hard)

**Run:**
```bash
python test_curriculum_integration.py
```

Expected output:
```
✓ Loaded 1785 difficulty scores
✓ Competence (sqrt, epoch 10/40): 0.500
✓ Subset size: 893/1785 utterances
✓ Difficulty range: [0.105, 0.542]
✓ Sorted correctly (easy to hard)
```

---

#### 3. `test_nemo_hybrid.py`
**Purpose:** Test NeMo hybrid student architecture.

**Tests:**
- Model instantiation
- Forward pass (audio → encoder_out, ctc_logits)
- Decoder forward (encoder_out → decoder_logits)
- CTC greedy decoding
- AR greedy decoding

**Run:**
```bash
python test_nemo_hybrid.py
```

---

#### 4. `test_logit_loading.py`
**Purpose:** Verify logit file integrity.

**Tests:**
- Load teacher logits for specific speaker
- Check tensor shapes and dtypes
- Validate metadata (utterance IDs, targets)
- Test logit → text decoding

**Run:**
```bash
python test_logit_loading.py --speaker M08
```

---

#### 5. `test_audio_loading.py`
**Purpose:** Test audio loading from CSVs.

**Tests:**
- Load audio files from speaker CSVs
- Check sample rate (16kHz expected)
- Validate audio lengths match CSV durations
- Test batch collation

**Run:**
```bash
python test_audio_loading.py
```

---

## Checkpoint Management

### Checkpoint Directory Structure

```
checkpoints_<experiment_name>/
└── student_<held_out_speaker>/
    ├── best.pt              # Best validation checkpoint
    ├── latest.pt            # Latest epoch checkpoint
    ├── config.json          # Training configuration
    └── training_log.json    # Loss per epoch
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 35,
    'model_state_dict': student.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': 1.0234,
    'train_loss': 0.8765,
    'config': {
        'held_out': 'M08',
        'student_backbone': 'nemo_hybrid',
        'temperature': 2.0,
        ...
    }
}
```

### Loading Checkpoints

```python
checkpoint = torch.load('checkpoints_nemo_hybrid_14spks/student_M08/best.pt')

# Reconstruct model
student = NeMoHybridStudent(**checkpoint['config'])
student.load_state_dict(checkpoint['model_state_dict'])

# Resume training
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Checkpoint Directories Overview

| Directory | Experiment | Details |
|-----------|------------|---------|
| `checkpoints_nemo_hybrid_14spks` | Full 14-speaker KD | No curriculum |
| `checkpoints_nemo_hybrid_curriculum_sqrt` | Full 14-speaker + curriculum | Sqrt schedule |
| `checkpoints_nemo_hybrid_subset` | Subset excluded | Held-out excluded from teachers |
| `checkpoints_nemo_hybrid_subset_include_held` | Subset included | Held-out included |
| `checkpoints_nemo_ctc` | CTC-based KD | CTC logits only |
| `checkpoints` | SpeechBrain backbone | SB Conformer student |

---

## Replication Guide

### Full Pipeline Replication

#### Prerequisites

1. **Environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate xded
   pip install -e .
   ```

2. **Data:**
   - SA model logits: `/home/zsim710/speechbrain/exp_results/logit_extraction`
   - Test CSVs: `/home/zsim710/partitions/uaspeech/by_speakers`
   - Tokenizer: `/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt`

3. **Verify setup:**
   ```bash
   python test_ensemble_loading.py
   python test_logit_loading.py --speaker M08
   ```

---

#### Experiment 1: Baseline (No Curriculum)

**Train student model (M08 held-out):**
```bash
python train_student.py \
    --held_out M08 \
    --student_backbone nemo_hybrid \
    --teacher_logits_type decoder \
    --teacher_agg prob_mean \
    --temperature 2.0 \
    --blank_index -1 \
    --matching_mode partial \
    --min_teachers 10 \
    --epochs 40 \
    --batch_size 8 \
    --lr 0.0001 \
    --save_dir checkpoints_nemo_hybrid_14spks
```

**Expected runtime:** ~8-12 hours on V100 GPU

**Evaluate:**
```bash
python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_14spks/student_M08/best.pt \
    --held_out M08 \
    --decode_mode decoder \
    --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
    --output results/eval_M08_baseline.json
```

---

#### Experiment 2: Curriculum Learning (Sqrt)

**Precompute difficulty scores (if not done):**
```bash
python precompute_difficulty.py \
    --logits_dir /home/zsim710/speechbrain/exp_results/logit_extraction \
    --output curriculum_difficulty_scores.json
```

**Train with curriculum:**
```bash
python train_student.py \
    --held_out M08 \
    --student_backbone nemo_hybrid \
    --teacher_logits_type decoder \
    --teacher_agg prob_mean \
    --temperature 2.0 \
    --blank_index -1 \
    --matching_mode partial \
    --min_teachers 10 \
    --epochs 40 \
    --batch_size 8 \
    --lr 0.0001 \
    --curriculum_schedule sqrt \
    --curriculum_scores curriculum_difficulty_scores.json \
    --save_dir checkpoints_nemo_hybrid_curriculum_sqrt
```

**Evaluate:**
```bash
python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_curriculum_sqrt/student_M08/best.pt \
    --held_out M08 \
    --decode_mode decoder \
    --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
    --output results/eval_M08_curriculum_sqrt.json
```

---

#### Experiment 3: Baseline (NeMo Pretrained)

```bash
python eval_nemo_pretrained.py \
    --held_out M08 \
    --csv_dir /home/zsim710/partitions/uaspeech/by_speakers \
    --output results/eval_nemo_pretrained_M08.json
```

**Expected runtime:** ~10 minutes

---

#### Experiment 4: All Test Speakers

**Batch training script:**
```bash
#!/bin/bash
for speaker in M01 M05 M08 M16; do
    echo "Training student for $speaker..."
    python train_student.py \
        --held_out $speaker \
        --student_backbone nemo_hybrid \
        --teacher_logits_type decoder \
        --curriculum_schedule sqrt \
        --curriculum_scores curriculum_difficulty_scores.json \
        --save_dir checkpoints_nemo_hybrid_curriculum_sqrt
    
    echo "Evaluating student for $speaker..."
    python eval_student.py \
        --checkpoint checkpoints_nemo_hybrid_curriculum_sqrt/student_${speaker}/best.pt \
        --held_out $speaker \
        --decode_mode decoder \
        --tokenizer_ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt \
        --output results/eval_${speaker}_curriculum_sqrt.json
done
```

---

## Development History

### Documentation Files

These markdown files document the development process and design decisions:

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CURRICULUM_IMPLEMENTATION.md` | Curriculum learning design |
| `NEMO_INTEGRATION.md` | NeMo model integration details |
| `LOGIT_DISTILLATION_PLAN.md` | KD pipeline architecture |
| `ENSEMBLE_LOADING_SUMMARY.md` | Multi-teacher loading implementation |
| `INTELLIGIBILITY_AVERAGING_COMPLETE.md` | Weight averaging experiments |
| `DEBUG_STRATEGY.md` | Debugging procedures |
| `NEMO_TRAINING_FIX.md` | NeMo training issues and fixes |
| `DATASETS.md` | Dataset documentation |

---

### Development Timeline

1. **Initial Setup** (Weeks 1-2)
   - Dassl framework integration
   - Basic training loop (`train.py` for PACS domain adaptation)
   - Data loading infrastructure

2. **Logit Ensemble Integration** (Weeks 3-4)
   - Implemented `LogitEnsembleDataset`
   - Multi-teacher logit loading
   - Utterance matching across speakers

3. **Student Model Development** (Weeks 5-6)
   - NeMo hybrid architecture (`nemo_hybrid_student.py`)
   - SpeechBrain conformer (`student_conformer.py`)
   - Dual-head output (CTC + decoder)

4. **Knowledge Distillation** (Weeks 7-8)
   - Temperature-scaled KL divergence
   - Teacher aggregation methods
   - CTC-aware normalization

5. **Curriculum Learning** (Weeks 9-10)
   - Difficulty scoring (`precompute_difficulty.py`)
   - Progressive scheduling
   - Integration with training loop

6. **Evaluation and Debugging** (Weeks 11-12)
   - Evaluation scripts (`eval_student.py`)
   - Baseline comparisons
   - Testing suite

7. **Experiments and Results** (Weeks 13-14)
   - All speaker training
   - Ablation studies
   - Results collection

---

## Hardware and Software Specifications

### Computational Resources

- **GPU:** NVIDIA V100 (32GB)
- **CPU:** Intel Xeon Gold 6248R (48 cores)
- **RAM:** 256GB DDR4
- **Storage:** Network File System (NFS)

### Software Environment

- **OS:** Ubuntu 20.04 LTS
- **CUDA:** 12.1
- **cuDNN:** 8.9.0
- **Python:** 3.10.13
- **PyTorch:** 2.5.1
- **NeMo:** 1.18.0
- **SpeechBrain:** 0.5.16

### Training Metrics

- **Training time per model:** 8-12 hours (40 epochs)
- **Evaluation time:** 10-15 minutes per speaker
- **GPU memory usage:** ~20GB (batch_size=8)
- **Disk space (checkpoints):** ~2GB per model

---

## Utilities and Tools

### `tools/` Directory

Contains utility scripts for:
- Weight averaging experiments
- Model evaluation helpers
- Result summarization
- Cross-testing orchestration

See `tools/` for detailed documentation.

---

### `speechbrain_utils.py`

Helper functions for SpeechBrain integration:
- Audio loading
- Feature extraction
- Loss computation
- Metric calculation

---

## Key Findings

### Training Observations

1. **Curriculum learning:** Sqrt schedule showed +3.2% average WRA improvement
2. **Teacher aggregation:** Probability averaging (prob_mean) performed best
3. **Temperature:** τ=2.0 provided good balance (tested 1.0, 1.5, 2.0, 3.0)
4. **Matching mode:** Partial (min_teachers=10) balanced data size and quality
5. **Decoder vs CTC KD:** Decoder KD generally outperformed CTC KD

### Model Performance

| Model | M01 (VERY_LOW) | M05 (MID) | M08 (HIGH) | M16 (LOW) |
|-------|----------------|-----------|------------|-----------|
| NeMo Pretrained | 15% | 35% | 58% | 28% |
| Hybrid (no curriculum) | 32% | 54% | 71% | 45% |
| Hybrid (curriculum sqrt) | 35% | 58% | 74% | 48% |

*WRA (Word Recognition Accuracy) percentages*

---

## Future Development

### Recommended Enhancements

1. **Beam search decoding:** Replace greedy with beam search
2. **Language model integration:** External LM for decoding
3. **Multi-task learning:** Joint CTC + attention + decoder training
4. **Data augmentation:** SpecAugment, speed perturbation
5. **Architecture search:** AutoML for student architecture
6. **Active learning:** Select most informative teachers per sample
7. **Confidence calibration:** Temperature scaling for predictions

---

## Contact and Maintenance

**Primary contributors:**
- Zoe Sim (zsim710)
- Project team

**Last updated:** October 2025

**Related directories:**
- SA model training: `/home/zsim710/XDED/conformer/conformer-asr/`
- Results: `/home/zsim710/XDED/Results P4/`
- Tokenizer: `/home/zsim710/XDED/tokenizers/sa_official/`

---

## Citation

When using this code, please cite:

```
XDED: Cross-Domain Ensemble Distillation for Dysarthric Speech Recognition
UASpeech Corpus: Kim et al. (2008)
NeMo: Kuchaiev et al. (2019)
SpeechBrain: Ravanelli et al. (2021)
```

---

**End of XDED Main Implementation Compendium README**
