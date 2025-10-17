# Curriculum Learning Implementation

## Overview
This document describes the curriculum learning implementation for the NeMoHybridStudent training pipeline. Curriculum learning trains the model progressively from easy to hard samples, improving generalization to unseen speakers.

## Components Implemented

### 1. precompute_difficulty.py
**Purpose**: Compute difficulty scores for each utterance based on speaker-level WER and utterance-level entropy.

**Location**: `/home/zsim710/XDED/XDED/precompute_difficulty.py`

**Key Features**:
- **Known Speaker WERs**: Hardcoded WER values for 14 speakers (F05: 2.05% to M04: 86.4%)
- **Utterance Entropy**: Computed from decoder logits [L, V] using: `entropy = -sum(p * log(p))`
- **Combined Score**: Weighted combination with alpha=0.6 (speaker) + beta=0.4 (utterance)
- **Output**: `curriculum_difficulty_scores.json` with per-utterance difficulty scores

**Formula**:
```
speaker_difficulty = speaker_wer / max_wer
utterance_difficulty = (entropy - min_entropy) / (max_entropy - min_entropy)
combined_score = 0.6 * speaker_difficulty + 0.4 * utterance_difficulty
```

**Usage**:
```bash
python /home/zsim710/XDED/XDED/precompute_difficulty.py
```

### 2. LogitEnsembleDataset (Modified)
**Purpose**: Dataset class extended to support curriculum learning.

**Location**: `/home/zsim710/XDED/XDED/dassl/data/datasets/logit_ensemble.py`

**New Parameters**:
- `curriculum_scores_file` (str, optional): Path to JSON file with difficulty scores

**New Methods**:
- `_load_curriculum_scores()`: Load difficulty scores from JSON
- `_create_curriculum_order()`: Sort utterances by difficulty (easy → hard)
- `get_curriculum_subset_indices(competence)`: Get indices for current competence level

**Key Features**:
- Automatically matches utterance IDs with or without speaker prefix
- Handles missing scores gracefully (defaults to 0.5)
- Prints statistics: easiest/hardest scores, mean, median
- Returns sorted indices for curriculum sampling

**Usage**:
```python
dataset = LogitEnsembleDataset(
    held_out_speaker="M08",
    split="train",
    curriculum_scores_file="curriculum_difficulty_scores.json"
)

# Get subset for current competence (0.0 = easiest only, 1.0 = all)
subset_indices = dataset.get_curriculum_subset_indices(competence=0.5)
```

### 3. train_student.py (To Be Modified)
**Status**: Pending implementation

**Required Changes**:
1. Add curriculum arguments:
   - `--curriculum_schedule`: Choice of ['none', 'linear', 'sqrt', 'step']
   - `--curriculum_scores`: Path to difficulty scores JSON
   
2. Add competence scheduler:
```python
def get_curriculum_competence(epoch, total_epochs, schedule='sqrt'):
    if schedule == 'none':
        return 1.0
    elif schedule == 'linear':
        return epoch / total_epochs
    elif schedule == 'sqrt':
        return (epoch / total_epochs) ** 0.5
    elif schedule == 'step':
        if epoch < total_epochs * 0.25:
            return 0.3
        elif epoch < total_epochs * 0.5:
            return 0.6
        elif epoch < total_epochs * 0.75:
            return 0.8
        else:
            return 1.0
```

3. Modify training loop to use curriculum sampling:
```python
# At the start of each epoch
competence = get_curriculum_competence(epoch, num_epochs, args.curriculum_schedule)
curriculum_indices = train_dataset.get_curriculum_subset_indices(competence)
train_sampler = SubsetRandomSampler(curriculum_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
```

## Curriculum Schedules

### Linear Schedule
- Competence = epoch / total_epochs
- Smoothly increases from 0 to 1
- Gradual increase in dataset size

### Square Root Schedule  
- Competence = sqrt(epoch / total_epochs)
- Faster initial growth, slower later
- Good for quick adaptation to easy samples

### Step Schedule
- 0-25% epochs: 30% of data (easiest)
- 25-50% epochs: 60% of data
- 50-75% epochs: 80% of data
- 75-100% epochs: 100% of data (all)

### None (Baseline)
- Competence = 1.0 always
- Uses all data from start (no curriculum)

## Difficulty Score Ranges

Based on known speaker WERs:
- **Easiest**: F05 (2.05% WER) - HIGH intelligibility
- **Mid-Easy**: M09 (3.35% WER) - HIGH intelligibility
- **Mid-Hard**: M05 (32.6% WER) - MID intelligibility
- **Hardest**: M04 (86.4% WER) - VERY_LOW intelligibility

Utterance entropy adds fine-grained difficulty within each speaker.

## Training Workflow

### Step 1: Precompute Difficulty Scores
```bash
cd /home/zsim710/XDED/XDED
python precompute_difficulty.py
```

**Expected Output**:
- `curriculum_difficulty_scores.json` with utterance scores
- Statistics: min/max/mean entropy and difficulty scores
- Example easiest/hardest utterances

### Step 2: Train with Curriculum Learning
```bash
CUDA_VISIBLE_DEVICES=1 python train_student.py \
    --student_type nemo_hybrid \
    --teacher_logits_type decoder \
    --held_out_speaker M08 \
    --num_epochs 40 \
    --batch_size 8 \
    --learning_rate 0.0002 \
    --temperature 2.0 \
    --teacher_agg logprob_mean \
    --curriculum_schedule sqrt \
    --curriculum_scores curriculum_difficulty_scores.json \
    --checkpoint_dir checkpoints_nemo_hybrid_14spks_curriculum/student_M08/
```

### Step 3: Evaluate
```bash
CUDA_VISIBLE_DEVICES=1 python eval_student.py \
    --checkpoint checkpoints_nemo_hybrid_14spks_curriculum/student_M08/best_model.pt \
    --held_out_speaker M08 \
    --output_file results_nemo_curriculum_M08.json \
    --spm_model /home/zsim710/XDED/tokenizers/sa_official/tokenizer
```

## Expected Improvements

Based on curriculum learning research:
1. **Better generalization**: Improved WER on unseen M08 speaker
2. **Faster convergence**: Lower validation loss in early epochs
3. **Stable training**: Reduced variance in loss curves
4. **Better feature learning**: More robust representations from structured learning

## File Structure

```
XDED/
├── precompute_difficulty.py          # Computes difficulty scores
├── curriculum_difficulty_scores.json # Output from precompute
├── train_student.py                  # Training script (needs modification)
├── eval_student.py                   # Evaluation script
├── CURRICULUM_IMPLEMENTATION.md      # This file
└── dassl/data/datasets/
    └── logit_ensemble.py             # Modified dataset class
```

## Next Steps

1. ✅ Created `precompute_difficulty.py` with hardcoded speaker WERs
2. ✅ Modified `LogitEnsembleDataset` to support curriculum learning
3. ⏳ **TODO**: Run `precompute_difficulty.py` to generate scores
4. ⏳ **TODO**: Modify `train_student.py` to integrate curriculum scheduler
5. ⏳ **TODO**: Train new model with curriculum learning
6. ⏳ **TODO**: Compare results with baseline (no curriculum)

## References

- Bengio et al. (2009): "Curriculum Learning"
- Speaker WERs from UASpeech evaluation results
- Entropy-based difficulty from Xu et al. (2020): "Curriculum Learning for Speech Recognition"
