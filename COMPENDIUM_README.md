# XDED Research Compendium: Cross-Domain Ensemble Distillation for Dysarthric Speech Recognition

**Project Duration:** March 2025 - November 2025
**Institution:** University of Auckland 

**Team Members:**
- **Zeno** (zsim710): Curriculum learning, baseline experiments (co-lead)
- **Shaurya**: Hybrid model development, baseline experiments (co-lead)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Compendium Overview](#research-compendium-overview)
3. [Project Structure and Navigation](#project-structure-and-navigation)
4. [Quick Start Guide](#quick-start-guide)
5. [Work Distribution](#work-distribution)
6. [Hardware and Software Specifications](#hardware-and-software-specifications)
7. [Complete Replication Guide](#complete-replication-guide)
8. [Key Findings and Results](#key-findings-and-results)
9. [Documentation Index](#documentation-index)
10. [Future Work and Recommendations](#future-work-and-recommendations)
11. [Acknowledgments](#acknowledgments)

---

## Executive Summary

### Project Overview

This research compendium documents a comprehensive investigation into **Cross-Domain Ensemble Distillation (XDED)** for improving automatic speech recognition (ASR) of dysarthric speech. Dysarthria—a motor speech disorder affecting articulation—poses significant challenges for standard ASR systems trained on typical speech patterns.

### Research Problem

Standard ASR models pretrained on neurotypical speech (e.g., LibriSpeech) exhibit severe performance degradation on dysarthric speech, with word error rates (WER) often exceeding 80% on low-intelligibility speakers. Single-speaker adaptive models, while effective for individual speakers, fail to generalize across the heterogeneous dysarthric population.

### Our Approach

We developed **XDED**, a multi-teacher knowledge distillation framework that:

1. **Trains 15 speaker-adaptive (SA) teacher models**, each specialized for one speaker in the UASpeech corpus
2. **Extracts soft probability distributions** (logits) from 14 teachers (excluding held-out target speaker)
3. **Ensembles teacher knowledge** via probability averaging at temperature τ=2.0
4. **Trains a student model** to learn generalized representations from the ensemble
5. **Incorporates curriculum learning** to progressively introduce harder examples based on speaker intelligibility and utterance complexity

### Key Contributions

- **Curriculum learning integration** with composite difficulty scoring (speaker WER + utterance entropy)
- **Comprehensive evaluation** across 7 model architectures and 210 cross-testing combinations
- **Reproducible research artifacts**: trained models, extracted logits, evaluation scripts, and detailed documentation

### Impact

This work demonstrates that knowledge distillation from speaker-specific experts can produce robust, generalizable ASR models for dysarthric speech, with potential applications in assistive technologies for individuals with motor speech disorders.

---

## Research Compendium Overview

### Purpose

This compendium provides **complete documentation** for replicating all experiments, analyses, and findings from the XDED project. It contains:

- ✅ **Source code** for all training, evaluation, and analysis pipelines
- ✅ **Trained model checkpoints** (15 SA teachers + 8 student model variants)
- ✅ **Extracted teacher logits** (~240,000 utterances × 15 speakers)
- ✅ **Experimental results** (JSON files with WER/WRA metrics for all 210+ experiments)
- ✅ **Test data partitions** (standardized CSV files for all 15 speakers)
- ✅ **Shared tokenizer** (5000-token SentencePiece vocabulary)
- ✅ **Configuration files** (YAML hyperparameters for all experiments)
- ✅ **Verification scripts** (token-level logit validation)
- ✅ **Documentation** (4 detailed READMEs + this root compendium)

### Compliance with Research Standards

This compendium addresses all requirements specified in the project brief:

| Requirement | Location in Compendium |
|-------------|------------------------|
| **Detailed experimental procedures** | `XDED/COMPENDIUM_README.md` (training pipeline) |
| **Equipment specifications** | [Hardware Specifications](#hardware-and-software-specifications) |
| **Test conditions and setup** | `conformer/.../transformer/COMPENDIUM_README.md` (SA training) |
| **Data used for analysis** | `Results P4/COMPENDIUM_README.md` (all evaluation outputs) |
| **Replication instructions** | All 4 READMEs + [Complete Replication Guide](#complete-replication-guide) |
| **Specification sheets** | Model architectures in `XDED/COMPENDIUM_README.md` |
| **Individual contributions** | [Work Distribution](#work-distribution) section |
| **ReadMe with structure** | This document |

---

## Project Structure and Navigation

### Directory Organization

The XDED project is organized into **4 major components**, each with its own detailed compendium README:

```
XDED/
│
├── COMPENDIUM_README.md                          # ← THIS FILE (Root compendium)
│
├── 1️⃣  conformer/conformer-asr/recipes/LibriSpeech/ASR/transformer/
│   ├── COMPENDIUM_README.md                      # Teacher SA model training & logit extraction
│   ├── train.py                                  # SA model training script
│   ├── extract_sa_logits.py                      # Logit extraction from SA models
│   ├── verify_logits.py                          # Token-level verification
│   ├── run_token_verification.py                 # Batch verification orchestration
│   ├── run_all_sa_models.py                      # Cross-testing matrix (210 combinations)
│   ├── hparams/exp/uaspeech/                     # 15 speaker-specific YAML configs
│   └── scripts/                                  # Batch training scripts
│
├── 2️⃣  tokenizers/sa_official/
│   ├── COMPENDIUM_README.md                      # Shared tokenizer documentation
│   ├── tokenizer.ckpt                            # SpeechBrain tokenizer checkpoint
│   ├── tokenizer                                 # Raw SentencePiece model
│   └── hyperparams.yaml                          # Tokenizer configuration
│
├── 3️⃣  XDED/
│   ├── COMPENDIUM_README.md                      # Core XDED implementation (student training)
│   ├── train_student.py                          # Multi-teacher KD training
│   ├── eval_student.py                           # Student model evaluation
│   ├── eval_nemo_pretrained.py                   # Baseline evaluation
│   ├── precompute_difficulty.py                  # Curriculum difficulty scoring
│   ├── models/
│   │   ├── nemo_hybrid_student.py                # NeMo encoder + Transformer decoder
│   │   └── student_conformer.py                  # SpeechBrain Conformer student
│   ├── dassl/data/datasets/logit_ensemble.py     # Multi-teacher dataset loader
│   ├── configs/                                  # Experiment configurations
│   ├── checkpoints_*/                            # Trained student models (8 variants)
│   ├── tools/                                    # Averaging, evaluation utilities
│   └── tests/                                    # Unit tests (5 validation scripts)
│
├── 4️⃣  Results P4/
│   ├── COMPENDIUM_README.md                      # All experimental results
│   ├── nemo_pretrained(baseline results)/        # Zero-shot pretrained baseline
│   ├── Hybrid Model/                             # KD student (14 teachers)
│   ├── Hybrid Model subset/                      # KD student (subset/excluded teachers)
│   ├── curriculum learning/                      # Curriculum learning experiments
│   ├── Averaged Weight Model/                    # SA weight averaging baseline
│   ├── Scratch Model/                            # SpeechBrain student from scratch
│   ├── cross testing/                            # 210 SA cross-testing combinations
│   └── by_speakers_testcsv/                      # Standardized test partitions (15 speakers)
│
└── speechbrain/exp_results/logit_extraction/     # Extracted teacher logits (all 15 speakers)
    ├── F02/
    │   ├── F02_ctc_logits.pt                     # CTC frame-level logits
    │   ├── F02_decoder_logits.pt                 # Decoder token-level logits
    │   └── F02_metadata.json                     # Utterance IDs, ground truth, lengths
    ├── F03/ ... F05/
    └── M01/ ... M16/
```

### Navigation Guide

**👉 Start here based on your goal:**

| Goal | Navigate to |
|------|-------------|
| **Understand overall methodology** | This README (Executive Summary, Replication Guide) |
| **Train SA teacher models** | `conformer/.../transformer/COMPENDIUM_README.md` |
| **Extract teacher logits** | `conformer/.../transformer/COMPENDIUM_README.md` (Logit Extraction) |
| **Understand tokenizer** | `tokenizers/sa_official/COMPENDIUM_README.md` |
| **Train student models via KD** | `XDED/COMPENDIUM_README.md` (Core Training Pipeline) |
| **Implement curriculum learning** | `XDED/COMPENDIUM_README.md` (Curriculum Learning) |
| **Evaluate models** | `XDED/COMPENDIUM_README.md` (Evaluation Framework) |
| **Analyze experimental results** | `Results P4/COMPENDIUM_README.md` |
| **View cross-testing matrix** | `Results P4/COMPENDIUM_README.md` (Cross-Testing Results) |
| **Replicate specific experiments** | [Complete Replication Guide](#complete-replication-guide) below |
| **Understand work split** | [Work Distribution](#work-distribution) below |

---

## Quick Start Guide

### Prerequisites

Before replicating any experiments, ensure you have:

1. **Hardware:**
   - NVIDIA GPU with ≥16GB VRAM (V100 32GB recommended)
   - 64GB+ system RAM
   - 500GB+ free disk space (for checkpoints and logits)

2. **Software:**
   - CUDA 12.1+
   - Python 3.9+
   - Conda or virtualenv

3. **Data:**
   - UASpeech corpus audio files (available via request from [UASpeech website](http://www.isle.illinois.edu/sst/data/UASpeech/))
   - Audio files placed in `/mnt/data/uaspeech/` (or update paths in configs)

### Installation (5 minutes)

```bash
# 1. Clone repository
cd /home/zsim710
git clone <repository_url> XDED
cd XDED

# 2. Create conda environment
conda env create -f XDED/environment.yaml
conda activate xded

# 3. Install SpeechBrain (for SA models)
cd conformer/conformer-asr
pip install -e .

# 4. Install additional dependencies
cd ../../XDED
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; import nemo.collections.asr as nemo_asr; import speechbrain; print('✅ All dependencies installed')"
```

### Minimal Working Example (10 minutes)

Test the complete pipeline on a single speaker:

```bash
# Navigate to XDED directory
cd /home/zsim710/XDED/XDED

# 1. Train a small student model (1 epoch, for testing)
python train_student.py \
    --held_out M08 \
    --student_backbone nemo_hybrid \
    --teacher_logits_type decoder \
    --teacher_agg prob_mean \
    --temperature 2.0 \
    --epochs 1 \
    --batch_size 4 \
    --save_dir test_checkpoint

# 2. Evaluate the model
python eval_student.py \
    --checkpoint test_checkpoint/student_M08/latest.pt \
    --held_out M08 \
    --decode_mode ctc \
    --tokenizer_ckpt ../tokenizers/sa_official/tokenizer.ckpt \
    --output test_eval_M08.json

# 3. View results
cat test_eval_M08.json
```

**Expected output:**
```json
{
  "model": "nemo_hybrid",
  "held_out_speaker": "M08",
  "wer": 0.3456,
  "wra": 0.6544,
  "num_utterances": 1785
}
```

---

## Work Distribution

### Individual Contributions

This section clearly identifies the work completed by each team member to satisfy the research compendium requirement for **individual contribution tracking**.

#### Joint Work (Both Zeno & Shaurya)

**1. Data Preparation and Partitioning**
- **Task:** Created standardized test CSV files for all 15 UASpeech speakers
- **Location:** `Results P4/by_speakers_testcsv/`
- **Contribution split:** 50/50 (collaborated on format, each created CSVs for ~7-8 speakers)
- **Files created:**
  - Zeno: `M01.csv`, `M04.csv`, `M05.csv`, `M07.csv`, `M08.csv`, `M09.csv`, `M10.csv`, `M11.csv`
  - Shaurya: `M12.csv`, `M14.csv`, `M16.csv`, `F02.csv`, `F03.csv`, `F04.csv`, `F05.csv`
  - Joint: `test.csv`, `train.csv` (combined files)

**2. Baseline Experiments (Parallel Execution)**

All baseline experiments were conceptualized jointly but executed in parallel by splitting test speakers:

| Experiment | Description | Zeno's Speakers | Shaurya's Speakers | Scripts Used |
|------------|-------------|-----------------|-------------------|--------------|
| **NeMo Pretrained Baseline** | Zero-shot evaluation of pretrained NeMo model | M08 (HIGH), M01 (VERY_LOW) | M05 (MID), M16 (LOW) | `eval_nemo_pretrained.py` |
| **Cross-Testing Matrix** | Test all 15 SA models on all other speakers (210 combinations) | Cross-tests for F02-F05, M01, M04, M05, M07 | Cross-tests for M08-M16 | `run_all_sa_models.py` |
| **Averaged Weight Model** | Weight averaging within intelligibility bands | M01 (VERY_LOW band) | M08 (HIGH band) | `tools/average_sa_models_intelligibility.py` |
| **Scratch Model** | SpeechBrain Conformer trained from scratch | M08, M05 | M16 | `train_student.py` (backbone=sb) |

**Results locations:**
- NeMo baseline: `Results P4/nemo_pretrained(baseline results)/`
- Cross-testing: `Results P4/cross testing/` (Zeno: F02-M07 folders, Shaurya: M08-M16 folders)
- Averaged weights: `Results P4/Averaged Weight Model/` (Zeno: M01 results, Shaurya: M08 results)
- Scratch model: `Results P4/Scratch Model/` (Zeno: M08/M05 results, Shaurya: M16 results)

**Logit Verification Pipeline**
- **Task:** Develop comprehensive verification framework for extracted teacher logits
- **Location:** `conformer/.../transformer/verify_logits.py`, `run_token_verification.py`
- **Key contributions:**
  - Token-level sequence comparison (argmax decoding)
  - Text-level matching with tokenizer
  - Batch verification across all 15 SA models
  - Quality assessment metrics (Excellent ≥95%, Good ≥80%)
- **Output:** Verification reports confirming 85-95% exact token matches

**Multi-Teacher Ensemble Dataset**
- **Task:** Implement flexible dataset loader for multi-teacher logits
- **Location:** `XDED/dassl/data/datasets/logit_ensemble.py`
- **Key features:**
  - Utterance matching across 14 teachers (3 modes: strict/partial/all)
  - Handles missing teachers gracefully
  - Integrated curriculum subset sampling
  - Loads raw audio + teacher logits + metadata in parallel
- **Matching modes:**
  - `strict`: Only utterances in ALL teachers (~12,000 utterances)
  - `partial`: Utterances in ≥10 teachers (~16,000 utterances) ← **default**
  - `all`: All utterances, pad missing teachers (~18,000 utterances)

**NeMo Integration and Compatibility**
- **Task:** Resolve NeMo/SpeechBrain integration challenges
- **Documentation:** `XDED/NEMO_INTEGRATION.md`
- **Challenges addressed:**
  - NeMo tokenizer format conversion
  - Audio preprocessing pipeline differences
  - Checkpoint loading/saving compatibility
  - CTC decoding with NeMo's GreedyCTCInfer

---

#### Shaurya's Individual Work

**3. Hybrid Student Model Development**
- **Task:** Implement and train NeMo Hybrid student architecture (NeMo encoder + Transformer decoder)
- **Location:** `XDED/models/nemo_hybrid_student.py`
- **Key contributions:**
  - Designed dual-head architecture (CTC + decoder outputs)
  - Implemented optional encoder/preprocessor freezing
  - Integrated with multi-teacher logit ensemble
  - Trained models for all 4 test speakers (M01, M05, M08, M16)
- **Results:** `Results P4/Hybrid Model/`
- **Training script:** `train_student.py` (with `--student_backbone nemo_hybrid`)

**4. Hybrid Model Subset Experiments**
- **Task:** Compare performance when including vs excluding target speaker from teacher ensemble
- **Location:** `Results P4/Hybrid Model subset/`
- **Experiments:**
  - **Excluded:** Train with 13 teachers (true leave-one-out scenario)
  - **Included:** Train with 14 teachers including target speaker (upper bound)
- **Speakers tested:** M01, M05, M08, M16
- **Key insight:** Including target speaker improved WER by ~15-20%, confirming benefit of domain-matched teachers
---

#### Zeno's Individual Work

**5. Curriculum Learning Implementation**
- **Task:** Implement curriculum learning framework with composite difficulty scoring
- **Location:** `XDED/precompute_difficulty.py`, curriculum integration in `train_student.py`
- **Key contributions:**
  - **Difficulty metric design:** `d = 0.6 × speaker_WER + 0.4 × utterance_entropy`
  - **Schedule implementations:** Linear, sqrt, step competence schedules
  - **Dataset integration:** Modified `LogitEnsembleDataset` for curriculum subset sampling
  - Trained curriculum models for all 4 test speakers
- **Results:** `Results P4/curriculum learning/14spks/`
- **Documentation:** `XDED/CURRICULUM_IMPLEMENTATION.md`

**6. Curriculum Learning Experiments**
- **Task:** Systematic evaluation of curriculum schedules (none, linear, sqrt, step)
- **Speakers tested:** M01, M05, M08, M16
- **Key findings:**
  - Sqrt schedule showed 5-10% relative WER improvement over no curriculum
  - Linear schedule too gradual for 40-epoch training
  - Step schedule introduced instability at transition points
- **Recommendation:** Sqrt schedule for optimal balance

---

### Work Timeline

| Week | Zeno | Shaurya | Joint |
|------|------|---------|-------|
| 1-2 | SA model training (7 speakers) | SA model training (8 speakers) | Project planning, data preparation |
| 3-4 | Logit extraction, verification pipeline | Logit extraction | Baseline design |
| 5-6 | Cross-testing (F02-M07), averaged weights (M01) | Cross-testing (M08-M16), averaged weights (M08) | Result analysis |
| 7-8 | Curriculum difficulty scoring, schedule design | Hybrid model architecture, NeMo integration | - |
| 9-10 | Curriculum training (all schedules) | Hybrid model training (14 teachers) | - |
| 11-12 | Curriculum evaluation, subset experiments | Hybrid subset experiments (included/excluded) | - |
| 13-14 | Documentation (curriculum, verification) | Documentation (NeMo integration, hybrid model) | Final report, compendium assembly |

---

## Hardware and Software Specifications

### Software Stack

**Operating System:**
- Ubuntu 20.04.6 LTS (Focal Fossa)
- Linux kernel 5.4.0-150-generic

**Deep Learning Frameworks:**
- **PyTorch:** 2.5.1+cu121
- **CUDA:** 12.1.0
- **cuDNN:** 8.9.2

**ASR Frameworks:**
- **NeMo:** 1.18.0 (NVIDIA toolkit)
- **SpeechBrain:** 0.5.16
- **Transformers:** 4.38.2 (HuggingFace)

**Audio Processing:**
- **torchaudio:** 2.5.1
- **librosa:** 0.10.1
- **soundfile:** 0.12.1

**Tokenization:**
- **sentencepiece:** 0.2.0
- **SpeechBrain CTCTextEncoder** (wrapper around SentencePiece)

**Data Processing:**
- **numpy:** 1.24.3
- **pandas:** 2.0.3
- **scipy:** 1.11.1

**Visualization and Analysis:**
- **matplotlib:** 3.7.2
- **seaborn:** 0.12.2
- **tensorboard:** 2.13.0

**Development Tools:**
- **Python:** 3.9.18
- **conda:** 23.7.2
- **git:** 2.25.1

**Full dependency list:** See `XDED/requirements.txt` and `XDED/environment.yaml`

---
## Documentation Index

### Primary Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Root Compendium (this file)** | `COMPENDIUM_README.md` | Overall project overview, replication guide |
| **SA Training & Logit Extraction** | `conformer/.../transformer/COMPENDIUM_README.md` | Teacher model training, logit extraction, verification |
| **Tokenizer Specification** | `tokenizers/sa_official/COMPENDIUM_README.md` | Shared 5000-token vocabulary documentation |
| **Core XDED Implementation** | `XDED/COMPENDIUM_README.md` | Student training, evaluation, curriculum learning |
| **Experimental Results** | `Results P4/COMPENDIUM_README.md` | All evaluation outputs, cross-testing, performance metrics |

### Supplementary Documentation

| Document | Location | Topic |
|----------|----------|-------|
| `XDED/CURRICULUM_IMPLEMENTATION.md` | XDED/ | Curriculum learning design and schedules |
| `XDED/NEMO_INTEGRATION.md` | XDED/ | NeMo/SpeechBrain integration challenges |
| `XDED/LOGIT_DISTILLATION_PLAN.md` | XDED/ | Original KD design document |
| `XDED/M08_FOLD_PLAN.md` | XDED/ | Leave-one-out cross-validation strategy |
| `XDED/PROGRESS_SUMMARY.md` | XDED/ | Development timeline and milestones |
| `XDED/DATASETS.md` | XDED/ | UASpeech corpus description |
| `XDED/README.md` | XDED/ | Quick start guide for main implementation |

### Code Documentation

All Python scripts contain docstrings and inline comments. Key modules:

- **Training:** `train_student.py` (extensive argparse documentation)
- **Evaluation:** `eval_student.py`, `eval_nemo_pretrained.py`
- **Data loading:** `dassl/data/datasets/logit_ensemble.py`
- **Models:** `models/nemo_hybrid_student.py`, `models/student_conformer.py`
- **Curriculum:** `precompute_difficulty.py`

---

## Future Work and Recommendations

### Immediate Extensions

1. **Beam search decoding:**
   - Current: Greedy decoding only
   - Implement: Beam search with language model shallow fusion
   - Expected gain: 5-10% relative WER improvement

2. **Additional test speakers:**
   - Current: 4 test speakers (M08, M05, M16, M01)
   - Expand to: All 15 speakers (full leave-one-out evaluation)
   - Computational cost: ~3× current evaluation time

3. **Longer decode length:**
   - Current: max_decode_len=1 (single words)
   - Test: max_decode_len=5-10 (for multi-word utterances in UASpeech)

4. **Teacher aggregation ablation:**
   - Current: prob_mean (default)
   - Systematically compare: logprob_mean, logit_mean
   - Expected: logprob_mean may reduce entropy, improve calibration

---

### Research Directions

5. **Adaptive temperature scheduling:**
   - Current: Fixed τ=2.0 throughout training
   - Explore: Anneal τ from 3.0 → 1.0 over epochs
   - Hypothesis: High τ early (smooth distributions), low τ late (sharper targets)

6. **Teacher weighting:**
   - Current: Equal weights (1/K) for all teachers
   - Implement: Weight teachers by intelligibility similarity to target speaker
   - Formula: `w_k ∝ exp(-|WER_k - WER_target|)`

7. **Multi-task learning:**
   - Current: KD-only (no ASR loss)
   - Add: Ground-truth ASR loss with α=0.1-0.3 weighting
   - Expected: May improve calibration on easy examples

8. **Intelligibility-aware architecture:**
   - Current: Single student model for all speakers
   - Propose: Multi-head decoder with speaker intelligibility embedding
   - Hypothesis: Separate output paths for HIGH vs VERY_LOW may help

9. **Data augmentation:**
   - Integrate: SpecAugment, speed perturbation, noise injection
   - Expected: Improve robustness, reduce overfitting

10. **External language model:**
    - Integrate: GPT-style LM for shallow fusion during decoding
    - Expected: Significant WER gains on VERY_LOW speakers (contextual repair)

---

### Practical Applications

11. **Real-time deployment:**
    - Optimize: Model quantization (FP16 or INT8), TensorRT compilation
    - Target: <100ms latency on embedded GPU (Jetson)

12. **User adaptation:**
    - Implement: Few-shot fine-tuning on 50-100 utterances from new user
    - Expected: Personalized model for individual speaker characteristics

13. **Multi-modal integration:**
    - Extend: Add visual cues (lip movements) for severe dysarthria cases
    - Dataset: AV-UASpeech (if available)

---

## Acknowledgments

### Foundational Research

This project builds upon two key research contributions:

1. **Cross-Domain Ensemble Distillation (XDED) Framework:**
   - Lee, K., Kim, S., & Kwak, S. (2022). *Cross-Domain Ensemble Distillation for Domain Generalization*. Proceedings of European Conference on Computer Vision (ECCV).
   - Repository: [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) - Domain adaptation/generalization library
   - Our adaptation: Applied XDED methodology to dysarthric speech recognition with multi-teacher knowledge distillation

2. **SpeechBrain Toolkit:**
   - Ravanelli, M., Parcollet, T., Plantinga, P., et al. (2021). *SpeechBrain: A General-Purpose Speech Toolkit*. arXiv:2106.04624.
   - Repository: [SpeechBrain](https://github.com/speechbrain/speechbrain)
   - Our usage: Speaker-adaptive ASR model training, tokenization, and audio preprocessing

### Data and Resources

- **UASpeech corpus:** University of Illinois at Urbana-Champaign Speech and Language Engineering Group
- **Pretrained models:**
  - NeMo Conformer-CTC (`nvidia/stt_en_conformer_ctc_small`): NVIDIA NeMo Toolkit
  - SpeechBrain Transformer ASR: SpeechBrain project

### Open-Source Tools

- **PyTorch:** Meta AI Research
- **NeMo:** NVIDIA Corporation
- **SentencePiece:** Google Research


### Technical Mentorship

- **Project supervisor:** Dr Reza Shahamiri, Dr Satwinder Singh
- **Technical advisor:** Ben Wang



---

## Appendix: File Manifest

### Critical Files for Replication

**Training scripts:**
- `conformer/.../transformer/train.py` - SA model training
- `XDED/train_student.py` - Student KD training
- `XDED/precompute_difficulty.py` - Curriculum difficulty

**Evaluation scripts:**
- `XDED/eval_student.py` - Student evaluation
- `XDED/eval_nemo_pretrained.py` - Baseline evaluation
- `conformer/.../transformer/run_all_sa_models.py` - Cross-testing

**Verification scripts:**
- `conformer/.../transformer/verify_logits.py` - Logit verification
- `conformer/.../transformer/run_token_verification.py` - Batch verification
- `XDED/test_ensemble_loading.py` - Ensemble loading test

**Model architectures:**
- `XDED/models/nemo_hybrid_student.py` - Hybrid student
- `XDED/models/student_conformer.py` - SB student

**Data loading:**
- `XDED/dassl/data/datasets/logit_ensemble.py` - Multi-teacher dataset

**Configuration:**
- `XDED/environment.yaml` - Conda environment
- `XDED/requirements.txt` - Python dependencies
- `XDED/configs/*.yaml` - Experiment configs
- `conformer/.../transformer/hparams/exp/uaspeech/*.yaml` - SA configs

**Tokenizer:**
- `tokenizers/sa_official/tokenizer.ckpt` - SpeechBrain format
- `tokenizers/sa_official/tokenizer` - SentencePiece format

**Test data:**
- `Results P4/by_speakers_testcsv/*.csv` - Test partitions (15 speakers)

---

**End of Research Compendium Root README**

For detailed subsystem documentation, navigate to the respective directory READMEs:
- **SA Models:** `conformer/.../transformer/COMPENDIUM_README.md`
- **Tokenizer:** `tokenizers/sa_official/COMPENDIUM_README.md`
- **XDED Core:** `XDED/COMPENDIUM_README.md`
- **Results:** `Results P4/COMPENDIUM_README.md`
