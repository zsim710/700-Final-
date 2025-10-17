# Speaker Model Averaging vs WRA Experiment

## Overview

This document outlines the experiment comparing two approaches to cross-speaker generalization in dysarthric speech recognition:

1. **Model Parameter Averaging**: Directly averaging model weights from multiple speaker-specific models
2. **Weighted Recognition Accuracy (WRA)**: Averaging the recognition results from multiple speaker-specific models

The goal is to determine which approach generalizes better to unseen dysarthric speakers.

## Implementation Details

We've created a set of tools to facilitate this experiment:

1. `tools/average_sa_models.py`: Averages multiple SA model checkpoints, excluding a specified held-out speaker
2. `tools/compare_sa_averaging.py`: Compares model parameter averaging vs WRA on a held-out speaker
3. `tools/prepare_checkpoint_mapping.py`: Prepares a mapping of speaker IDs to checkpoint paths
4. `speechbrain_utils.py`: Utility functions for working with SpeechBrain models
5. `run_sa_averaging_experiment.sh`: Script that orchestrates the entire experiment workflow

## Experiment Workflow

1. **Create a checkpoint mapping**: Map each speaker ID to its corresponding SA model checkpoint
2. **For each held-out speaker**:
   - Create an averaged model by averaging parameters from all other speakers
   - Evaluate each individual speaker model on the held-out speaker
   - Compute WRA using different weighting schemes
   - Compare model averaging vs WRA performance
3. **Aggregate results** across all held-out speakers

## Running the Experiment

1. Update the SA checkpoint paths in the script with the ones your supervisor provided:
   ```bash
   # Base directory for SA model checkpoints
   SA_CHECKPOINT_DIR="/mnt/Research/qwan121/ICASSP_SA"
   ```

2. Run the experiment script:
   ```bash
   chmod +x /home/zsim710/XDED/XDED/run_sa_averaging_experiment.sh
   ./run_sa_averaging_experiment.sh
   ```

## Expected Results

The experiment will produce:
- Averaged model checkpoints for each held-out speaker
- Individual comparison results for each held-out speaker
- An aggregated summary across all held-out speakers

Key metrics include:
- WER for each individual model on held-out speakers
- WER for the averaged model
- WER for the WRA approach (with different weighting schemes)
- Difference between model averaging and WRA

## Analysis

To interpret the results:
1. If model averaging WER < WRA WER: Parameter averaging generalizes better
2. If WRA WER < model averaging WER: Ensemble methods generalize better
3. Look at the difference between best individual speaker model and the averaging methods

The results will provide insight into whether model averaging is a viable approach for cross-speaker generalization in dysarthric speech recognition, potentially offering a computationally efficient alternative to ensemble methods.