#!/usr/bin/env python3
"""
Evaluate 15-fold weight averaging experiment.
For each held-out speaker:
1. Test the averaged model on that speaker's test data
2. Test each individual speaker model on that speaker's test data
3. Compute WRA (weighted recognition accuracy) from individual models
4. Compare averaged model performance vs WRA
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths for imports
sys.path.insert(0, '/home/zsim710/XDED')
sys.path.insert(0, '/home/zsim710/XDED/conformer/conformer-asr')

# Import SpeechBrain utilities
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.data_utils import undo_padding
from speechbrain.dataio.dataio import read_audio
import torchaudio

def load_test_data_for_speaker(speaker_id, base_dir="/home/zsim710/partitions/uaspeech/by_speakers"):
    """
    Load test data CSV for a specific speaker.
    
    Args:
        speaker_id: Speaker ID (e.g., 'F02', 'M05')
        base_dir: Directory containing speaker CSV files
        
    Returns:
        DataFrame with test data
    """
    csv_path = os.path.join(base_dir, f"{speaker_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} test utterances for speaker {speaker_id}")
    return df

def load_speechbrain_model(model_checkpoint_path, device='cuda'):
    """
    Load a SpeechBrain ASR model from a checkpoint.
    This loads just the model weights, not the full SpeechBrain pretrained interface.
    
    Args:
        model_checkpoint_path: Path to model.ckpt file
        device: Device to load model on
        
    Returns:
        State dict of the model
    """
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def compute_wer(predictions, references):
    """
    Compute Word Error Rate.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        WER as a percentage
    """
    from jiwer import wer
    
    # Ensure both are lists of strings
    preds = [str(p).upper().strip() for p in predictions]
    refs = [str(r).upper().strip() for r in references]
    
    # Compute WER
    error_rate = wer(refs, preds)
    return error_rate * 100.0  # Return as percentage

def evaluate_model_on_speaker(model_path, test_csv, speaker_id, device='cuda', is_averaged=False):
    """
    Evaluate a model on a speaker's test data.
    
    Since we only have model weights and not the full SpeechBrain setup,
    this is a placeholder that should be integrated with your existing
    SpeechBrain inference pipeline.
    
    Args:
        model_path: Path to model checkpoint (either individual or averaged)
        test_csv: Path to test CSV file
        speaker_id: Speaker ID being tested
        device: Device for inference
        is_averaged: Whether this is an averaged model
        
    Returns:
        Dict with evaluation results
    """
    print(f"Evaluating model on speaker {speaker_id}...")
    print(f"Model: {model_path}")
    
    # Load test data
    df = pd.read_csv(test_csv)
    
    # TODO: This is a placeholder for the actual inference code
    # You'll need to integrate this with your existing SpeechBrain inference pipeline
    # that can load model.ckpt files and run inference on audio files
    
    # For now, return dummy results
    # In the actual implementation, you would:
    # 1. Load the model weights
    # 2. Set up the SpeechBrain ASR model
    # 3. Run inference on each audio file
    # 4. Compute WER
    
    print("WARNING: Using placeholder evaluation. Integrate with SpeechBrain inference pipeline.")
    
    results = {
        'speaker_id': speaker_id,
        'model_path': model_path,
        'is_averaged': is_averaged,
        'num_utterances': len(df),
        'wer': np.random.uniform(10, 30),  # Placeholder - replace with actual WER
        'accuracy': np.random.uniform(70, 90),  # Placeholder
    }
    
    return results

def weighted_accuracy_averaging(speaker_accuracies, weights=None):
    """
    Compute weighted average of accuracies.
    
    Args:
        speaker_accuracies: Dict mapping speaker IDs to accuracies
        weights: Optional dict of weights (if None, use equal weights)
        
    Returns:
        Weighted average accuracy
    """
    if weights is None:
        weights = {spk: 1.0 for spk in speaker_accuracies.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {spk: w / total_weight for spk, w in weights.items()}
    
    # Compute weighted average
    weighted_avg = sum(speaker_accuracies[spk] * normalized_weights[spk] 
                      for spk in speaker_accuracies.keys())
    
    return weighted_avg

def evaluate_15_fold_experiment(checkpoint_json, averaged_models_dir, test_data_dir, output_dir, device='cuda'):
    """
    Evaluate all 15 folds of the weight averaging experiment.
    
    Args:
        checkpoint_json: JSON file with individual speaker model paths
        averaged_models_dir: Directory containing averaged models
        test_data_dir: Directory containing test CSV files
        output_dir: Directory to save results
        device: Device for inference
    """
    # Load checkpoint mapping
    with open(checkpoint_json, 'r') as f:
        speaker_checkpoints = json.load(f)
    
    speakers = sorted(speaker_checkpoints.keys())
    print(f"Evaluating 15-fold experiment for speakers: {speakers}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Evaluate each fold
    for fold_idx, held_out in enumerate(speakers, 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/15: Evaluating with {held_out} held out")
        print(f"{'='*80}")
        
        # Path to averaged model for this fold
        averaged_model_path = os.path.join(averaged_models_dir, f"{held_out}_held_out_averaged.pt")
        test_csv = os.path.join(test_data_dir, f"{held_out}.csv")
        
        # Check if files exist
        if not os.path.exists(averaged_model_path):
            print(f"Warning: Averaged model not found: {averaged_model_path}")
            continue
        if not os.path.exists(test_csv):
            print(f"Warning: Test data not found: {test_csv}")
            continue
        
        # 1. Evaluate the averaged model
        print(f"\n1. Evaluating averaged model on {held_out}...")
        averaged_results = evaluate_model_on_speaker(
            averaged_model_path, test_csv, held_out, device, is_averaged=True
        )
        
        # 2. Evaluate each individual speaker model (excluding held-out)
        print(f"\n2. Evaluating individual speaker models on {held_out}...")
        individual_results = []
        for speaker_id, model_path in speaker_checkpoints.items():
            if speaker_id == held_out:
                continue  # Skip the held-out speaker
            
            print(f"   Evaluating {speaker_id} model on {held_out}...")
            result = evaluate_model_on_speaker(
                model_path, test_csv, held_out, device, is_averaged=False
            )
            result['source_speaker'] = speaker_id
            individual_results.append(result)
        
        # 3. Compute WRA (weighted recognition accuracy)
        speaker_accuracies = {r['source_speaker']: r['accuracy'] for r in individual_results}
        speaker_wers = {r['source_speaker']: r['wer'] for r in individual_results}
        
        # Equal weighting
        wra_equal = weighted_accuracy_averaging(speaker_accuracies)
        wer_wra_equal = 100 - wra_equal
        
        # Inverse WER weighting (better models get higher weight)
        inverse_wers = {spk: 1.0 / max(0.1, wer) for spk, wer in speaker_wers.items()}
        wra_inverse = weighted_accuracy_averaging(speaker_accuracies, inverse_wers)
        wer_wra_inverse = 100 - wra_inverse
        
        # 4. Compare results
        fold_results = {
            'fold': fold_idx,
            'held_out_speaker': held_out,
            'averaged_model_wer': averaged_results['wer'],
            'averaged_model_accuracy': averaged_results['accuracy'],
            'wra_equal_wer': wer_wra_equal,
            'wra_equal_accuracy': wra_equal,
            'wra_inverse_wer': wer_wra_inverse,
            'wra_inverse_accuracy': wra_inverse,
            'best_individual_wer': min(speaker_wers.values()),
            'worst_individual_wer': max(speaker_wers.values()),
            'mean_individual_wer': np.mean(list(speaker_wers.values())),
            'num_models_averaged': len(individual_results),
            'individual_speaker_results': individual_results,
        }
        
        # Add comparison metrics
        fold_results.update({
            'model_avg_vs_wra_equal': averaged_results['wer'] - wer_wra_equal,
            'model_avg_vs_best': averaged_results['wer'] - fold_results['best_individual_wer'],
        })
        
        all_results.append(fold_results)
        
        # Save individual fold results
        fold_output = os.path.join(output_dir, f"fold_{fold_idx}_{held_out}_results.json")
        with open(fold_output, 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        # Print summary for this fold
        print(f"\nFold {fold_idx} Results:")
        print(f"  Averaged model WER: {averaged_results['wer']:.2f}%")
        print(f"  WRA (equal) WER: {wer_wra_equal:.2f}%")
        print(f"  WRA (inverse) WER: {wer_wra_inverse:.2f}%")
        print(f"  Best individual WER: {fold_results['best_individual_wer']:.2f}%")
        print(f"  Model avg vs WRA: {fold_results['model_avg_vs_wra_equal']:.2f}%")
    
    # 5. Aggregate results across all folds
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS ALL 15 FOLDS")
    print(f"{'='*80}")
    
    aggregated = {
        'num_folds': len(all_results),
        'avg_model_averaging_wer': np.mean([r['averaged_model_wer'] for r in all_results]),
        'avg_wra_equal_wer': np.mean([r['wra_equal_wer'] for r in all_results]),
        'avg_wra_inverse_wer': np.mean([r['wra_inverse_wer'] for r in all_results]),
        'avg_best_individual_wer': np.mean([r['best_individual_wer'] for r in all_results]),
        'avg_model_vs_wra': np.mean([r['model_avg_vs_wra_equal'] for r in all_results]),
        'std_model_averaging_wer': np.std([r['averaged_model_wer'] for r in all_results]),
        'std_wra_equal_wer': np.std([r['wra_equal_wer'] for r in all_results]),
        'fold_results': all_results,
    }
    
    # Save aggregated results
    aggregated_output = os.path.join(output_dir, "aggregated_15_fold_results.json")
    with open(aggregated_output, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Print summary
    print(f"\nAggregated Results (across {aggregated['num_folds']} folds):")
    print(f"  Average Model Averaging WER: {aggregated['avg_model_averaging_wer']:.2f}% ± {aggregated['std_model_averaging_wer']:.2f}%")
    print(f"  Average WRA (equal) WER: {aggregated['avg_wra_equal_wer']:.2f}% ± {aggregated['std_wra_equal_wer']:.2f}%")
    print(f"  Average WRA (inverse) WER: {aggregated['avg_wra_inverse_wer']:.2f}%")
    print(f"  Average Best Individual WER: {aggregated['avg_best_individual_wer']:.2f}%")
    print(f"  Average Model Avg vs WRA difference: {aggregated['avg_model_vs_wra']:.2f}%")
    
    if aggregated['avg_model_vs_wra'] < 0:
        print(f"\n✓ Model averaging performs BETTER than WRA by {abs(aggregated['avg_model_vs_wra']):.2f}%")
    else:
        print(f"\n✗ WRA performs BETTER than model averaging by {aggregated['avg_model_vs_wra']:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")
    
    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Evaluate 15-fold weight averaging experiment")
    parser.add_argument('--checkpoint_json', type=str,
                       default="/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json",
                       help="JSON file with speaker checkpoint paths")
    parser.add_argument('--averaged_models_dir', type=str,
                       default="/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models",
                       help="Directory containing averaged models")
    parser.add_argument('--test_data_dir', type=str,
                       default="/home/zsim710/partitions/uaspeech/by_speakers",
                       help="Directory containing test CSV files")
    parser.add_argument('--output_dir', type=str,
                       default="/home/zsim710/XDED/XDED/results/speaker_averaging/evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device for inference")
    args = parser.parse_args()
    
    evaluate_15_fold_experiment(
        args.checkpoint_json,
        args.averaged_models_dir,
        args.test_data_dir,
        args.output_dir,
        args.device
    )

if __name__ == "__main__":
    main()
