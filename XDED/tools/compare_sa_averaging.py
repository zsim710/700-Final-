#!/usr/bin/env python3
"""
Script to compare WRA (weighted recognition accuracy) with model parameter averaging
for cross-speaker generalization.
"""

import os
import torch
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Add XDED to path for importing modules
sys.path.insert(0, '/home/zsim710/XDED')

from XDED.options import parse_options
from XDED.speechbrain_utils import setup_inference_with_checkpoint
from XDED.speechbrain_utils import compute_wer_for_testset

def weighted_accuracy_averaging(speaker_accuracies, weights=None):
    """
    Compute a weighted average of per-speaker accuracies.
    
    Args:
        speaker_accuracies: Dict mapping speaker IDs to their accuracies (as percentages)
        weights: Optional dict mapping speaker IDs to weights. If None, equal weights are used.
        
    Returns:
        Weighted average accuracy
    """
    if weights is None:
        # Equal weighting
        weights = {spk: 1.0 for spk in speaker_accuracies.keys()}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {spk: w / total_weight for spk, w in weights.items()}
    
    # Compute weighted average
    weighted_avg = sum(speaker_accuracies[spk] * normalized_weights[spk] 
                     for spk in speaker_accuracies.keys())
    
    return weighted_avg

def evaluate_model_on_held_out_speaker(model_path, held_out_speaker, testset_path, device='cuda'):
    """
    Evaluate a model on the test set for a specific held-out speaker.
    
    Args:
        model_path: Path to the model checkpoint
        held_out_speaker: Speaker ID of the held-out speaker
        testset_path: Path to the test set
        device: Device for computation
        
    Returns:
        Dict with evaluation results, including WER
    """
    print(f"Evaluating model {model_path} on held-out speaker {held_out_speaker}")
    
    # Set up SA model for inference
    asr_model = setup_inference_with_checkpoint(model_path)
    
    # Load test set
    # This would depend on your specific data format
    test_data = load_testset(testset_path, held_out_speaker)
    
    # Compute WER
    results = compute_wer_for_testset(asr_model, test_data)
    
    return results

def compare_averaging_methods(checkpoint_paths, held_out_speaker, testset_path, 
                             output_dir, device='cuda'):
    """
    Compare model weight averaging vs. WRA averaging for cross-speaker generalization.
    
    Args:
        checkpoint_paths: Dict mapping speaker IDs to model checkpoint paths
        held_out_speaker: Speaker ID to use as held-out speaker
        testset_path: Path to test data
        output_dir: Directory to save results
        device: Device for computation
    """
    print(f"Comparing averaging methods for held-out speaker: {held_out_speaker}")
    
    # Exclude the held-out speaker from the averaging
    filtered_checkpoints = {spk: path for spk, path in checkpoint_paths.items() 
                           if spk != held_out_speaker}
    
    # 1. Individual speaker model evaluation
    print("Evaluating individual speaker models...")
    speaker_results = {}
    for spk, path in tqdm(filtered_checkpoints.items()):
        try:
            results = evaluate_model_on_held_out_speaker(path, held_out_speaker, testset_path, device)
            speaker_results[spk] = results
        except Exception as e:
            print(f"Error evaluating model for speaker {spk}: {str(e)}")
    
    # Extract WER for each speaker model on the held-out speaker
    speaker_wers = {spk: res['wer'] for spk, res in speaker_results.items()}
    
    # 2. Compute WRA (weighted recognition accuracy)
    # Convert WER to accuracy: Accuracy = 100 - WER
    speaker_accuracies = {spk: 100 - wer for spk, wer in speaker_wers.items()}
    
    # Equal weighting for WRA
    wra_equal = weighted_accuracy_averaging(speaker_accuracies)
    wer_wra_equal = 100 - wra_equal
    
    # Inverse WER weighting (higher weight to better models)
    inverse_wers = {spk: 1.0 / max(0.1, wer) for spk, wer in speaker_wers.items()}
    wra_inverse = weighted_accuracy_averaging(speaker_accuracies, inverse_wers)
    wer_wra_inverse = 100 - wra_inverse
    
    # 3. Evaluate the averaged model
    # Path to the averaged model (this should be created by average_sa_models.py)
    averaged_model_path = os.path.join(output_dir, f"{held_out_speaker}_held_out_averaged.pt")
    
    # Check if averaged model exists
    if not os.path.exists(averaged_model_path):
        print(f"Averaged model not found at {averaged_model_path}")
        print("Please run average_sa_models.py first to create the averaged model.")
        averaged_model_results = {'wer': float('nan')}
    else:
        # Evaluate the averaged model
        averaged_model_results = evaluate_model_on_held_out_speaker(
            averaged_model_path, held_out_speaker, testset_path, device
        )
    
    # 4. Compare results
    results = {
        'held_out_speaker': held_out_speaker,
        'individual_speaker_wers': speaker_wers,
        'weighted_avg_wer_equal': wer_wra_equal,
        'weighted_avg_wer_inverse': wer_wra_inverse,
        'averaged_model_wer': averaged_model_results['wer'],
        'num_speakers_averaged': len(filtered_checkpoints),
    }
    
    # Add summary metrics
    best_individual = min(speaker_wers.values())
    results.update({
        'best_individual_wer': best_individual,
        'wra_vs_best': wer_wra_equal - best_individual,
        'model_avg_vs_best': averaged_model_results['wer'] - best_individual,
        'model_avg_vs_wra': averaged_model_results['wer'] - wer_wra_equal,
    })
    
    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{held_out_speaker}_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\nResults Summary:")
    print(f"Held-out speaker: {held_out_speaker}")
    print(f"Best individual speaker WER: {best_individual:.2f}%")
    print(f"WRA (equal weights) WER: {wer_wra_equal:.2f}%")
    print(f"WRA (inverse WER weights) WER: {wer_wra_inverse:.2f}%")
    print(f"Averaged model WER: {averaged_model_results['wer']:.2f}%")
    print(f"Model averaging vs WRA difference: {results['model_avg_vs_wra']:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare weight averaging vs WRA for cross-speaker generalization")
    parser.add_argument('--checkpoint_paths_file', type=str, required=True,
                       help="JSON file containing checkpoint paths for each speaker")
    parser.add_argument('--held_out', type=str, required=True, 
                        help="Held-out speaker to evaluate on")
    parser.add_argument('--testset_path', type=str, required=True,
                        help="Path to test set for evaluation")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save results")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for computation")
    args = parser.parse_args()
    
    # Load checkpoint paths from JSON file
    with open(args.checkpoint_paths_file, 'r') as f:
        speaker_checkpoints = json.load(f)
    
    # Compare averaging methods
    compare_averaging_methods(
        speaker_checkpoints, 
        args.held_out,
        args.testset_path,
        args.output_dir,
        args.device
    )

if __name__ == "__main__":
    main()