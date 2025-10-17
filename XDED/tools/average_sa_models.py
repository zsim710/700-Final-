#!/usr/bin/env python3
"""
Script to average weights from SA model checkpoints for generalization experiments.
Compares model weight averaging vs WRA averaging for cross-speaker generalization.
"""

import os
import torch
import argparse
import json
from pathlib import Path
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from tqdm import tqdm

def load_sa_model_checkpoint(checkpoint_path, device='cuda'):
    """
    Load an SA model checkpoint and extract the model state dict.
    SpeechBrain model checkpoints are dictionaries mapping parameter names to tensors.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load full checkpoint - this is already a state dict for SpeechBrain models
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # SpeechBrain model.ckpt files are directly state dicts (dict of tensors)
    # Verify it's a valid state dict
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")
    
    # Count tensor parameters
    num_params = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
    print(f"  Loaded {num_params} parameter tensors")
    
    return checkpoint

def create_base_transformer_asr():
    """Create a base TransformerASR model with the same architecture as SA models"""
    model = TransformerASR(
        input_size=80,  # Mel features
        tgt_vocab=5000,  # Vocabulary size
        d_model=144,    # Model dimension
        nhead=4,        # Number of attention heads 
        num_encoder_layers=12,  # Same as SA models
        num_decoder_layers=4,   # Same as SA models
        d_ffn=1024,     # FFN dimension
        dropout=0.1,    # Dropout rate
        activation="relu",
        encoder_module="transformer",
        attention_type="regularMHA",
    )
    return model

def average_model_weights(checkpoint_paths, output_path, held_out_speaker, device='cuda'):
    """
    Average the weights of multiple speaker-specific models and save the result.
    Excludes the held-out speaker from averaging.
    """
    # Filter out the held-out speaker
    filtered_paths = [path for path in checkpoint_paths if held_out_speaker not in path]
    
    if not filtered_paths:
        raise ValueError(f"No checkpoints found after filtering out {held_out_speaker}")
    
    print(f"Averaging {len(filtered_paths)} models (excluding {held_out_speaker})...")
    
    # Load all model state dicts
    state_dicts = []
    for path in tqdm(filtered_paths, desc="Loading checkpoints"):
        try:
            state_dict = load_sa_model_checkpoint(path, device)
            state_dicts.append(state_dict)
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")
    
    if not state_dicts:
        raise ValueError("No valid state dicts could be loaded")
    
    # Get reference keys from the first state dict
    reference_keys = list(state_dicts[0].keys())
    
    # Initialize averaged state dict
    averaged_state = {}
    
    # Average each parameter
    print("Averaging parameters...")
    for key in tqdm(reference_keys):
        try:
            # Check if this key exists in all state dicts
            if all(key in sd for sd in state_dicts):
                # Stack and average tensors
                tensors = [sd[key] for sd in state_dicts]
                averaged_state[key] = sum(tensors) / len(tensors)
            else:
                missing = [i for i, sd in enumerate(state_dicts) if key not in sd]
                print(f"Warning: Key {key} missing in some state dicts: {missing}")
                # Use tensors from models where the key exists
                available_tensors = [sd[key] for sd in state_dicts if key in sd]
                if available_tensors:
                    averaged_state[key] = sum(available_tensors) / len(available_tensors)
        except Exception as e:
            print(f"Error averaging key {key}: {str(e)}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the averaged model
    print(f"Saving averaged model to: {output_path}")
    torch.save({
        'model': averaged_state,
        'held_out': held_out_speaker,
        'num_models_averaged': len(state_dicts),
        'model_paths': filtered_paths,
    }, output_path)
    
    print("Done!")
    return averaged_state

def main():
    parser = argparse.ArgumentParser(description="Average weights from multiple SA models")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Path to save the averaged model")
    parser.add_argument('--held_out', type=str, required=True, 
                        help="Held-out speaker to exclude from averaging")
    parser.add_argument('--device', type=str, default='cuda', 
                        help="Device for computation")
    parser.add_argument('--checkpoint_paths_file', type=str, required=True,
                       help="JSON file containing checkpoint paths for each speaker")
    parser.add_argument('--include_speakers', type=str, nargs='*', default=None,
                        help="Optional list of speaker IDs to include in averaging (e.g., M04 F03 M12)")
    args = parser.parse_args()
    
    # Load checkpoint paths from JSON file
    with open(args.checkpoint_paths_file, 'r') as f:
        speaker_checkpoints = json.load(f)
    
    # Convert to a list of paths
    if args.include_speakers:
        missing = [s for s in args.include_speakers if s not in speaker_checkpoints]
        if missing:
            raise ValueError(f"Speakers not found in mapping: {missing}")
        checkpoint_paths = [speaker_checkpoints[s] for s in args.include_speakers]
    else:
        checkpoint_paths = list(speaker_checkpoints.values())
    
    # Average the model weights
    average_model_weights(checkpoint_paths, args.output_path, args.held_out, args.device)

if __name__ == "__main__":
    main()