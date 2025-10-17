#!/usr/bin/env python3
"""
Average SA model weights by intelligibility band for generalization baseline.
"""

import os
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Intelligibility bands
INTELLIGIBILITY_BANDS = {
    "HIGH": ["M09", "M14", "M10", "M08", "F05"],
    "VERY_LOW": ["M12", "M01", "F03", "M04"]
}

def load_sa_model_checkpoint(checkpoint_path, device='cuda'):
    """Load an SA model checkpoint and extract the model state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")
    
    num_params = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
    print(f"  Loaded {num_params} parameter tensors")
    
    return checkpoint

def average_by_intelligibility_band(
    speaker_checkpoints: dict,
    band: str,
    output_path: str,
    exclude_speaker: str = None,
    device: str = 'cuda'
):
    """
    Average model weights from speakers in a specific intelligibility band.
    
    Args:
        speaker_checkpoints: Dict mapping speaker IDs to checkpoint paths
        band: Intelligibility band (HIGH or VERY_LOW)
        output_path: Path to save averaged model
        exclude_speaker: Optional speaker to exclude from averaging
        device: Device for computation
    """
    if band not in INTELLIGIBILITY_BANDS:
        raise ValueError(f"Invalid band: {band}. Must be one of {list(INTELLIGIBILITY_BANDS.keys())}")
    
    speakers_in_band = INTELLIGIBILITY_BANDS[band]
    
    # Filter by band and optionally exclude speaker
    filtered_speakers = [s for s in speakers_in_band if s in speaker_checkpoints]
    if exclude_speaker:
        filtered_speakers = [s for s in filtered_speakers if s != exclude_speaker]
    
    if not filtered_speakers:
        raise ValueError(f"No speakers found for band {band} after filtering")
    
    print(f"\n{'='*80}")
    print(f"Averaging {band} band models")
    print(f"{'='*80}")
    print(f"Speakers in band: {speakers_in_band}")
    print(f"Available speakers: {filtered_speakers}")
    if exclude_speaker:
        print(f"Excluding: {exclude_speaker}")
    print(f"Total models to average: {len(filtered_speakers)}")
    
    # Load all state dicts
    state_dicts = []
    for speaker in tqdm(filtered_speakers, desc="Loading checkpoints"):
        try:
            checkpoint_path = speaker_checkpoints[speaker]
            state_dict = load_sa_model_checkpoint(checkpoint_path, device)
            state_dicts.append(state_dict)
        except Exception as e:
            print(f"Failed to load {speaker}: {str(e)}")
    
    if not state_dicts:
        raise ValueError("No valid state dicts could be loaded")
    
    # Get reference keys from first state dict
    reference_keys = list(state_dicts[0].keys())
    
    # Initialize averaged state dict
    averaged_state = {}
    
    # Average each parameter
    print("Averaging parameters...")
    for key in tqdm(reference_keys):
        try:
            if all(key in sd for sd in state_dicts):
                tensors = [sd[key] for sd in state_dicts]
                averaged_state[key] = sum(tensors) / len(tensors)
            else:
                # Handle missing keys
                available_tensors = [sd[key] for sd in state_dicts if key in sd]
                if available_tensors:
                    averaged_state[key] = sum(available_tensors) / len(available_tensors)
        except Exception as e:
            print(f"Error averaging key {key}: {str(e)}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save averaged model
    print(f"Saving averaged model to: {output_path}")
    torch.save({
        'model': averaged_state,
        'intelligibility_band': band,
        'speakers_averaged': filtered_speakers,
        'num_models_averaged': len(state_dicts),
        'excluded_speaker': exclude_speaker,
    }, output_path)
    
    print(f"✓ Saved averaged {band} band model")
    return averaged_state

def main():
    parser = argparse.ArgumentParser(description="Average SA models by intelligibility band")
    parser.add_argument('--speaker_checkpoints_json', type=str, required=True,
                       help="JSON file mapping speaker IDs to checkpoint paths")
    parser.add_argument('--band', type=str, required=True,
                       choices=['HIGH', 'VERY_LOW'],
                       help="Intelligibility band to average (HIGH or VERY_LOW)")
    parser.add_argument('--output_path', type=str, required=True,
                       help="Path to save averaged model")
    parser.add_argument('--exclude_speaker', type=str, default=None,
                       help="Speaker to exclude from averaging (optional)")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device for computation")
    args = parser.parse_args()
    
    # Load speaker checkpoint mapping
    with open(args.speaker_checkpoints_json, 'r') as f:
        speaker_checkpoints = json.load(f)
    
    # Average by band
    average_by_intelligibility_band(
        speaker_checkpoints,
        args.band,
        args.output_path,
        args.exclude_speaker,
        args.device
    )

if __name__ == "__main__":
    main()
