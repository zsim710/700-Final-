#!/usr/bin/env python3
"""
Script to find all SA model checkpoints and create a JSON mapping.
"""

import os
import json
from pathlib import Path

def find_sa_checkpoints(base_dir, output_path):
    """
    Find all SA model checkpoints and create a JSON mapping.
    
    Args:
        base_dir: Base directory containing speaker-specific SA models
        output_path: Path to save the JSON mapping
    """
    checkpoint_mapping = {}
    
    # List all speaker directories
    speaker_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('val_uncommon_')])
    
    print(f"Found {len(speaker_dirs)} speaker directories")
    
    for speaker_dir in speaker_dirs:
        # Extract speaker ID (e.g., F03, M05)
        speaker_id = speaker_dir.split('_')[2]  # val_uncommon_F03_E0D3 -> F03
        
        # Find the checkpoint path
        speaker_path = os.path.join(base_dir, speaker_dir)
        
        # Look for the checkpoint directory pattern: 7775/save/CKPT+*
        save_dir = os.path.join(speaker_path, '7775', 'save')
        
        if os.path.exists(save_dir):
            # Find CKPT+* directories
            ckpt_dirs = sorted([d for d in os.listdir(save_dir) if d.startswith('CKPT+')])
            
            if ckpt_dirs:
                # Use the latest checkpoint (last in sorted order)
                ckpt_dir = ckpt_dirs[-1]
                ckpt_path = os.path.join(save_dir, ckpt_dir)
                
                # Check if model.ckpt exists
                model_ckpt = os.path.join(ckpt_path, 'model.ckpt')
                if os.path.exists(model_ckpt):
                    checkpoint_mapping[speaker_id] = model_ckpt
                    print(f"  {speaker_id}: {model_ckpt}")
                else:
                    print(f"  Warning: model.ckpt not found for {speaker_id}")
            else:
                print(f"  Warning: No CKPT+ directory found for {speaker_id}")
        else:
            print(f"  Warning: Save directory not found for {speaker_id}")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(checkpoint_mapping, f, indent=2)
    
    print(f"\nCheckpoint mapping saved to: {output_path}")
    print(f"Total speakers found: {len(checkpoint_mapping)}")
    
    return checkpoint_mapping

if __name__ == "__main__":
    base_dir = "/mnt/Research/qwan121/ICASSP_SA"
    output_path = "/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
    
    find_sa_checkpoints(base_dir, output_path)
