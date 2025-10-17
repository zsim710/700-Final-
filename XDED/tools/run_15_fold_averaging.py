#!/usr/bin/env python3
"""
Script to run 15-fold weight averaging experiment.
For each of the 15 speakers, leave that speaker out and average the other 14 models.
"""

import os
import json
import subprocess
from pathlib import Path

def run_15_fold_averaging():
    """
    Run weight averaging for all 15 folds (one held-out speaker at a time).
    """
    # Paths
    checkpoint_json = "/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
    output_dir = "/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load speaker checkpoints
    with open(checkpoint_json, 'r') as f:
        speaker_checkpoints = json.load(f)
    
    speakers = list(speaker_checkpoints.keys())
    print(f"Running 15-fold averaging for speakers: {speakers}")
    print(f"Total speakers: {len(speakers)}\n")
    
    # Run averaging for each held-out speaker
    for i, held_out in enumerate(speakers, 1):
        print(f"="*80)
        print(f"FOLD {i}/15: Averaging models with {held_out} held out")
        print(f"="*80)
        
        # Output path for this fold
        output_path = os.path.join(output_dir, f"{held_out}_held_out_averaged.pt")
        
        # Run the averaging script
        cmd = [
            "python3",
            "/home/zsim710/XDED/XDED/tools/average_sa_models.py",
            "--checkpoint_paths_file", checkpoint_json,
            "--output_path", output_path,
            "--held_out", held_out,
            "--device", "cuda"
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)
            print(f"✓ Successfully averaged models for {held_out} held-out\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error averaging models for {held_out}:")
            print(e.stdout)
            print(e.stderr)
            print()
    
    print("="*80)
    print("All 15 folds completed!")
    print(f"Averaged models saved in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    run_15_fold_averaging()
