#!/usr/bin/env python3
"""
Run intelligibility-based weight averaging experiment for HIGH and VERY_LOW bands.
Creates averaged models for each band with/without held-out speaker excluded.
"""

import os
import json
import subprocess

# Test speakers for HIGH and VERY_LOW bands only
TEST_SPEAKERS = {
    'HIGH': 'M08',
    'VERY_LOW': 'M01'
}

def run_intelligibility_averaging(exclude_held_out: bool = True):
    """
    Run weight averaging for HIGH and VERY_LOW intelligibility bands.
    
    Args:
        exclude_held_out: If True, exclude test speaker from averaging
    """
    # Paths
    checkpoint_json = "/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json"
    output_dir = "/home/zsim710/XDED/XDED/results/intelligibility_averaging"
    
    if exclude_held_out:
        output_subdir = os.path.join(output_dir, "excluded")
    else:
        output_subdir = os.path.join(output_dir, "included")
    
    os.makedirs(output_subdir, exist_ok=True)
    
    print("="*80)
    print(f"INTELLIGIBILITY-BASED WEIGHT AVERAGING")
    print(f"Bands: HIGH and VERY_LOW")
    print(f"Exclude held-out speaker: {exclude_held_out}")
    print("="*80)
    
    # Run averaging for each band
    for band, test_speaker in TEST_SPEAKERS.items():
        print(f"\n{'='*80}")
        print(f"Band: {band} | Test Speaker: {test_speaker}")
        print(f"{'='*80}")
        
        # Output path
        if exclude_held_out:
            output_path = os.path.join(output_subdir, f"{band}_averaged_exclude_{test_speaker}.pt")
            exclude_arg = ["--exclude_speaker", test_speaker]
        else:
            output_path = os.path.join(output_subdir, f"{band}_averaged_include_{test_speaker}.pt")
            exclude_arg = []
        
        # Run averaging
        cmd = [
            "python3",
            "/home/zsim710/XDED/XDED/tools/average_sa_models_intelligibility.py",
            "--speaker_checkpoints_json", checkpoint_json,
            "--band", band,
            "--output_path", output_path,
            "--device", "cuda"
        ] + exclude_arg
        
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)
            print(f"✓ Successfully averaged {band} band models\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error averaging {band} band:")
            print(e.stdout)
            print(e.stderr)
    
    print("="*80)
    print("Both bands completed!")
    print(f"Models saved in: {output_subdir}")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude_held_out', action='store_true',
                       help='Exclude test speaker from averaging')
    parser.add_argument('--include_held_out', action='store_true',
                       help='Include test speaker in averaging')
    args = parser.parse_args()
    
    # Run both scenarios if neither specified
    if not args.exclude_held_out and not args.include_held_out:
        print("Running both scenarios: excluded and included\n")
        run_intelligibility_averaging(exclude_held_out=True)
        print("\n" + "="*80 + "\n")
        run_intelligibility_averaging(exclude_held_out=False)
    elif args.exclude_held_out:
        run_intelligibility_averaging(exclude_held_out=True)
    else:
        run_intelligibility_averaging(exclude_held_out=False)
