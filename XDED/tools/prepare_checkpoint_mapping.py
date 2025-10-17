#!/usr/bin/env python3
"""
Script to prepare speaker model checkpoints mapping for SA model averaging experiment.
Creates a JSON file mapping speaker IDs to checkpoint paths.
"""

import os
import argparse
import json
from pathlib import Path
import sys

# Add XDED to path for importing modules
sys.path.insert(0, '/home/zsim710/XDED')
from XDED.speechbrain_utils import prepare_checkpoint_json

def main():
    parser = argparse.ArgumentParser(description="Prepare speaker checkpoint mapping JSON file")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Base directory containing speaker checkpoints")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the JSON file")
    parser.add_argument('--speakers', type=str, nargs='+', required=True,
                        help="List of speaker IDs to include")
    args = parser.parse_args()
    
    prepare_checkpoint_json(args.speakers, args.base_dir, args.output_path)

if __name__ == "__main__":
    main()