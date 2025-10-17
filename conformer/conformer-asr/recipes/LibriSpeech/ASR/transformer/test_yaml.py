#!/usr/bin/env python3
"""Test script to debug YAML parsing issues."""

import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

def test_yaml_parsing():
    """Test if the YAML file can be parsed correctly."""
    print("Testing YAML parsing...")
    
    try:
        # Test if the original YAML file works
        print("1. Testing if we can run train.py with --help...")
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, 'train.py', 
            'hparams/exp/uaspeech/ua_SA_val_uncommon_WRA.yaml', 
            '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ train.py --help works fine")
        else:
            print(f"✗ train.py --help failed with return code {result.returncode}")
            print("STDERR:", result.stderr[:1000])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ train.py --help timed out")
        return False
    except Exception as e:
        print(f"✗ Exception during train.py test: {e}")
        return False
    
    print("✓ All tests passed")
    return True

if __name__ == "__main__":
    test_yaml_parsing()
