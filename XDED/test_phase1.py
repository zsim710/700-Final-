#!/usr/bin/env python3
"""
Phase 1 Test Script: Verify audio domain adaptation pipeline works.

This script tests:
1. Audio dataset loading
2. Audio feature extraction with mel-spectrograms + CNN
3. Basic domain adaptation training setup
4. Data flow through the pipeline
"""

import os
import sys
import torch
import pandas as pd

# Add current directory to path
sys.path.append('/home/zsim710/XDED/XDED')

def test_csv_files():
    """Test 1: Check if CSV files exist and are readable."""
    print("🔍 Test 1: Checking CSV files...")
    
    partition_dir = '/home/zsim710/partitions/uaspeech/by_speakers'
    test_speakers = ['F03', 'F04', 'F05', 'M01']
    
    for speaker in test_speakers:
        csv_file = os.path.join(partition_dir, f"{speaker}.csv")
        if os.path.exists(csv_file):
            print(f"  ✅ {speaker}.csv found")
            try:
                df = pd.read_csv(csv_file)
                print(f"     - {len(df)} samples, columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"     - Sample row: {dict(df.iloc[0])}")
            except Exception as e:
                print(f"     ❌ Error reading CSV: {e}")
        else:
            print(f"  ❌ {speaker}.csv NOT found at {csv_file}")
    
    return True

def test_audio_loading():
    """Test 2: Check if we can load audio files."""
    print("\n🎵 Test 2: Testing audio loading...")
    
    try:
        import torchaudio
        print("  ✅ torchaudio available")
        
        # Try to load a sample audio file
        csv_file = '/home/zsim710/partitions/uaspeech/by_speakers/F03.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if len(df) > 0 and 'wav' in df.columns:
                sample_audio = df.iloc[0]['wav']
                print(f"  🔍 Trying to load: {sample_audio}")
                
                if os.path.exists(sample_audio):
                    audio, sr = torchaudio.load(sample_audio)
                    print(f"  ✅ Audio loaded: shape={audio.shape}, sr={sr}")
                    return True
                else:
                    print(f"  ❌ Audio file not found: {sample_audio}")
                    return False
            else:
                print(f"  ❌ No audio files in CSV or missing 'wav' column")
                return False
        else:
            print(f"  ❌ CSV file not found: {csv_file}")
            return False
            
    except ImportError:
        print("  ❌ torchaudio not available")
        return False
    except Exception as e:
        print(f"  ❌ Audio loading error: {e}")
        return False

def test_dataset_import():
    """Test 3: Check if our dataset can be imported and initialized."""
    print("\n📦 Test 3: Testing dataset import...")
    
    try:
        from dassl.data.datasets.da.uaspeech import UASpeechDataset
        print("  ✅ UASpeechDataset import successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_backbone_import():
    """Test 4: Check if our audio backbone can be imported."""
    print("\n🧠 Test 4: Testing audio backbone import...")
    
    try:
        from dassl.modeling.backbone.audio_cnn import AudioCNN
        print("  ✅ AudioCNN import successful")
        
        # Try to create the backbone
        backbone = AudioCNN()
        print(f"  ✅ AudioCNN created: out_features={backbone.out_features}")
        
        # Test with dummy audio
        dummy_audio = torch.randn(2, 16000)  # 2 samples, 1 second each
        features = backbone(dummy_audio)
        print(f"  ✅ Feature extraction: {dummy_audio.shape} → {features.shape}")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Runtime error: {e}")
        return False

def test_config_loading():
    """Test 5: Check if our config file loads correctly.""" 
    print("\n⚙️  Test 5: Testing config loading...")
    
    try:
        from dassl.config import get_cfg_default
        
        cfg = get_cfg_default()
        config_file = '/home/zsim710/XDED/XDED/configs/datasets/uaspeech.yaml'
        
        if os.path.exists(config_file):
            cfg.merge_from_file(config_file)
            print(f"  ✅ Config loaded from {config_file}")
            print(f"     - Dataset: {cfg.DATASET.NAME}")
            print(f"     - Backbone: {cfg.MODEL.BACKBONE.NAME}")
            return True
        else:
            print(f"  ❌ Config file not found: {config_file}")
            return False
            
    except Exception as e:
        print(f"  ❌ Config loading error: {e}")
        return False

def test_minimal_dataset_creation():
    """Test 6: Try to create a minimal dataset instance."""
    print("\n🗂️  Test 6: Testing minimal dataset creation...")
    
    try:
        from dassl.config import get_cfg_default
        from dassl.data.datasets.da.uaspeech import UASpeechDataset
        
        # Create minimal config
        cfg = get_cfg_default()
        cfg.DATASET.SOURCE_DOMAINS = ['F03']  # Just one speaker for testing
        cfg.DATASET.TARGET_DOMAINS = ['F04']  # Just one target speaker
        
        print("  🔍 Creating UASpeechDataset...")
        dataset = UASpeechDataset(cfg)
        
        print(f"  ✅ Dataset created successfully!")
        print(f"     - Training samples: {len(dataset.train_x) if dataset.train_x else 0}")
        print(f"     - Test samples: {len(dataset.test) if dataset.test else 0}")
        
        if dataset.train_x and len(dataset.train_x) > 0:
            sample = dataset.train_x[0]
            print(f"     - Sample: domain={sample.domain}, label={sample.label}, classname={sample.classname}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests."""
    print("🎯 PHASE 1 TESTING: Audio Domain Adaptation Pipeline")
    print("=" * 70)
    
    tests = [
        test_csv_files,
        test_audio_loading, 
        test_dataset_import,
        test_backbone_import,
        test_config_loading,
        test_minimal_dataset_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  💥 Test failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"🏆 PHASE 1 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 pipeline is ready.")
        print("✨ You can now proceed with:")
        print("   - Running basic audio domain adaptation training")
        print("   - Phase 2: Adding SA model logit distillation")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)