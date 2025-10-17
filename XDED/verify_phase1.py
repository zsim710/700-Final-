#!/usr/bin/env python3

"""
Quick Phase 1 Verification Script
"""

import sys
import os

def main():
    print("🎯 PHASE 1 VERIFICATION")
    print("=" * 50)
    
    try:
        # Test 1: Dataset import
        print("📦 Testing UASpeechDataset import...")
        from dassl.data.datasets.da.uaspeech import UASpeechDataset
        print("  ✅ UASpeechDataset imported successfully")
        
        # Test 2: AudioCNN import
        print("🧠 Testing AudioCNN import...")
        from dassl.modeling.backbone.audio_cnn import AudioCNN
        model = AudioCNN()
        print(f"  ✅ AudioCNN created with {model.out_features} features")
        
        # Test 3: Audio processing
        print("🎵 Testing audio processing...")
        import torch
        import torchaudio
        
        # Create fake audio data
        fake_audio = torch.randn(2, 16000)  # 2 samples, 1 second at 16kHz
        features = model(fake_audio)
        print(f"  ✅ Audio processing: {fake_audio.shape} → {features.shape}")
        
        # Test 4: Config file exists
        print("⚙️  Testing config file...")
        config_path = "/home/zsim710/XDED/XDED/configs/datasets/uaspeech.yaml"
        if os.path.exists(config_path):
            print("  ✅ Config file exists")
        else:
            print("  ❌ Config file missing")
            return False
        
        # Test 5: Training script exists
        print("🚀 Testing training script...")
        train_script = "/home/zsim710/XDED/XDED/uaspeech_train.py"
        if os.path.exists(train_script):
            print("  ✅ Training script exists")
        else:
            print("  ❌ Training script missing")
            return False
        
        print("\n🏆 PHASE 1 VERIFICATION: ALL TESTS PASSED! 🎉")
        print("Phase 1 is ready for training. You can proceed to Phase 2!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)