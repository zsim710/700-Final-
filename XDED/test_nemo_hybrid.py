#!/usr/bin/env python3
"""Quick smoke test for NeMoHybridStudent model."""

import torch
import sys

print("🧪 Testing NeMoHybridStudent...")

# Import the model
try:
    from models.nemo_hybrid_student import NeMoHybridStudent
    print("✓ NeMoHybridStudent imported successfully")
except Exception as e:
    print(f"✗ Failed to import NeMoHybridStudent: {e}")
    sys.exit(1)

# Create model
print("\n📦 Creating model (this will download pretrained checkpoint if needed)...")
try:
    model = NeMoHybridStudent(
        nemo_model_name="nvidia/stt_en_conformer_ctc_small",
        vocab_size=5000,
        num_decoder_layers=4,
        nhead=4,
        d_ffn=1024,
        dropout=0.1,
        freeze_nemo_encoder=False,
        device=torch.device('cpu')  # Use CPU for smoke test
    )
    print(f"✓ Model created successfully")
    print(f"  Model d_model: {model.d_model}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Test forward pass
print("\n🔍 Testing forward pass...")
batch_size = 2
audio_len = 16000  # 1 second at 16kHz

try:
    dummy_audio = torch.randn(batch_size, audio_len)
    wav_lens = torch.tensor([1.0, 0.8])
    
    encoder_out, ctc_logits = model(dummy_audio, wav_lens)
    print(f"✓ Forward pass successful")
    print(f"  Encoder output shape: {encoder_out.shape}")
    print(f"  CTC logits shape: {ctc_logits.shape}")
    print(f"  Expected vocab size (5000): {ctc_logits.shape[-1] == 5000}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test decoder forward
print("\n🔍 Testing decoder forward...")
try:
    targets = torch.randint(0, 5000, (batch_size, 10))
    decoder_logits = model.forward_decoder(encoder_out, targets, wav_lens)
    print(f"✓ Decoder forward successful")
    print(f"  Decoder logits shape: {decoder_logits.shape}")
    print(f"  Expected shape: [B={batch_size}, T=10, V=5000]")
except Exception as e:
    print(f"✗ Decoder forward failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test greedy decoding
print("\n🔍 Testing greedy decoding...")
try:
    predictions = model.decode_greedy(encoder_out, max_len=5)
    print(f"✓ Greedy decoding successful")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  First prediction: {predictions[0].tolist()[:10]}")
except Exception as e:
    print(f"✗ Greedy decoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All tests passed!")
print("\n✅ NeMoHybridStudent is ready for training with decoder-KD")
