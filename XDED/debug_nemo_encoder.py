#!/usr/bin/env python3
"""Debug NeMo encoder output shapes."""

import torch
import nemo.collections.asr as nemo_asr

print("Loading NeMo model...")
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_small")
model.cpu()

print("\nModel encoder config:")
if hasattr(model.encoder, '_cfg'):
    print(f"  Encoder cfg: {model.encoder._cfg}")
if hasattr(model.encoder, 'd_model'):
    print(f"  Encoder d_model attribute: {model.encoder.d_model}")

print("\nRunning test forward...")
with torch.no_grad():
    dummy_wav = torch.randn(1, 16000)  # 1 second
    dummy_len = torch.tensor([16000], dtype=torch.long)
    
    print(f"Input shape: {dummy_wav.shape}, length: {dummy_len}")
    
    proc_sig, proc_len = model.preprocessor(input_signal=dummy_wav, length=dummy_len)
    print(f"After preprocessor: {proc_sig.shape}, length: {proc_len}")
    
    enc_out, enc_len = model.encoder(audio_signal=proc_sig, length=proc_len)
    print(f"After encoder: {enc_out.shape}, length: {enc_len}")
    print(f"  enc_out.size(0) (batch): {enc_out.size(0)}")
    print(f"  enc_out.size(1) (time): {enc_out.size(1)}")
    print(f"  enc_out.size(2) (features): {enc_out.size(2)}")
    
print("\nConclusion:")
print(f"NeMo encoder d_model should be: {enc_out.size(2)}")
