"""
Student Conformer Model for Knowledge Distillation

Lightweight version of SA Conformer architecture for ensemble distillation.
Based on the SA model config but with reduced capacity:
- d_model: 144 (same as SA template)
- Encoder layers: 8 (vs 12 in SA)
- Decoder layers: 4 (same as SA)
- Attention heads: 4 (same as SA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
from speechbrain.nnet.linear import Linear


class StudentConformer(nn.Module):
    """
    Lightweight Conformer for ensemble distillation.
    
    Architecture:
    - CNN frontend (2 blocks, stride 2)
    - Conformer encoder (8 layers, d_model=144)
    - Transformer decoder (4 layers, d_model=144)
    - CTC head for auxiliary loss
    """
    
    def __init__(
        self,
        vocab_size=5000,
        sample_rate=16000,
        n_mels=80,
        d_model=144,
        num_encoder_layers=8,
        num_decoder_layers=4,
        nhead=4,
        d_ffn=1024,
        dropout=0.1,
        activation=nn.GELU
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.d_model = d_model
        
        # Feature extraction (mel-spectrogram)
        self.compute_features = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=0,
            f_max=8000
        )
        
        # CNN frontend (matches SA model config)
        self.CNN = ConvolutionFrontEnd(
            input_shape=(8, 10, n_mels),
            num_blocks=2,
            num_layers_per_block=1,
            out_channels=(64, 32),
            kernel_sizes=(3, 3),
            strides=(2, 2),
            residuals=(False, False)
        )
        
        # Conformer encoder + decoder (using SpeechBrain's TransformerASR)
        self.Transformer = TransformerASR(
            input_size=640,  # CNN output: 32 channels * 20 time features
            tgt_vocab=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            encoder_module='conformer',
            attention_type='RelPosMHAXL',
            normalize_before=True,
            causal=False
        )
        
        # CTC projection head
        self.ctc_lin = Linear(
            input_size=d_model,
            n_neurons=vocab_size
        )
        
        # Decoder output projection
        self.seq_lin = Linear(
            input_size=d_model,
            n_neurons=vocab_size
        )
    
    def forward(self, wavs, wav_lens=None):
        """
        Forward pass through the model.
        
        Args:
            wavs: [batch, time] raw audio waveform
            wav_lens: [batch] relative lengths (0-1)
        
        Returns:
            encoder_out: [batch, seq_len, d_model]
            ctc_logits: [batch, seq_len, vocab_size]
        """
        # Extract mel-spectrogram features
        feats = self._extract_features(wavs)  # [batch, time, n_mels]
        
        # Apply CNN frontend
        x = self.CNN(feats)  # [batch, time/4, 640]
        
        # Encode using TransformerASR's encode method
        # TransformerASR has encode() method that returns encoder output
        encoder_out = self.Transformer.encode(x, wav_lens)  # [batch, seq_len, d_model]
        
        # CTC logits for auxiliary loss or CTC decoding
        ctc_logits = self.ctc_lin(encoder_out)  # [batch, seq_len, vocab_size]
        
        return encoder_out, ctc_logits
    
    def forward_decoder(self, encoder_out, targets, wav_lens=None, tgt_mask=None):
        """
        Forward pass through decoder (for teacher-forced training).
        
        Args:
            encoder_out: [batch, seq_len, d_model]
            targets: [batch, target_len] token IDs
            wav_lens: [batch] encoder output lengths (optional)
            tgt_mask: Optional mask for target sequence
        
        Returns:
            decoder_logits: [batch, target_len, vocab_size]
        """
        # Prepare src_length for padding mask inside TransformerASR
        # - If wav_lens is float in [0,1], convert to encoder frame counts
        # - If wav_lens is int/long tensor of lengths, use as-is
        # - If None, use full length for all (no padding mask)
        B, Lenc, _ = encoder_out.shape
        if wav_lens is None:
            src_length = torch.full((B,), Lenc, dtype=torch.long, device=encoder_out.device)
        else:
            if wav_lens.dtype.is_floating_point:
                src_length = torch.round(wav_lens.clamp(0, 1) * Lenc).long().clamp(min=1, max=Lenc)
            else:
                src_length = wav_lens.to(torch.long)

        # Decode using TransformerASR's decode method
        # Note: decode() method signature: decode(tgt, encoder_out, src_length)
        decoder_out = self.Transformer.decode(targets, encoder_out, src_length)
        # Some SpeechBrain implementations return a tuple (decoder_out, attn/...)
        if isinstance(decoder_out, (tuple, list)):
            decoder_out = decoder_out[0]
        # decoder_out: [batch, target_len, d_model]
        
        # Project to vocabulary
        decoder_logits = self.seq_lin(decoder_out)  # [batch, target_len, vocab_size]
        
        return decoder_logits
    
    def _extract_features(self, wavs):
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            wavs: [batch, time] raw waveform
        
        Returns:
            features: [batch, time_frames, n_mels]
        """
        # Compute mel-spectrogram
        mel = self.compute_features(wavs)  # [batch, n_mels, time_frames]
        
        # Transpose to [batch, time_frames, n_mels]
        mel = mel.transpose(1, 2)
        
        # Log compression
        mel = torch.log(mel + 1e-8)
        
        return mel
    
    def decode_greedy(self, encoder_out, max_len=100):
        """
        Greedy decoding for inference.
        
        Args:
            encoder_out: [batch, seq_len, d_model]
            max_len: Maximum decode length
        
        Returns:
            predictions: [batch, max_decode_len]
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        
        # Start with BOS token (assuming index 1)
        predictions = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Forward through decoder
            decoder_logits = self.forward_decoder(encoder_out, predictions)
            
            # Get next token (greedy)
            next_token = decoder_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to predictions
            predictions = torch.cat([predictions, next_token], dim=1)
            
            # Stop if all sequences have EOS (assuming index 2)
            if (next_token == 2).all():
                break
        
        return predictions
    
    def decode_ctc_greedy(self, ctc_logits):
        """
        Simple CTC greedy decoding.
        
        Args:
            ctc_logits: [batch, seq_len, vocab_size]
        
        Returns:
            predictions: List of token sequences
        """
        # Greedy decode: take argmax at each timestep
        predictions = ctc_logits.argmax(dim=-1)  # [batch, seq_len]
        
        # Remove blanks (index 0) and repetitions
        decoded = []
        for pred in predictions:
            # Remove blanks
            non_blank = pred[pred != 0]
            
            # Remove consecutive duplicates
            if len(non_blank) > 0:
                unique = [non_blank[0].item()]
                for i in range(1, len(non_blank)):
                    if non_blank[i] != non_blank[i-1]:
                        unique.append(non_blank[i].item())
                decoded.append(unique)
            else:
                decoded.append([])
        
        return decoded


if __name__ == "__main__":
    """Quick test of the model."""
    print("🧪 Testing StudentConformer...")
    
    # Create model
    model = StudentConformer(vocab_size=5000)
    print(f"✅ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    audio_len = 16000  # 1 second at 16kHz
    
    dummy_audio = torch.randn(batch_size, audio_len)
    print(f"\n🔍 Testing forward pass with audio shape: {dummy_audio.shape}")
    
    encoder_out, ctc_logits = model(dummy_audio)
    print(f"✅ Encoder output shape: {encoder_out.shape}")
    print(f"✅ CTC logits shape: {ctc_logits.shape}")
    
    # Test CTC decoding
    decoded = model.decode_ctc_greedy(ctc_logits)
    print(f"✅ CTC decoded (first sample): {decoded[0][:10]}...")  # First 10 tokens
    
    print("\n🎉 Model test complete!")
