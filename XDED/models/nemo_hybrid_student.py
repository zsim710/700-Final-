"""
NeMo-hybrid Student model

This model reuses a pretrained NeMo Conformer-CTC encoder and attaches:
- our own CTC head (to match the project's 5000-token vocab), and
- a lightweight autoregressive Transformer decoder head (for decoder-KD).

Rationale:
- Keeps tokenizer/vocab consistent with teacher logits (5000 tokens),
  avoiding any need to remap NeMo's BPE tokens.
- Allows initializing from a strong pretrained acoustic encoder while
  training the decoder via KD from teacher decoder logits.

Dependencies:
- NVIDIA NeMo (nemo_toolkit) must be installed when using this model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer decoder."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, D]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class NeMoHybridStudent(nn.Module):
    """
    Student model that uses NeMo Conformer-CTC encoder and custom decoder/CTC heads.

    Interface matches the existing StudentConformer:
    - forward(wavs, wav_lens) -> encoder_out, ctc_logits
    - forward_decoder(encoder_out, targets, wav_lens=None) -> decoder_logits
    - decode_greedy(encoder_out, max_len)
    """

    def __init__(
        self,
        nemo_model_name: str = "nvidia/stt_en_conformer_ctc_small",
        vocab_size: int = 5000,
        num_decoder_layers: int = 4,
        nhead: int = 4,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        freeze_nemo_preprocessor: bool = False,
        freeze_nemo_encoder: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Lazy import NeMo to keep it optional
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "NeMo is required for NeMoHybridStudent. Install with: pip install 'nemo_toolkit[all]'"
            ) from e

        # Instantiate pretrained NeMo model
        self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(nemo_model_name)

        # Optionally freeze components
        if freeze_nemo_preprocessor and hasattr(self.nemo_model, 'preprocessor'):
            for p in self.nemo_model.preprocessor.parameters():
                p.requires_grad = False
        if freeze_nemo_encoder and hasattr(self.nemo_model, 'encoder'):
            for p in self.nemo_model.encoder.parameters():
                p.requires_grad = False

        # Determine encoder output dim (d_model) by probing with a tiny dummy forward
        # NeMo encoder d_model may not be directly exposed; safer to probe
        # Keep NeMo on CPU during probe to avoid device conflicts
        self.vocab_size = vocab_size
        self.nemo_model.cpu()
        with torch.no_grad():
            dummy_wav = torch.randn(1, 16000)
            dummy_len = torch.tensor([16000], dtype=torch.long)
            proc_sig, proc_len = self.nemo_model.preprocessor(input_signal=dummy_wav, length=dummy_len)
            enc_out, enc_len = self.nemo_model.encoder(audio_signal=proc_sig, length=proc_len)
            # NeMo encoder returns [B, D, T] - transpose to [B, T, D]
            enc_out = enc_out.transpose(1, 2)
            nemo_d_model = enc_out.size(2)
        del dummy_wav, dummy_len, proc_sig, proc_len, enc_out, enc_len
        self.d_model = nemo_d_model

        # Ensure nhead is compatible with d_model
        if self.d_model % nhead != 0:
            # Find closest divisor of d_model near the requested nhead
            divisors = [i for i in range(1, self.d_model + 1) if self.d_model % i == 0]
            nhead = min(divisors, key=lambda x: abs(x - nhead))
            print(f"  ⚠️  Adjusted nhead to {nhead} (d_model={self.d_model} requires divisor)")

        # Our CTC projection head to 5000-token vocab
        self.ctc_lin = nn.Linear(self.d_model, vocab_size)

        # Our autoregressive Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=d_ffn, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.token_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, dropout=dropout)
        self.generator = nn.Linear(self.d_model, vocab_size)

        # Move to device if provided
        if device is not None:
            self.to(device)

    def _to_lengths(self, wavs: torch.Tensor, wav_lens: Optional[torch.Tensor]) -> torch.Tensor:
        B, T = wavs.shape
        if wav_lens is None:
            return torch.full((B,), T, dtype=torch.long, device=wavs.device)
        if wav_lens.dtype.is_floating_point:
            return torch.round(wav_lens.clamp(0, 1) * T).long().clamp(min=1, max=T)
        return wav_lens.to(torch.long)

    def forward(self, wavs: torch.Tensor, wav_lens: Optional[torch.Tensor] = None):
        """
        Args:
            wavs: [B, T] raw waveforms
            wav_lens: [B] relative (0..1) or absolute lengths

        Returns:
            encoder_out: [B, S, D]
            ctc_logits:   [B, S, V]
        """
        lengths = self._to_lengths(wavs, wav_lens)

        # NeMo preprocessor -> encoder
        proc_sig, proc_len = self.nemo_model.preprocessor(input_signal=wavs, length=lengths)
        enc_out, enc_len = self.nemo_model.encoder(audio_signal=proc_sig, length=proc_len)
        # NeMo encoder returns [B, D, T]; transpose to [B, T, D]
        enc_out = enc_out.transpose(1, 2)  # [B, S, D]

        # Our CTC head
        ctc_logits = self.ctc_lin(enc_out)
        return enc_out, ctc_logits

    def forward_decoder(
        self,
        encoder_out: torch.Tensor,
        targets: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced decoder forward.

        Args:
            encoder_out: [B, S, D]
            targets: [B, T] token IDs (BOS at t=0, next tokens following)
            wav_lens: [B] relative or absolute lengths of input waveforms (for src key padding mask)
            tgt_mask: optional subsequent mask for target (will be auto-generated if None)

        Returns:
            decoder_logits: [B, T, V]
        """
        B, S, D = encoder_out.shape
        src_key_padding_mask = None
        if wav_lens is not None:
            if wav_lens.dtype.is_floating_point:
                src_len = torch.round(wav_lens.clamp(0, 1) * S).long().clamp(min=1, max=S)
            else:
                src_len = wav_lens.to(torch.long).clamp(min=1, max=S)
            # True for padded positions
            src_key_padding_mask = torch.arange(S, device=encoder_out.device).expand(B, S) >= src_len.unsqueeze(1)

        # Prepare tgt embeddings with positional encoding
        # nn.Transformer expects [T, B, D]
        tgt_emb = self.token_emb(targets).transpose(0, 1)  # [T, B, D]
        tgt_emb = self.pos_enc(tgt_emb)

        # Build causal mask if not provided
        if tgt_mask is None:
            T = tgt_emb.size(0)
            causal = torch.full((T, T), float('-inf'), device=encoder_out.device)
            causal = torch.triu(causal, diagonal=1)
            tgt_mask = causal

        memory = encoder_out.transpose(0, 1)  # [S, B, D]

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # [T, B, D]

        dec_out = dec_out.transpose(0, 1)  # [B, T, D]
        decoder_logits = self.generator(dec_out)  # [B, T, V]
        return decoder_logits

    def decode_greedy(self, encoder_out: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """
        Greedy autoregressive decoding using our Transformer decoder.

        Returns:
            predictions: [B, <=max_len+1] including BOS at index 0
        """
        B, S, D = encoder_out.shape
        device = encoder_out.device
        preds = torch.ones(B, 1, dtype=torch.long, device=device)  # BOS=1 assumption
        for _ in range(max_len):
            logits = self.forward_decoder(encoder_out, preds)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            preds = torch.cat([preds, next_token], dim=1)
            if (next_token == 2).all():  # EOS=2
                break
        return preds
