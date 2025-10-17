#!/usr/bin/env python3
"""
Run WRA (Weighted/naive Recognition Averaging) across 14 SA models for a held-out speaker.

For each utterance:
- Load 14 SA checkpoints (all except held-out) with the SA architecture.
- Compute CTC logits per model.
- Aggregate across models by prob_mean, logprob_mean, or logit_mean.
- Greedy decode and compute WER.

Assumptions:
- All SA models share the same architecture (CNN 2 blocks (64,32), Transformer d_model=144, 12/4, input_size=640)
- Use a consistent tokenizer across models (provided via --tokenizer or loaded from averaged checkpoint dir)
- Use per-utterance mean/std normalization by default (robust across missing normalizer stats)
"""

import os
import sys
import json
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# Add SpeechBrain paths
sys.path.insert(0, '/home/zsim710/XDED/conformer/conformer-asr')

from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.features import Fbank
from speechbrain.dataio.dataio import read_audio
import sentencepiece
from jiwer import wer


def build_sa_modules(device='cuda'):
    """Create SA model modules (CNN, Transformer, CTC head) with correct config."""
    cnn = ConvolutionFrontEnd(
        input_shape=(8, 10, 80),
        num_blocks=2,
        num_layers_per_block=1,
        out_channels=(64, 32),
        kernel_sizes=(3, 3),
        strides=(2, 2),
        residuals=(False, False)
    ).to(device)

    transformer = TransformerASR(
        input_size=640,
        tgt_vocab=5000,
        d_model=144,
        nhead=4,
        num_encoder_layers=12,
        num_decoder_layers=4,
        d_ffn=1024,
        dropout=0.1,
        activation=torch.nn.GELU,
        encoder_module='transformer',
        attention_type='regularMHA',
        normalize_before=True,
        causal=False
    ).to(device)

    ctc_lin = Linear(input_size=144, n_neurons=5000).to(device)

    cnn.eval(); transformer.eval(); ctc_lin.eval()
    return cnn, transformer, ctc_lin


def load_sa_checkpoint_into_modules(ckpt_path: str, cnn, transformer, ctc_lin, device='cuda'):
    """Load numeric-prefixed SA checkpoint weights into modules."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Handle possible nested 'model' key
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']

    cnn_state = {k.replace('0.', ''): v for k, v in sd.items() if k.startswith('0.')}
    tr_state = {k.replace('1.', ''): v for k, v in sd.items() if k.startswith('1.')}
    ctc_state = {k.replace('2.', ''): v for k, v in sd.items() if k.startswith('2.')}

    cnn.load_state_dict(cnn_state, strict=False)
    transformer.load_state_dict(tr_state, strict=False)
    ctc_lin.load_state_dict(ctc_state, strict=False)


def extract_ctc_logits(audio_path: str, cnn, transformer, ctc_lin, fbank, device='cuda') -> torch.Tensor:
    """Compute CTC logits [T, V] for one audio file using given modules."""
    with torch.no_grad():
        signal = read_audio(audio_path).to(device).unsqueeze(0)
        feats = fbank(signal)
        # Simple per-utterance normalization (robust to missing global stats)
        mean = feats.mean(dim=(1, 2), keepdim=True)
        std = feats.std(dim=(1, 2), keepdim=True) + 1e-5
        feats = (feats - mean) / std

        src = cnn(feats)
        enc = transformer.encode(src)
        logits = ctc_lin(enc).squeeze(0)  # [T, V]
        return logits


def aggregate_logits(models_logits: List[torch.Tensor], how: str = 'logit_mean') -> torch.Tensor:
    """Aggregate a list of [T, V] logits/probabilities across models."""
    # Ensure same length
    T = min([x.shape[0] for x in models_logits])
    if any(x.shape[0] != T for x in models_logits):
        models_logits = [x[:T] for x in models_logits]

    if how == 'prob_mean':
        probs = [F.softmax(x, dim=-1) for x in models_logits]
        avg = torch.stack(probs, dim=0).mean(dim=0)
        return torch.log(avg + 1e-12)
    elif how == 'logprob_mean':
        logprobs = [F.log_softmax(x, dim=-1) for x in models_logits]
        return torch.stack(logprobs, dim=0).mean(dim=0)
    else:  # 'logit_mean'
        logits = torch.stack(models_logits, dim=0).mean(dim=0)
        return F.log_softmax(logits, dim=-1)


def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int = 0) -> List[int]:
    """Greedy CTC collapse from framewise log-probs [T, V] -> token ids."""
    ids = torch.argmax(log_probs, dim=-1).tolist()
    out, prev = [], None
    for t in ids:
        if t != blank_id and t != prev:
            out.append(t)
        prev = t
    return out


def run_wra_for_speaker(held_out: str, speaker_ckpt_json: str, test_csv: str, tokenizer_path: str,
                        agg: str = 'logit_mean', device: str = 'cuda', limit: int = None,
                        verbose_every: int = 50) -> Dict:
    """Run WRA on one held-out speaker and compute WER."""
    with open(speaker_ckpt_json, 'r') as f:
        spk2ckpt = json.load(f)

    donor_speakers = [s for s in spk2ckpt.keys() if s != held_out]
    donor_ckpts = [spk2ckpt[s] for s in donor_speakers]
    print(f"Held-out: {held_out} | Using {len(donor_ckpts)} donor SA models")

    # Tokenizer
    tokenizer = sentencepiece.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    # Feature extractor
    fbank = Fbank(sample_rate=16000, n_fft=400, n_mels=80).to(device)
    fbank.eval()

    # Read test CSV
    df = pd.read_csv(test_csv)
    if limit is not None:
        df = df.head(limit)

    predictions, references, details = [], [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"WRA {held_out}"):
        wav = row['wav']
        ref = str(row['wrd']).strip().upper()

        # Collect logits from all donor models
        logits_list = []
        for ckpt in donor_ckpts:
            try:
                cnn, tr, ctc = build_sa_modules(device)
                load_sa_checkpoint_into_modules(ckpt, cnn, tr, ctc, device)
                logits = extract_ctc_logits(wav, cnn, tr, ctc, fbank, device)
                logits_list.append(logits)
            except Exception as e:
                print(f"Warning: failed model {ckpt}: {e}")
                continue

        if len(logits_list) == 0:
            hyp = ""
        else:
            agg_lp = aggregate_logits(logits_list, how=agg)  # [T, V] log-probs
            ids = ctc_greedy_decode(agg_lp, blank_id=0)
            hyp = tokenizer.decode(ids) if len(ids) > 0 else ""

        hyp = hyp.strip().upper()
        predictions.append(hyp)
        references.append(ref)
        details.append({'audio_path': wav, 'reference': ref, 'prediction': hyp, 'num_models': len(logits_list)})

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{i+1}] REF: {ref} | HYP: {hyp}")

    wer_pct = wer(references, predictions) * 100.0
    result = {
        'held_out': held_out,
        'num_utts': len(df),
        'wer': wer_pct,
        'accuracy': 100.0 - wer_pct,
        'agg': agg,
        'details': details,
    }
    print(f"\nWRA on {held_out} | {agg} | WER: {wer_pct:.2f}% | Acc: {100.0-wer_pct:.2f}%")
    return result


def main():
    ap = argparse.ArgumentParser(description='Run WRA across 14 SA models for a held-out speaker')
    ap.add_argument('--held_out', type=str, required=True)
    ap.add_argument('--speaker_ckpt_json', type=str, default='/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json')
    ap.add_argument('--test_csv_dir', type=str, default='/home/zsim710/partitions/uaspeech/by_speakers')
    ap.add_argument('--tokenizer', type=str, default='/home/zsim710/XDED/tokenizers/sa_official/tokenizer')
    ap.add_argument('--agg', type=str, default='logit_mean', choices=['prob_mean', 'logprob_mean', 'logit_mean'])
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    test_csv = os.path.join(args.test_csv_dir, f"{args.held_out}.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    res = run_wra_for_speaker(
        held_out=args.held_out,
        speaker_ckpt_json=args.speaker_ckpt_json,
        test_csv=test_csv,
        tokenizer_path=args.tokenizer,
        agg=args.agg,
        device=args.device,
        limit=args.limit,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        import json as _json
        with open(args.output, 'w') as f:
            _json.dump(res, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()
