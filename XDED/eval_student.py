#!/usr/bin/env python3
"""
Evaluate the trained student Conformer model on M08 test set.
Generate predictions using CTC greedy decoding and check if outputs are reasonable.
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torchaudio
import sentencepiece as spm

# Import student model and dataset
from models.student_conformer import StudentConformer
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset

# Lazy import for NeMo model (optional dependency)
try:
    from models.nemo_hybrid_student import NeMoHybridStudent
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    NeMoHybridStudent = None

# Import SpeechBrain tokenizer for decoding
import speechbrain as sb
from speechbrain.dataio.encoder import CTCTextEncoder
from speechbrain.utils import edit_distance
import torch.nn.functional as F




def build_decoder(spm_model_path=None, tokenizer_ckpt=None, vocab_offset=1, tokens_file=None, units_file=None):
    """Build a decoding function to map token IDs -> text.

        Preference order:
            1) SentencePiece model (.model)
            2) CTCTextEncoder checkpoint (.ckpt)
            3) tokens.txt / units.txt (one token per line, SPM-like pieces)

    Args:
        spm_model_path: Path to SentencePiece .model file used by SA teachers.
        tokenizer_ckpt: Path to SpeechBrain CTCTextEncoder checkpoint.
        vocab_offset: Offset between student vocab IDs and tokenizer IDs.
                      Typically 1 when CTC blank_index=0 and SPM ids start at 0.

    Returns:
        decode_fn: callable(List[int]) -> str, or None if not available
        info: dict with tokenizer metadata
    """

    if spm_model_path and os.path.exists(spm_model_path):
        print(f"Using SentencePiece model: {spm_model_path}")
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)

        def decode_fn(token_ids):
            if not token_ids:
                return ""
            # Map student ids -> spm ids by subtracting offset
            spm_ids = [tid - vocab_offset for tid in token_ids]
            # Guard against out-of-range after offset
            spm_ids = [i for i in spm_ids if 0 <= i < sp.vocab_size()]
            return sp.decode(spm_ids)

        return decode_fn, {"type": "sentencepiece", "path": spm_model_path, "vocab_offset": vocab_offset}

    if tokenizer_ckpt and os.path.exists(tokenizer_ckpt):
        # Many SpeechBrain repos save the SPM model as tokenizer.ckpt (binary),
        # so try loading it as SentencePiece first; if that fails, fall back to CTCTextEncoder
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_ckpt)
            print(f"Using SentencePiece model (ckpt): {tokenizer_ckpt}")

            def decode_fn(token_ids):
                if not token_ids:
                    return ""
                spm_ids = [tid - vocab_offset for tid in token_ids]
                spm_ids = [i for i in spm_ids if 0 <= i < sp.vocab_size()]
                return sp.decode(spm_ids)

            return decode_fn, {"type": "sentencepiece_ckpt", "path": tokenizer_ckpt, "vocab_offset": vocab_offset}
        except Exception:
            print(f"Falling back to CTCTextEncoder for: {tokenizer_ckpt}")
            enc = CTCTextEncoder()
            enc.load(tokenizer_ckpt)

            def decode_fn(token_ids):
                if not token_ids:
                    return ""
                t = torch.tensor([token_ids], dtype=torch.long)
                out = enc.decode_ndim(t)
                return out[0] if isinstance(out, list) else str(out)

            return decode_fn, {"type": "ctc_text_encoder", "path": tokenizer_ckpt}

    # Try tokens.txt / units.txt mapping
    token_list_path = tokens_file or units_file
    if token_list_path and os.path.exists(token_list_path):
        print(f"Using token list: {token_list_path}")
        with open(token_list_path, 'r') as f:
            tokens = [line.strip() for line in f if line.strip() != ""]

        def pieces_to_text(pieces):
            # Simple SPM-style reconstruction: '▁' denotes word boundary/space
            text = ""
            for p in pieces:
                if p.startswith("▁"):
                    if len(text) > 0:
                        text += " "
                    text += p.lstrip("▁")
                else:
                    text += p
            return text

        def decode_fn(token_ids):
            if not token_ids:
                return ""
            idxs = [tid - vocab_offset for tid in token_ids]
            pieces = []
            for i in idxs:
                if 0 <= i < len(tokens):
                    pieces.append(tokens[i])
            return pieces_to_text(pieces)

        return decode_fn, {"type": "tokens_list", "path": token_list_path, "vocab_offset": vocab_offset}

    print("WARNING: No tokenizer provided or file not found. Will keep token IDs only.")
    return None, {"type": "none"}


def greedy_decode_ctc(logits, blank_index=0):
    """
    Greedy CTC decoding: take argmax and collapse repeats.
    
    Args:
        logits: [seq_len, vocab_size] - raw CTC logits
        blank_index: index of blank token (usually 0)
    
    Returns:
        List of token IDs (non-blank, non-repeated)
    """
    # Take argmax to get most likely token at each timestep
    predictions = torch.argmax(logits, dim=-1)  # [seq_len]
    
    # Collapse consecutive duplicates
    collapsed = []
    prev_token = None
    for token_id in predictions.tolist():
        if token_id != prev_token:
            if token_id != blank_index:  # Skip blank tokens
                collapsed.append(token_id)
            prev_token = token_id
    
    return collapsed


def ctc_greedy_collapse(ids, blank_id=0):
    out=[]
    prev=None
    for i in ids:
        if i==blank_id: 
            prev=None
            continue
        if prev==i: 
            continue
        out.append(i)
        prev=i
    return out


def decode_teacher_batch(teacher_logits, lengths_list, num_teachers, blank_index, vocab_offset, sp_decode_ids):
    # teacher_logits: [B, T, L, V]
    B, T, L, V = teacher_logits.shape
    device = teacher_logits.device
    with torch.no_grad():
        probs = F.softmax(teacher_logits, dim=-1)  # [B,T,L,V]
        # mask per teacher length
        mask = torch.zeros((B,T,L), dtype=torch.bool, device=device)
        for b in range(B):
            nt = int(num_teachers[b].item())
            lens = lengths_list[b]
            for t in range(nt):
                l = int(lens[t].item())
                if l>0:
                    mask[b,t,:min(l,L)] = True
        probs = probs * mask.unsqueeze(-1).to(probs.dtype)
        cnt = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,L,1]
        avg = probs.sum(dim=1) / cnt  # [B,L,V]
        pred = avg.argmax(dim=-1)     # [B,L]
        # collapse CTC and decode
        texts=[]
        ids_collapsed=[]
        for b in range(B):
            seq = ctc_greedy_collapse(pred[b].tolist(), blank_id=blank_index)
            ids_collapsed.append(seq)
            # map offset before decode
            sp_ids = [i - vocab_offset for i in seq if (i - vocab_offset) >= 0]
            texts.append(sp_decode_ids(sp_ids))
        return texts, ids_collapsed


def evaluate_model(checkpoint_path, test_dataset, decode_fn, tokenizer_info, device='cuda', num_samples=20, blank_index=0, decode_mode='ctc', max_decode_len=20):
    """
    Evaluate student model on test set.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_dataset: LogitEnsembleDataset for M08 test set
        tokenizer: Tokenizer for decoding
        device: Device to run on
        num_samples: Number of samples to show detailed predictions
    
    Returns:
        Dictionary with evaluation results
    """
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model backbone from checkpoint keys
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    is_nemo_backbone = any('nemo_model' in key for key in state_dict_keys)
    
    # Initialize model based on backbone
    if is_nemo_backbone:
        if not NEMO_AVAILABLE:
            raise RuntimeError("This checkpoint uses NeMo backbone, but NeMo is not installed. "
                             "Please install NeMo: pip install nemo_toolkit[asr]")
        print("Detected NeMo backbone. Initializing NeMoHybridStudent...")
        model = NeMoHybridStudent(
            vocab_size=5000,
            nemo_model_name='nvidia/stt_en_conformer_ctc_small',
            num_decoder_layers=4,
            dropout=0.1
        ).to(device)
    else:
        print("Detected SpeechBrain backbone. Initializing StudentConformer...")
        model = StudentConformer(
            vocab_size=5000,
            d_model=144,
            nhead=4,
            num_encoder_layers=8,
            num_decoder_layers=4,
            d_ffn=1024,  # Must match training config
            dropout=0.1
        ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluation loop
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    print(f"Showing detailed predictions for first {num_samples} samples:\n")
    
    all_predictions_text = []
    all_predictions_tokens = []
    all_targets = []
    all_utt_ids = []
    wer_S = wer_D = wer_I = 0
    wer_N = 0
    # UA-Speech is a single-word task; compute Word Recognition Accuracy (WRA)
    wra_total = 0
    wra_correct = 0

    def _normalize_word(s: str) -> str:
        return (s or "").strip().lower()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            # Get batch
            batch = test_dataset[idx]
            
            # Prepare inputs
            audio = batch['audio'].unsqueeze(0).to(device)  # [1, time]
            audio_lens = torch.tensor([1.0]).to(device)  # Full length
            
            # Forward pass through student model
            encoder_out, ctc_logits = model(audio, audio_lens)

            # Choose decoding mode
            if decode_mode == 'decoder':
                # Autoregressive greedy decode using student decoder
                dec_tokens = model.decode_greedy(encoder_out, max_len=max_decode_len)[0].tolist()
                # strip BOS/EOS (1/2) and any padding zeros
                predicted_tokens = [t for t in dec_tokens if t not in (0, 1, 2)]
            else:
                # Greedy CTC decode
                # ctc_logits: [1, T, V] -> [T, V]
                ctc_logits_squeezed = ctc_logits.squeeze(0)
                predicted_tokens = greedy_decode_ctc(ctc_logits_squeezed, blank_index=blank_index)
            
            # Decode to text if tokenizer available
            target_text = batch['target_text']
            if decode_fn is not None:
                predicted_text = decode_fn(predicted_tokens)
            else:
                predicted_text = f"Tokens: {predicted_tokens[:50]}"
            
            # Store results
            all_predictions_tokens.append(predicted_tokens)
            all_predictions_text.append(predicted_text)
            all_targets.append(target_text)
            all_utt_ids.append(batch['utterance_id'])
            
            # Compute per-utt WER stats if possible
            if decode_fn is not None and isinstance(target_text, str) and target_text.strip() != "":
                ref_words = target_text.strip().split()
                hyp_words = predicted_text.strip().split()
                # Get alignment details to aggregate corpus-level S, D, I
                try:
                    (details,) = edit_distance.wer_details_for_batch(
                        ["utt"], [ref_words], [hyp_words], compute_alignments=True
                    )
                    alignment = details.get("alignment", [])
                    for item in alignment:
                        # alignment entries may be tuples like (op, ref, hyp) or dicts
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            op = item[0]
                        elif isinstance(item, dict):
                            op = item.get("type") or item.get("op")
                        else:
                            op = None
                        if op == "S":
                            wer_S += 1
                        elif op == "D":
                            wer_D += 1
                        elif op == "I":
                            wer_I += 1
                except Exception:
                    # If WER computation fails for any reason, skip counting for this utterance
                    pass
                # N is total reference word count across utterances
                wer_N += len(ref_words)

            # Compute WRA on first decoded word (single-word setting)
            if isinstance(target_text, str) and target_text.strip() != "":
                ref_first = _normalize_word(target_text)
                hyp_first = _normalize_word(predicted_text.split()[0] if predicted_text.strip() else "")
                wra_total += 1
                if ref_first == hyp_first:
                    wra_correct += 1

            # Show detailed output for first N samples
            if idx < num_samples:
                print(f"\n{'='*80}")
                print(f"Sample {idx+1}/{num_samples}")
                print(f"Utterance ID: {batch['utterance_id']}")
                print(f"Speaker: {batch['speaker']}")
                print(f"Target: {target_text}")
                print(f"Predicted: {predicted_text}")
                print(f"Predicted tokens ({len(predicted_tokens)}): {predicted_tokens[:20]}...")
                print(f"CTC logits shape: {ctc_logits.shape}")
                print(f"Encoder output shape: {encoder_out.shape}")
    
    # Summary statistics based on token sequences
    num_samples_eval = len(all_predictions_tokens)
    avg_pred_tokens = sum(len(toks) for toks in all_predictions_tokens) / max(1, num_samples_eval)
    empty_pred_frac = sum(1 for toks in all_predictions_tokens if len(toks) == 0) / max(1, num_samples_eval)
    unique_token_seqs = len({tuple(toks) for toks in all_predictions_tokens})

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {num_samples_eval}")
    print(f"Average predicted tokens: {avg_pred_tokens:.2f}")
    print(f"Empty predictions: {empty_pred_frac*100:.1f}%")
    print(f"Unique token sequences: {unique_token_seqs}/{num_samples_eval}")
    # Hypothesis word stats (after decoding)
    try:
        avg_hyp_words = sum(len((txt or "").strip().split()) for txt in all_predictions_text) / max(1, num_samples_eval)
        print(f"Average decoded words per utt: {avg_hyp_words:.2f}")
    except Exception:
        pass

    wer_value = None
    if wer_N > 0:
        wer_value = (wer_S + wer_D + wer_I) / max(1, wer_N)
        print(f"Corpus WER: {wer_value*100:.2f}%  (S={wer_S}, D={wer_D}, I={wer_I}, N={wer_N})")
    else:
        print("Corpus WER: n/a (no reference text available)")

    # WRA for single-word evaluation
    wra_value = (wra_correct / max(1, wra_total)) if wra_total > 0 else None
    if wra_value is not None:
        print(f"WRA (first-word match, case-insensitive): {wra_value*100:.2f}%  ({wra_correct}/{wra_total})")
    
    if unique_token_seqs < 10:
        print("\n⚠️  WARNING: Very few unique predictions! Model may not be learning.")
    elif unique_token_seqs > num_samples_eval * 0.5:
        print("\n✅ Good diversity in predictions!")
    
    # Save detailed results
    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'num_samples': num_samples_eval,
        'tokenizer': tokenizer_info,
        'wer': {
            'value': wer_value,
            'S': wer_S,
            'D': wer_D,
            'I': wer_I,
            'N': wer_N,
        },
        'wra': {
            'value': wra_value,
            'correct': wra_correct,
            'total': wra_total,
        },
        'predictions': [
            {
                'utterance_id': utt_id,
                'target': target,
                'prediction_tokens': toks,
                'prediction_text': txt,
            }
            for utt_id, target, toks, txt in zip(all_utt_ids, all_targets, all_predictions_tokens, all_predictions_text)
        ]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate student Conformer model')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/student_M08/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--held_out', type=str, default='M08',
                       help='Held-out speaker to evaluate on')
    parser.add_argument('--matching_mode', type=str, default='partial',
                       choices=['strict', 'partial', 'all'],
                       help='Matching mode for ensemble loading')
    parser.add_argument('--min_teachers', type=int, default=10,
                       help='Minimum teachers for partial mode')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of detailed samples to show')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--output', type=str, default='eval_results_M08.json',
                       help='Output file for results')
    parser.add_argument('--blank_index', type=int, default=0, help='CTC blank index')
    parser.add_argument('--decode_mode', type=str, default='ctc', choices=['ctc','decoder'], help='Use CTC greedy or decoder greedy for eval')
    parser.add_argument('--max_decode_len', type=int, default=20, help='Max length for decoder greedy decoding')
    # Tokenizer / decoding options
    parser.add_argument('--spm_model', type=str, default=None,
                        help='Path to SentencePiece .model used by SA teachers')
    parser.add_argument('--tokenizer_ckpt', type=str, default=None,
                        help='Path to SpeechBrain CTCTextEncoder checkpoint')
    parser.add_argument("--vocab_offset", type=int, default=None, help="student_id -> spm_id offset; default 1 if blank_index==0 else 0")
    parser.add_argument("--report_teacher", action="store_true", help="decode teacher ensemble and report WER/WRA")
    parser.add_argument('--tokens_file', type=str, default=None,
                        help='Path to tokens.txt/units.txt (one token per line) for piece-based decoding')
    parser.add_argument('--units_file', type=str, default=None,
                        help='Alias for tokens_file; path to units.txt file')
    
    args = parser.parse_args()
    # default offset
    if args.vocab_offset is None:
        args.vocab_offset = 1 if args.blank_index == 0 else 0
    
    # Configuration
    logit_root_dir = "/home/zsim710/XDED/speechbrain/exp_results/logit_extraction"
    csv_dir = "/home/zsim710/partitions/uaspeech/by_speakers"
    
    print("="*80)
    print("STUDENT MODEL EVALUATION")
    print("="*80)
    print(f"Held-out speaker: {args.held_out}")
    print(f"Matching mode: {args.matching_mode}")
    print(f"Min teachers: {args.min_teachers}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Load tokenizer
    print("\nPreparing tokenizer/decoder...")
    # If user did not specify vocab_offset but blank_index != 0, auto-adjust
    vocab_offset = args.vocab_offset
    if args.blank_index != 0 and args.vocab_offset == 1:
        # Common case: when blank is not 0, there may be no offset
        vocab_offset = 0
    decode_fn, tok_info = build_decoder(
        args.spm_model, args.tokenizer_ckpt, vocab_offset=vocab_offset,
        tokens_file=args.tokens_file, units_file=args.units_file
    )
    # expose a low-level id->text for teacher decode
    spm_proc = tok_info.get("spm_proc")
    def sp_decode_ids(sp_ids):
        if spm_proc is not None:
            return spm_proc.DecodeIds(sp_ids).upper()
        # fallback token list
        pieces = [tok_info["id2piece"].get(i, "") for i in sp_ids]
        return ("".join(pieces)).replace("▁"," ").strip().upper()

    # Create test dataset for held-out speaker
    print(f"\nCreating test dataset for {args.held_out}...")
    test_dataset = LogitEnsembleDataset(
        logit_root_dir=logit_root_dir,
        held_out_speaker=args.held_out,
        split='test',
        matching_mode=args.matching_mode,
        min_teachers=args.min_teachers,
        exclude_speakers=['M01'],  # M01 is outlier
        csv_dir=csv_dir
    )
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Evaluate model
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        test_dataset=test_dataset,
        decode_fn=decode_fn,
        tokenizer_info=tok_info,
        device=args.device,
        num_samples=args.num_samples,
        blank_index=args.blank_index,
        decode_mode=args.decode_mode,
        max_decode_len=args.max_decode_len,
    )
    
    # Save results
    output_path = args.output
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving detailed results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print(f"\nNext steps based on results:")
    print("1. Check if predictions look like real words")
    print("2. If predictions are reasonable: fix LR scheduler and continue training")
    print("3. If predictions are garbage: debug model architecture or loss function")
    print("4. If predictions are good: compute WER and compare to baseline")


if __name__ == '__main__':
    main()
