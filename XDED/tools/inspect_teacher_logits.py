#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/zsim710/XDED/XDED')
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits

try:
    import sentencepiece as spm
except Exception:
    spm = None

def ctc_greedy_collapse(ids, blank_id=0):
    out, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if prev == i:
            continue
        out.append(i)
        prev = i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held_out', type=str, default='M08')
    ap.add_argument('--split', type=str, default='test', choices=['train','test'])
    ap.add_argument('--matching_mode', type=str, default='partial', choices=['strict','partial','all'])
    ap.add_argument('--min_teachers', type=int, default=10)
    ap.add_argument('--exclude_speakers', nargs='+', default=['M01'])
    ap.add_argument('--blank_index', type=int, default=0)
    ap.add_argument('--vocab_offset', type=int, default=1, help='teacher_id -> spm_id offset (1 if blank at 0)')
    ap.add_argument('--tokenizer_ckpt', type=str, required=False)
    ap.add_argument('--max_samples', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--use_decoder_logits', action='store_true', help='Use decoder logits instead of CTC logits')
    ap.add_argument('--agg', type=str, default='prob_mean', choices=['prob_mean','logprob_mean','logit_mean'])
    args = ap.parse_args()

    ds = LogitEnsembleDataset(
        held_out_speaker=args.held_out,
        split=args.split,
        use_decoder_logits=args.use_decoder_logits,
        matching_mode=args.matching_mode,
        min_teachers=args.min_teachers,
        exclude_speakers=args.exclude_speakers,
    )

    if args.max_samples:
        ds = torch.utils.data.Subset(ds, list(range(min(args.max_samples, len(ds)))))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_logits)

    sp = None
    if args.tokenizer_ckpt and spm is not None:
        sp = spm.SentencePieceProcessor()
        sp.Load(args.tokenizer_ckpt)

    def decode_ids(sp_ids):
        if sp is None:
            return ""
        return sp.DecodeIds(sp_ids).upper()

    first_word_ctr = Counter()
    top_id_ctr = Counter()
    total_frames = 0
    avg_entropy = 0.0

    total_utts = 0
    wra_correct = 0

    with torch.no_grad():
        for batch in tqdm(dl, desc='Inspect-teacher'):
            if 'teacher_logits' in batch:
                tlog = batch['teacher_logits']  # [B, T, L, V]
                lengths_list = batch['lengths']
                num_teachers = batch['num_teachers']
                B, T, L, V = tlog.shape
            else:
                # test split: wrap single-stream logits into teacher dimension
                logits = batch['logits']  # [B, L, V]
                lengths = batch['lengths']  # [B]
                B, L, V = logits.shape
                tlog = logits.unsqueeze(1)  # [B, 1, L, V]
                num_teachers = torch.ones(B, dtype=torch.long)
                lengths_list = [torch.tensor([int(lengths[i].item())]) for i in range(B)]
                T = 1

            probs = F.softmax(tlog, dim=-1)  # [B, T, L, V]
            log_probs = F.log_softmax(tlog, dim=-1)
            # Mask invalid frames per-teacher
            mask = torch.zeros((B, T, L), dtype=torch.bool)
            for b in range(B):
                nt = int(num_teachers[b])
                lens = lengths_list[b]
                for t in range(nt):
                    l = int(lens[t])
                    if l > 0:
                        mask[b, t, :min(l, L)] = True
            cnt = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

            if args.agg == 'prob_mean':
                tp = probs * mask.unsqueeze(-1).to(probs.dtype)
                avg = tp.sum(dim=1) / cnt
            elif args.agg == 'logprob_mean':
                tlp = log_probs * mask.unsqueeze(-1).to(log_probs.dtype)
                logp_avg = tlp.sum(dim=1) / cnt
                avg = torch.exp(logp_avg)
                avg = avg / avg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:  # logit_mean
                tl = tlog * mask.unsqueeze(-1).to(tlog.dtype)
                logit_avg = tl.sum(dim=1) / cnt
                avg = F.softmax(logit_avg, dim=-1)

            # Entropy over frames
            ent = -(avg.clamp(min=1e-8).log() * avg.clamp(min=1e-8)).sum(dim=-1)  # [B, L]
            avg_entropy += ent.sum().item()
            total_frames += int(mask.sum().item())

            pred = avg.argmax(dim=-1)  # [B, L]

            for i in range(B):
                seq = ctc_greedy_collapse(pred[i].tolist(), blank_id=args.blank_index)
                for tid in seq:
                    top_id_ctr[tid] += 1
                if sp is not None:
                    sp_ids = [j - args.vocab_offset for j in seq if (j - args.vocab_offset) >= 0]
                    hyp = decode_ids(sp_ids)
                    hyp_first = (hyp.split()[:1] or [""])[0]
                    if hyp_first:
                        first_word_ctr[hyp_first] += 1
                total_utts += 1

    print('\nTeacher logits inspection:')
    if total_frames > 0:
        print(f'- Avg teacher-avg entropy: {avg_entropy/total_frames:.3f} nats/frame')
    print('- Top-10 predicted token IDs after CTC collapse:')
    for tid, c in top_id_ctr.most_common(10):
        print(f'  id={tid}\tcount={c}')
    if first_word_ctr:
        print('- Top-20 first words:')
        for w, c in first_word_ctr.most_common(20):
            print(f'  {w}\t{c}')

if __name__ == '__main__':
    main()
