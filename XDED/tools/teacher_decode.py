#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sentencepiece as spm

import sys
sys.path.insert(0, '/home/zsim710/XDED/XDED')
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits

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
    ap.add_argument("--held_out", type=str, default="M08")
    ap.add_argument("--matching_mode", type=str, default="partial", choices=["strict","partial","all"])
    ap.add_argument("--min_teachers", type=int, default=10)
    ap.add_argument("--blank_index", type=int, default=0)
    ap.add_argument("--tokenizer_ckpt", type=str, required=True)
    ap.add_argument("--vocab_offset", type=int, default=1, help="teacher_id -> spm_id")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    ds = LogitEnsembleDataset(
        held_out_speaker=args.held_out,
        split="test",
        use_decoder_logits=False,
        matching_mode=args.matching_mode,
        min_teachers=args.min_teachers,
        exclude_speakers=['M01'],
    )
    if args.max_samples:
        ds = torch.utils.data.Subset(ds, list(range(min(args.max_samples, len(ds)))))

    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_logits,
                    num_workers=args.num_workers)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_ckpt)

    def decode_ids(sp_ids):
        return sp.DecodeIds(sp_ids).upper()

    total = 0
    wra_correct = 0
    wer_N = 0
    wer_edits = 0
    from collections import Counter
    first_word_ctr = Counter()

    with torch.no_grad():
        for batch in tqdm(dl, desc="Teacher-decode"):
            tlog = batch["teacher_logits"]            # [B, T, L, V]
            lengths_list = batch["lengths"]           # list of len B; each is [Ti]
            num_teachers = batch["num_teachers"]      # [B]
            B, T, L, V = tlog.shape

            probs = F.softmax(tlog, dim=-1)           # [B, T, L, V]
            mask = torch.zeros((B, T, L), dtype=torch.bool)
            for b in range(B):
                nt = int(num_teachers[b])
                lens = lengths_list[b]
                for t in range(nt):
                    l = int(lens[t])
                    if l > 0:
                        mask[b, t, :min(l, L)] = True
            probs = probs * mask.unsqueeze(-1).to(probs.dtype)
            cnt = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, L, 1]
            avg = probs.sum(dim=1) / cnt                      # [B, L, V]
            pred = avg.argmax(dim=-1)                         # [B, L]

            for i in range(B):
                ids = ctc_greedy_collapse(pred[i].tolist(), blank_id=args.blank_index)
                sp_ids = [j - args.vocab_offset for j in ids if (j - args.vocab_offset) >= 0]
                hyp = decode_ids(sp_ids)
                ref = (batch.get("target_text", [""]*B)[i] or "").strip().upper()

                # stats
                hyp_first = (hyp.split()[:1] or [""])[0]
                ref_first = (ref.split()[:1] or [""])[0]
                if hyp_first:
                    first_word_ctr[hyp_first] += 1

                if ref:
                    # simple WRA
                    total += 1
                    if hyp_first == ref_first:
                        wra_correct += 1
                    # word-level Levenshtein
                    r = ref.split(); h = hyp.split()
                    m = [[0]*(len(h)+1) for _ in range(len(r)+1)]
                    for a in range(len(r)+1): m[a][0]=a
                    for b in range(len(h)+1): m[0][b]=b
                    for a in range(1,len(r)+1):
                        for b2 in range(1,len(h)+1):
                            cost = 0 if r[a-1]==h[b2-1] else 1
                            m[a][b2] = min(m[a-1][b2]+1, m[a][b2-1]+1, m[a-1][b2-1]+cost)
                    wer_edits += m[-1][-1]
                    wer_N += len(r)

    print("\nTeacher ensemble summary")
    if total:
        print(f"- WRA (first word): {100.0*wra_correct/total:.2f}%  ({wra_correct}/{total})")
    if wer_N:
        print(f"- WER: {100.0*wer_edits/wer_N:.2f}%  (edits={wer_edits}, N={wer_N})")
    print("- Top-20 first words:")
    for w,c in first_word_ctr.most_common(20):
        print(f"  {w}\t{c}")

if __name__ == "__main__":
    main()