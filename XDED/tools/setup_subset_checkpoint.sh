#!/bin/sh
set -eu

HELD_OUT="M01"
AVG_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models"
OUT_DIR="/home/zsim710/XDED/XDED/results/speaker_averaging/inference_checkpoints/${HELD_OUT}_subset"
TOKENIZER_DIR="/home/zsim710/XDED/tokenizers/sa_official"
REF_CKPT="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00"

echo "Creating checkpoint dir: $OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Copying tokenizer + hyperparams"
cp "$TOKENIZER_DIR/hyperparams.yaml" "$OUT_DIR/"
cp "$TOKENIZER_DIR/tokenizer.ckpt" "$OUT_DIR/"
cp -r "$TOKENIZER_DIR/tokenizer" "$OUT_DIR/" 2>/dev/null || true

echo "Copying normalizer"
cp "$REF_CKPT/normalizer.ckpt" "$OUT_DIR/"

echo "Placing averaged model as model.ckpt"
cp "$AVG_DIR/${HELD_OUT}_subset_samelevel.pt" "$OUT_DIR/model.ckpt"

echo "Listing files"
ls -lh "$OUT_DIR"

echo "Done"
