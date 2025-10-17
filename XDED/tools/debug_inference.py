"""
Debug script to check what the model is predicting.

Extended to support encoder-only averaging: use averaged encoder weights
while taking CNN/positional/head weights from a donor SA checkpoint.
Also adds simple per-utterance normalization as a toggle.
"""
import torch
import sys
sys.path.insert(0, '/home/zsim710/XDED/conformer/conformer-asr')
sys.path.insert(0, '/home/zsim710/XDED/XDED')

from speechbrain.dataio.dataio import read_audio
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
from speechbrain.nnet.linear import Linear
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.features import Fbank
from speechbrain.nnet.normalization import LayerNorm
import sentencepiece as spm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--avg_ckpt_dir', type=str, required=True, help='Path to averaged checkpoint directory (contains model.ckpt, tokenizer, normalizer.ckpt)')
parser.add_argument('--donor_ckpt', type=str, default=None, help='Path to donor SA model.ckpt (numeric prefix format)')
parser.add_argument('--encoder_only_avg', action='store_true', help='Compose averaged encoder with donor CNN/non-encoder/heads')
parser.add_argument('--simple_norm', action='store_true', help='Use simple per-utterance normalization instead of global normalizer')
parser.add_argument('--audio', type=str, default="/mnt/DataSets/UASpeech/audio/F02/F02_B3_C10_M2.wav", help='Path to a WAV file to probe')
args = parser.parse_args()

# Paths
checkpoint_dir = args.avg_ckpt_dir
test_audio = args.audio  # e.g., "LINE"

device = 'cuda'

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(f"{checkpoint_dir}/tokenizer")

# Load normalizer (used unless --simple_norm)
normalizer_state = torch.load(f"{checkpoint_dir}/normalizer.ckpt", map_location=device, weights_only=False)
normalizer = InputNormalization(norm_type='global')
normalizer.load_state_dict(normalizer_state, strict=False)
normalizer = normalizer.to(device)
normalizer.eval()

# Create feature extractor
compute_features = Fbank(n_mels=80).to(device)

# Create CNN - CORRECT configuration from SA models
cnn = ConvolutionFrontEnd(
    input_shape=(None, None, 80),
    num_blocks=2,  # CORRECT: SA models use 2, not 3!
    num_layers_per_block=1,
    out_channels=(64, 32),  # CORRECT: SA models use (64, 32), not (64, 64, 64)!
    kernel_sizes=(3, 3),  # CORRECT: SA models use (3, 3), not (5, 5, 1)!
    strides=(2, 2),  # Both stride 2
    residuals=(False, False),
    norm=LayerNorm
).to(device)

# Create transformer
transformer = TransformerASR(
    input_size=640,  # CORRECT: 20 freq bins × 32 channels = 640, not 1280!
    tgt_vocab=5000,
    d_model=144,  # CORRECT: SA models use 144, not 512!
    nhead=4,
    num_encoder_layers=12,
    num_decoder_layers=4,  # CORRECT: SA models use 4, not 6!
    d_ffn=1024,  # CORRECT: SA models use 1024, not 2048!
    dropout=0.1,
    activation=torch.nn.GELU,
    encoder_module='transformer',
    attention_type='regularMHA',
    normalize_before=True,
    causal=False
).to(device)

# Output layers
ctc_lin = Linear(input_size=144, n_neurons=5000).to(device)  # d_model=144
seq_lin = Linear(input_size=144, n_neurons=5000).to(device)  # d_model=144

print("Loading averaged checkpoint...")
avg_ckpt = torch.load(f"{checkpoint_dir}/model.ckpt", map_location=device)
print(f"Checkpoint top-level keys: {list(avg_ckpt.keys())}")

# Averaged model packs state under 'model'
if 'model' in avg_ckpt:
    print("Found 'model' key in averaged checkpoint")
    model_state = avg_ckpt['model']
    print(f"Held out speaker: {avg_ckpt.get('held_out', 'unknown')}")
    print(f"Number of models averaged: {avg_ckpt.get('num_models_averaged', 'unknown')}")
else:
    model_state = avg_ckpt

print(f"Model state keys (first 20): {list(model_state.keys())[:20]}")
print(f"Total keys in model state: {len(model_state.keys())}")

# Prepare averaged partitions: 0.=CNN, 1.=Transformer, 2.=CTC, 3.=Seq
avg_cnn = {k.replace('0.', ''): v for k, v in model_state.items() if k.startswith('0.')}
avg_tr_all = {k.replace('1.', ''): v for k, v in model_state.items() if k.startswith('1.')}
avg_tr_enc = {k.replace('1.', ''): v for k, v in model_state.items() if k.startswith('1.encoder.')}
avg_ctc = {k.replace('2.', ''): v for k, v in model_state.items() if k.startswith('2.')}
avg_seq = {k.replace('3.', ''): v for k, v in model_state.items() if k.startswith('3.')}

donor_sd = None
if args.donor_ckpt and os.path.isfile(args.donor_ckpt):
    print(f"Loading donor checkpoint: {args.donor_ckpt}")
    donor_loaded = torch.load(args.donor_ckpt, map_location=device)
    # Some checkpoints may pack under 'model'
    donor_sd = donor_loaded.get('model', donor_loaded)

if args.encoder_only_avg and donor_sd is not None:
    print("Composing encoder-only averaged model (avg encoder + donor others)...")
    donor_cnn = {k.replace('0.', ''): v for k, v in donor_sd.items() if k.startswith('0.')}
    donor_tr_nonenc = {k.replace('1.', ''): v for k, v in donor_sd.items() if k.startswith('1.') and not k.startswith('1.encoder.')}
    donor_ctc = {k.replace('2.', ''): v for k, v in donor_sd.items() if k.startswith('2.')}
    donor_seq = {k.replace('3.', ''): v for k, v in donor_sd.items() if k.startswith('3.')}

    cnn_state = donor_cnn if len(donor_cnn) > 0 else avg_cnn
    transformer_state = donor_tr_nonenc.copy()
    transformer_state.update(avg_tr_enc)
    ctc_state = donor_ctc if len(donor_ctc) > 0 else avg_ctc
    seq_state = donor_seq if len(donor_seq) > 0 else avg_seq
else:
    # Default: fully averaged
    cnn_state = avg_cnn
    transformer_state = avg_tr_all
    ctc_state = avg_ctc
    seq_state = avg_seq

print(f"Loading CNN with {len(cnn_state)} parameters")
print(f"Loading Transformer with {len(transformer_state)} parameters")
print(f"Loading CTC with {len(ctc_state)} parameters")

# Check a sample weight before loading
print(f"\nBefore loading - CTC weight sample: {ctc_lin.w.weight[0, :5]}")

cnn.load_state_dict(cnn_state, strict=False)
transformer.load_state_dict(transformer_state, strict=False)
ctc_lin.load_state_dict(ctc_state, strict=False)

# Check the same weight after loading
print(f"After loading - CTC weight sample: {ctc_lin.w.weight[0, :5]}")
if 'w.weight' in ctc_state:
    print(f"CTC weight from checkpoint sample: {ctc_state['w.weight'][0, :5]}")

cnn.eval()
transformer.eval()
ctc_lin.eval()

print("✓ Model loaded")
print()

# Load and process audio
print(f"Processing: {test_audio}")
print(f"Expected: 'LINE'")
print()

signal = read_audio(test_audio)
signal = signal.to(device).unsqueeze(0)

with torch.no_grad():
    # Extract features
    feats = compute_features(signal)
    print(f"Features shape: {feats.shape}")
    
    # Normalize
    if not args.simple_norm:
        feats = normalizer(feats, torch.tensor([1.0]).to(device))
    else:
        mean = feats.mean(dim=(1, 2), keepdim=True)
        std = feats.std(dim=(1, 2), keepdim=True) + 1e-5
        feats = (feats - mean) / std
    print(f"Normalized features shape: {feats.shape}")
    
    # CNN
    src = cnn(feats)
    print(f"CNN output shape: {src.shape}")
    
    # Encoder
    enc_out = transformer.encode(src)
    print(f"Encoder output shape: {enc_out.shape}")
    
    # CTC
    logits = ctc_lin(enc_out)
    print(f"CTC logits shape: {logits.shape}")
    
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get predictions
    predicted_tokens = torch.argmax(log_probs, dim=-1).squeeze(0)
    print(f"Predicted tokens shape: {predicted_tokens.shape}")
    print(f"First 20 predicted tokens: {predicted_tokens[:20].tolist()}")
    print(f"Unique tokens in prediction: {torch.unique(predicted_tokens).tolist()[:20]}")
    print()
    
    # CTC decode
    decoded_tokens = []
    prev_token = None
    for token in predicted_tokens:
        token = token.item()
        if token != 0 and token != prev_token:  # 0 is blank
            decoded_tokens.append(token)
        prev_token = token
    
    print(f"After CTC collapse (removing blanks and repeats): {decoded_tokens}")
    print()
    
    # Decode with tokenizer
    if len(decoded_tokens) > 0:
        # Try decoding
        transcription = tokenizer.decode(decoded_tokens)
        print(f"Decoded text: '{transcription}'")
    else:
        print("No tokens after CTC decode!")
    
    # Also try decoding ALL non-blank tokens to see what they are
    all_non_blank = [t.item() for t in predicted_tokens if t.item() != 0]
    if len(all_non_blank) > 0:
        print(f"\nAll non-blank tokens (first 50): {all_non_blank[:50]}")
        sample_decode = tokenizer.decode(all_non_blank[:20])
        print(f"Sample decode of first 20 non-blank tokens: '{sample_decode}'")
