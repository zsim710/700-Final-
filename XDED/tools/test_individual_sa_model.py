"""
Test an individual SA model (not averaged) to verify inference works
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

# Test with F03 individual model
checkpoint_path = "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00/model.ckpt"
tokenizer_path = "/home/zsim710/XDED/tokenizers/sa_official/tokenizer"
test_audio = "/mnt/DataSets/UASpeech/audio/F02/F02_B3_C10_M2.wav"  # "LINE"

device = 'cuda'

print(f"Testing individual SA model (F03)")
print(f"Checkpoint: {checkpoint_path}")
print()

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)

# Create normalizer (we'll use a simple one for testing)
normalizer = InputNormalization(norm_type='global')
normalizer = normalizer.to(device)
normalizer.eval()

# Create feature extractor
compute_features = Fbank(n_mels=80).to(device)

# Create CNN
cnn = ConvolutionFrontEnd(
    input_shape=(None, None, 80),
    num_blocks=2,
    num_layers_per_block=1,
    out_channels=(64, 32),
    kernel_sizes=(3, 3),
    strides=(2, 2),
    residuals=(False, False),
    norm=LayerNorm
).to(device)

# Create transformer
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

# Output layers
ctc_lin = Linear(input_size=144, n_neurons=5000).to(device)

# Load checkpoint
print("Loading individual SA model checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract weights with correct prefix mapping
cnn_state = {k.replace('0.', ''): v for k, v in checkpoint.items() if k.startswith('0.')}
transformer_state = {k.replace('1.', ''): v for k, v in checkpoint.items() if k.startswith('1.')}
ctc_state = {k.replace('2.', ''): v for k, v in checkpoint.items() if k.startswith('2.')}

print(f"Loading {len(cnn_state)} CNN parameters")
print(f"Loading {len(transformer_state)} Transformer parameters")
print(f"Loading {len(ctc_state)} CTC parameters")

cnn.load_state_dict(cnn_state, strict=False)
transformer.load_state_dict(transformer_state, strict=False)
ctc_lin.load_state_dict(ctc_state, strict=False)

cnn.eval()
transformer.eval()
ctc_lin.eval()

print("✓ Model loaded")
print()

# Test inference
print(f"Processing: {test_audio}")
print(f"Expected: 'LINE'")
print()

signal = read_audio(test_audio)
signal = signal.to(device).unsqueeze(0)

with torch.no_grad():
    # Extract features
    feats = compute_features(signal)
    print(f"Features shape: {feats.shape}")
    
    # Normalize - simple standardization without proper stats
    feats = (feats - feats.mean()) / (feats.std() + 1e-5)
    print(f"Normalized features shape: {feats.shape}")
    
    # CNN expects (batch, time, freq) - add batch dim if needed
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    
    print(f"Input to CNN shape: {feats.shape}")
    
    # CNN
    src = cnn(feats)
    
    # Encoder
    enc_out = transformer.encode(src)
    
    # CTC
    logits = ctc_lin(enc_out)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get predictions
    predicted_tokens = torch.argmax(log_probs, dim=-1).squeeze(0)
    print(f"First 20 predicted tokens: {predicted_tokens[:20].tolist()}")
    print(f"Unique tokens (first 20): {sorted(torch.unique(predicted_tokens).tolist())[:20]}")
    print()
    
    # CTC decode
    decoded_tokens = []
    prev_token = None
    for token in predicted_tokens:
        token = token.item()
        if token != 0 and token != prev_token:  # 0 is blank
            decoded_tokens.append(token)
        prev_token = token
    
    print(f"After CTC collapse: {decoded_tokens[:20]}")
    print()
    
    # Decode with tokenizer
    if len(decoded_tokens) > 0:
        transcription = tokenizer.decode(decoded_tokens)
        print(f"✓ Decoded text: '{transcription}'")
    else:
        print("✗ No tokens after CTC decode!")
