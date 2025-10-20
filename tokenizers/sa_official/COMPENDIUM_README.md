# Research Compendium: Tokenizer Configuration

## Directory Overview

This directory contains the standardized tokenizer used across all XDED experiments. The tokenizer was extracted from the trained speaker-adaptive (SA) SpeechBrain Conformer models to ensure consistency in vocabulary and token mapping across all training and evaluation pipelines.

**Location:** `/home/zsim710/XDED/tokenizers/sa_official`

---

## Table of Contents

1. [Purpose and Rationale](#purpose-and-rationale)
2. [Tokenizer Specifications](#tokenizer-specifications)
3. [File Descriptions](#file-descriptions)
4. [Extraction Process](#extraction-process)
5. [Usage in Experiments](#usage-in-experiments)
6. [Verification and Validation](#verification-and-validation)
7. [Integration with Models](#integration-with-models)
8. [Troubleshooting](#troubleshooting)

---

## Purpose and Rationale

### Why a Shared Tokenizer?

In the XDED pipeline, maintaining vocabulary consistency is critical for knowledge distillation:

1. **Teacher-Student Alignment**: All 14 SA teacher models must share the same vocabulary space with the student model
2. **Logit Compatibility**: Pre-extracted logits from SA models must map to the same token indices as the student's output layer
3. **Cross-Model Evaluation**: Consistent tokenization enables fair comparison across different model architectures
4. **Reproducibility**: A fixed tokenizer ensures results can be replicated exactly

### Origin

This tokenizer was extracted from the **best-performing SA model** (speaker F02, validation set with uncommon words) after training on the UASpeech corpus. The choice of F02 was arbitrary—all SA models were trained with the same tokenizer, but F02's checkpoint was selected as the canonical source.

---

## Tokenizer Specifications

### Type
**SentencePiece BPE (Byte-Pair Encoding)**

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Vocabulary Size** | 5000 | Total number of unique tokens |
| **Model Type** | Unigram/BPE | SentencePiece subword tokenization |
| **Special Tokens** | `<blank>`, `<s>`, `</s>`, `<unk>` | CTC blank, BOS, EOS, unknown |
| **Token Indices** | 0-4999 | Zero-indexed vocabulary |
| **Blank Index** | 0 | CTC blank token (for CTC-based models) |
| **BOS Index** | 1 | Beginning-of-sequence token |
| **EOS Index** | 2 | End-of-sequence token |
| **UNK Index** | 0 or 3 | Unknown token (implementation-dependent) |
| **Character Coverage** | 1.0 | 100% character coverage for English |
| **Input Format** | Lowercase text | Normalized transcripts |

### Vocabulary Composition

The 5000-token vocabulary includes:
- **Subword units**: Most common syllables and morphemes from UASpeech transcripts
- **Whole words**: Common single-word utterances (e.g., "THIS", "THAT", "YES", "NO")
- **Character sequences**: Rare or out-of-vocabulary patterns
- **Special tokens**: Control tokens for sequence modeling

### Training Corpus

The tokenizer was trained on:
- **Dataset**: UASpeech training transcripts (all 15 speakers)
- **Utterances**: ~18,000 training utterances
- **Vocabulary**: 455 unique words in the corpus
- **Text normalization**: Uppercase → lowercase, punctuation removed

---

## File Descriptions

### 1. `tokenizer.ckpt`

**Type:** SpeechBrain checkpoint file (PyTorch state_dict wrapper)

**Contents:**
- SentencePiece model binary
- Vocabulary mappings (token ID ↔ subword string)
- Special token configurations
- Encoding/decoding parameters

**Loading method:**
```python
from speechbrain.dataio.encoder import CTCTextEncoder

tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file="tokenizer.ckpt",
    sequences=None,  # Not needed when loading existing
)

# Example usage
text = "this is a test"
tokens = tokenizer.encode_sequence(text)
decoded = tokenizer.decode_ids(tokens)
```

**Purpose:**
- Primary tokenizer file for all SpeechBrain-based models
- Used in SA model training and evaluation
- Used in student model evaluation via `eval_student.py`

---

### 2. `tokenizer`

**Type:** Raw SentencePiece model file (`.model` format)

**Contents:**
- Binary SentencePiece model
- Can be loaded directly with `sentencepiece` library

**Loading method:**
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('tokenizer')

# Example usage
text = "this is a test"
tokens = sp.encode_as_ids(text)
decoded = sp.decode_ids(tokens)
```

**Purpose:**
- Alternative loading method for non-SpeechBrain pipelines
- Used in verification scripts
- Useful for standalone tokenization tasks

---

### 3. `hyperparams.yaml`

**Type:** YAML configuration file

**Contents:**
- Tokenizer training hyperparameters
- Special token definitions
- Path configurations
- Integration settings for SpeechBrain

**Key sections:**
```yaml
# Tokenizer configuration
tokenizer: !new:speechbrain.dataio.encoder.CTCTextEncoder

# Special tokens
blank_index: 0
bos_index: 1
eos_index: 2
unk_index: 0

# Vocabulary
vocab_size: 5000

# Paths
tokenizer_file: tokenizer.ckpt
save_folder: .
```

**Purpose:**
- Documents tokenizer creation parameters
- Reference for reproducing tokenizer if needed
- Used by SpeechBrain to load tokenizer in training/evaluation

---

## Extraction Process

### How the Tokenizer Was Obtained

The tokenizer was extracted from a trained SA model checkpoint following these steps:

#### Step 1: Identify Source Checkpoint

```bash
# SA model checkpoint directory
SOURCE_CHECKPOINT="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00"
```

This checkpoint contains:
- `model.ckpt`: Model weights
- `tokenizer.ckpt`: Tokenizer state
- `hyperparams.yaml`: Full configuration

#### Step 2: Copy Tokenizer Files

```bash
# Create tokenizer directory
mkdir -p /home/zsim710/XDED/tokenizers/sa_official

# Copy tokenizer files
cp ${SOURCE_CHECKPOINT}/tokenizer.ckpt /home/zsim710/XDED/tokenizers/sa_official/
cp ${SOURCE_CHECKPOINT}/hyperparams.yaml /home/zsim710/XDED/tokenizers/sa_official/
```

#### Step 3: Verify Tokenizer

```python
from speechbrain.dataio.encoder import CTCTextEncoder

# Load tokenizer
tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file="/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt"
)

# Verify vocabulary size
assert len(tokenizer.lab2ind) == 5000, "Vocab size mismatch!"

# Test encoding/decoding
test_text = "this is a test"
tokens = tokenizer.encode_sequence(test_text)
decoded = tokenizer.decode_ids(tokens)
print(f"Original: {test_text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
```

#### Step 4: Validate Consistency

```bash
# Verify all SA models use the same tokenizer
for speaker in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
    echo "Checking ${speaker}..."
    CHECKPOINT_DIR="/mnt/Research/qwan121/ICASSP_SA/val_uncommon_${speaker}_*/save/CKPT*"
    diff ${CHECKPOINT_DIR}/tokenizer.ckpt /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt
done
```

All SA models were confirmed to use identical tokenizers.

---

## Usage in Experiments

### 1. SA Model Training

The tokenizer is loaded during SA model training via the hyperparams YAML:

```yaml
# In hparams/exp/uaspeech/ua_SA_val_uncommon_<SPEAKER>.yaml
tokenizer: !new:speechbrain.dataio.encoder.CTCTextEncoder
    save_path: !ref <save_folder>/tokenizer.ckpt
```

During training, the model:
1. Loads existing tokenizer (if available)
2. Maps ground-truth text to token sequences
3. Computes CTC/cross-entropy losses over token space
4. Saves tokenizer alongside model checkpoint

### 2. Logit Extraction

When extracting teacher logits, the tokenizer ensures consistent token indexing:

```python
# In extract_sa_logits.py
from speechbrain.dataio.encoder import CTCTextEncoder

tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file=f"{checkpoint_dir}/tokenizer.ckpt"
)

# Ground truth tokens (for metadata)
ground_truth_text = "this"
ground_truth_tokens = tokenizer.encode_sequence(ground_truth_text)

# Extracted logits shape: [seq_len, 5000]
# Each position scores over the same 5000-token vocabulary
```

### 3. Student Training

The student model is configured with the same vocabulary size:

```python
# In train_student.py
from models.nemo_hybrid_student import NeMoHybridStudent

# Vocabulary must match tokenizer
vocab_size = 5000
blank_index = 0

student = NeMoHybridStudent(
    vocab_size=vocab_size,
    blank_index=blank_index,
    # ... other params
)

# Teacher logits: [batch, teachers, seq_len, 5000]
# Student logits: [batch, seq_len, 5000]
# Both operate in same token space
```

### 4. Student Evaluation

The tokenizer converts predicted token IDs back to text:

```python
# In eval_student.py
import sentencepiece as spm

# Option 1: Load via SentencePiece
sp = spm.SentencePieceProcessor()
sp.load('/home/zsim710/XDED/tokenizers/sa_official/tokenizer')

# Option 2: Load via SpeechBrain
from speechbrain.dataio.encoder import CTCTextEncoder
tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file='/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt'
)

# Decode predictions
predicted_tokens = [45, 123, 456, 789, 2]  # Example token IDs
predicted_text = sp.decode_ids(predicted_tokens)
# OR
predicted_text = tokenizer.decode_ids(predicted_tokens)
```

### 5. Verification Scripts

Tokenizer is used to validate extracted logits:

```python
# In verify_logits.py
def load_tokenizer(hparams_file):
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)
    return hparams['tokenizer']

tokenizer = load_tokenizer('/home/zsim710/XDED/tokenizers/sa_official/hyperparams.yaml')

# Decode logits via argmax
predicted_tokens = logits.argmax(dim=-1)  # [seq_len]
predicted_text = tokenizer.decode_ids(predicted_tokens.tolist())
```

---

## Verification and Validation

### Tokenizer Integrity Checks

#### 1. Vocabulary Size Verification

```python
from speechbrain.dataio.encoder import CTCTextEncoder

tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file="tokenizer.ckpt"
)

vocab_size = len(tokenizer.lab2ind)
assert vocab_size == 5000, f"Expected 5000 tokens, got {vocab_size}"
print(f"✓ Vocabulary size: {vocab_size}")
```

#### 2. Special Token Verification

```python
# Check special tokens
assert tokenizer.lab2ind.get("<blank>", tokenizer.lab2ind.get("", None)) == 0
assert tokenizer.lab2ind["<s>"] == 1
assert tokenizer.lab2ind["</s>"] == 2
print("✓ Special tokens correctly indexed")
```

#### 3. Encoding/Decoding Round-Trip

```python
test_phrases = [
    "this",
    "that",
    "yes",
    "no",
    "the quick brown fox"
]

for phrase in test_phrases:
    tokens = tokenizer.encode_sequence(phrase)
    decoded = tokenizer.decode_ids(tokens)
    
    # Allow minor differences (lowercase, spacing)
    assert phrase.lower().replace(" ", "") == decoded.lower().replace(" ", "")
    print(f"✓ Round-trip: '{phrase}' → {tokens} → '{decoded}'")
```

#### 4. Cross-Model Consistency

```python
# Verify tokenizer matches SA model checkpoints
import torch

sa_tokenizer_path = "/mnt/Research/qwan121/ICASSP_SA/val_uncommon_M08_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00/tokenizer.ckpt"
official_tokenizer_path = "/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt"

sa_tok = torch.load(sa_tokenizer_path)
official_tok = torch.load(official_tokenizer_path)

# Compare vocab mappings
assert sa_tok['lab2ind'] == official_tok['lab2ind']
assert sa_tok['ind2lab'] == official_tok['ind2lab']
print("✓ Tokenizer matches SA model checkpoints")
```

---

## Integration with Models

### Model-Specific Integration

#### SpeechBrain Models (SA models, SB student)

```python
# Load tokenizer via hyperparams
with open('hyperparams.yaml') as f:
    hparams = load_hyperpyyaml(f)

tokenizer = hparams['tokenizer']

# Use in dataloader
def text_pipeline(text):
    return tokenizer.encode_sequence(text)
```

#### NeMo Hybrid Student

```python
# NeMo uses different tokenizer format
# We provide SentencePiece model for compatibility
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('/home/zsim710/XDED/tokenizers/sa_official/tokenizer')

# Encode text
tokens = sp.encode_as_ids("this is a test")

# Decode tokens
text = sp.decode_ids(tokens)
```

#### Evaluation Scripts

```python
# eval_student.py supports multiple tokenizer formats
def load_tokenizer(tokenizer_path):
    if tokenizer_path.endswith('.ckpt'):
        # SpeechBrain format
        tokenizer = CTCTextEncoder()
        tokenizer.load_or_create(label_encoder_file=tokenizer_path)
        return tokenizer
    elif tokenizer_path.endswith('.model') or os.path.isfile(tokenizer_path):
        # SentencePiece format
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        return sp
    else:
        raise ValueError(f"Unknown tokenizer format: {tokenizer_path}")
```

### Vocabulary Offset Handling

Some models use a vocabulary offset to account for special tokens:

```python
# In eval_student.py
vocab_offset = 1 if blank_index == 0 else 0

# When decoding student predictions
student_token_ids = [45, 123, 456, 2]  # From model output
spm_token_ids = [tid - vocab_offset for tid in student_token_ids]
decoded_text = sp.decode_ids(spm_token_ids)
```

This ensures:
- Student's token 0 = blank (CTC)
- Student's token 1 = SentencePiece token 0
- Correct alignment between model output and tokenizer vocabulary

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Tokenizer Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'tokenizer.ckpt'
```

**Solution:**
```bash
# Ensure tokenizer is in the correct location
ls /home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt

# Or update path in your script
TOKENIZER_PATH="/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt"
```

---

#### Issue 2: Vocabulary Size Mismatch

**Symptom:**
```
RuntimeError: size mismatch for output_layer.weight: copying a param with shape torch.Size([5000, 512]) from checkpoint, the shape in current model is torch.Size([5001, 512]).
```

**Solution:**
Ensure student model is initialized with `vocab_size=5000`:
```python
student = NeMoHybridStudent(
    vocab_size=5000,  # Must match tokenizer
    blank_index=0,
)
```

---

#### Issue 3: Decoding Returns Garbage

**Symptom:**
Decoded text contains unexpected characters or tokens:
```
Predicted: "▁th▁is▁i▁s▁te▁st"
```

**Solution:**
This is normal SentencePiece behavior. The `▁` character represents spaces. Clean the output:
```python
decoded = sp.decode_ids(tokens).replace('▁', ' ').strip()
```

---

#### Issue 4: Special Token Confusion

**Symptom:**
Predictions include `<s>`, `</s>`, or `<unk>` tokens in text output.

**Solution:**
Filter special tokens before decoding:
```python
# Remove special tokens
tokens_filtered = [t for t in tokens if t not in [0, 1, 2]]  # blank, BOS, EOS
decoded = sp.decode_ids(tokens_filtered)
```

---

#### Issue 5: Tokenizer Version Mismatch

**Symptom:**
```
ModuleNotFoundError: No module named 'sentencepiece'
```

**Solution:**
Install SentencePiece:
```bash
pip install sentencepiece==0.1.99
```

Or use SpeechBrain's CTCTextEncoder (already includes SentencePiece).

---

## Reproducibility Guidelines

### To Use This Tokenizer in Future Work

1. **Load the tokenizer:**
   ```python
   from speechbrain.dataio.encoder import CTCTextEncoder
   
   tokenizer = CTCTextEncoder()
   tokenizer.load_or_create(
       label_encoder_file="/home/zsim710/XDED/tokenizers/sa_official/tokenizer.ckpt"
   )
   ```

2. **Verify vocabulary size:**
   ```python
   assert len(tokenizer.lab2ind) == 5000
   ```

3. **Use in your model:**
   ```python
   vocab_size = 5000
   model = YourModel(vocab_size=vocab_size)
   ```

4. **Encode text:**
   ```python
   tokens = tokenizer.encode_sequence("your text here")
   ```

5. **Decode predictions:**
   ```python
   text = tokenizer.decode_ids(token_ids)
   ```

### To Recreate the Tokenizer (if needed)

If the tokenizer files are lost or corrupted, you can recreate it from UASpeech transcripts:

```python
from speechbrain.dataio.encoder import CTCTextEncoder

# Collect all training transcripts
transcripts = []
for csv_file in glob.glob("/home/zsim710/partitions/uaspeech/by_speakers/*.csv"):
    df = pd.read_csv(csv_file)
    transcripts.extend(df['wrd'].tolist())

# Train tokenizer
tokenizer = CTCTextEncoder()
tokenizer.load_or_create(
    label_encoder_file="new_tokenizer.ckpt",
    sequences=transcripts,
    vocab_size=5000,
)

# Save
tokenizer.save("new_tokenizer.ckpt")
```

**Note:** The recreated tokenizer will have a slightly different vocabulary distribution, so pretrained models won't be compatible. Always use the official tokenizer for consistency.

---

## File Checksums (for verification)

To verify file integrity:

```bash
# Generate checksums
md5sum tokenizer.ckpt
md5sum tokenizer
md5sum hyperparams.yaml
```

Expected checksums (store these for future verification):
```
<md5_hash>  tokenizer.ckpt
<md5_hash>  tokenizer
<md5_hash>  hyperparams.yaml
```

If checksums don't match, the files may have been corrupted or modified.

---

## Contact and Maintenance

**Primary maintainer:** Zoe Sim (zsim710)

**Source checkpoint:** 
- Speaker: F02
- Experiment ID: E0D2
- Checkpoint: `/mnt/Research/qwan121/ICASSP_SA/val_uncommon_F02_E0D2/7775/save/CKPT+2024-07-11+18-53-02+00`

**Last verified:** October 2025

**Related directories:**
- SA model training: `/home/zsim710/XDED/conformer/conformer-asr/`
- Student training: `/home/zsim710/XDED/XDED/`
- Evaluation scripts: `/home/zsim710/XDED/XDED/eval_student.py`

---

## References

1. SentencePiece: A simple and language independent tokenization library (Kudo & Richardson, 2018)
2. SpeechBrain: A General-Purpose Speech Toolkit (Ravanelli et al., 2021)
3. UASpeech corpus documentation

---

**End of Tokenizer Compendium README**
