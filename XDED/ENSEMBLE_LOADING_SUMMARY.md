# Ensemble Loading Implementation Summary

## ✅ Completed Tasks

### 1. Fixed Utterance Matching Analysis
- **Script**: `check_utterance_matching.py`
- **Changes**: 
  - Updated to parse `utterance_ids` key (not `utterances`)
  - Implemented core-ID normalization by removing speaker prefix
  - Analyzed cross-speaker utterance overlap
  
**Results**:
```
- 693 core-IDs in all 15 speakers (38.8%)
- 1506 core-IDs in ≥12/14 teachers (84.3%)  
- 1530 core-IDs in ≥10/14 teachers (85.6%)
- M01 identified as outlier (765 vs 1275-1785 utterances)
```

### 2. Implemented Cross-Speaker Ensemble Loading
- **File**: `XDED/dassl/data/datasets/logit_ensemble.py`
- **Key Features**:

#### Core-ID Mapping
```python
# Extracts core-ID by removing speaker prefix
# Example: "M08_B3_D3_M2" -> "B3_D3_M2"
self.coreid_to_speakers = {}  # {core_id: {speaker: local_idx}}
```

#### Matching Modes
1. **Strict Mode** (`matching_mode="strict"`):
   - Requires ALL teachers (14 for M08 fold)
   - **693 training samples**
   - Average: 14.0 teachers per utterance
   - Use when: Maximum teacher coverage needed

2. **Partial Mode** (`matching_mode="partial"`, `min_teachers=10`):
   - Requires ≥min_teachers (configurable, default=10)
   - **1530 samples (min_teachers=10)** or **1506 samples (min_teachers=12)**
   - Average: 13.2 teachers per utterance (min=10), 13.3 (min=12)
   - Use when: Balance between coverage and data quantity
   - **Recommended for M08 fold**

3. **All Mode** (`matching_mode="all"`):
   - No matching requirements, uses all utterances
   - **1785 training samples**
   - Average: 12.6 teachers per utterance
   - Use when: Maximum data needed regardless of teacher coverage

#### Speaker Exclusion
```python
# Exclude outliers or problematic speakers
LogitEnsembleDataset(
    held_out_speaker="M08",
    matching_mode="partial",
    min_teachers=10,
    exclude_speakers=["M01"]  # Exclude M01 outlier
)
```

#### Ensemble Data Structure
For each training sample, returns:
```python
{
    'teacher_logits': [num_teachers, max_seq_len, vocab_size],  # Padded ensemble
    'num_teachers': int,  # Number of teachers for this utterance
    'teacher_speakers': List[str],  # Teacher speaker IDs
    'lengths': Tensor[num_teachers],  # Original sequence lengths
    'core_id': str,  # Core utterance ID (e.g., "B3_D3_M2")
    'target_text': str  # Ground truth text (if available)
}
```

Batch collation handles variable numbers of teachers:
```python
{
    'teacher_logits': [batch, max_teachers, max_seq_len, vocab_size],
    'num_teachers': [batch],  # Number of teachers per sample
    'teacher_speakers': List[List[str]],
    'lengths': List[Tensor],
    'core_ids': List[str],
    'target_texts': List[str]
}
```

### 3. Comprehensive Testing
- **Script**: `test_ensemble_loading.py`
- **Tests**:
  1. ✅ Strict matching (693 samples, all 14 teachers)
  2. ✅ Partial matching ≥10 teachers (1530 samples, avg 13.2 teachers)
  3. ✅ Partial matching ≥12 teachers (1506 samples, avg 13.3 teachers)
  4. ✅ All mode (1785 samples, avg 12.6 teachers)
  5. ✅ Speaker exclusion (M01 excluded, 1530 samples with 13 teachers)
  6. ✅ Core-ID consistency (verified same utterance across speakers)

**All 6/6 tests passed!**

## 📊 Comparison of Matching Modes

| Mode | Min Teachers | Total Samples | Avg Teachers | Coverage | Use Case |
|------|--------------|---------------|--------------|----------|----------|
| Strict | 14 (all) | 693 | 14.0 | 38.8% | Maximum teacher consensus |
| Partial (≥12) | 12 | 1506 | 13.3 | 84.3% | **Recommended** - high coverage |
| Partial (≥10) | 10 | 1530 | 13.2 | 85.6% | Good coverage + more data |
| All | 1+ | 1785 | 12.6 | 100% | Maximum data quantity |

## 🎯 Recommended Configuration for M08 Fold

```python
from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset, collate_logits
from torch.utils.data import DataLoader

# Training dataset
train_dataset = LogitEnsembleDataset(
    logit_root_dir="/home/zsim710/XDED/speechbrain/exp_results/logit_extraction",
    held_out_speaker="M08",
    split="train",
    matching_mode="partial",      # Partial matching
    min_teachers=12,              # ≥12/14 teachers (can adjust to 10)
    exclude_speakers=["M01"]      # Optional: exclude M01 outlier
)

# DataLoader with custom collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_logits,
    num_workers=4
)

# Example batch
for batch in train_loader:
    teacher_logits = batch['teacher_logits']  # [16, 14, max_len, 5000]
    num_teachers = batch['num_teachers']      # [16]
    lengths = batch['lengths']                # List of 16 tensors
    # ... train student model
```

## 🔄 Next Steps

1. **Implement Student Model**
   - Design architecture (simple MLP or small Transformer)
   - Decide on ensemble fusion strategy (averaging vs attention)
   
2. **Implement Training Loop**
   - KL divergence loss with temperature scaling
   - Ensemble averaging or weighted fusion
   - Validation on M08 test set
   
3. **Extend to Other Folds**
   - M05 fold (MID intelligibility)
   - M16 fold (LOW intelligibility)
   - M01 fold (VERY_LOW intelligibility)

## 📝 Implementation Notes

### Core-ID Normalization
- Removes speaker prefix to create shared identifier
- Example: `F03_B3_D3_M2` → `B3_D3_M2`
- Enables cross-speaker utterance matching

### M01 Outlier
- Only 765 utterances (vs 1275-1785 for others)
- Significantly reduces available training data in strict mode
- **Recommendation**: Exclude M01 using `exclude_speakers=["M01"]`

### Configurable Thresholds
The `min_teachers` parameter can be easily adjusted:
```python
# More data, lower coverage
LogitEnsembleDataset(..., matching_mode="partial", min_teachers=10)

# Higher coverage, less data  
LogitEnsembleDataset(..., matching_mode="partial", min_teachers=12)

# Maximum coverage, minimal data
LogitEnsembleDataset(..., matching_mode="strict")
```

### Variable Teacher Counts
- Different utterances may have different numbers of teachers
- `num_teachers` field tracks actual teacher count per sample
- Batch collation pads to `max_teachers` in batch
- Use `lengths` tensor to identify valid (non-padded) portions

## ✅ Verification

All ensemble loading functionality has been verified:
- ✅ Core-ID extraction and matching works correctly
- ✅ Multiple teachers loaded for each utterance
- ✅ Batch collation handles variable teacher counts
- ✅ All three matching modes work as expected
- ✅ Speaker exclusion filters correctly
- ✅ Data structure matches expected format

Ready to proceed with student model implementation!
