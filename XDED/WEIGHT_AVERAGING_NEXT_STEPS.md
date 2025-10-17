# 15-Fold Weight Averaging Evaluation - Next Steps

## Current Status

✅ **Completed:**
1. All 15 speaker checkpoint paths have been identified and mapped
2. All 15 averaged models have been successfully created (each excluding one held-out speaker)
3. Averaged models saved in: `/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models/`

## What We Have

**Averaged Models:**
- F02_held_out_averaged.pt (average of 14 models, excluding F02)
- F03_held_out_averaged.pt (average of 14 models, excluding F03)
- ...and so on for all 15 speakers

**Test Data:**
- Test CSV files for each speaker in: `/home/zsim710/partitions/uaspeech/by_speakers/`
- Format: ID,duration,wav,spk_id,wrd

**Individual SA Models:**
- All 15 individual speaker-adaptive models in: `/mnt/Research/qwan121/ICASSP_SA/`

## What Needs to Be Done

### Step 1: Integrate Averaged Models with SpeechBrain Inference

The averaged models are just the model weights (`model.ckpt` format). To use them for inference, you need to:

1. Load the averaged model weights into a SpeechBrain ASR model
2. Load the necessary hyperparameters (from one of the original SA model directories)
3. Load the tokenizer
4. Set up the inference pipeline

**Challenge:** SpeechBrain models need more than just weights - they need:
- Hyperparameters (hyperparams.yaml)
- Tokenizer (tokenizer.ckpt)
- Normalizer stats (normalizer.ckpt)
- Model architecture configuration

**Solution Options:**

#### Option A: Copy full checkpoint structure
For each averaged model, create a full SpeechBrain checkpoint directory:
```bash
# Example for F02 held-out
mkdir -p /home/zsim710/XDED/XDED/results/speaker_averaging/checkpoints/F02_held_out/
# Copy hyperparams.yaml from any SA model (they should all be the same)
cp /mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00/hyperparams.yaml \
   /home/zsim710/XDED/XDED/results/speaker_averaging/checkpoints/F02_held_out/
# Copy tokenizer
cp /mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00/tokenizer.ckpt \
   /home/zsim710/XDED/XDED/results/speaker_averaging/checkpoints/F02_held_out/
# Copy normalizer  
cp /mnt/Research/qwan121/ICASSP_SA/val_uncommon_F03_E0D3/7775/save/CKPT+2024-07-11+20-46-30+00/normalizer.ckpt \
   /home/zsim710/XDED/XDED/results/speaker_averaging/checkpoints/F02_held_out/
# Copy our averaged model weights
cp /home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models/F02_held_out_averaged.pt \
   /home/zsim710/XDED/XDED/results/speaker_averaging/checkpoints/F02_held_out/model.ckpt
```

#### Option B: Use existing SA test script
Modify the SpeechBrain test script (like `/home/zsim710/XDED/conformer/conformer-asr/recipes/LibriSpeech/ASR/transformer/test.py`) to:
1. Load your averaged model weights
2. Run inference on the test CSVs
3. Compute WER

### Step 2: Run Inference for Each Fold

For each of the 15 folds:

1. **Test averaged model on held-out speaker**
   - Load: `F02_held_out_averaged.pt`
   - Test on: `/home/zsim710/partitions/uaspeech/by_speakers/F02.csv`
   - Compute WER

2. **Test each individual model on held-out speaker** (for WRA comparison)
   - For F02 held-out, test these 14 models on F02 data:
     - F03 model → F02 data
     - F04 model → F02 data
     - F05 model → F02 data
     - M01 model → F02 data
     - ... (all except F02)
   - Compute WER for each

3. **Compute WRA**
   - Equal weighting: Average of all individual model accuracies
   - Inverse WER weighting: Better models get higher weight

4. **Compare**
   - Averaged model WER vs WRA WER
   - Which approach generalizes better?

### Step 3: Recommended Approach

Since you already have working SA models and know how to run inference with them, I recommend:

1. **Create a test script** based on your existing inference code that:
   ```python
   def test_model_on_speaker(model_checkpoint_dir, test_csv, output_file):
       # Load model using SpeechBrain
       asr_model = load_speechbrain_model(model_checkpoint_dir)
       
       # Load test data
       test_data = pd.read_csv(test_csv)
       
       # Run inference
       predictions = []
       references = []
       for row in test_data.iterrows():
           audio_path = row['wav']
           reference = row['wrd']
           
           # Transcribe
           prediction = asr_model.transcribe_file(audio_path)
           
           predictions.append(prediction)
           references.append(reference)
       
       # Compute WER
       wer = compute_wer(predictions, references)
       
       return {'wer': wer, 'predictions': predictions, 'references': references}
   ```

2. **Run for all 15 folds:**
   ```bash
   # For each speaker
   for speaker in F02 F03 F04 F05 M01 M04 M05 M07 M08 M09 M10 M11 M12 M14 M16; do
       # Test averaged model
       python test_averaged_model.py \
           --model_dir /path/to/averaged/checkpoint/${speaker}_held_out \
           --test_csv /home/zsim710/partitions/uaspeech/by_speakers/${speaker}.csv \
           --output results/${speaker}_averaged.json
       
       # Test each individual model (for WRA)
       for other_speaker in [all speakers except $speaker]; do
           python test_individual_model.py \
               --model_dir /mnt/Research/qwan121/ICASSP_SA/val_uncommon_${other_speaker}_* \
               --test_csv /home/zsim710/partitions/uaspeech/by_speakers/${speaker}.csv \
               --output results/${speaker}_${other_speaker}.json
       done
   done
   ```

3. **Aggregate results** using the `evaluate_15_fold_averaging.py` script (after adding real WER computation)

## Files Created

1. `/home/zsim710/XDED/XDED/tools/average_sa_models.py` - Averages model weights ✅
2. `/home/zsim710/XDED/XDED/tools/run_15_fold_averaging.py` - Runs all 15 folds ✅
3. `/home/zsim710/XDED/XDED/tools/evaluate_15_fold_averaging.py` - Evaluation framework (needs inference integration)
4. `/home/zsim710/XDED/XDED/results/speaker_averaging/speaker_checkpoints.json` - Checkpoint mapping ✅
5. `/home/zsim710/XDED/XDED/results/speaker_averaging/averaged_models/*.pt` - 15 averaged models ✅

## Next Action Items

🔲 **Ask your supervisor:** Do you have an existing inference script for the SA models? 
   - This would be the easiest way to test the averaged models

🔲 **Alternative:** Adapt the SpeechBrain test.py script to work with your averaged models

🔲 **Create full checkpoint directories** for the averaged models (Option A above)

🔲 **Run inference** and collect WER results for all 15 folds

🔲 **Aggregate results** and compare weight averaging vs WRA averaging

## Expected Results

Once inference is running, you'll be able to answer:

1. **Does weight averaging work?** Compare averaged model WER to best individual model WER
2. **Weight averaging vs WRA?** Which approach gives better cross-speaker generalization?
3. **Is it worth it?** Weight averaging is computationally free (just tensor averaging), while WRA requires running 14 models at inference time

## Questions for Discussion

1. Do you have working inference code for the SA models that we can adapt?
2. Should we create full checkpoint directories for the averaged models?
3. What format do you want the final results in?
