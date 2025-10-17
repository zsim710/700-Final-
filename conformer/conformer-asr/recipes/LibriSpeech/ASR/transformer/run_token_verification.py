#!/usr/bin/env python3
"""
Script to run token-level verification on extracted logits.
This is the most accurate verification method for checking logit extraction quality.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from verify_logits import verify_model_tokens, load_tokenizer

def main():
    """Run token-level verification on all available models."""
    
    print("🎯 TOKEN-LEVEL LOGIT VERIFICATION")
    print("="*80)
    print("This script performs the most accurate verification by comparing")
    print("token sequences between ground truth and decoded logits.")
    print("="*80)
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("❌ Failed to load tokenizer. Cannot proceed.")
        return
    
    print("✅ Tokenizer loaded successfully")
    
    # Define models to verify (completed extractions)
    models_to_verify = ["F03", "F04", "F05", "M01"]
    
    print(f"\n🔍 Will verify {len(models_to_verify)} models with token-level analysis:")
    for model in models_to_verify:
        print(f"  - {model}")
    
    # Verify each model
    results = {}
    
    for model in models_to_verify:
        print(f"\n{'='*100}")
        try:
            result = verify_model_tokens(model, tokenizer)
            if result:
                results[model] = result
                print(f"✅ {model}: Token verification completed")
            else:
                print(f"❌ {model}: Token verification failed")
        except Exception as e:
            print(f"💥 {model}: Exception during verification: {e}")
    
    # Final comprehensive summary
    print(f"\n{'='*100}")
    print("🏆 COMPREHENSIVE TOKEN-LEVEL VERIFICATION SUMMARY")
    print("="*100)
    
    if not results:
        print("❌ No successful verifications completed.")
        return
    
    print(f"Successfully verified {len(results)}/{len(models_to_verify)} models:")
    print()
    
    for model, result in results.items():
        ctc_stats = result['ctc_stats']
        decoder_stats = result['decoder_stats']
        
        print(f"📊 {model}:")
        print(f"  CTC Results:")
        print(f"    - Exact matches: {ctc_stats['exact_matches']}/{ctc_stats['total_utterances']} ({100*ctc_stats['exact_matches']/ctc_stats['total_utterances']:.1f}%)")
        print(f"    - Token accuracy: {100*ctc_stats['token_accuracy']:.1f}%")
        
        print(f"  Decoder Results:")
        print(f"    - Exact matches: {decoder_stats['exact_matches']}/{decoder_stats['total_utterances']} ({100*decoder_stats['exact_matches']/decoder_stats['total_utterances']:.1f}%)")
        print(f"    - Token accuracy: {100*decoder_stats['token_accuracy']:.1f}%")
        print()
    
    # Quality assessment
    print("🎯 QUALITY ASSESSMENT:")
    excellent_models = []
    good_models = []
    fair_models = []
    
    for model, result in results.items():
        decoder_accuracy = result['decoder_stats']['token_accuracy']
        if decoder_accuracy >= 0.95:
            excellent_models.append((model, decoder_accuracy))
        elif decoder_accuracy >= 0.80:
            good_models.append((model, decoder_accuracy))
        else:
            fair_models.append((model, decoder_accuracy))
    
    if excellent_models:
        print(f"🌟 Excellent quality (≥95% token accuracy): {len(excellent_models)} models")
        for model, acc in excellent_models:
            print(f"   - {model}: {100*acc:.1f}%")
    
    if good_models:
        print(f"✅ Good quality (≥80% token accuracy): {len(good_models)} models")  
        for model, acc in good_models:
            print(f"   - {model}: {100*acc:.1f}%")
    
    if fair_models:
        print(f"⚠️  Fair quality (<80% token accuracy): {len(fair_models)} models")
        for model, acc in fair_models:
            print(f"   - {model}: {100*acc:.1f}%")
    
    print(f"\n✨ Token-level verification complete! This provides the most accurate")
    print(f"   assessment of logit extraction quality for knowledge distillation.")

if __name__ == "__main__":
    main()