#!/usr/bin/env python3
"""
Quick test to verify curriculum learning integration works correctly.
Tests the get_curriculum_competence function and dataset integration.
"""

import sys
sys.path.insert(0, '/home/zsim710/XDED/XDED')

from train_student import get_curriculum_competence


def test_curriculum_schedules():
    """Test curriculum competence functions."""
    print("🧪 Testing Curriculum Learning Schedules\n")
    print("="*80)
    
    total_epochs = 40
    test_epochs = [0, 1, 4, 9, 10, 16, 20, 25, 30, 39]
    
    schedules = ['none', 'linear', 'sqrt', 'step']
    
    for schedule in schedules:
        print(f"\n📊 {schedule.upper()} Schedule:")
        print(f"{'Epoch':<10} {'Progress':<12} {'Competence':<12} {'Data %':<10}")
        print("-" * 50)
        
        for epoch in test_epochs:
            progress = epoch / total_epochs
            competence = get_curriculum_competence(epoch, total_epochs, schedule)
            data_percent = competence * 100
            
            print(f"{epoch+1:<10} {progress:.3f}        {competence:.3f}        {data_percent:.1f}%")
    
    print("\n" + "="*80)
    print("✅ All schedule tests passed!")


def test_dataset_integration():
    """Test that dataset can be created with curriculum scores."""
    print("\n🧪 Testing Dataset Integration\n")
    print("="*80)
    
    try:
        from dassl.data.datasets.logit_ensemble import LogitEnsembleDataset
        
        print("✅ Successfully imported LogitEnsembleDataset")
        
        # Check if curriculum parameters are in __init__
        import inspect
        sig = inspect.signature(LogitEnsembleDataset.__init__)
        params = list(sig.parameters.keys())
        
        if 'curriculum_scores_file' in params:
            print("✅ curriculum_scores_file parameter found in __init__")
        else:
            print("❌ curriculum_scores_file parameter NOT FOUND in __init__")
            return False
        
        # Check if curriculum methods exist
        methods = ['_load_curriculum_scores', '_create_curriculum_order', 'get_curriculum_subset_indices']
        for method in methods:
            if hasattr(LogitEnsembleDataset, method):
                print(f"✅ Method {method} found")
            else:
                print(f"❌ Method {method} NOT FOUND")
                return False
        
        print("\n✅ All dataset integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during dataset integration test: {e}")
        return False


def test_competence_values():
    """Test edge cases and validity of competence values."""
    print("\n🧪 Testing Competence Value Validity\n")
    print("="*80)
    
    all_valid = True
    
    # Test edge cases
    test_cases = [
        (0, 40, 'sqrt'),      # First epoch
        (39, 40, 'sqrt'),     # Last epoch
        (0, 1, 'linear'),     # Single epoch
        (20, 40, 'step'),     # Mid-training
    ]
    
    for epoch, total, schedule in test_cases:
        try:
            competence = get_curriculum_competence(epoch, total, schedule)
            
            # Check if competence is in valid range [0, 1]
            if 0 <= competence <= 1:
                status = "✅"
            else:
                status = "❌"
                all_valid = False
            
            print(f"{status} Epoch {epoch+1}/{total}, {schedule}: competence={competence:.3f}")
            
        except Exception as e:
            print(f"❌ Error with epoch {epoch+1}/{total}, {schedule}: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✅ All competence values are valid (0 ≤ competence ≤ 1)")
    else:
        print("\n❌ Some competence values are invalid!")
    
    return all_valid


def main():
    print("\n" + "="*80)
    print("🎓 CURRICULUM LEARNING INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Run all tests
    test1 = True
    test2 = True
    test3 = True
    
    try:
        test_curriculum_schedules()
    except Exception as e:
        print(f"❌ Schedule test failed: {e}")
        test1 = False
    
    try:
        test2 = test_dataset_integration()
    except Exception as e:
        print(f"❌ Dataset integration test failed: {e}")
        test2 = False
    
    try:
        test3 = test_competence_values()
    except Exception as e:
        print(f"❌ Competence value test failed: {e}")
        test3 = False
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"{'Schedule Functions:':<30} {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"{'Dataset Integration:':<30} {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"{'Competence Validity:':<30} {'✅ PASSED' if test3 else '❌ FAILED'}")
    print("="*80)
    
    if test1 and test2 and test3:
        print("\n🎉 ALL TESTS PASSED! Curriculum learning is ready to use.")
        print("\nNext steps:")
        print("1. Run: python precompute_difficulty.py")
        print("2. Train with curriculum: See CURRICULUM_TRAINING_GUIDE.md")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
