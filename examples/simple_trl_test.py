"""
Simple test script to verify TRL integration
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing RLlama imports...")
    print(f"Project root: {project_root}")
    
    try:
        from rllama.rewards.composer import RewardComposer
        print("✅ RewardComposer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RewardComposer: {e}")
        return False
    
    try:
        from rllama.rewards.shaper import RewardShaper
        print("✅ RewardShaper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RewardShaper: {e}")
        return False
    
    try:
        from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor
        print("✅ TRLRllamaRewardProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TRLRllamaRewardProcessor: {e}")
        print("This is expected if TRL wrapper needs to be updated")
        return False
    
    return True

def test_basic_functionality():
    """Test basic RLlama functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from rllama.rewards.composer import RewardComposer
        from rllama.rewards.shaper import RewardShaper
        
        # Create a simple config
        config = {
            'composer': {
                'components': [
                    {'type': 'CoherenceReward', 'weight': 1.0},
                    {'type': 'HelpfulnessReward', 'weight': 0.8}
                ]
            },
            'shaper': {
                'normalization_method': 'standard'
            }
        }
        
        # Test composer
        composer = RewardComposer(config['composer'])
        print(f"✅ Composer initialized with {len(composer.components)} components")
        
        # Test shaper
        shaper = RewardShaper(config['shaper'])
        print("✅ Shaper initialized successfully")
        
        # Test basic reward computation
        prompts = ["What is machine learning?"]
        responses = ["Machine learning is a subset of AI."]
        
        raw_reward = composer.compose(prompts[0], responses[0])
        shaped_rewards = shaper.shape([raw_reward])
        
        print(f"✅ Raw reward: {raw_reward:.4f}")
        print(f"✅ Shaped reward: {shaped_rewards[0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🦙 RLlama Basic Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed.")
        print("Let's try basic functionality without TRL...")
        
        # Try basic functionality even if TRL wrapper fails
        if test_basic_functionality():
            print("\n✅ Core RLlama functionality works!")
            print("Only TRL integration needs fixing.")
        else:
            print("\n❌ Core functionality also failed.")
        return
    
    print("\n🎉 All imports passed!")

if __name__ == "__main__":
    main()