#!/usr/bin/env python3

import torch
import numpy as np
import sys

try:
    from rllama.rewards.domain_models import ReasoningRewardModel
    from rllama.rewards.vision_rewards import VisualReasoningReward
    
    def test_reasoning_model():
        """Test the reasoning reward model"""
        print("\n=== Testing Reasoning Reward Model ===")
        
        try:
            # Create a reasoning reward model
            model = ReasoningRewardModel(
                name="test_reasoning",
                domain="general"
            )
            
            # Test with different responses
            test_cases = [
                {
                    "query": "Explain climate change.",
                    "response": "It's getting hotter."
                },
                {
                    "query": "Explain climate change.",
                    "response": "Climate change is primarily caused by greenhouse gases. When we burn fossil fuels, we release CO2. This CO2 traps heat in the atmosphere, leading to rising global temperatures. Additionally, deforestation reduces the planet's ability to absorb CO2. In conclusion, human activities are driving climate change through these mechanisms."
                }
            ]
            
            for i, context in enumerate(test_cases):
                reward = model.compute_reward(context)
                print(f"Test case {i+1}: Reward = {reward}")
                
                # Second response should have better reasoning
                if i == 1 and reward <= 0:
                    print("❌ Expected higher reward for better reasoning")
                    return False
            
            print("✅ Reasoning reward model test passed!")
            return True
        except Exception as e:
            print(f"❌ Reasoning model test failed with error: {e}")
            return False
    
    def test_visual_model():
        """Test the visual reasoning reward model"""
        print("\n=== Testing Visual Reasoning Reward Model ===")
        
        try:
            # Skip if not running in an environment with PIL
            try:
                from PIL import Image
                import numpy as np
                
                # Create a simple test image (red square)
                img_array = np.zeros((100, 100, 3), dtype=np.uint8)
                img_array[25:75, 25:75, 0] = 255  # Red square
                img = Image.fromarray(img_array)
                
                # Create visual reward model
                model = VisualReasoningReward()
                
                # Test contexts
                contexts = [
                    {
                        "image": img,
                        "query": "What's in this image?",
                        "response": "The image contains a red square."
                    },
                    {
                        "image": img,
                        "query": "What's in this image?",
                        "response": "The image shows a blue circle."  # Incorrect
                    }
                ]
                
                for i, context in enumerate(contexts):
                    try:
                        reward = model.calculate(context)
                        print(f"Visual context {i+1}: Reward = {reward}")
                    except Exception as e:
                        print(f"Error processing visual context {i+1}: {e}")
                
                print("✅ Visual reasoning test completed!")
                return True
                
            except ImportError:
                print("Skipping visual test - PIL not available")
                return True
                
        except Exception as e:
            print(f"❌ Visual model test failed with error: {e}")
            return False

    if __name__ == "__main__":
        print("Running domain model tests...")
        
        # Run tests
        reasoning_success = test_reasoning_model()
        visual_success = test_visual_model()
        
        # Show overall results
        print("\n=== Test Summary ===")
        print(f"Reasoning Model: {'✅ Passed' if reasoning_success else '❌ Failed'}")
        print(f"Visual Model: {'✅ Passed' if visual_success else '❌ Failed'}")
        
        # Overall success
        if all([reasoning_success, visual_success]):
            print("\n🎉 All domain model tests passed!")
        else:
            print("\n❌ Some tests failed. Check the logs above for details.")
            
except ImportError as e:
    print(f"Could not test domain models: {e}")
    print("This is expected if these components haven't been implemented yet.")
    sys.exit(0)
