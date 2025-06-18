#!/usr/bin/env python3

import torch
import numpy as np
from rllama.rewards.components.advanced_components import ExplorationReward, DiversityReward
from rllama.rewards.optimizer import BayesianRewardOptimizer

def test_advanced_rewards():
    """Test advanced reward components"""
    print("\n=== Testing Advanced Reward Components ===")
    
    try:
        # Test exploration reward
        exploration_reward = ExplorationReward(reward_scale=0.2)
        
        # Test with different states
        states = [f"state_{i}" for i in range(5)]
        
        print("Testing ExplorationReward...")
        for i in range(10):
            state_idx = i % len(states)
            context = {"state_hash": states[state_idx]}
            reward = exploration_reward.calculate(context)
            print(f"  State {states[state_idx]} (visit {i//len(states) + 1}): Reward = {reward}")
            
            # We should see decreasing rewards for repeat visits
            if i >= len(states) and reward >= exploration_reward.reward_scale:
                print("  ❌ Expected lower reward for repeat visits")
                return False
        
        # Test diversity reward
        diversity_reward = DiversityReward(target_key="action", reward_scale=0.5)
        
        print("\nTesting DiversityReward...")
        actions = ["jump", "run", "walk", "jump", "sit", "jump"]
        
        for action in actions:
            context = {"action": action}
            reward = diversity_reward.calculate(context)
            print(f"  Action '{action}': Reward = {reward}")
        
        print("✅ Advanced rewards test passed!")
        return True
    except Exception as e:
        print(f"❌ Advanced rewards test failed with error: {e}")
        return False

def test_optimizer():
    """Test the optimizer"""
    print("\n=== Testing Reward Optimizer ===")
    
    try:
        # Define a simple parameter space
        param_space = {
            "weight_a": (0.1, 1.0),
            "weight_b": (0.5, 2.0)
        }
        
        # Define evaluation function with known optimum
        def eval_func(params):
            # This function has an optimum at weight_a=0.3, weight_b=1.5
            a_term = -((params["weight_a"] - 0.3) ** 2)
            b_term = -((params["weight_b"] - 1.5) ** 2)
            
            # Add some noise
            noise = np.random.normal(0, 0.1)
            return 2.0 + a_term + b_term + noise
        
        # Create optimizer with minimal trials for quick testing
        optimizer = BayesianRewardOptimizer(
            param_space=param_space,
            eval_function=eval_func,
            direction="maximize",
            n_trials=5  # Small number for quick testing
        )
        
        print("Running optimization...")
        results = optimizer.optimize(show_progress_bar=True)
        
        print(f"Best parameters: {results.best_params}")
        print(f"Best value: {results.best_value}")
        
        # Generate config and save
        config = optimizer.generate_config("./test_output/optimized_config.yaml")
        print("Generated optimized config and saved to ./test_output/optimized_config.yaml")
        
        print("✅ Optimizer test passed!")
        return True
    except Exception as e:
        print(f"❌ Optimizer test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running advanced RLlama tests...")
    
    # Run tests
    rewards_success = test_advanced_rewards()
    optimizer_success = test_optimizer()
    
    # Show overall results
    print("\n=== Test Summary ===")
    print(f"Advanced Rewards: {'✅ Passed' if rewards_success else '❌ Failed'}")
    print(f"Optimizer: {'✅ Passed' if optimizer_success else '❌ Failed'}")
    
    # Overall success
    if all([rewards_success, optimizer_success]):
        print("\n🎉 All tests passed! RLlama advanced features are functioning correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
