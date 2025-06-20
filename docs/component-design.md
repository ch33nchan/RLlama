# Reward Component Design

This guide explains how to design custom reward components in RLlama, including best practices and advanced techniques.

## Basic Component Structure

Every reward component in RLlama inherits from the `BaseReward` class:

```python
from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def compute(self, context):
        # Extract relevant information from context
        # Perform reward calculation
        # Return a numerical reward value
        return reward_value
```

## The Context Object

The `context` parameter passed to the `compute` method is a dictionary containing all the information needed to calculate rewards. Common keys include:

- `state`: The current state of the environment
- `action`: The action taken by the agent
- `next_state`: The resulting state after taking the action
- `done`: A boolean indicating if the episode is complete
- `response`: For language models, the generated text
- `history`: For sequential tasks, the history of previous states/responses

Your component should extract the information it needs from this context object.

## Best Practices

1. **Single Responsibility**: Each component should focus on a single aspect of behavior
2. **Robust to Missing Data**: Handle cases where expected data is missing
3. **Configurable Parameters**: Make important values configurable
4. **Normalized Output Range**: Keep rewards in a consistent range
5. **Documentation**: Document what your component does and how it works
