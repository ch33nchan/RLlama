# Basic Examples

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward

# Create a reward engine
engine = RewardEngine()
engine.add_component(LengthReward(target_length=100))

# Compute rewards
context = {"response": "This is a test response"}
reward = engine.compute(context)
```
