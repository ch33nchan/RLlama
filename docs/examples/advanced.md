# Advanced Examples

```python
from rllama import RewardEngine
from rllama.rewards.optimizer import RewardOptimizer

# Create optimizer
optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(evaluate_weights, n_trials=100)
```
