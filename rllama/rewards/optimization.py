from typing import Dict, List, Callable
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianRewardOptimizer:
    def __init__(self, reward_space: dict, n_initial_points: int = 5):
        self.reward_space = reward_space
        self.X = []
        self.y = []
        self.n_initial_points = n_initial_points
        
    def suggest(self) -> Dict[str, float]:
        """Suggest next reward configuration to try"""
        if len(self.X) < self.n_initial_points:
            return self._random_config()
            
        # Gaussian Process optimization
        X = np.array(self.X)
        y = np.array(self.y)
        
        def expected_improvement(x):
            mu, sigma = self._predict(x, X, y)
            mu_sample = max(y)
            
            with np.errstate(divide='warn'):
                imp = mu - mu_sample
                Z = imp / sigma if sigma > 0 else 0
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
                
            return -ei
            
        x_next = self._optimize_acquisition(expected_improvement)
        return self._config_from_array(x_next)
        
    def update(self, config: Dict[str, float], result: float):
        """Update optimizer with results"""
        self.X.append(self._array_from_config(config))
        self.y.append(result)