# RLlama Documentation: Reward Engineering Framework

## Overview

RLlama is a composable reward engineering framework that lets you declaratively combine, weight, schedule, and optimize multiple reward signals (LLM-based, rule-based, or custom) for any reinforcement learning workflow.

**Think of it as "LangChain but for RL rewards"** - instead of writing custom reward calculation code every time, you declare your reward strategy in YAML and RLlama handles the composition, normalization, scheduling, and hyperparameter tuning automatically.

## Core Concepts

### 1. Framework-Agnostic Design

RLlama works with any RL framework:
- **TRL (Transformers Reinforcement Learning)**
- **Stable Baselines3**
- **Ray RLlib** 
- **Custom RL implementations**

### 2. Declarative Configuration

Instead of writing reward functions in code, you configure them in YAML:

```yaml
composer:
  components:
    - type: "CoherenceReward"
      weight: 1.0
    - type: "HelpfulnessReward"
      weight: 0.8
    - type: "DiversityReward"
      weight: 0.6
```

## How RLlama Works

### Training Flow

```
1. PROMPT: "Describe a beautiful sunset"
   ↓
2. MODEL GENERATES: "The sky turns orange and red colors..."
   ↓
3. RLLAMA EVALUATES: CoherenceReward=0.8, HelpfulnessReward=0.9, etc.
   ↓
4. TOTAL REWARD: 2.3 (sum of all weighted component rewards)
   ↓
5. PPO UPDATES: Model learns "this response got high reward"
```

### Component Responsibilities

- **Your Model (e.g., GPT-2)**: Generates the response text
- **RLlama**: Evaluates how "good" that response is using multiple criteria
- **RL Framework (TRL/PPO)**: Uses RLlama's scores to train the model

## Built-in Reward Components

### 1. CoherenceReward

**What it measures**: Response structure and logical flow

**Built-in definition**:
```python
def _coherence_reward(self, prompt: str, response: str) -> float:
    score = 0.0
    
    # Rule 1: Must have multiple sentences
    sentences = response.split('.')
    if len(sentences) > 1:
        score += 0.3  # "Good! Has multiple sentences"
    
    # Rule 2: Should have transition words
    transitions = ['however', 'therefore', 'moreover', 'furthermore']
    if any(word in response.lower() for word in transitions):
        score += 0.2  # "Good! Uses connecting words"
    
    # Rule 3: Appropriate length
    word_count = len(response.split())
    if 10 <= word_count <= 50:
        score += 0.5  # "Good! Right length"
    
    return score  # Returns 0.0 to 1.0
```

**Example evaluation**:
- **Response A**: "Orange sky nice." → CoherenceReward = 0.0
- **Response B**: "The sunset creates beautiful colors. However, the sky also shows purple hues." → CoherenceReward = 1.0

### 2. HelpfulnessReward

**What it measures**: How well the response addresses the prompt

**Logic**:
- Calculates word overlap between prompt and response
- Bonus for answering questions (looks for explanatory words)
- Penalizes off-topic responses

### 3. DiversityReward

**What it measures**: Lexical diversity and vocabulary richness

**Logic**:
- Ratio of unique words to total words
- Bonus for varied vocabulary
- Penalizes repetitive language

### 4. ConcisenessReward

**What it measures**: Response length appropriateness

**Logic**:
- Optimal range: 5-25 words (configurable)
- Penalizes very short responses (incomplete)
- Penalizes very long responses (verbose)

### 5. FactualityReward

**What it measures**: Basic factual content indicators

**Logic**:
- Looks for factual patterns (years, percentages, citations)
- Rewards definitive statements when facts are requested
- Penalizes uncertain language in factual contexts

### 6. LengthReward

**What it measures**: Target length compliance

**Parameters**:
- `optimal_length`: Target word count (default: 25)
- Normalized score based on deviation from target

### 7. EntropyBonus

**What it measures**: Information entropy in word usage

**Logic**:
- Calculates Shannon entropy of word distribution
- Rewards diverse word usage patterns
- Penalizes repetitive content

## Weight Scheduling

Dynamic weight adjustment over training steps:

### Schedule Types

#### 1. Exponential Decay
```yaml
- type: "DiversityReward"
  weight: 2.0
  schedule:
    type: "exponential_decay"
    decay_rate: 0.98  # weight = 2.0 * (0.98 ^ step)
```

#### 2. Linear Decay
```yaml
- type: "ExplorationBonus"
  weight: 1.0
  schedule:
    type: "linear_decay"
    decay_steps: 100  # Linearly decrease to 0 over 100 steps
```

#### 3. Curriculum Learning
```yaml
- type: "FactualityReward"
  weight: 0.1
  schedule:
    type: "curriculum"
    max_steps: 200  # Gradually increase importance
```

#### 4. Step Schedule
```yaml
- type: "StyleReward"
  weight: 0.5
  schedule:
    type: "step_schedule"
    steps:
      - step: 50
        weight: 1.0
      - step: 100
        weight: 1.5
```

## Configuration File Structure

### Complete YAML Configuration

```yaml
# Training configuration (optional)
training:
  model:
    name: "microsoft/DialoGPT-medium"
    type: "causal_lm"
    load_in_8bit: false
    
  dataset:
    name: "daily_dialog"
    split: "train"
    text_column: "dialog"
    max_samples: 1000

# Reward composition
composer:
  components:
    - type: "CoherenceReward"
      weight: 1.0
      params:
        min_sentences: 2
        transition_bonus: 0.3
    
    - type: "HelpfulnessReward"
      weight: 0.8
      params:
        overlap_weight: 0.7
        question_bonus: 0.4
    
    - type: "DiversityReward"
      weight: 0.6
      schedule:
        type: "exponential_decay"
        decay_rate: 0.98

# Reward normalization
shaper:
  normalization_method: "standard"  # Options: none, standard, minmax, robust

# Hyperparameter optimization (optional)
optimizer:
  enabled: false
```

## Framework Integrations

### TRL Integration

```python
from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor
from trl import PPOTrainer, PPOConfig

# Initialize RLlama processor
rllama_processor = TRLRllamaRewardProcessor(
    rllama_config_path="config.yaml"
)

# Training loop
for step in range(num_steps):
    # Generate responses
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    # Calculate RLlama rewards
    rewards = rllama_processor.compute_rewards(
        prompts_text=query_texts,
        responses_text=response_texts
    )
    
    # Convert to tensors for TRL
    scores = [torch.tensor(float(r), device=device) for r in rewards]
    
    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, scores)
```

### Stable Baselines3 Integration

```python
from rllama.integrations.stable_baselines3_wrapper import SB3RllamaWrapper
from stable_baselines3 import PPO

# Wrap your environment
env = SB3RllamaWrapper(
    base_env=your_text_env,
    config_path="config.yaml"
)

# Train with RLlama rewards
model = PPO('MlpPolicy', env)
model.learn(total_timesteps=10000)
```

### Ray RLlib Integration

```python
from rllama.integrations.rllib_wrapper import RLlibRllamaCallback

config = {
    "callbacks": RLlibRllamaCallback,
    "callback_config": {"config_path": "rewards.yaml"}
}

trainer = ppo.PPOTrainer(config=config, env="TextEnv-v0")
trainer.train()
```

## Evaluation Pipeline

### Step-by-Step Process

1. **Component Calculation**: Each reward component scores the response (0.0-1.0)
2. **Weight Scheduling**: Apply dynamic weights based on training step
3. **Composition**: Sum weighted scores from all components
4. **Normalization**: Apply statistical normalization for stable training
5. **Analysis Tracking**: Store component contributions for analysis

### Example Evaluation

**Prompt**: "Write a movie review"  
**Response**: "This movie was really good. However, the acting could be better. Overall, I recommend it."

```
RLlama Evaluation:

CoherenceReward (weight: 1.0):
- Multiple sentences? YES → +0.3
- Transition words? YES ("However", "Overall") → +0.2  
- Right length? YES (16 words) → +0.5
- Component Score: 1.0 × 1.0 = 1.0

HelpfulnessReward (weight: 0.8):
- Addresses "movie review"? YES → +0.7
- Uses review vocabulary? YES → +0.3
- Component Score: 1.0 × 0.8 = 0.8

DiversityReward (weight: 0.6):
- Unique words / total words = 14/16 = 0.875
- Component Score: 0.875 × 0.6 = 0.525

Total Raw Reward = 1.0 + 0.8 + 0.525 = 2.325
Normalized Reward = (2.325 - batch_mean) / batch_std
```

## Custom Reward Components

### Creating Custom Components

```python
from rllama.rewards.base_rewards import BaseReward

class MyCustomReward(BaseReward):
    """Custom reward component for domain-specific evaluation"""
    
    def __init__(self, domain_keywords=None, **kwargs):
        super().__init__(**kwargs)
        self.domain_keywords = domain_keywords or []
    
    def calculate(self, prompt: str, response: str, **kwargs) -> float:
        score = 0.0
        
        # Custom logic for your domain
        for keyword in self.domain_keywords:
            if keyword.lower() in response.lower():
                score += 0.2
        
        # Additional custom rules
        if len(response.split()) > 10:
            score += 0.3
            
        return min(1.0, score)

# Register the component
from rllama.core.registry import register_component
register_component("MyCustomReward", MyCustomReward)
```

### Using Custom Components

```yaml
composer:
  components:
    - type: "MyCustomReward"
      weight: 1.5
      params:
        domain_keywords: ["medical", "diagnosis", "treatment"]
```

## Analysis and Monitoring

### Component Analysis

```python
# Get training analysis
analysis = rllama_processor.get_component_analysis()

print(f"Total training steps: {analysis['total_steps']}")
print("Component average contributions:")
for comp_name, avg_reward in analysis['avg_rewards'].items():
    print(f"  {comp_name}: {avg_reward:.4f}")
```

### Batch-Level Details

```python
# Get detailed info from last batch
last_batch = rllama_processor.get_last_batch_detailed_infos()
if last_batch:
    print(f"Step {last_batch['step']}:")
    print(f"  Raw rewards: {last_batch['raw_rewards']}")
    print(f"  Normalized rewards: {last_batch['normalized_rewards']}")
```

## Real-World Use Cases

### 1. Content Moderation Training
```yaml
composer:
  components:
    - type: "ToxicityReward"      # Penalize toxic content
      weight: -2.0
    - type: "CoherenceReward"     # Reward coherent responses  
      weight: 1.0
    - type: "FactualityReward"    # Reward factual accuracy
      weight: 1.5
```

### 2. Customer Service Chatbot
```yaml
composer:
  components:
    - type: "HelpfulnessReward"   # Primary objective
      weight: 2.0
    - type: "PolitenessReward"    # Secondary objective
      weight: 1.0
    - type: "ConcisenessReward"   # Avoid verbosity
      weight: 0.5
```

### 3. Educational Content Generation
```yaml
composer:
  components:
    - type: "AccuracyReward"
      weight: 1.0
    - type: "SimplicityReward"    # Start high, decay over time
      weight: 2.0
      schedule:
        type: "exponential_decay"
        decay_rate: 0.95
```

## Benefits Over Manual Approach

### Traditional RL (Without RLlama)
```python
# Manual reward calculation - lots of boilerplate
def compute_rewards(queries, responses):
    rewards = []
    for query, response in zip(queries, responses):
        # Manual scoring
        toxicity_score = toxicity_model(response)
        coherence_score = coherence_model(response) 
        helpfulness_score = helpfulness_model(query, response)
        
        # Manual combination and normalization
        reward = -2.0 * toxicity_score + 1.0 * coherence_score + 1.5 * helpfulness_score
        # Manual normalization, scheduling, tracking...
        rewards.append(reward)
    return rewards
```

### RLlama Approach
```python
# Declarative - just configure and use
rllama_processor = TRLRllamaRewardProcessor("config.yaml")
rewards = rllama_processor.compute_rewards(query_texts, response_texts)
# Automatic: composition, weighting, scheduling, normalization, analysis
```

## Getting Started

### Quick Start Example

1. **Install RLlama**:
```bash
pip install rllama
```

2. **Create config file** (`rewards.yaml`):
```yaml
composer:
  components:
    - type: "CoherenceReward"
      weight: 1.0
    - type: "HelpfulnessReward"
      weight: 0.8

shaper:
  normalization_method: "standard"
```

3. **Use in your training**:
```python
from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor

processor = TRLRllamaRewardProcessor("rewards.yaml")
rewards = processor.compute_rewards(prompts, responses)
```

### Next Steps

- Explore advanced scheduling strategies
- Create custom reward components for your domain
- Set up hyperparameter optimization
- Integrate with your preferred RL framework
- Monitor component contributions and adjust weights

## Conclusion

RLlama transforms reward engineering from a manual, error-prone process into a declarative, composable, and optimizable system. By providing pre-built components, automatic scheduling, and framework integrations, it enables researchers and practitioners to focus on their core objectives rather than reward function implementation details.

The framework's design philosophy of "configuration over implementation" makes it accessible to both RL experts and newcomers while maintaining the flexibility needed for advanced research applications.