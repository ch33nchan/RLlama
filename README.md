<p align="center">
  <img src="https://raw.githubusercontent.com/ch33nchan/RLlama/main/rllama.jpg" height="250" alt="RLlama" />
</p>
<p align="center">
  <em>Empowering LLMs with Memory-Augmented Reinforcement Learning</em>
</p>
<p align="center">
    <a href="https://pypi.org/project/rllama/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/rllama?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://github.com/ch33nchan/RLlama">🔗 GitHub Repository</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://pypi.org/project/rllama">📦 PyPI Package</a>
</p>

# RLlama

RLlama introduces a novel approach to training Large Language Models (LLMs) by combining memory-augmented learning with reinforcement learning techniques. Our framework simplifies the process of implementing and experimenting with various RL algorithms while maintaining the context and memory of previous interactions.

## Features

- 🧠 Memory-Augmented Learning
- 🎮 Multiple RL Algorithms (PPO, DQN, A2C, SAC, REINFORCE)
- 🔄 Online Learning Support
- 🎯 Seamless Integration with Gymnasium
- 🚀 Multi-Modal Support (Coming Soon)

## Quick Start

Get started with RLlama in seconds:

```bash
pip install rllama
```

## Usage

Here's how to create a simple blackjack agent:

```python
from rllama import RLlamaAgent

class BlackjackAgent(RLlamaAgent):
    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Follow these rules:
        1. ALWAYS hit if your total is 11 or below
        2. With 12-16: hit if dealer shows 7+, stay if 6 or lower
        3. ALWAYS stay if your total is 17+ without an ace
        4. With a usable ace: hit if total is 17 or below"""

    def format_observation(self, observation) -> str:
        return f"Current hand total: {observation[0]}\nDealer's card: {observation[1]}\nUsable ace: {'yes' if observation[2] else 'no'}"

    def extract_action(self, response: str):
        return 0 if "stay" in response.lower() else 1
```

Train your agent:

```python
import gymnasium as gym
from transformers import AutoTokenizer, AutoModelForCausalLMWithValueHead

# Initialize model and agent
model = AutoModelForCausalLMWithValueHead.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
agent = BlackjackAgent(model, tokenizer, "cuda", algorithm="ppo")

# Training loop
env = gym.make("Blackjack-v1")
for episode in range(1000):
    observation, info = env.reset()
    done = False
    
    while not done:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        agent.assign_reward(reward)
        done = terminated or truncated
    
    agent.terminate_episode()
```

## Examples

Check out our example implementations:
- [Blackjack Agent](/examples/blackjack.py)
- [Text World Agent](/examples/textworld_agent.py) (Coming Soon)
- [Multi-Modal Agent](/examples/multimodal_agent.py) (Coming Soon)

## Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Relevant Work

- [Grounding Large Language Models with Online Reinforcement Learning](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)
- [Lamorel: Language Models for Reinforcement Learning](https://github.com/flowersteam/lamorel)

## Citation

```bibtex
@misc{ch33nchan2024rllama,
    title = {RLlama: Memory-Augmented Reinforcement Learning Framework for LLMs},
    author = {Ch33nchan},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/ch33nchan/RLlama}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```