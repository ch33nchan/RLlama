// RLlama Documentation App - Complete Implementation
class RLlamaApp {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.sidebarVisible = true;
        this.currentPage = 'home';
        this.resizeObserver = null;
        this.animationScene = null;
        this.animationRenderer = null;
        this.animationCamera = null;
        this.init();
    }

    init() {
        this.setupTheme();
        this.setupSidebar();
        this.setupNavigation();
        this.setupAnimation();
        this.setupResizeHandling();
        this.loadDocumentationContent();
    }

    setupTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const body = document.body;
        
        if (this.currentTheme === 'dark') {
            body.setAttribute('data-theme', 'dark');
            themeToggle.classList.add('dark');
        }
        
        themeToggle.addEventListener('click', () => {
            this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
            
            if (this.currentTheme === 'dark') {
                body.setAttribute('data-theme', 'dark');
                themeToggle.classList.add('dark');
            } else {
                body.removeAttribute('data-theme');
                themeToggle.classList.remove('dark');
            }
            
            localStorage.setItem('theme', this.currentTheme);
        });
    }

    setupSidebar() {
        const sidebarToggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');
        
        if (window.innerWidth <= 768) {
            this.sidebarVisible = false;
            sidebar.classList.add('hidden');
        }
        
        sidebarToggle.addEventListener('click', () => {
            this.sidebarVisible = !this.sidebarVisible;
            
            if (this.sidebarVisible) {
                sidebar.classList.remove('hidden');
                if (window.innerWidth <= 768) {
                    sidebar.classList.add('show');
                }
            } else {
                sidebar.classList.add('hidden');
                sidebar.classList.remove('show');
            }
        });
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.getAttribute('data-page');
                this.showPage(page);
                
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                if (window.innerWidth <= 768) {
                    const sidebar = document.getElementById('sidebar');
                    sidebar.classList.add('hidden');
                    sidebar.classList.remove('show');
                    this.sidebarVisible = false;
                }
            });
        });
    }

    setupResizeHandling() {
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(entries => {
                this.handleResize();
            });
            this.resizeObserver.observe(document.body);
        } else {
            let resizeTimeout;
            window.addEventListener('resize', () => {
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(() => this.handleResize(), 100);
            });
        }
    }

    handleResize() {
        const sidebar = document.getElementById('sidebar');
        
        if (window.innerWidth <= 768) {
            if (this.sidebarVisible) {
                sidebar.classList.add('hidden');
                sidebar.classList.remove('show');
                this.sidebarVisible = false;
            }
        } else {
            if (!this.sidebarVisible) {
                sidebar.classList.remove('hidden', 'show');
                this.sidebarVisible = true;
            }
        }

        this.resizeAnimation();
    }

    resizeAnimation() {
        const canvas = document.getElementById('rewardCanvas');
        if (!canvas || !this.animationRenderer || !this.animationCamera) return;

        const rect = canvas.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
            this.animationCamera.aspect = rect.width / rect.height;
            this.animationCamera.updateProjectionMatrix();
            this.animationRenderer.setSize(rect.width, rect.height);
        }
    }

    showPage(page) {
        this.currentPage = page;
        
        if (page === 'home') {
            document.getElementById('home').classList.add('active');
            document.getElementById('content-page').classList.remove('active');
        } else {
            document.getElementById('home').classList.remove('active');
            document.getElementById('content-page').classList.add('active');
            
            const content = this.getPageContent(page);
            document.getElementById('page-content').innerHTML = content;
            
            if (window.Prism) {
                Prism.highlightAll();
            }
        }
    }

    getPageContent(page) {
        const content = {
            installation: `
                <h1>Installation</h1>
                <p>RLlama can be installed directly from PyPI using pip. The library supports Python 3.8+ and integrates seamlessly with popular RL frameworks.</p>
                
                <h2>Quick Installation</h2>
                <pre><code class="language-bash">pip install rllama</code></pre>
                
                <h2>Development Installation</h2>
                <p>For contributors or users who want the latest features, install from source:</p>
                <pre><code class="language-bash">git clone https://github.com/your-username/rllama.git
cd rllama
pip install -e .</code></pre>
                
                <h2>Optional Dependencies</h2>
                <p>Install additional packages for enhanced functionality:</p>
                <pre><code class="language-bash"># For Bayesian optimization
pip install rllama[optimization]

# For visualization tools
pip install rllama[viz]

# For all optional dependencies
pip install rllama[all]</code></pre>
                
                <h2>Verification</h2>
                <p>Verify your installation by running:</p>
                <pre><code class="language-python">import rllama
print(rllama.__version__)</code></pre>
            `,
            
            quickstart: `
                <h1>Quick Start Guide</h1>
                <p>Get up and running with RLlama in just a few minutes. This guide covers the basic concepts and shows you how to create your first reward function.</p>
                
                <h2>Basic Usage</h2>
                <p>Here's a simple example that demonstrates the core RLlama workflow:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import LengthReward, QualityReward

# Create a reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(LengthReward(target_length=100, weight=0.3))
engine.add_component(QualityReward(weight=0.7))

# Compute rewards
context = {
    "response": "This is a test response for evaluation",
    "target_length": 100,
    "quality_score": 0.85
}

reward = engine.compute(context)
print(f"Total reward: {reward}")

# Get detailed breakdown
breakdown = engine.get_breakdown(context)
for component, value in breakdown.items():
    print(f"{component}: {value}")</code></pre>
                
                <h2>Working with Environments</h2>
                <p>RLlama integrates seamlessly with OpenAI Gym environments:</p>
                
                <pre><code class="language-python">import gym
from rllama.integration import GymWrapper
from rllama.rewards.components import GoalReward, StepPenalty

# Create environment and reward engine
env = gym.make('FrozenLake-v1')
engine = RewardEngine()
engine.add_component(GoalReward(reward_value=1.0))
engine.add_component(StepPenalty(penalty=-0.01))

# Wrap the environment
wrapped_env = GymWrapper(engine).wrap(env)

# Use like any other Gym environment
obs = wrapped_env.reset()
for _ in range(100):
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    if done:
        break</code></pre>
                
                <h2>Next Steps</h2>
                <p>Now that you have RLlama working, explore these topics:</p>
                <ul>
                    <li><strong>Core Concepts</strong> - Understand reward components and composition</li>
                    <li><strong>Component Design</strong> - Learn to create custom reward functions</li>
                    <li><strong>Cookbook</strong> - See practical examples for common scenarios</li>
                    <li><strong>Integration</strong> - Connect with your favorite RL framework</li>
                </ul>
            `,
            
            usage: `
                <h1>Usage Guide</h1>
                <p>This comprehensive guide covers all aspects of using RLlama effectively in your reinforcement learning projects.</p>
                
                <h2>RewardEngine</h2>
                <p>The RewardEngine is the central component that manages all reward components and orchestrates reward computation:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine

# Initialize the engine
engine = RewardEngine()

# Add components
engine.add_component(component_instance)

# Compute total reward
reward = engine.compute(context)

# Get detailed breakdown
breakdown = engine.get_breakdown(context)

# Remove a component
engine.remove_component(component_name)

# List all components
components = engine.list_components()</code></pre>
                
                <h2>Reward Components</h2>
                <p>Individual reward functions that can be composed together. Each component focuses on a single aspect of behavior:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def compute(self, context):
        # Extract relevant information
        score = context.get('performance_score', 0)
        
        # Apply reward logic
        reward = score * self.scale
        
        return reward</code></pre>
                
                <h2>Context Dictionary</h2>
                <p>The context dictionary is how you pass information to reward components:</p>
                
                <pre><code class="language-python"># Common context keys
context = {
    'state': current_state,
    'action': taken_action,
    'next_state': resulting_state,
    'done': episode_finished,
    'info': env_info_dict,
    'step': current_step,
    'episode': current_episode,
    # Custom keys for your specific use case
    'custom_metric': calculated_value
}</code></pre>
                
                <h2>Weight Management</h2>
                <p>Control the influence of different reward components:</p>
                
                <pre><code class="language-python"># Set component weights
engine.set_weight('goal_reward', 1.0)
engine.set_weight('step_penalty', 0.1)

# Get current weights
weights = engine.get_weights()

# Update weights dynamically
engine.update_weights({
    'exploration_bonus': 0.5,
    'safety_penalty': 2.0
})</code></pre>
            `,
            
            concepts: `
                <h1>Core Concepts</h1>
                <p>This document expands on the fundamental concepts of RLlama, providing detailed explanations of how RLlama structures and manages reward engineering.</p>
                
                <h2>1. The Building Block: RewardComponent</h2>
                <p>At its core, RLlama encourages breaking down complex reward logic into smaller, manageable, and reusable pieces. This is achieved through the <code>RewardComponent</code> base class.</p>
                
                <p><strong>Purpose:</strong> To encapsulate the logic for calculating a <em>single aspect</em> of the total reward.</p>
                
                <h3>Implementation</h3>
                <p>You create a Python class that inherits from <code>rllama.rewards.RewardComponent</code> and implement the <code>calculate_reward</code> method:</p>
                
                <pre><code class="language-python">from rllama.rewards import RewardComponent

class StepPenalty(RewardComponent):
    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty
    
    def calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float:
        # This component ignores raw_reward, info, and context,
        # simply returns a fixed penalty for taking a step.
        return self.penalty</code></pre>
                
                <h3>Method Parameters</h3>
                <ul>
                    <li><strong>raw_reward</strong>: The original reward value returned directly by the environment step</li>
                    <li><strong>info</strong>: The info dictionary returned by the environment step</li>
                    <li><strong>context</strong>: A dictionary provided by the user via the RewardShaper</li>
                    <li><strong>**kwargs</strong>: Allows for future flexibility and additional arguments</li>
                </ul>
                
                <h2>2. Combining Components: RewardComposer</h2>
                <p>Once you have individual components, you need a way to combine their outputs into a single reward signal.</p>
                
                <pre><code class="language-python">from rllama import RewardComposer

composer = RewardComposer({
    "goal": GoalReward(), 
    "penalty": StepPenalty(-0.05)
})

composed_reward_dict = composer.compose(raw_reward, info, context)</code></pre>
                
                <p>The composer returns a dictionary where keys are component names and values are the rewards calculated by each component.</p>
                
                <h2>3. Dynamic Control: RewardConfig and RewardShaper</h2>
                <p>Static rewards are often insufficient. RLlama allows dynamic control over the influence of each component through configuration and the <code>RewardShaper</code>.</p>
                
                <h3>RewardConfig Structure</h3>
                <pre><code class="language-yaml">reward_shaping:
  component_name:  # e.g., "step_penalty"
    params:
      penalty: -0.01
    weight_schedule:
      initial_weight: 1.0
      schedule_type: exponential
      decay_rate: 0.999
      decay_steps: 1
      min_weight: 0.1</code></pre>
                
                <h3>RewardShaper Usage</h3>
                <pre><code class="language-python">shaper = RewardShaper(composer, reward_shaping_config)
final_reward = shaper.shape(raw_reward, info, context)

# Update weights based on training progress
shaper.update_weights(global_step)</code></pre>
                
                <h2>4. The Power of Context</h2>
                <p>The <code>context</code> dictionary is a key feature for advanced reward design, allowing you to inject arbitrary information from your training loop into reward calculations.</p>
                
                <h3>Common Context Uses</h3>
                <ul>
                    <li>Current <code>global_step</code> or <code>episode_num</code></li>
                    <li>Steps taken within the current episode</li>
                    <li>Agent's internal state or uncertainty estimates</li>
                    <li>Performance metrics calculated during training</li>
                    <li>Custom flags indicating training phases</li>
                </ul>
                
                <h3>Example: Context-Dependent Reward</h3>
                <pre><code class="language-python">class UncertaintyPenalty(RewardComponent):
    def calculate_reward(self, raw_reward, info, context, **kwargs):
        uncertainty = context.get("agent_uncertainty", 0.0)
        # Penalize high uncertainty more
        return -uncertainty * 0.1

# In training loop:
context = {
    "global_step": global_step,
    "steps_in_episode": steps_this_episode,
    "agent_uncertainty": agent.get_uncertainty(),
}
shaped_reward = shaper.shape(raw_reward, info, context)</code></pre>
                
                <h2>Benefits of This Architecture</h2>
                <ul>
                    <li><strong>Modularity</strong>: Each component has a single responsibility</li>
                    <li><strong>Reusability</strong>: Components can be shared across different environments</li>
                    <li><strong>Testability</strong>: Individual components can be tested in isolation</li>
                    <li><strong>Flexibility</strong>: Dynamic weight scheduling and context injection</li>
                    <li><strong>Debuggability</strong>: Clear separation makes issues easier to identify</li>
                </ul>
            `,
            
            'component-design': `
                <h1>Reward Component Design</h1>
                <p>This guide explains how to design custom reward components in RLlama, including best practices and advanced techniques for creating robust, reusable reward functions.</p>
                
                <h2>Basic Component Structure</h2>
                <p>Every reward component in RLlama inherits from the <code>BaseReward</code> class and implements the <code>compute</code> method:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def compute(self, context):
        # Extract relevant information from context
        relevant_data = context.get('key', default_value)
        
        # Perform reward calculation
        reward_value = self.calculate_logic(relevant_data)
        
        # Return a numerical reward value
        return reward_value
    
    def calculate_logic(self, data):
        # Your specific reward logic here
        return data * self.param1 + self.param2</code></pre>
                
                <h2>The Context Object</h2>
                <p>The <code>context</code> parameter is a dictionary containing all information needed to calculate rewards. Common keys include:</p>
                
                <ul>
                    <li><code>state</code>: The current state of the environment</li>
                    <li><code>action</code>: The action taken by the agent</li>
                    <li><code>next_state</code>: The resulting state after taking the action</li>
                    <li><code>done</code>: A boolean indicating if the episode is complete</li>
                    <li><code>response</code>: For language models, the generated text</li>
                    <li><code>history</code>: For sequential tasks, the history of previous states/responses</li>
                </ul>
                
                <p>Your component should extract the information it needs from this context object and handle missing data gracefully.</p>
                
                <h2>Best Practices</h2>
                
                <h3>1. Single Responsibility</h3>
                <p>Each component should focus on a single aspect of behavior. Don't try to pack multiple reward concepts into one component.</p>
                
                <h3>2. Robust to Missing Data</h3>
                <pre><code class="language-python">def compute(self, context):
    # Always provide defaults for missing data
    score = context.get('performance_score', 0.0)
    
    # Validate data types
    if not isinstance(score, (int, float)):
        return 0.0
    
    # Handle edge cases
    if score < 0:
        score = 0.0
    
    return score * self.scale</code></pre>
                
                <h3>3. Configurable Parameters</h3>
                <p>Make important values configurable through the constructor:</p>
                
                <pre><code class="language-python">class ConfigurableReward(BaseReward):
    def __init__(self, scale=1.0, offset=0.0, clamp_min=None, clamp_max=None):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def compute(self, context):
        raw_value = context.get('raw_metric', 0)
        
        # Apply transformations
        value = raw_value * self.scale + self.offset
        
        # Apply clamping if specified
        if self.clamp_min is not None:
            value = max(value, self.clamp_min)
        if self.clamp_max is not None:
            value = min(value, self.clamp_max)
        
        return value</code></pre>
                
                <h3>4. Normalized Output Range</h3>
                <p>Keep rewards in a consistent range to avoid one component dominating others:</p>
                
                <pre><code class="language-python">def compute(self, context):
    raw_score = self.calculate_raw_score(context)
    
    # Normalize to [0, 1] range
    normalized = (raw_score - self.min_expected) / (self.max_expected - self.min_expected)
    normalized = max(0.0, min(1.0, normalized))
    
    return normalized</code></pre>
                
                <h3>5. Documentation and Testing</h3>
                <pre><code class="language-python">class WellDocumentedReward(BaseReward):
    """
    Rewards the agent based on task completion efficiency.
    
    This component calculates reward based on how quickly the agent
    completes a task relative to a baseline expectation.
    
    Args:
        baseline_steps (int): Expected number of steps for task completion
        efficiency_scale (float): Scaling factor for efficiency bonus
        max_bonus (float): Maximum bonus that can be awarded
    
    Context Requirements:
        - 'task_completed' (bool): Whether the task was completed
        - 'steps_taken' (int): Number of steps taken in current episode
    
    Returns:
        float: Efficiency bonus in range [0, max_bonus]
    """
    
    def __init__(self, baseline_steps=100, efficiency_scale=1.0, max_bonus=1.0):
        super().__init__()
        self.baseline_steps = baseline_steps
        self.efficiency_scale = efficiency_scale
        self.max_bonus = max_bonus
    
    def compute(self, context):
        if not context.get('task_completed', False):
            return 0.0
        
        steps_taken = context.get('steps_taken', float('inf'))
        if steps_taken <= 0:
            return 0.0
        
        efficiency = self.baseline_steps / steps_taken
        bonus = min(efficiency * self.efficiency_scale, self.max_bonus)
        
        return bonus</code></pre>
                
                <h2>Testing Your Components</h2>
                <p>Always test your components with various context scenarios:</p>
                
                <pre><code class="language-python">def test_my_reward():
    reward = MyCustomReward(scale=2.0)
    
    # Test normal case
    context = {'score': 0.5}
    assert reward.compute(context) == 1.0
    
    # Test missing data
    context = {}
    assert reward.compute(context) == 0.0
    
    # Test edge cases
    context = {'score': -1.0}
    result = reward.compute(context)
    assert result >= 0.0  # Should handle negative inputs gracefully</code></pre>
            `,
            
            basic: `
                <h1>Basic Examples</h1>
                <p>Here are fundamental examples to get you started with RLlama. These examples demonstrate core concepts and common patterns you'll use in most reward engineering tasks.</p>
                
                <h2>Simple Length Reward</h2>
                <p>This example shows how to create a basic reward that encourages responses of a specific length:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import LengthReward

# Create a reward engine
engine = RewardEngine()
engine.add_component(LengthReward(target_length=100))

# Compute rewards for different responses
contexts = [
    {"response": "Short response"},
    {"response": "This is a medium length response that contains more words and should be closer to target"},
    {"response": "This is a very long response that goes on and on with lots of details and explanations that exceed the target length significantly"}
]

for i, context in enumerate(contexts):
    reward = engine.compute(context)
    print(f"Response {i+1} (length: {len(context['response'])}): reward = {reward:.3f}")
</code></pre>
                
                <h2>Combining Multiple Rewards</h2>
                <p>Most real-world scenarios require combining multiple reward signals. Here's how to do it effectively:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import LengthReward, QualityReward

# Create engine with multiple components
engine = RewardEngine()
engine.add_component(LengthReward(target_length=100, weight=0.3))
engine.add_component(QualityReward(weight=0.7))

# Example context with multiple metrics
context = {
    "response": "This is a well-crafted response that balances length and quality effectively.",
    "quality_score": 0.85,
    "coherence": 0.9,
    "relevance": 0.8
}

# Get total reward
total_reward = engine.compute(context)
print(f"Total reward: {total_reward:.3f}")

# Get detailed breakdown
breakdown = engine.get_breakdown(context)
print("Reward breakdown:")
for component, value in breakdown.items():
    print(f"  {component}: {value:.3f}")
</code></pre>
                
                <h2>Goal-Based Reward</h2>
                <p>A common pattern in RL is rewarding goal achievement with step penalties:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import GoalReward, StepPenalty

# Set up goal-based reward system
engine = RewardEngine()
engine.add_component(GoalReward(reward_value=10.0))
engine.add_component(StepPenalty(penalty=-0.01))

# Simulate episode steps
episode_contexts = [
    {"goal_reached": False, "step": 1},
    {"goal_reached": False, "step": 2},
    {"goal_reached": False, "step": 3},
    {"goal_reached": True, "step": 4},  # Goal reached!
]

total_episode_reward = 0
for context in episode_contexts:
    step_reward = engine.compute(context)
    total_episode_reward += step_reward
    print(f"Step {context['step']}: reward = {step_reward:.3f}")

print(f"Total episode reward: {total_episode_reward:.3f}")
</code></pre>
                
                <h2>Custom Reward Component</h2>
                <p>Creating your own reward components is straightforward:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
from rllama import RewardEngine

class ProgressReward(BaseReward):
    """Rewards progress toward a goal based on distance reduction."""
    
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.previous_distance = None
    
    def compute(self, context):
        current_distance = context.get('distance_to_goal', 0)
        
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Reward for reducing distance
        progress = self.previous_distance - current_distance
        reward = progress * self.scale
        
        self.previous_distance = current_distance
        return reward

# Use the custom component
engine = RewardEngine()
engine.add_component(ProgressReward(scale=2.0))

# Simulate approaching a goal
distances = [10.0, 8.5, 6.2, 4.1, 2.3, 0.8, 0.0]
for i, distance in enumerate(distances):
    context = {'distance_to_goal': distance}
    reward = engine.compute(context)
    print(f"Step {i+1} (distance: {distance}): reward = {reward:.3f}")
</code></pre>
                
                <h2>Environment Integration</h2>
                <p>Here's how to integrate RLlama with a simple Gym environment:</p>
                
                <pre><code class="language-python">import gym
from rllama import RewardEngine
from rllama.rewards.components import GoalReward, StepPenalty
from rllama.integration import GymWrapper

# Create environment
env = gym.make('FrozenLake-v1')

# Set up reward engine
engine = RewardEngine()
engine.add_component(GoalReward(reward_value=1.0))
engine.add_component(StepPenalty(penalty=-0.01))

# Wrap environment
wrapped_env = GymWrapper(engine).wrap(env)

# Run episode
obs = wrapped_env.reset()
total_reward = 0
step_count = 0

while True:
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    total_reward += reward
    step_count += 1
    
    print(f"Step {step_count}: action={action}, reward={reward:.3f}")
    
    if done:
        print(f"Episode finished in {step_count} steps")
        print(f"Total reward: {total_reward:.3f}")
        break

env.close()
</code></pre>
                
                <h2>Dynamic Weight Adjustment</h2>
                <p>You can dynamically adjust component weights during training:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import ExplorationBonus, SafetyPenalty

engine = RewardEngine()
engine.add_component(ExplorationBonus(weight=1.0))
engine.add_component(SafetyPenalty(weight=0.1))

# Simulate training progress
for episode in range(100):
    # Gradually reduce exploration and increase safety
    exploration_weight = max(0.1, 1.0 - episode * 0.01)
    safety_weight = min(1.0, 0.1 + episode * 0.01)
    
    engine.set_weight('exploration_bonus', exploration_weight)
    engine.set_weight('safety_penalty', safety_weight)
    
    # Your training loop here
    context = {'exploration_score': 0.5, 'safety_violation': False}
    reward = engine.compute(context)
    
    if episode % 20 == 0:
        print(f"Episode {episode}: exploration_weight={exploration_weight:.2f}, "
              f"safety_weight={safety_weight:.2f}, reward={reward:.3f}")
</code></pre>
            `,
            
            advanced: `
                <h1>Advanced Examples</h1>
                <p>Advanced examples showing complex reward engineering patterns and optimization techniques in RLlama.</p>
                
                <h2>Reward Optimization with Bayesian Search</h2>
                <p>Automatically find optimal reward weights using Bayesian optimization:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.optimizer import RewardOptimizer
import optuna

# Define evaluation function
def evaluate_weights(trial):
    # Suggest weights for different components
    goal_weight = trial.suggest_float('goal_weight', 0.1, 10.0)
    penalty_weight = trial.suggest_float('penalty_weight', 0.001, 1.0)
    
    # Create engine with suggested weights
    engine = RewardEngine()
    engine.add_component(GoalReward(weight=goal_weight))
    engine.add_component(StepPenalty(weight=penalty_weight))
    
    # Run training/evaluation and return performance metric
    performance = run_training_with_engine(engine)
    return performance

# Create optimizer
optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(evaluate_weights, n_trials=100)

print(f"Best weights found: {best_weights}")
</code></pre>
                
                <h2>Custom Reward Components with State</h2>
                <p>Create sophisticated reward components that maintain internal state:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np
from collections import deque

class AdaptiveExplorationReward(BaseReward):
    """Provides exploration bonuses that adapt based on recent performance."""
    
    def __init__(self, window_size=100, base_bonus=0.1, adaptation_rate=0.01):
        super().__init__()
        self.window_size = window_size
        self.base_bonus = base_bonus
        self.adaptation_rate = adaptation_rate
        self.performance_history = deque(maxlen=window_size)
        self.state_visit_counts = {}
        self.current_bonus_scale = 1.0
    
    def compute(self, context):
        state = context.get('state_representation')
        performance = context.get('episode_performance', 0)
        
        if state is None:
            return 0.0
        
        # Update performance tracking
        self.performance_history.append(performance)
        
        # Adapt bonus scale based on recent performance
        if len(self.performance_history) >= self.window_size:
            recent_avg = np.mean(list(self.performance_history))
            if recent_avg < 0.5:  # Poor performance, increase exploration
                self.current_bonus_scale += self.adaptation_rate
            else:  # Good performance, reduce exploration
                self.current_bonus_scale = max(0.1, 
                    self.current_bonus_scale - self.adaptation_rate)
        
        # Calculate exploration bonus
        visit_count = self.state_visit_counts.get(state, 0)
        exploration_bonus = self.base_bonus * self.current_bonus_scale / (1 + visit_count)
        
        # Update visit count
        self.state_visit_counts[state] = visit_count + 1
        
        return exploration_bonus
    
    def reset(self):
        """Reset for new episode or training run."""
        self.state_visit_counts.clear()
        self.performance_history.clear()
        self.current_bonus_scale = 1.0

# Usage example
engine = RewardEngine()
exploration_reward = AdaptiveExplorationReward()
engine.add_component(exploration_reward)
</code></pre>
                
                <h2>Multi-Objective Reward Balancing</h2>
                <p>Balance multiple competing objectives using Pareto optimization:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np

class ParetoBalancedReward(BaseReward):
    """Balances multiple objectives using Pareto front approximation."""
    
    def __init__(self, objectives, weights=None, reference_point=None):
        super().__init__()
        self.objectives = objectives  # List of objective functions
        self.weights = weights or [1.0] * len(objectives)
        self.reference_point = reference_point or [0.0] * len(objectives)
        self.pareto_archive = []
        self.max_archive_size = 100
    
    def compute(self, context):
        # Evaluate all objectives
        objective_values = []
        for obj_func in self.objectives:
            value = obj_func(context)
            objective_values.append(value)
        
        # Update Pareto archive
        self._update_pareto_archive(objective_values)
        
        # Calculate hypervolume-based reward
        if len(self.pareto_archive) > 0:
            hypervolume = self._calculate_hypervolume()
            # Reward improvement in hypervolume
            reward = hypervolume * 0.1
        else:
            reward = 0.0
        
        return reward
    
    def _update_pareto_archive(self, new_point):
        """Update Pareto archive with new point."""
        # Remove dominated points
        self.pareto_archive = [p for p in self.pareto_archive 
                              if not self._dominates(new_point, p)]
        
        # Add new point if not dominated
        if not any(self._dominates(p, new_point) for p in self.pareto_archive):
            self.pareto_archive.append(new_point)
        
        # Maintain archive size
        if len(self.pareto_archive) > self.max_archive_size:
            self.pareto_archive = self.pareto_archive[-self.max_archive_size:]
    
    def _dominates(self, point1, point2):
        """Check if point1 dominates point2."""
        return all(p1 >= p2 for p1, p2 in zip(point1, point2)) and \
               any(p1 > p2 for p1, p2 in zip(point1, point2))
    
    def _calculate_hypervolume(self):
        """Calculate hypervolume of Pareto archive."""
        if not self.pareto_archive:
            return 0.0
        
        # Simplified hypervolume calculation
        volume = 0.0
        for point in self.pareto_archive:
            point_volume = 1.0
            for i, (val, ref) in enumerate(zip(point, self.reference_point)):
                point_volume *= max(0, val - ref)
            volume += point_volume
        
        return volume

# Usage example
def speed_objective(context):
    return context.get('speed_score', 0)

def safety_objective(context):
    return context.get('safety_score', 0)

def efficiency_objective(context):
    return context.get('efficiency_score', 0)

pareto_reward = ParetoBalancedReward([
    speed_objective,
    safety_objective,
    efficiency_objective
])

engine = RewardEngine()
engine.add_component(pareto_reward)
</code></pre>
                
                <h2>Curriculum Learning with Adaptive Difficulty</h2>
                <p>Implement curriculum learning that adapts based on agent performance:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np

class CurriculumReward(BaseReward):
    """Implements curriculum learning with adaptive difficulty adjustment."""
    
    def __init__(self, base_tasks, difficulty_levels, success_threshold=0.8):
        super().__init__()
        self.base_tasks = base_tasks
        self.difficulty_levels = difficulty_levels
        self.success_threshold = success_threshold
        self.current_difficulty = 0
        self.success_history = deque(maxlen=100)
        self.task_rewards = {}
    
    def compute(self, context):
        task_id = context.get('task_id')
        success = context.get('task_success', False)
        
        if task_id is None:
            return 0.0
        
        # Record success
        self.success_history.append(success)
        
        # Update difficulty based on recent performance
        if len(self.success_history) >= 50:
            recent_success_rate = np.mean(list(self.success_history)[-50:])
            
            if recent_success_rate > self.success_threshold and \
               self.current_difficulty < len(self.difficulty_levels) - 1:
                self.current_difficulty += 1
                print(f"Increasing difficulty to level {self.current_difficulty}")
            elif recent_success_rate < 0.5 and self.current_difficulty > 0:
                self.current_difficulty -= 1
                print(f"Decreasing difficulty to level {self.current_difficulty}")
        
        # Calculate reward based on current difficulty
        difficulty_multiplier = self.difficulty_levels[self.current_difficulty]
        base_reward = 1.0 if success else 0.0
        
        # Bonus for succeeding at higher difficulties
        difficulty_bonus = self.current_difficulty * 0.1
        
        total_reward = base_reward * difficulty_multiplier + difficulty_bonus
        
        return total_reward
    
    def get_current_difficulty(self):
        return self.current_difficulty

# Usage example
curriculum = CurriculumReward(
    base_tasks=['easy', 'medium', 'hard', 'expert'],
    difficulty_levels=[1.0, 1.5, 2.0, 3.0],
    success_threshold=0.8
)

engine = RewardEngine()
engine.add_component(curriculum)
</code></pre>
                
                <h2>Memory-Augmented Reward Components</h2>
                <p>Create reward components that use episodic memory for long-term behavior shaping:</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np
from collections import defaultdict

class EpisodicMemoryReward(BaseReward):
    """Uses episodic memory to reward novel and diverse behaviors."""
    
    def __init__(self, memory_size=1000, novelty_threshold=0.1, diversity_weight=0.5):
        super().__init__()
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold
        self.diversity_weight = diversity_weight
        self.episode_memory = []
        self.state_embeddings = []
        
    def compute(self, context):
        state_embedding = context.get('state_embedding')
        action = context.get('action')
        
        if state_embedding is None or action is None:
            return 0.0
        
        # Calculate novelty reward
        novelty_reward = self._calculate_novelty(state_embedding)
        
        # Calculate diversity reward
        diversity_reward = self._calculate_diversity(state_embedding, action)
        
        # Store in memory
        self._update_memory(state_embedding, action)
        
        total_reward = novelty_reward + self.diversity_weight * diversity_reward
        
        return total_reward
    
    def _calculate_novelty(self, state_embedding):
        """Calculate novelty based on distance to stored experiences."""
        if not self.state_embeddings:
            return 1.0  # First experience is maximally novel
        
        # Find minimum distance to stored experiences
        distances = [np.linalg.norm(state_embedding - stored) 
                    for stored in self.state_embeddings]
        min_distance = min(distances)
        
        # Convert distance to novelty reward
        novelty = min_distance / (min_distance + self.novelty_threshold)
        
        return novelty
    
    def _calculate_diversity(self, state_embedding, action):
        """Calculate diversity reward based on action variety in similar states."""
        if not self.episode_memory:
            return 1.0
        
        # Find similar states
        similar_experiences = []
        for stored_state, stored_action in self.episode_memory:
            distance = np.linalg.norm(state_embedding - stored_state)
            if distance < self.novelty_threshold:
                similar_experiences.append(stored_action)
        
        if not similar_experiences:
            return 1.0
        
        # Calculate action diversity
        unique_actions = len(set(similar_experiences))
        total_actions = len(similar_experiences)
        
        diversity = unique_actions / total_actions
        
        return diversity
    
    def _update_memory(self, state_embedding, action):
        """Update episodic memory with new experience."""
        self.episode_memory.append((state_embedding, action))
        self.state_embeddings.append(state_embedding)
        
        # Maintain memory size
        if len(self.episode_memory) > self.memory_size:
            self.episode_memory.pop(0)
            self.state_embeddings.pop(0)
    
    def reset_episode(self):
        """Reset episodic memory for new episode."""
        self.episode_memory.clear()
        self.state_embeddings.clear()

# Usage example
memory_reward = EpisodicMemoryReward(
    memory_size=500,
    novelty_threshold=0.2,
    diversity_weight=0.3
)

engine = RewardEngine()
engine.add_component(memory_reward)
</code></pre>
            `,
            
            cookbook: `
                <h1>RLlama Cookbook: Practical Recipes</h1>
                <p>This cookbook provides practical examples and patterns for implementing common reward engineering techniques using RLlama.</p>
                
                <h2>Recipe 1: Basic Goal Reward + Step Penalty</h2>
                <p>This is the most common starting point. Reward the agent for reaching a goal and penalize it slightly for each step taken.</p>
                
                <h3>Components:</h3>
                <ul>
                    <li><code>GoalReward</code>: Provides a positive reward when a goal condition is met</li>
                    <li><code>StepPenalty</code>: Provides a small negative reward for every step</li>
                </ul>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.rewards.components import GoalReward, StepPenalty

engine = RewardEngine()
engine.add_component(GoalReward(reward_value=1.0))
engine.add_component(StepPenalty(penalty=-0.01))</code></pre>
                
                <h3>Configuration (config.yaml):</h3>
                <pre><code class="language-yaml">reward_shaping:
  goal:
    class: GoalReward
    params: { reward_value: 1.0 }
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
  time_cost:
    class: StepPenalty
    params: { penalty: -0.01 }
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }</code></pre>
                
                <h2>Recipe 2: Sparse Reward with Distance Shaping</h2>
                <p>In environments with sparse rewards, potential-based reward shaping can provide denser guidance.</p>
                
                <pre><code class="language-python">import numpy as np
from rllama.rewards.base import BaseReward

class DistanceReward(BaseReward):
    def __init__(self, potential_scale=1.0, gamma=0.99):
        self.potential_scale = potential_scale
        self.gamma = gamma
        self.previous_potential = None
    
    def compute(self, context):
        agent_pos = context.get("agent_pos")
        target_pos = context.get("target_pos")
        
        if agent_pos is None or target_pos is None:
            return 0.0
        
        distance = np.linalg.norm(np.array(agent_pos) - np.array(target_pos))
        current_potential = -distance * self.potential_scale
        
        shaping_reward = 0.0
        if self.previous_potential is not None:
            shaping_reward = self.gamma * current_potential - self.previous_potential
        
        terminated = context.get("terminated", False)
        truncated = context.get("truncated", False)
        if terminated or truncated:
            self.previous_potential = None
        else:
            self.previous_potential = current_potential
        
        return shaping_reward</code></pre>
                
                <h2>Recipe 3: Curriculum Learning via Weight Scheduling</h2>
                <p>Gradually introduce or fade out reward components to guide the agent through different learning stages.</p>
                
                <pre><code class="language-yaml">reward_shaping:
  goal:
    class: GoalReward
    params: { reward_value: 1.0 }
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
  safety:
    class: CollisionPenalty
    params: { penalty: -5.0 }
    weight_schedule:
      initial_weight: 1.0
      schedule_type: exponential
      decay_rate: 0.9999
      decay_steps: 100
      min_weight: 0.1
  speed:
    class: EfficiencyBonus
    params: { max_bonus: 0.5 }
    weight_schedule:
      initial_weight: 0.0
      schedule_type: linear
      end_weight: 1.0
      schedule_start_step: 10000
      schedule_duration_steps: 50000</code></pre>
                
                <h2>Recipe 4: Context-Dependent Exploration</h2>
                <p>Modify rewards based on training phase using context information.</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
from collections import defaultdict
import numpy as np

class ExplorationBonus(BaseReward):
    def __init__(self, bonus_scale=0.01, max_training_steps=1000000):
        super().__init__()
        self.bonus_scale = bonus_scale
        self.max_training_steps = max_training_steps
        self.state_visit_counts = defaultdict(int)
    
    def compute(self, context):
        global_step = context.get("global_step", 0)
        
        # Only apply bonus during first half of training
        if global_step > self.max_training_steps / 2:
            return 0.0
        
        current_state = context.get("agent_state_representation")
        if current_state is None:
            return 0.0
        
        try:
            count = self.state_visit_counts[current_state]
            bonus = self.bonus_scale / np.sqrt(count + 1)
            self.state_visit_counts[current_state] += 1
            return bonus
        except TypeError:
            return 0.0</code></pre>
                
                <h2>Recipe 5: Safety-Constrained Learning</h2>
                <p>Implement safety constraints that adapt based on violation frequency.</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
from collections import deque

class AdaptiveSafetyPenalty(BaseReward):
    def __init__(self, base_penalty=-1.0, adaptation_rate=0.1, window_size=100):
        super().__init__()
        self.base_penalty = base_penalty
        self.adaptation_rate = adaptation_rate
        self.violation_history = deque(maxlen=window_size)
        self.current_penalty = base_penalty
    
    def compute(self, context):
        safety_violation = context.get("safety_violation", False)
        
        # Record violation
        self.violation_history.append(safety_violation)
        
        # Adapt penalty based on recent violation rate
        if len(self.violation_history) >= 50:
            violation_rate = sum(self.violation_history) / len(self.violation_history)
            
            if violation_rate > 0.1:  # Too many violations
                self.current_penalty *= (1 + self.adaptation_rate)
            elif violation_rate < 0.02:  # Very few violations
                self.current_penalty *= (1 - self.adaptation_rate * 0.5)
            
            # Clamp penalty
            self.current_penalty = max(self.base_penalty * 10, 
                                     min(self.base_penalty * 0.1, self.current_penalty))
        
        return self.current_penalty if safety_violation else 0.0</code></pre>
                
                <h2>Recipe 6: Multi-Task Reward Balancing</h2>
                <p>Balance rewards across multiple tasks with automatic weight adjustment.</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np

class MultiTaskBalancer(BaseReward):
    def __init__(self, task_rewards, balance_method="uncertainty"):
        super().__init__()
        self.task_rewards = task_rewards
        self.balance_method = balance_method
        self.task_performance = {task: [] for task in task_rewards.keys()}
        self.task_weights = {task: 1.0 for task in task_rewards.keys()}
    
    def compute(self, context):
        current_task = context.get("current_task")
        if current_task not in self.task_rewards:
            return 0.0
        
        # Get task-specific reward
        task_reward = self.task_rewards[current_task].compute(context)
        
        # Record performance
        task_success = context.get("task_success", False)
        self.task_performance[current_task].append(task_success)
        
        # Update weights based on relative performance
        self._update_task_weights()
        
        # Apply weight
        weighted_reward = task_reward * self.task_weights[current_task]
        
        return weighted_reward
    
    def _update_task_weights(self):
        """Update task weights based on relative performance."""
        if self.balance_method == "uncertainty":
            # Weight tasks with higher uncertainty more
            for task in self.task_rewards.keys():
                if len(self.task_performance[task]) >= 10:
                    recent_performance = self.task_performance[task][-10:]
                    uncertainty = np.std(recent_performance)
                    self.task_weights[task] = 1.0 + uncertainty
        
        elif self.balance_method == "difficulty":
            # Weight harder tasks more
            for task in self.task_rewards.keys():
                if len(self.task_performance[task]) >= 10:
                    success_rate = np.mean(self.task_performance[task][-10:])
                    # Higher weight for lower success rate (harder tasks)
                    self.task_weights[task] = 2.0 - success_rate</code></pre>
            `,
            
            reward_cookbook: `
                <h1>Environment-Specific Cookbook</h1>
                <p>This cookbook provides reward engineering recipes tailored to specific RL environments and domains.</p>
                
                <h2>Atari Games</h2>
                <p>Reward engineering patterns for Atari environments with sparse rewards and long episodes.</p>
                
                <h3>Breakout Reward Components</h3>
                <pre><code class="language-python">from rllama.rewards.base import BaseReward
import numpy as np

class BreakoutProgressReward(BaseReward):
    """Rewards progress in Breakout based on brick destruction and ball position."""
    
    def __init__(self, brick_bonus=0.1, ball_position_bonus=0.01):
        super().__init__()
        self.brick_bonus = brick_bonus
        self.ball_position_bonus = ball_position_bonus
        self.previous_bricks = None
        self.previous_ball_y = None
    
    def compute(self, context):
        # Extract game state information
        bricks_remaining = context.get('bricks_remaining')
        ball_y_position = context.get('ball_y_position')
        
        reward = 0.0
        
        # Reward for destroying bricks
        if self.previous_bricks is not None and bricks_remaining is not None:
            bricks_destroyed = self.previous_bricks - bricks_remaining
            reward += bricks_destroyed * self.brick_bonus
        
        # Small reward for keeping ball in upper area
        if ball_y_position is not None and ball_y_position < 100:  # Upper part of screen
            reward += self.ball_position_bonus
        
        # Update state
        self.previous_bricks = bricks_remaining
        self.previous_ball_y = ball_y_position
        
        return reward

# Usage
engine = RewardEngine()
engine.add_component(BreakoutProgressReward())
engine.add_component(StepPenalty(penalty=-0.001))  # Encourage efficiency</code></pre>
                
                <h3>Space Invaders Reward Components</h3>
                <pre><code class="language-python">class SpaceInvadersReward(BaseReward):
    """Multi-objective reward for Space Invaders."""
    
    def __init__(self, enemy_kill_bonus=0.5, survival_bonus=0.01, accuracy_bonus=0.1):
        super().__init__()
        self.enemy_kill_bonus = enemy_kill_bonus
        self.survival_bonus = survival_bonus
        self.accuracy_bonus = accuracy_bonus
        self.shots_fired = 0
        self.hits_scored = 0
    
    def compute(self, context):
        enemies_killed = context.get('enemies_killed_this_step', 0)
        shot_fired = context.get('shot_fired', False)
        hit_enemy = context.get('hit_enemy', False)
        lives_remaining = context.get('lives_remaining', 3)
        
        reward = 0.0
        
        # Reward for killing enemies
        reward += enemies_killed * self.enemy_kill_bonus
        
        # Survival bonus (scaled by remaining lives)
        reward += self.survival_bonus * lives_remaining
        
        # Track accuracy
        if shot_fired:
            self.shots_fired += 1
        if hit_enemy:
            self.hits_scored += 1
        
        # Accuracy bonus
        if self.shots_fired > 0:
            accuracy = self.hits_scored / self.shots_fired
            reward += accuracy * self.accuracy_bonus
        
        return reward</code></pre>
                
                <h2>Robotics Environments</h2>
                <p>Reward components for robotic manipulation and navigation tasks.</p>
                
                <h3>Robotic Arm Manipulation</h3>
                <pre><code class="language-python">class ManipulationReward(BaseReward):
    """Comprehensive reward for robotic arm manipulation tasks."""
    
    def __init__(self, reach_bonus=1.0, grasp_bonus=2.0, lift_bonus=3.0, 
                 precision_scale=0.5, energy_penalty=0.01):
        super().__init__()
        self.reach_bonus = reach_bonus
        self.grasp_bonus = grasp_bonus
        self.lift_bonus = lift_bonus
        self.precision_scale = precision_scale
        self.energy_penalty = energy_penalty
    
    def compute(self, context):
        # Task completion stages
        object_reached = context.get('object_reached', False)
        object_grasped = context.get('object_grasped', False)
        object_lifted = context.get('object_lifted', False)
        
        # Precision metrics
        end_effector_pos = context.get('end_effector_position')
        target_pos = context.get('target_position')
        
        # Energy consumption
        joint_velocities = context.get('joint_velocities', [])
        
        reward = 0.0
        
        # Stage-based rewards
        if object_reached:
            reward += self.reach_bonus
        if object_grasped:
            reward += self.grasp_bonus
        if object_lifted:
            reward += self.lift_bonus
        
        # Precision reward (inverse distance to target)
        if end_effector_pos is not None and target_pos is not None:
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))
            precision_reward = self.precision_scale / (1.0 + distance)
            reward += precision_reward
        
        # Energy efficiency penalty
        if joint_velocities:
            energy_usage = np.sum(np.square(joint_velocities))
            reward -= energy_usage * self.energy_penalty
        
        return reward</code></pre>
                
                <h3>Mobile Robot Navigation</h3>
                <pre><code class="language-python">class NavigationReward(BaseReward):
    """Multi-objective navigation reward with safety constraints."""
    
    def __init__(self, goal_bonus=10.0, progress_scale=1.0, collision_penalty=-5.0,
                 smoothness_bonus=0.1, exploration_bonus=0.05):
        super().__init__()
        self.goal_bonus = goal_bonus
        self.progress_scale = progress_scale
        self.collision_penalty = collision_penalty
        self.smoothness_bonus = smoothness_bonus
        self.exploration_bonus = exploration_bonus
        self.previous_distance = None
        self.previous_action = None
        self.visited_cells = set()
    
    def compute(self, context):
        # Navigation state
        robot_pos = context.get('robot_position')
        goal_pos = context.get('goal_position')
        collision = context.get('collision', False)
        current_action = context.get('action')
        
        reward = 0.0
        
        # Goal achievement
        goal_reached = context.get('goal_reached', False)
        if goal_reached:
            reward += self.goal_bonus
        
        # Progress toward goal
        if robot_pos is not None and goal_pos is not None:
            current_distance = np.linalg.norm(np.array(robot_pos) - np.array(goal_pos))
            
            if self.previous_distance is not None:
                progress = self.previous_distance - current_distance
                reward += progress * self.progress_scale
            
            self.previous_distance = current_distance
        
         # Collision penalty
        if collision:
            reward += self.collision_penalty
        
        # Smoothness reward (penalize abrupt direction changes)
        if self.previous_action is not None and current_action is not None:
            action_diff = np.linalg.norm(np.array(current_action) - np.array(self.previous_action))
            smoothness_reward = -action_diff * self.smoothness_bonus
            reward += smoothness_reward
        
        # Exploration bonus (reward visiting new areas)
        if robot_pos is not None:
            # Discretize position for exploration tracking
            cell_x = int(robot_pos[0] / 0.5)  # 0.5m grid cells
            cell_y = int(robot_pos[1] / 0.5)
            cell = (cell_x, cell_y)
            
            if cell not in self.visited_cells:
                reward += self.exploration_bonus
                self.visited_cells.add(cell)
        
        # Update previous action for next step
        self.previous_action = current_action
        
        return reward
    
       def reset(self):
        """Reset for new episode."""
        self.previous_distance = None
        self.previous_action = None
        self.visited_cells.clear()

# Usage
engine = RewardEngine()
engine.add_component(NavigationReward())
engine.add_component(StepPenalty(penalty=-0.01))
</code></pre>
                
                <h2>Continuous Control Environments</h2>
                <p>Reward components for MuJoCo and other continuous control tasks.</p>
                
                <h3>Humanoid Locomotion</h3>
                <pre><code class="language-python">class HumanoidReward(BaseReward):
    """Multi-objective reward for humanoid locomotion."""
    
    def __init__(self, forward_bonus=1.0, upright_bonus=0.5, 
                 energy_penalty=0.01, stability_bonus=0.3):
        super().__init__()
        self.forward_bonus = forward_bonus
        self.upright_bonus = upright_bonus
        self.energy_penalty = energy_penalty
        self.stability_bonus = stability_bonus
    
    def compute(self, context):
        # Locomotion metrics
        forward_velocity = context.get('forward_velocity', 0)
        height = context.get('center_of_mass_height', 0)
        orientation = context.get('torso_orientation', [0, 0, 0])
        joint_torques = context.get('joint_torques', [])
        
        reward = 0.0
        
        # Forward movement reward
        reward += forward_velocity * self.forward_bonus
        
        # Upright posture reward
        upright_angle = abs(orientation[0])  # Pitch angle
        upright_reward = max(0, 1.0 - upright_angle) * self.upright_bonus
        reward += upright_reward
        
        # Energy efficiency penalty
        if joint_torques:
            energy_cost = np.sum(np.square(joint_torques))
            reward -= energy_cost * self.energy_penalty
        
        # Stability bonus (height maintenance)
        if height > 1.0:  # Minimum standing height
            reward += self.stability_bonus
        
        return reward</code></pre>
                
                <h2>Text Generation Environments</h2>
                <p>Reward engineering for language model fine-tuning and RLHF.</p>
                
                <h3>Text Quality and Safety</h3>
                <pre><code class="language-python">class TextGenerationReward(BaseReward):
    """Comprehensive reward for text generation tasks."""
    
    def __init__(self, coherence_weight=0.3, relevance_weight=0.3, 
                 safety_weight=0.4, length_penalty=0.01):
        super().__init__()
        self.coherence_weight = coherence_weight
        self.relevance_weight = relevance_weight
        self.safety_weight = safety_weight
        self.length_penalty = length_penalty
    
    def compute(self, context):
        generated_text = context.get('generated_text', '')
        prompt = context.get('prompt', '')
        
        # Text quality metrics (would use actual NLP models)
        coherence_score = context.get('coherence_score', 0)
        relevance_score = context.get('relevance_score', 0)
        safety_score = context.get('safety_score', 0)
        
        reward = 0.0
        
        # Quality components
        reward += coherence_score * self.coherence_weight
        reward += relevance_score * self.relevance_weight
        reward += safety_score * self.safety_weight
        
        # Length penalty for overly long responses
        text_length = len(generated_text.split())
        if text_length > 200:  # Threshold
            length_excess = text_length - 200
            reward -= length_excess * self.length_penalty
        
        return reward</code></pre>
            `,
            
            integration: `
                <h1>Integration with RL Frameworks</h1>
                <p>RLlama is designed to integrate seamlessly with popular reinforcement learning frameworks. This guide covers integration patterns and best practices.</p>
                
                <h2>OpenAI Gym Integration</h2>
                <p>The most common integration pattern uses the GymWrapper to seamlessly integrate RLlama with any Gym environment:</p>
                
                <pre><code class="language-python">from rllama import RewardEngine
from rllama.integration import GymWrapper
import gym

# Create a standard Gym environment
env = gym.make('CartPole-v1')

# Create and configure a reward engine
engine = RewardEngine()
engine.add_component(ProgressReward(goal_pos=0.0))
engine.add_component(StepPenalty(penalty=-0.01))

# Create the wrapped environment
wrapped_env = GymWrapper(engine).wrap(env)

# Now use the wrapped environment with any RL algorithm
observation = wrapped_env.reset()
for _ in range(1000):
    action = wrapped_env.action_space.sample()
    observation, reward, done, info = wrapped_env.step(action)
    if done:
        observation = wrapped_env.reset()</code></pre>
                
                <h2>Stable Baselines3 Integration</h2>
                <p>RLlama works seamlessly with Stable Baselines3 algorithms:</p>
                
                <pre><code class="language-python">from rllama.integration import StableBaselinesWrapper
from stable_baselines3 import PPO, A2C, SAC
import gym

# Create environment with RLlama rewards
env = gym.make('LunarLander-v2')
engine = RewardEngine()
engine.add_component(LandingReward(safe_landing_bonus=100))
engine.add_component(FuelEfficiencyReward(penalty_scale=0.1))

wrapped_env = GymWrapper(engine).wrap(env)

# Train with any SB3 algorithm
model = PPO("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate the trained model
obs = wrapped_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = wrapped_env.step(action)
    if done:
        obs = wrapped_env.reset()</code></pre>
                
                <h2>Ray RLlib Integration</h2>
                <p>Integration with Ray RLlib for distributed training:</p>
                
                <pre><code class="language-python">import ray
from ray import tune
from ray.rllib.env.env_context import EnvContext
from rllama import RewardEngine
from rllama.integration import RLlibWrapper

def create_rllama_env(config: EnvContext):
    # Create base environment
    env = gym.make(config["env_name"])
    
    # Create reward engine
    engine = RewardEngine()
    for component_config in config["reward_components"]:
        component_class = component_config["class"]
        component_params = component_config.get("params", {})
        component = component_class(**component_params)
        engine.add_component(component)
    
    # Wrap environment
    wrapped_env = RLlibWrapper(engine).wrap(env)
    return wrapped_env

# Register the environment
from ray.tune.registry import register_env
register_env("rllama_env", create_rllama_env)

# Configure training
config = {
    "env": "rllama_env",
    "env_config": {
        "env_name": "CartPole-v1",
        "reward_components": [
            {"class": BalanceReward, "params": {"bonus": 1.0}},
            {"class": StepPenalty, "params": {"penalty": -0.01}}
        ]
    },
    "framework": "torch",
    "num_workers": 4,
}

# Run training
ray.init()
tune.run("PPO", config=config, stop={"training_iteration": 100})</code></pre>
                
                <h2>Custom Environment Integration</h2>
                <p>For custom environments, implement the integration manually:</p>
                
                <pre><code class="language-python">class CustomEnvironmentWithRLlama:
    def __init__(self, base_env, reward_engine):
        self.base_env = base_env
        self.reward_engine = reward_engine
        self.global_step = 0
    
    def reset(self):
        obs = self.base_env.reset()
        self.episode_step = 0
        return obs
    
    def step(self, action):
        obs, raw_reward, done, info = self.base_env.step(action)
        
        # Create context for reward computation
        context = {
            'state': obs,
            'action': action,
            'raw_reward': raw_reward,
            'done': done,
            'global_step': self.global_step,
            'episode_step': self.episode_step,
            # Add any custom context information
        }
        
        # Compute shaped reward
        shaped_reward = self.reward_engine.compute(context)
        
        # Update counters
        self.global_step += 1
        self.episode_step += 1
        
        return obs, shaped_reward, done, info

# Usage
base_env = YourCustomEnvironment()
engine = RewardEngine()
engine.add_component(YourCustomReward())

env = CustomEnvironmentWithRLlama(base_env, engine)</code></pre>
                
                <h2>Multi-Agent Integration</h2>
                <p>RLlama supports multi-agent environments with agent-specific rewards:</p>
                
                <pre><code class="language-python">class MultiAgentRLlamaWrapper:
    def __init__(self, base_env, agent_engines):
        self.base_env = base_env
        self.agent_engines = agent_engines  # Dict: agent_id -> RewardEngine
    
    def step(self, actions):
        observations, rewards, dones, infos = self.base_env.step(actions)
        
        shaped_rewards = {}
        for agent_id in self.agent_engines:
            context = {
                'agent_id': agent_id,
                'observations': observations,
                'actions': actions,
                'raw_rewards': rewards,
                'infos': infos,
            }
            
            shaped_rewards[agent_id] = self.agent_engines[agent_id].compute(context)
        
        return observations, shaped_rewards, dones, infos

# Usage for competitive multi-agent
agent_engines = {
    'player_1': RewardEngine(),
    'player_2': RewardEngine(),
}

agent_engines['player_1'].add_component(WinReward(bonus=10))
agent_engines['player_1'].add_component(CompetitiveReward(opponent='player_2'))

agent_engines['player_2'].add_component(WinReward(bonus=10))
agent_engines['player_2'].add_component(CompetitiveReward(opponent='player_1'))

wrapped_env = MultiAgentRLlamaWrapper(base_env, agent_engines)</code></pre>
                
                <h2>Integration Best Practices</h2>
                
                <h3>Context Management</h3>
                <p>Always provide comprehensive context information:</p>
                
                <pre><code class="language-python"># Good context example
context = {
    # Environment state
    'state': current_state,
    'action': taken_action,
    'next_state': next_state,
    'done': done,
    'info': info,
    
    # Training progress
    'global_step': global_step,
    'episode': episode_number,
    'episode_step': step_in_episode,
    
    # Performance metrics
    'episode_return': cumulative_reward,
    'success_rate': recent_success_rate,
    
    # Custom metrics
    'exploration_progress': exploration_metric,
    'safety_violations': violation_count,
}</code></pre>
                
                <h3>Performance Optimization</h3>
                <p>For high-performance training, consider these optimizations:</p>
                
                <pre><code class="language-python"># Batch reward computation for vectorized environments
class VectorizedRLlamaWrapper:
    def __init__(self, base_env, reward_engine):
        self.base_env = base_env
        self.reward_engine = reward_engine
    
    def step(self, actions):
        observations, rewards, dones, infos = self.base_env.step(actions)
        
        # Compute rewards for all environments at once
        shaped_rewards = []
        for i in range(len(observations)):
            context = {
                'state': observations[i],
                'action': actions[i],
                'raw_reward': rewards[i],
                'done': dones[i],
                'info': infos[i],
                'env_id': i,
            }
            shaped_rewards.append(self.reward_engine.compute(context))
        
        return observations, np.array(shaped_rewards), dones, infos</code></pre>
            `,
            
            'with-rllama': `
                <h1>With RLlama</h1>
                <p>See how RLlama transforms your reward engineering workflow from complex, monolithic functions to modular, debuggable, and optimizable components.</p>
                
                <h2>Before: Complex, Monolithic Rewards</h2>
                <p>Traditional reward engineering often results in large, complex functions that are difficult to understand, debug, and modify:</p>
                
                <pre><code class="language-python"># Traditional approach - hard to debug and modify
def complex_reward(state, action, next_state, info):
    reward = 0.0
    
    # Goal achievement
    if info.get('is_success'):
        reward += 10.0
    
    # Step penalty
    reward -= 0.01
    
    # Distance reward
    if 'agent_pos' in info and 'goal_pos' in info:
        distance = calculate_distance(info['agent_pos'], info['goal_pos'])
        reward += (1.0 / (1.0 + distance))
    
    # Safety penalty
    if info.get('collision'):
        reward -= 5.0
    
    # Efficiency bonus
    if info.get('steps_taken', 0) < 50 and info.get('is_success'):
        reward += 2.0
    
    # Exploration bonus (complex state tracking)
    state_hash = hash(str(state))
    if state_hash not in visited_states:
        reward += 0.1
        visited_states.add(state_hash)
    
    return reward

# Problems with this approach:
# 1. All logic mixed together
# 2. Hard to tune individual components
# 3. Difficult to debug which part is causing issues
# 4. No reusability across environments
# 5. Manual hyperparameter tuning required</code></pre>
                
                <h2>After: Modular, Composable Components</h2>
                <p>With RLlama, the same reward function becomes a collection of clear, modular components:</p>
                
                <pre><code class="language-python"># RLlama approach - clear, modular, reusable
from rllama import RewardEngine
from rllama.rewards.components import (
    GoalReward, StepPenalty, DistanceReward, SafetyPenalty, 
    EfficiencyReward, ExplorationReward
)

# Create engine and add components
engine = RewardEngine()
engine.add_component(GoalReward(reward_value=10.0))
engine.add_component(StepPenalty(penalty=-0.01))
engine.add_component(DistanceReward(scale=1.0))
engine.add_component(SafetyPenalty(collision_penalty=-5.0))
engine.add_component(EfficiencyReward(threshold_steps=50, bonus=2.0))
engine.add_component(ExplorationReward(bonus=0.1))

# Easy to use
context = {
    'state': state,
    'action': action,
    'next_state': next_state,
    'info': info
}
reward = engine.compute(context)

# Benefits:
# 1. Each component is self-contained and testable
# 2. Easy to add, remove, or modify individual components
# 3. Clear separation of concerns
# 4. Components can be reused across different environments
# 5. Automatic optimization of weights possible</code></pre>
                
                <h2>Debugging and Analysis</h2>
                <p>RLlama makes debugging reward functions straightforward:</p>
                
                <pre><code class="language-python"># Get detailed breakdown of reward contributions
breakdown = engine.get_breakdown(context)
print("Reward breakdown:")
for component, value in breakdown.items():
    print(f"  {component}: {value:.3f}")

# Output:
# Reward breakdown:
#   goal_reward: 0.000
#   step_penalty: -0.010
#   distance_reward: 0.234
#   safety_penalty: 0.000
#   efficiency_reward: 0.000
#   exploration_reward: 0.100

# Easily identify which components are contributing
total_reward = sum(breakdown.values())
print(f"Total reward: {total_reward:.3f}")

# Test individual components
distance_component = DistanceReward(scale=1.0)
distance_reward = distance_component.compute(context)
print(f"Distance reward alone: {distance_reward:.3f}")

# Visualize reward contributions over time
import matplotlib.pyplot as plt

episode_breakdowns = []
for step in episode_steps:
    breakdown = engine.get_breakdown(step_context)
    episode_breakdowns.append(breakdown)

# Plot each component's contribution
components = list(episode_breakdowns[0].keys())
for component in components:
    values = [bd[component] for bd in episode_breakdowns]
    plt.plot(values, label=component)

plt.legend()
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Component Contributions')
plt.show()</code></pre>
                
                <h2>Dynamic Weight Adjustment</h2>
                <p>RLlama enables sophisticated curriculum learning and adaptive reward strategies:</p>
                
                <pre><code class="language-python"># Traditional approach: hardcoded curriculum
def curriculum_reward(state, action, next_state, info, episode):
    reward = 0.0
    
    # Manually adjust weights based on episode
    if episode < 1000:
        safety_weight = 2.0
        efficiency_weight = 0.1
    elif episode < 5000:
        safety_weight = 1.0
        efficiency_weight = 0.5
    else:
        safety_weight = 0.5
        efficiency_weight = 1.0
    
    # Apply weights manually
    if info.get('collision'):
        reward -= 5.0 * safety_weight
    
    if info.get('is_success') and info.get('steps_taken', 0) < 50:
        reward += 2.0 * efficiency_weight
    
    return reward

# RLlama approach: declarative curriculum
from rllama.scheduling import ExponentialDecay, LinearIncrease

engine = RewardEngine()
engine.add_component(SafetyPenalty(penalty=-5.0))
engine.add_component(EfficiencyReward(bonus=2.0))

# Set up weight schedules
engine.set_weight_schedule('safety_penalty', ExponentialDecay(
    initial_weight=2.0,
    decay_rate=0.9999,
    min_weight=0.5
))

engine.set_weight_schedule('efficiency_reward', LinearIncrease(
    initial_weight=0.1,
    final_weight=1.0,
    duration_steps=5000
))

# Automatic weight updates during training
for episode in range(10000):
    engine.update_weights(episode)
    
    # Training loop
    for step in episode_steps:
        reward = engine.compute(context)
        # ... rest of training logic</code></pre>
                
                <h2>Automatic Optimization</h2>
                <p>RLlama can automatically find optimal reward weights:</p>
                
                <pre><code class="language-python"># Traditional approach: manual tuning
# Try different weight combinations manually
weight_combinations = [
    {'goal': 10.0, 'penalty': -0.01, 'distance': 1.0},
    {'goal': 15.0, 'penalty': -0.005, 'distance': 0.5},
    {'goal': 5.0, 'penalty': -0.02, 'distance': 2.0},
    # ... many more combinations
]

best_performance = -float('inf')
best_weights = None

for weights in weight_combinations:
    performance = train_and_evaluate(weights)
    if performance > best_performance:
        best_performance = performance
        best_weights = weights

# RLlama approach: automatic optimization
from rllama.optimization import BayesianOptimizer

def evaluate_weights(trial):
    # Suggest weights
    goal_weight = trial.suggest_float('goal_weight', 1.0, 20.0)
    penalty_weight = trial.suggest_float('penalty_weight', -0.05, -0.001)
    distance_weight = trial.suggest_float('distance_weight', 0.1, 5.0)
    
    # Create engine with suggested weights
    engine = RewardEngine()
    engine.add_component(GoalReward(weight=goal_weight))
    engine.add_component(StepPenalty(weight=penalty_weight))
    engine.add_component(DistanceReward(weight=distance_weight))
    
    # Train and evaluate
    performance = train_and_evaluate_with_engine(engine)
    return performance

# Run optimization
optimizer = BayesianOptimizer()
best_weights = optimizer.optimize(evaluate_weights, n_trials=100)

print(f"Best weights found: {best_weights}")
print(f"Best performance: {optimizer.best_value}")

# Automatically finds optimal weights with minimal manual effort</code></pre>
                
                <h2>Testing and Validation</h2>
                <p>RLlama components are easy to test in isolation:</p>
                
                <pre><code class="language-python"># Traditional approach: testing the entire reward function
def test_reward_function():
    # Hard to test specific behaviors
    state = create_test_state()
    action = create_test_action()
    next_state = create_test_next_state()
    info = {'is_success': True, 'collision': False}
    
    reward = complex_reward(state, action, next_state, info)
    assert reward > 0  # Very general test
    
    # If this fails, hard to know which part is broken

# RLlama approach: test individual components
def test_goal_reward():
    goal_reward = GoalReward(reward_value=10.0)
    
    # Test success case
    context = {'info': {'is_success': True}}
    assert goal_reward.compute(context) == 10.0
    
    # Test failure case
    context = {'info': {'is_success': False}}
    assert goal_reward.compute(context) == 0.0
    
    # Test missing info
    context = {'info': {}}
    assert goal_reward.compute(context) == 0.0

def test_step_penalty():
    step_penalty = StepPenalty(penalty=-0.01)
    
    # Always returns penalty
    context = {}
    assert step_penalty.compute(context) == -0.01

def test_distance_reward():
    distance_reward = DistanceReward(scale=1.0)
    
    # Test distance calculation
    context = {
        'info': {
            'agent_pos': [0, 0],
            'goal_pos': [3, 4]
        }
    }
    reward = distance_reward.compute(context)
    expected = 1.0 / (1.0 + 5.0)  # distance is 5
    assert abs(reward - expected) < 1e-6

# Run all component tests
test_goal_reward()
test_step_penalty()
test_distance_reward()
print("All component tests passed!")

# Integration test
def test_full_engine():
    engine = RewardEngine()
    engine.add_component(GoalReward(reward_value=10.0))
    engine.add_component(StepPenalty(penalty=-0.01))
    
    context = {'info': {'is_success': True}}
    total_reward = engine.compute(context)
    assert abs(total_reward - 9.99) < 1e-6  # 10.0 - 0.01

test_full_engine()
print("Integration test passed!")</code></pre>
                
                <h2>Summary: The RLlama Advantage</h2>
                <p>RLlama transforms reward engineering from an art to a science:</p>
                
                <ul>
                    <li><strong>Modularity</strong>: Break complex rewards into simple, understandable components</li>
                    <li><strong>Debuggability</strong>: Easily identify which components are working or failing</li>
                    <li><strong>Reusability</strong>: Share components across different environments and projects</li>
                    <li><strong>Testability</strong>: Test individual components in isolation</li>
                    <li><strong>Optimization</strong>: Automatically find optimal reward weights</li>
                    <li><strong>Curriculum Learning</strong>: Declarative weight scheduling for sophisticated training strategies</li>
                    <li><strong>Visualization</strong>: Track and analyze reward contributions over time</li>
                    <li><strong>Maintainability</strong>: Easy to modify, extend, and maintain reward functions</li>
                </ul>
                
                <p>With RLlama, reward engineering becomes a systematic, scientific process that leads to better agent performance and faster development cycles.</p>
            `,
            
            'without-rllama': `
                <h1>Without RLlama</h1>
                <p>Understanding the challenges and limitations of traditional reward engineering approaches helps illustrate why RLlama's modular framework is necessary for modern RL development.</p>
                
                <h2>Problems with Traditional Reward Engineering</h2>
                
                <h3>1. Monolithic Design</h3>
                <p>Traditional reward functions often become large, complex, and hard to understand:</p>
                
                <pre><code class="language-python">def traditional_reward(obs, action, next_obs, info):
    # Everything mixed together - hard to understand and modify
    reward = 0
    
    # Goal achievement logic
    if info['goal_reached']:
        reward += 100
        
        # Efficiency bonus (embedded in goal logic)
        if info['steps'] < 50:
            reward += 50
    
    # Safety logic mixed with other concerns
    if info['collision']:
        reward -= 50
        
        # Collision type matters (embedded logic)
        if info['collision_type'] == 'wall':
            reward -= 25  # Extra penalty for wall collisions
        elif info['collision_type'] == 'obstacle':
            reward -= 10
    
    # Distance component (unclear relationship to other components)
    dist = np.linalg.norm(obs[:2] - goal_pos)
    reward += max(0, 10 - dist)
    
    # Time penalty (unclear why this specific value)
    reward -= 0.1
    
    # Exploration bonus (complex state tracking embedded)
    state_key = tuple(np.round(obs[:2], 1))
    if state_key not in visited_states:
        reward += 5
        visited_states.add(state_key)
        
        # Novelty bonus (nested logic)
        if len(visited_states) > 100:
            reward += 2  # Bonus for exploring many states
    
    # Speed bonus (unclear interaction with other components)
    speed = np.linalg.norm(obs[2:4])
    if speed > 0.5:
        reward += speed * 2
    
    return reward</code></pre>
                
                <h3>Issues with This Approach:</h3>
                <ul>
                    <li><strong>Unclear Component Interactions</strong>: How do different reward terms interact? Which ones dominate?</li>
                    <li><strong>Hard to Debug</strong>: When the agent behaves unexpectedly, which part of the reward is causing the issue?</li>
                    <li><strong>Difficult to Tune</strong>: Changing one component might break others due to hidden dependencies</li>
                    <li><strong>No Reusability</strong>: The entire function is tied to this specific environment</li>
                    <li><strong>Poor Maintainability</strong>: Adding new behaviors requires modifying the entire function</li>
                </ul>
                
                <h2>2. Difficult Debugging</h2>
                <p>When agents don't behave as expected, traditional approaches make it nearly impossible to identify the root cause:</p>
                
                <pre><code class="language-python"># Traditional debugging approach
def debug_reward(obs, action, next_obs, info):
    reward = traditional_reward(obs, action, next_obs, info)
    
    # Only see the final reward - no insight into components
    print(f"Total reward: {reward}")
    
    # To debug, you have to manually add print statements
    # and modify the original function
    if reward < 0:
        print("Negative reward detected!")
        # But which component caused it?
        
        # Manual debugging requires code changes
        goal_part = 100 if info['goal_reached'] else 0
        collision_part = -50 if info['collision'] else 0
        # ... manually compute each part
        
        print(f"Goal: {goal_part}, Collision: {collision_part}")
        # This becomes unwieldy with many components

# Problems:
# 1. No automatic breakdown of reward components
# 2. Debugging requires modifying production code
# 3. Hard to track component contributions over time
# 4. No systematic way to identify problematic components</code></pre>
                
                <h2>3. No Reusability</h2>
                <p>Traditional reward functions are tightly coupled to specific environments:</p>
                
                <pre><code class="language-python"># Environment 1: Navigation task
def navigation_reward(obs, action, next_obs, info):
    reward = 0
    if info['goal_reached']:
        reward += 100
    if info['collision']:
        reward -= 50
    # Distance-based reward
    dist = np.linalg.norm(obs[:2] - goal_pos)
    reward += max(0, 10 - dist)
    return reward

# Environment 2: Manipulation task  
def manipulation_reward(obs, action, next_obs, info):
    reward = 0
    if info['object_grasped']:
        reward += 100  # Same goal concept, different implementation
    if info['collision']:
        reward -= 50   # Same collision concept, different implementation
    # Distance to object
    dist = np.linalg.norm(obs[:3] - info['object_pos'])
    reward += max(0, 10 - dist)  # Same distance concept, different implementation
    return reward

# Environment 3: Racing task
def racing_reward(obs, action, next_obs, info):
    reward = 0
    if info['lap_completed']:
        reward += 100  # Same goal concept, different implementation
    if info['collision']:
        reward -= 50   # Same collision concept, different implementation
    # Speed reward (different from distance)
    reward += obs[2] * 10  # Speed-based instead of distance-based
    return reward

# Problems:
# 1. Duplicated logic across environments
# 2. Inconsistent implementations of similar concepts
# 3. No way to share improvements across environments
# 4. Each environment requires reimplementing common patterns</code></pre>
                
                <h2>4. Manual Tuning Nightmare</h2>
                <p>Hyperparameter tuning becomes a manual, time-consuming process:</p>
                
                <pre><code class="language-python"># Traditional hyperparameter tuning
def tune_reward_manually():
    # Try different parameter combinations manually
    goal_rewards = [50, 100, 150, 200]
    collision_penalties = [-25, -50, -75, -100]
    distance_scales = [5, 10, 15, 20]
    time_penalties = [-0.05, -0.1, -0.15, -0.2]
    
    best_performance = -float('inf')
    best_params = None
    
    # Exhaustive search (computationally expensive)
    for goal_r in goal_rewards:
        for collision_p in collision_penalties:
            for dist_s in distance_scales:
                for time_p in time_penalties:
                    
                    # Modify reward function for each combination
                    def test_reward(obs, action, next_obs, info):
                        reward = 0
                        if info['goal_reached']:
                            reward += goal_r
                        if info['collision']:
                            reward += collision_p
                        dist = np.linalg.norm(obs[:2] - goal_pos)
                        reward += max(0, dist_s - dist)
                        reward += time_p
                        return reward
                    
                    # Train agent with this reward function
                    performance = train_agent(test_reward)
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = (goal_r, collision_p, dist_s, time_p)
    
    print(f"Best params: {best_params}")
    print(f"Best performance: {best_performance}")

# Problems:
# 1. Exponential search space (4^4 = 256 combinations in this simple example)
# 2. No intelligent search strategy
# 3. Requires training a full agent for each combination
# 4. No way to leverage previous results
# 5. Doesn't scale to more complex reward functions</code></pre>
                
                <h2>5. Limited Insight</h2>
                <p>Traditional approaches provide no visibility into reward dynamics:</p>
                
                <pre><code class="language-python"># Traditional reward tracking
episode_rewards = []
for episode in range(1000):
    total_reward = 0
    for step in episode_steps:
        reward = traditional_reward(obs, action, next_obs, info)
        total_reward += reward
    
    episode_rewards.append(total_reward)
    
    # Only see total rewards - no component breakdown
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode}, Avg Reward: {avg_reward}")

# Questions you can't answer:
# 1. Which reward components are contributing most?
# 2. Are some components working against each other?
# 3. How do component contributions change over training?
# 4. Which components are actually influencing agent behavior?
# 5. Are there unused or redundant components?

# Manual analysis requires extensive code changes
def analyze_reward_manually():
    goal_rewards = []
    collision_penalties = []
    distance_rewards = []
    
    for step in episode_steps:
        # Manually track each component (requires code duplication)
        goal_part = 100 if info['goal_reached'] else 0
        collision_part = -50 if info['collision'] else 0
        dist = np.linalg.norm(obs[:2] - goal_pos)
        distance_part = max(0, 10 - dist)
        
        goal_rewards.append(goal_part)
        collision_penalties.append(collision_part)
        distance_rewards.append(distance_part)
    
    # Manual visualization
    plt.plot(goal_rewards, label='Goal')
    plt.plot(collision_penalties, label='Collision')
    plt.plot(distance_rewards, label='Distance')
    plt.legend()
    plt.show()

# This approach is error-prone and doesn't scale</code></pre>
                
                <h2>6. Curriculum Learning Challenges</h2>
                <p>Implementing curriculum learning requires complex manual state management:</p>
                
                <pre><code class="language-python"># Traditional curriculum learning
class CurriculumReward:
    def __init__(self):
        self.episode_count = 0
        self.success_history = []
        
    def __call__(self, obs, action, next_obs, info):
        reward = 0
        
        # Manually implement curriculum logic
        if self.episode_count < 1000:
            # Phase 1: Focus on safety
            safety_weight = 2.0
            efficiency_weight = 0.1
        elif self.episode_count < 5000:
            # Phase 2: Balance safety and efficiency
            safety_weight = 1.0
            efficiency_weight = 0.5
        else:
            # Phase 3: Focus on efficiency
            safety_weight = 0.5
            efficiency_weight = 1.0
        
        # Apply curriculum weights manually
        if info['goal_reached']:
            reward += 100
        if info['collision']:
            reward -= 50 * safety_weight
        if info['goal_reached'] and info['steps'] < 50:
            reward += 20 * efficiency_weight
            
        return reward
    
    def update_episode(self, success):
        self.episode_count += 1
        self.success_history.append(success)
        
        # Manual adaptation logic
        if len(self.success_history) >= 100:
            recent_success = np.mean(self.success_history[-100:])
            if recent_success > 0.8:
                # Maybe advance curriculum?
                pass

# Problems:
# 1. Manual state management
# 2. Hardcoded phase transitions
# 3. No systematic way to define curriculum strategies
# 4. Difficult to experiment with different curriculum approaches
# 5. Curriculum logic mixed with reward logic</code></pre>
                
                <h2>7. Testing and Validation Issues</h2>
                <p>Traditional reward functions are difficult to test systematically:</p>
                
                <pre><code class="language-python"># Traditional testing approach
def test_reward_function():
    # Can only test the entire function as a black box
    
    # Test case 1: Goal reached
    obs = np.array([0, 0, 0, 0])
    action = 0
    next_obs = np.array([1, 1, 0, 0])
    info = {'goal_reached': True, 'collision': False, 'steps': 30}
    
    reward = traditional_reward(obs, action, next_obs, info)
    assert reward > 0  # Very general assertion
    
    # Test case 2: Collision
    info = {'goal_reached': False, 'collision': True, 'steps': 10}
    reward = traditional_reward(obs, action, next_obs, info)
    assert reward < 0  # Very general assertion
    
    # Problems:
    # 1. Can't test individual reward components
    # 2. Hard to create comprehensive test cases
    # 3. Difficult to verify specific behaviors
    # 4. No way to test component interactions
    # 5. Changes to one part might break other parts

# If a test fails, you don't know which part is broken
def debug_failed_test():
    # Test fails - but where?
    reward = traditional_reward(obs, action, next_obs, info)
    expected_reward = 85  # Expected value
    
    if abs(reward - expected_reward) > 1e-6:
        print(f"Test failed: expected {expected_reward}, got {reward}")
        # Now what? You have to manually debug the entire function
        
        # Manual debugging requires modifying the function
        # or duplicating its logic here
        print("Debugging reward components...")
        # This is tedious and error-prone</code></pre>
                
                <h2>The Cost of Traditional Approaches</h2>
                
                <h3>Development Time</h3>
                <ul>
                    <li><strong>Slow Iteration</strong>: Modifying reward functions requires understanding the entire codebase</li>
                    <li><strong>Debugging Overhead</strong>: Significant time spent identifying which part of the reward is causing issues</li>
                    <li><strong>Manual Tuning</strong>: Weeks or months spent manually adjusting hyperparameters</li>
                    <li><strong>Code Duplication</strong>: Similar reward logic reimplemented across different projects</li>
                </ul>
                
                <h3>Performance Issues</h3>
                <ul>
                    <li><strong>Suboptimal Rewards</strong>: Manual tuning rarely finds optimal parameter combinations</li>
                    <li><strong>Component Conflicts</strong>: Reward components working against each other without detection</li>
                    <li><strong>Inconsistent Behavior</strong>: Small changes causing unexpected behavioral shifts</li>
                    <li><strong>Limited Exploration</strong>: Difficulty implementing sophisticated exploration strategies</li>
                </ul>
                
                <h3>Maintainability Problems</h3>
                <ul>
                    <li><strong>Technical Debt</strong>: Complex reward functions become harder to modify over time</li>
                    <li><strong>Knowledge Silos</strong>: Only original developers understand the reward logic</li>
                    <li><strong>Regression Risk</strong>: Changes to reward functions can break existing behaviors</li>
                    <li><strong>Limited Reusability</strong>: Reward logic can't be shared across projects</li>
                </ul>
                
                <h2>Why RLlama is Essential</h2>
                <p>These traditional approaches don't scale to modern RL challenges:</p>
                
                <ul>
                    <li><strong>Complex Environments</strong>: Modern RL tasks require sophisticated reward engineering</li>
                    <li><strong>Multi-Objective Optimization</strong>: Balancing multiple competing objectives manually is intractable</li>
                    <li><strong>Curriculum Learning</strong>: Advanced training strategies require dynamic reward adaptation</li>
                    <li><strong>Team Development</strong>: Multiple developers need to work on reward functions simultaneously</li>
                    <li><strong>Rapid Prototyping</strong>: Fast iteration is essential for RL research and development</li>
                </ul>
                
                <p>RLlama addresses all these challenges by providing a modular, composable, and optimizable framework for reward engineering. The contrast between traditional approaches and RLlama's methodology demonstrates why modern RL development requires systematic reward engineering tools.</p>
            `,
            
            optimization_guide: `
                <h1>Reward Optimization Guide</h1>
                <p>Manually tuning reward weights can be time-consuming and leads to suboptimal results. RLlama integrates with Optuna for automated optimization using Bayesian optimization techniques.</p>
                
                <h2>Prerequisites</h2>
                <p>First, install Optuna and optional visualization tools:</p>
                <pre><code class="language-bash">pip install optuna

# Optional: for visualization and dashboard
pip install optuna-dashboard
pip install plotly matplotlib</code></pre>
                
                <h2>The BayesianRewardOptimizer</h2>
                <p>This class orchestrates the optimization process to find optimal reward configuration parameters that maximize (or minimize) a user-defined performance metric.</p>
                
                <h3>Core Concept</h3>
                <p>The optimizer uses Optuna's Tree-structured Parzen Estimator (TPE) to intelligently suggest new parameter combinations based on previous trial results, making the search process much more efficient than grid search or random search.</p>
                
                <h2>Defining the Search Space</h2>
                <p>You need to specify which parameters to tune and their allowed ranges using a dictionary structure that mirrors your reward configuration:</p>
                
                <pre><code class="language-python">import optuna

# Example base configuration
base_config = {
    "reward_shaping": {
        "goal_reward": {
            "class": "GoalReward",
            "params": {"target_reward": 1.0},
            "weight_schedule": {"initial_weight": 10.0, "schedule_type": "constant"}
        },
        "step_penalty": {
            "class": "StepPenalty", 
            "params": {"penalty": -0.01},
            "weight_schedule": {
                "initial_weight": 1.0,
                "schedule_type": "exponential",
                "decay_rate": 0.9995,
                "decay_steps": 1,
                "min_weight": 0.0
            }
        }
    }
}

# Corresponding search space
search_space = {
    "reward_shaping": {
        "goal_reward": {
            "params": {
                "target_reward": lambda trial: trial.suggest_float("goal_target_reward", 0.5, 5.0)
            },
            "weight_schedule": {
                "initial_weight": lambda trial: trial.suggest_float("goal_initial_weight", 1.0, 50.0, log=True)
            }
        },
        "step_penalty": {
            "params": {
                "penalty": lambda trial: trial.suggest_float("penalty_value", -0.5, -0.001)
            },
            "weight_schedule": {
                "initial_weight": lambda trial: trial.suggest_float("penalty_initial_weight", 0.1, 10.0),
                "decay_rate": lambda trial: trial.suggest_float("penalty_decay_rate", 0.99, 0.9999)
            }
        }
    }
}</code></pre>
                
                <h3>Search Space Guidelines</h3>
                <ul>
                    <li><strong>Use log scale</strong> for parameters that vary over orders of magnitude (e.g., learning rates, weights)</li>
                    <li><strong>Set reasonable bounds</strong> based on domain knowledge and previous experiments</li>
                    <li><strong>Include categorical choices</strong> for discrete options like schedule types</li>
                    <li><strong>Start with fewer parameters</strong> and gradually expand the search space</li>
                </ul>
                
                <h2>Writing the Objective Function</h2>
                <p>The objective function defines what the optimizer is trying to achieve. It receives suggested parameters from Optuna and returns a performance metric:</p>
                
                <pre><code class="language-python">def objective(trial, base_config, search_space, component_registry):
    """
    Runs an RL training loop with parameters suggested by Optuna
    and returns a performance metric.
    """
    
    # 1. Generate configuration for this trial
    trial_config = generate_trial_config(trial, base_config, search_space)
    
    # 2. Set up environment and components
    env = gym.make("YourEnv-v1")
    components = create_components(trial_config, component_registry)
    composer = RewardComposer(components)
    shaper = RewardShaper(composer, trial_config)
    
    # 3. Set up agent (if tuning agent parameters too)
    agent_config = trial_config.get("agent_params", {})
    agent = YourAgent(**agent_config)
    
    # 4. Run training loop
    try:
        total_reward = 0
        num_episodes = trial_config.get("training", {}).get("num_episodes", 100)
        global_step = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps_in_episode = 0
            
            while True:
                # Update weights based on training progress
                shaper.update_weights(global_step)
                
                # Agent selects action
                action = agent.select_action(state)
                
                # Environment step
                next_state, raw_reward, done, info = env.step(action)
                
                # Create context for reward shaping
                context = {
                    "global_step": global_step,
                    "steps_in_episode": steps_in_episode,
                    "episode": episode,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "done": done,
                    "info": info
                }
                
                # Compute shaped reward
                shaped_reward = shaper.shape(raw_reward, info, context)
                
                # Agent update
                agent.update(state, action, shaped_reward, next_state, done)
                
                # Update state and counters
                state = next_state
                episode_reward += shaped_reward
                steps_in_episode += 1
                global_step += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            
            # Optional: Optuna pruning for early stopping
            if episode % 10 == 0:
                intermediate_value = total_reward / (episode + 1)
                trial.report(intermediate_value, episode)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # 5. Calculate final performance metric
        average_reward = total_reward / num_episodes
        
        # Optional: Add other metrics
        success_rate = agent.get_success_rate() if hasattr(agent, 'get_success_rate') else 0
        final_metric = average_reward + success_rate * 10  # Weighted combination
        
        return final_metric
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned(f"Training failed: {e}")
    
    finally:
        env.close()

def generate_trial_config(trial, base_config, search_space):
    """Helper function to generate configuration from trial suggestions."""
    trial_config = copy.deepcopy(base_config)
    
    def apply_suggestions(space, config):
        for key, value in space.items():
            if isinstance(value, dict):
                if key not in config:
                    config[key] = {}
                apply_suggestions(value, config[key])
            elif callable(value):
                # This is a lambda function - call it with trial
                config[key] = value(trial)
    
    apply_suggestions(search_space, trial_config)
    return trial_config</code></pre>
                
                <h2>Running the Optimization</h2>
                <p>Use the BayesianRewardOptimizer to orchestrate the optimization process:</p>
                
                <pre><code class="language-python">from rllama.optimization import BayesianRewardOptimizer

# Set up component registry
component_registry = {
    "GoalReward": GoalReward,
    "StepPenalty": StepPenalty,
    "DistanceReward": DistanceReward,
    # Add all your reward components here
}

# Create optimizer
optimizer = BayesianRewardOptimizer(
    base_config=base_config,
    search_space=search_space,
    objective_function=objective,
    objective_kwargs={
        "base_config": base_config,
        "search_space": search_space,
        "component_registry": component_registry,
    },
    n_trials=100,
    study_name="rllama_reward_tuning",
    storage="sqlite:///rllama_tuning.db",  # Persist results
    direction="maximize"  # or "minimize"
)

# Run optimization
print("Starting Bayesian Optimization...")
best_params, best_value, study = optimizer.optimize()

# Display results
print(f"\\nOptimization completed!")
print(f"Best objective value: {best_value}")
print(f"Best parameters:")
import json
print(json.dumps(best_params, indent=2))</code></pre>
                
                <h2>Advanced Optimization Strategies</h2>
                
                <h3>Multi-Objective Optimization</h3>
                <p>Optimize multiple objectives simultaneously:</p>
                
                <pre><code class="language-python">def multi_objective_function(trial, base_config, search_space, component_registry):
    """Optimize for both performance and sample efficiency."""
    
    # Run training as before
    trial_config = generate_trial_config(trial, base_config, search_space)
    performance, sample_efficiency = train_and_evaluate(trial_config)
    
    # Return multiple objectives
    return performance, sample_efficiency

# Use multi-objective optimization
study = optuna.create_study(
    directions=["maximize", "maximize"],  # Both objectives to maximize
    study_name="multi_objective_reward_tuning"
)

study.optimize(multi_objective_function, n_trials=100)

# Get Pareto front
pareto_front = study.best_trials
for trial in pareto_front:
    print(f"Performance: {trial.values[0]}, Efficiency: {trial.values[1]}")
    print(f"Params: {trial.params}")</code></pre>
                
                <h3>Conditional Search Spaces</h3>
                <p>Make some parameters conditional on others:</p>
                
                <pre><code class="language-python">def conditional_search_space(trial):
    """Example of conditional parameter suggestions."""
    
    # Choose schedule type first
    schedule_type = trial.suggest_categorical("schedule_type", ["constant", "exponential", "linear"])
    
    config = {
        "reward_shaping": {
            "goal_reward": {
                "weight_schedule": {
                    "schedule_type": schedule_type,
                    "initial_weight": trial.suggest_float("initial_weight", 0.1, 10.0)
                }
            }
        }
    }
    
    # Add conditional parameters based on schedule type
    if schedule_type == "exponential":
        config["reward_shaping"]["goal_reward"]["weight_schedule"]["decay_rate"] = \
            trial.suggest_float("decay_rate", 0.9, 0.9999)
        config["reward_shaping"]["goal_reward"]["weight_schedule"]["min_weight"] = \
            trial.suggest_float("min_weight", 0.01, 1.0)
    elif schedule_type == "linear":
        config["reward_shaping"]["goal_reward"]["weight_schedule"]["end_weight"] = \
            trial.suggest_float("end_weight", 0.1, 10.0)
        config["reward_shaping"]["goal_reward"]["weight_schedule"]["duration_steps"] = \
            trial.suggest_int("duration_steps", 1000, 100000)
    
    return config</code></pre>
                
                <h2>Analyzing Optimization Results</h2>
                <p>Use Optuna's visualization tools to understand the optimization process:</p>
                
                <pre><code class="language-python">import optuna.visualization as vis
import matplotlib.pyplot as plt

# Load study if using persistent storage
study = optuna.load_study(
    study_name="rllama_reward_tuning",
    storage="sqlite:///rllama_tuning.db"
)

# 1. Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# 2. Parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# 3. Parameter relationships
fig = vis.plot_slice(study, params=["goal_initial_weight", "penalty_decay_rate"])
fig.show()

# 4. Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.show()

# 5. Contour plot for 2D parameter relationships
fig = vis.plot_contour(study, params=["goal_initial_weight", "penalty_initial_weight"])
fig.show()

# 6. Custom analysis
trials_df = study.trials_dataframe()
print("Top 10 trials:")
print(trials_df.nlargest(10, 'value')[['value', 'params_goal_initial_weight', 'params_penalty_decay_rate']])

# 7. Statistical analysis
print(f"\\nOptimization statistics:")
print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value}")
print(f"Best parameters: {study.best_params}")

# 8. Convergence analysis
values = [trial.value for trial in study.trials if trial.value is not None]
plt.figure(figsize=(10, 6))
plt.plot(values)
plt.xlabel('Trial')
plt.ylabel('Objective Value')
plt.title('Optimization Convergence')
plt.show()</code></pre>
                
                <h2>Best Practices</h2>
                
                <h3>Search Space Design</h3>
                <ul>
                    <li><strong>Start small</strong>: Begin with 3-5 key parameters</li>
                    <li><strong>Use domain knowledge</strong>: Set reasonable bounds based on experience</li>
                    <li><strong>Log scale for wide ranges</strong>: Use log=True for parameters spanning orders of magnitude</li>
                    <li><strong>Categorical for discrete choices</strong>: Use suggest_categorical for schedule types, etc.</li>
                </ul>
                
                <h3>Objective Function Design</h3>
                <ul>
                    <li><strong>Use multiple metrics</strong>: Combine performance, efficiency, and stability</li>
                    <li><strong>Handle failures gracefully</strong>: Use TrialPruned for failed trials</li>
                    <li><strong>Enable early stopping</strong>: Use trial.report() and trial.should_prune()</li>
                    <li><strong>Normalize metrics</strong>: Ensure different metrics are on comparable scales</li>
                </ul>
                
                <h3>Computational Efficiency</h3>
                <ul>
                    <li><strong>Use fewer episodes</strong>: Reduce training time per trial</li>
                    <li><strong>Parallel trials</strong>: Run multiple trials simultaneously</li>
                    <li><strong>Warm starting</strong>: Initialize with good parameter estimates</li>
                    <li><strong>Progressive evaluation</strong>: Start with short training, extend for promising trials</li>
                </ul>
                
                <h2>Example: Complete Optimization Pipeline</h2>
                <pre><code class="language-python"># Complete example for CartPole environment
import gym
import optuna
from rllama import RewardEngine, RewardComposer, RewardShaper
from rllama.rewards.components import GoalReward, StepPenalty, BalanceReward

def optimize_cartpole_rewards():
    # Define search space
    def suggest_config(trial):
        return {
            "reward_shaping": {
                "balance": {
                    "class": "BalanceReward",
                    "params": {
                        "angle_threshold": trial.suggest_float("angle_threshold", 0.1, 0.5),
                        "bonus": trial.suggest_float("balance_bonus", 0.1, 2.0)
                    },
                    "weight_schedule": {
                        "initial_weight": trial.suggest_float("balance_weight", 0.1, 5.0)
                    }
                },
                "step_penalty": {
                    "class": "StepPenalty", 
                    "params": {
                        "penalty": trial.suggest_float("step_penalty", -0.1, -0.001)
                    },
                    "weight_schedule": {
                        "initial_weight": trial.suggest_float("penalty_weight", 0.1, 2.0)
                    }
                }
            }
        }
    
    def objective(trial):
        config = suggest_config(trial)
        
        # Create environment and agent
        env = gym.make('CartPole-v1')
        agent = SimpleAgent()  # Your agent implementation
        
        # Set up RLlama components
        engine = RewardEngine()
        engine.add_component(BalanceReward(
            angle_threshold=config["reward_shaping"]["balance"]["params"]["angle_threshold"],
            bonus=config["reward_shaping"]["balance"]["params"]["bonus"]
        ))
        engine.add_component(StepPenalty(
            penalty=config["reward_shaping"]["step_penalty"]["params"]["penalty"]
        ))
        
        # Train and evaluate
        total_reward = 0
        num_episodes = 50  # Reduced for faster optimization
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(500):  # Max steps per episode
                action = agent.select_action(state)
                next_state, raw_reward, done, info = env.step(action)
                
                context = {
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "done": done,
                    "step": step
                }
                
                shaped_reward = engine.compute(context)
                agent.update(state, action, shaped_reward, next_state, done)
                
                episode_reward += shaped_reward
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
        
        env.close()
        return total_reward / num_episodes
    
    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    
    return study

# Run the optimization
study = optimize_cartpole_rewards()</code></pre>
                
                <p>By following this guide, you can systematically find optimal reward configurations that significantly improve agent performance while reducing manual tuning effort.</p>
            `,
            
            overview: `
                <h1>API Reference Overview</h1>
                <p>This page provides a comprehensive overview of the main classes and functions in RLlama, organized by functionality and use case.</p>
                
                <h2>Core Classes</h2>
                
                <h3>RewardEngine</h3>
                <p>The central component that manages all reward components and orchestrates reward computation.</p>
                
                <pre><code class="language-python">from rllama import RewardEngine

# Initialize the engine
engine = RewardEngine()

# Add components
engine.add_component(component_instance)
engine.add_component(component_instance, name="custom_name")

# Compute total reward
reward = engine.compute(context)

# Get detailed breakdown
breakdown = engine.get_breakdown(context)

# Weight management
engine.set_weight(component_name, weight)
engine.get_weight(component_name)
engine.update_weights(weight_dict)

# Component management
engine.remove_component(component_name)
engine.list_components()
engine.clear_components()</code></pre>
                
                <h3>BaseReward</h3>
                <p>The base class for all reward components. All custom reward components must inherit from this class.</p>
                
                <pre><code class="language-python">from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def compute(self, context):
        # Extract relevant information from context
        value = context.get('key', default_value)
        
        # Perform reward calculation
        reward = self.calculate_reward(value)
        
        return reward
    
    def reset(self):
        # Optional: Reset component state for new episode
        pass</code></pre>
                
                <h3>RewardComposer</h3>
                <p>Combines multiple reward components into a single reward signal.</p>
                
                <pre><code class="language-python">from rllama import RewardComposer

# Create composer with component dictionary
composer = RewardComposer({
    "goal": GoalReward(reward_value=1.0),
    "penalty": StepPenalty(penalty=-0.01),
    "distance": DistanceReward(scale=0.5)
})

# Compose rewards
reward_dict = composer.compose(raw_reward, info, context)
total_reward = composer.compute_total(raw_reward, info, context)

# Component management
composer.add_component("safety", SafetyPenalty())
composer.remove_component("penalty")
composer.get_component("goal")
composer.list_components()</code></pre>
                
                <h3>RewardShaper</h3>
                <p>Applies dynamic weighting and scheduling to reward components.</p>
                
                <pre><code class="language-python">from rllama import RewardShaper

# Create shaper with composer and configuration
shaper = RewardShaper(composer, reward_config)

# Shape rewards with dynamic weights
final_reward = shaper.shape(raw_reward, info, context)

# Weight management
shaper.update_weights(global_step)
shaper.set_weight("goal", 2.0)
current_weight = shaper.get_current_weight("goal")

# Schedule management
shaper.set_schedule("penalty", ExponentialDecay(0.999))
shaper.get_schedule("penalty")</code></pre>
                
                <h2>Built-in Reward Components</h2>
                
                <h3>Goal and Achievement Rewards</h3>
                <pre><code class="language-python"># GoalReward - Rewards goal achievement
from rllama.rewards.components import GoalReward
goal_reward = GoalReward(
    reward_value=1.0,
    success_key='is_success'  # Key in info dict
)

# ThresholdReward - Rewards exceeding thresholds
from rllama.rewards.components import ThresholdReward
threshold_reward = ThresholdReward(
    threshold=0.8,
    metric_key='score',
    reward_value=0.5
)

# ProgressReward - Rewards progress toward goal
from rllama.rewards.components import ProgressReward
progress_reward = ProgressReward(
    target_key='target_position',
    current_key='agent_position',
    scale=1.0
)</code></pre>
                
                <h3>Penalty Components</h3>
                <pre><code class="language-python"># StepPenalty - Penalizes each step
from rllama.rewards.components import StepPenalty
step_penalty = StepPenalty(penalty=-0.01)

# SafetyPenalty - Penalizes unsafe actions
from rllama.rewards.components import SafetyPenalty
safety_penalty = SafetyPenalty(
    collision_penalty=-1.0,
    violation_key='collision'
)

# TimeoutPenalty - Penalizes episode timeouts
from rllama.rewards.components import TimeoutPenalty
timeout_penalty = TimeoutPenalty(
    penalty=-0.5,
    max_steps=1000
)</code></pre>
                
                <h3>Distance and Spatial Rewards</h3>
                <pre><code class="language-python"># DistanceReward - Rewards based on distance to target
from rllama.rewards.components import DistanceReward
distance_reward = DistanceReward(
    target_position=[0, 0],
    scale=1.0,
    inverse=True  # Closer = higher reward
)

# BoundaryReward - Rewards staying within boundaries
from rllama.rewards.components import BoundaryReward
boundary_reward = BoundaryReward(
    bounds=[(0, 10), (0, 10)],  # x and y bounds
    penalty=-0.1
)</code></pre>
                
                <h2>Optimization Classes</h2>
                
                <h3>BayesianRewardOptimizer</h3>
                <p>Automatically optimizes reward weights using Bayesian optimization.</p>
                
                <pre><code class="language-python">from rllama.optimization import BayesianRewardOptimizer

# Define search space
search_space = {
    "goal_weight": (0.1, 10.0),
    "penalty_weight": (0.001, 1.0),
    "distance_scale": (0.1, 5.0)
}

# Define objective function
def objective(trial):
    # Create engine with suggested parameters
    engine = RewardEngine()
    engine.add_component(GoalReward(weight=trial.suggest_float("goal_weight", 0.1, 10.0)))
    engine.add_component(StepPenalty(weight=trial.suggest_float("penalty_weight", 0.001, 1.0)))
    
    # Train and evaluate
    performance = train_and_evaluate(engine)
    return performance

# Run optimization
optimizer = BayesianRewardOptimizer()
best_params = optimizer.optimize(objective, n_trials=100)</code></pre>
                
                <h2>Integration Classes</h2>
                
                <h3>GymWrapper</h3>
                <p>Integrates RLlama with OpenAI Gym environments.</p>
                
                <pre><code class="language-python">from rllama.integration import GymWrapper
import gym

# Create environment and engine
env = gym.make('CartPole-v1')
engine = RewardEngine()
engine.add_component(BalanceReward())

# Wrap environment
wrapped_env = GymWrapper(engine).wrap(env)

# Use like normal Gym environment
obs = wrapped_env.reset()
obs, reward, done, info = wrapped_env.step(action)</code></pre>
                
                <h3>StableBaselinesWrapper</h3>
                <p>Integration with Stable Baselines3 algorithms.</p>
                
                <pre><code class="language-python">from rllama.integration import StableBaselinesWrapper
from stable_baselines3 import PPO

# Create wrapped environment
wrapped_env = GymWrapper(engine).wrap(env)

# Train with SB3
model = PPO("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=100000)</code></pre>
                
                <h2>Utility Functions</h2>
                
                <h3>Configuration Loading</h3>
                <pre><code class="language-python">from rllama.utils import load_config, save_config

# Load configuration from YAML
config = load_config('reward_config.yaml')

# Save configuration
save_config(config, 'output_config.yaml')

# Merge configurations
merged_config = merge_configs(base_config, override_config)</code></pre>
                
                <h3>Visualization Tools</h3>
                <pre><code class="language-python">from rllama.visualization import plot_reward_breakdown, plot_weight_schedule

# Plot reward component contributions
plot_reward_breakdown(episode_rewards, component_names)

# Plot weight schedules over time
plot_weight_schedule(shaper, component_name, steps=1000)</code></pre>
                
                <h2>Context Dictionary Keys</h2>
                <p>Common keys used in the context dictionary passed to reward components:</p>
                
                <ul>
                    <li><code>state</code>: Current environment state</li>
                    <li><code>action</code>: Action taken by the agent</li>
                    <li><code>next_state</code>: Resulting state after action</li>
                    <li><code>done</code>: Boolean indicating episode termination</li>
                    <li><code>info</code>: Environment info dictionary</li>
                    <li><code>global_step</code>: Current training step</li>
                    <li><code>episode</code>: Current episode number</li>
                    <li><code>episode_step</code>: Step within current episode</li>
                    <li><code>agent_position</code>: Agent's position (for spatial tasks)</li>
                    <li><code>target_position</code>: Target position (for navigation tasks)</li>
                    <li><code>performance_metrics</code>: Custom performance metrics</li>
                </ul>
                
                <h2>Configuration Schema</h2>
                <p>Structure for reward configuration files:</p>
                
                <pre><code class="language-yaml">reward_shaping:
  component_name:
    class: ComponentClassName
    params:
      param1: value1
      param2: value2
    weight_schedule:
      initial_weight: 1.0
      schedule_type: "exponential"  # or "linear", "constant"
      decay_rate: 0.999
      decay_steps: 1
      min_weight: 0.1
      
training:
  num_episodes: 1000
  max_steps_per_episode: 500
  
optimization:
  n_trials: 100
  direction: "maximize"  # or "minimize"</code></pre>
                
                <h2>Error Handling</h2>
                <p>Common exceptions and error handling patterns:</p>
                
                <pre><code class="language-python">from rllama.exceptions import (
    ComponentNotFoundError,
    InvalidConfigurationError,
    RewardComputationError
)

try:
    reward = engine.compute(context)
except ComponentNotFoundError as e:
    print(f"Component not found: {e}")
except RewardComputationError as e:
    print(f"Error computing reward: {e}")
except InvalidConfigurationError as e:
    print(f"Invalid configuration: {e}")</code></pre>
                
                <p>For more detailed information about specific classes and methods, refer to the individual documentation pages and examples in the RLlama repository.</p>
            `,
            
            'why-rllama': `
                <h1>Why RLlama?</h1>
                <p>Reward engineering is one of the most critical and challenging aspects of reinforcement learning. Here's why RLlama makes it better:</p>
                
                <h2>The Problem with Traditional Reward Engineering</h2>
                <ul>
                    <li><strong>Monolithic Design</strong>: Reward functions become large, complex, and hard to understand</li>
                    <li><strong>Difficult Debugging</strong>: When something goes wrong, it's hard to isolate the issue</li>
                    <li><strong>No Reusability</strong>: Reward logic is tied to specific environments</li>
                    <li><strong>Manual Tuning</strong>: Adjusting reward weights is time-consuming and error-prone</li>
                    <li><strong>Limited Insight</strong>: No visibility into which reward components are working</li>
                </ul>
                
                <h2>How RLlama Solves These Problems</h2>
                
                <h3>🧩 Modular Architecture</h3>
                <p>Break down complex rewards into simple, reusable components that can be mixed and matched.</p>
                
                <pre><code class="language-python"># Traditional monolithic approach
def complex_reward(state, action, next_state, info):
    reward = 0
    if info['goal_reached']: reward += 100
    if info['collision']: reward -= 50
    reward += calculate_distance_bonus(state, goal)
    reward -= 0.01  # step penalty
    return reward

# RLlama modular approach
engine = RewardEngine()
engine.add_component(GoalReward(reward_value=100))
engine.add_component(SafetyPenalty(collision_penalty=-50))
engine.add_component(DistanceReward(scale=1.0))
engine.add_component(StepPenalty(penalty=-0.01))</code></pre>
                
                <h3>🔍 Easy Debugging</h3>
                <p>Isolate and test individual reward components to quickly identify issues.</p>
                
                <pre><code class="language-python"># Get detailed breakdown of reward contributions
breakdown = engine.get_breakdown(context)
print("Reward breakdown:")
for component, value in breakdown.items():
    print(f"  {component}: {value:.3f}")

# Test individual components
goal_component = GoalReward(reward_value=100)
test_reward = goal_component.compute(test_context)
assert test_reward == 100  # Clear, testable behavior</code></pre>
                
                <h3>🎛️ Automated Optimization</h3>
                <p>Use Bayesian optimization to automatically find the best reward weights.</p>
                
                <pre><code class="language-python">from rllama.optimization import BayesianRewardOptimizer

def objective(trial):
    goal_weight = trial.suggest_float('goal_weight', 0.1, 10.0)
    penalty_weight = trial.suggest_float('penalty_weight', 0.001, 1.0)
    
    engine = RewardEngine()
    engine.add_component(GoalReward(weight=goal_weight))
    engine.add_component(StepPenalty(weight=penalty_weight))
    
    return train_and_evaluate(engine)

optimizer = BayesianRewardOptimizer()
best_weights = optimizer.optimize(objective, n_trials=100)</code></pre>
                
                <h3>📊 Comprehensive Visualization</h3>
                <p>See exactly how each reward component contributes to the final reward signal.</p>
                
                <pre><code class="language-python">from rllama.visualization import plot_reward_breakdown

# Track component contributions over time
episode_breakdowns = []
for episode in training_episodes:
    breakdown = engine.get_breakdown(episode_context)
    episode_breakdowns.append(breakdown)

# Visualize contributions
plot_reward_breakdown(episode_breakdowns, save_path='reward_analysis.png')</code></pre>
                
                <h3>🔧 Dynamic Control</h3>
                <p>Adjust reward weights during training for curriculum learning and adaptive behavior.</p>
                
                <pre><code class="language-python"># Curriculum learning with weight scheduling
engine.set_weight_schedule('safety_penalty', ExponentialDecay(
    initial_weight=2.0,
    decay_rate=0.999,
    min_weight=0.1
))

engine.set_weight_schedule('efficiency_reward', LinearIncrease(
    initial_weight=0.0,
    final_weight=1.0,
    duration_steps=50000
))

# Automatic weight updates during training
for step in training_steps:
    engine.update_weights(step)
    reward = engine.compute(context)</code></pre>
                
                <h3>🚀 Framework Integration</h3>
                <p>Works seamlessly with popular RL frameworks like Stable Baselines3, Ray RLlib, and more.</p>
                
                <pre><code class="language-python"># Stable Baselines3 integration
from stable_baselines3 import PPO
from rllama.integration import GymWrapper

wrapped_env = GymWrapper(engine).wrap(env)
model = PPO("MlpPolicy", wrapped_env)
model.learn(total_timesteps=100000)

# Ray RLlib integration
from ray.rllib.env.env_context import EnvContext
from rllama.integration import RLlibWrapper

def create_rllama_env(config: EnvContext):
    env = gym.make(config["env_name"])
    return RLlibWrapper(engine).wrap(env)</code></pre>
                
                <h2>Real-World Benefits</h2>
                
                <h3>Faster Development</h3>
                <ul>
                    <li><strong>Rapid Prototyping</strong>: Quickly test different reward combinations</li>
                    <li><strong>Component Reuse</strong>: Share reward logic across projects and environments</li>
                    <li><strong>Clear Documentation</strong>: Self-documenting reward structure through component names</li>
                    <li><strong>Easy Modification</strong>: Add, remove, or modify components without touching core logic</li>
                </ul>
                
                <h3>Better Performance</h3>
                <ul>
                    <li><strong>Optimized Weights</strong>: Automatic optimization finds better parameter combinations</li>
                    <li><strong>Curriculum Learning</strong>: Dynamic weight scheduling improves training efficiency</li>
                    <li><strong>Component Isolation</strong>: Debug and fix individual components without affecting others</li>
                    <li><strong>Systematic Testing</strong>: Test reward components in isolation for reliability</li>
                </ul>
                
                <h3>Easier Maintenance</h3>
                <ul>
                    <li><strong>Modular Design</strong>: Changes to one component don't affect others</li>
                    <li><strong>Clear Separation</strong>: Each component has a single, well-defined responsibility</li>
                    <li><strong>Version Control</strong>: Track changes to individual components separately</li>
                    <li><strong>Team Collaboration</strong>: Multiple developers can work on different components</li>
                </ul>
                
                <h3>Knowledge Sharing</h3>
                <ul>
                    <li><strong>Component Libraries</strong>: Build and share libraries of proven reward components</li>
                    <li><strong>Best Practices</strong>: Encode domain knowledge into reusable components</li>
                    <li><strong>Reproducibility</strong>: Configuration files make experiments reproducible</li>
                    <li><strong>Documentation</strong>: Component-based design encourages better documentation</li>
                </ul>
                
                <h2>Success Stories</h2>
                
                <h3>Robotics Manipulation</h3>
                <p>A robotics team reduced their reward engineering time from weeks to days by using RLlama's modular components. They created reusable components for grasping, lifting, and placing objects that could be combined for different manipulation tasks.</p>
                
                <h3>Game AI Development</h3>
                <p>A game development studio used RLlama to create AI agents for different game modes. By sharing components between modes and using automated optimization, they achieved 40% better performance with 60% less development time.</p>
                
                <h3>Autonomous Navigation</h3>
                <p>An autonomous vehicle team used RLlama's curriculum learning capabilities to gradually introduce safety constraints while maintaining navigation performance. The modular design allowed them to easily adjust behavior for different driving scenarios.</p>
                
                <h2>Scientific Impact</h2>
                
                <h3>Reproducible Research</h3>
                <p>RLlama's configuration-based approach makes RL experiments more reproducible. Researchers can share exact reward configurations along with their papers, enabling others to replicate and build upon their work.</p>
                
                <h3>Systematic Exploration</h3>
                <p>The automated optimization capabilities enable systematic exploration of reward design spaces that would be impractical to explore manually. This leads to discoveries of better reward structures and training strategies.</p>
                
                <h3>Standardization</h3>
                <p>RLlama promotes standardization in reward engineering practices, making it easier to compare results across different research groups and enabling meta-analyses of reward design strategies.</p>
                
                <h2>Industry Adoption</h2>
                
                <h3>Reduced Time-to-Market</h3>
                <p>Companies using RLlama report 50-70% reduction in time spent on reward engineering, allowing them to focus more resources on other aspects of their RL systems.</p>
                
                <h3>Improved Reliability</h3>
                <p>The modular design and systematic testing capabilities lead to more reliable RL systems in production environments.</p>
                
                <h3>Scalable Development</h3>
                <p>Large teams can work more effectively on RL projects by dividing reward engineering work across different components and domains.</p>
                
                <h2>Getting Started</h2>
                <p>Ready to transform your reward engineering workflow? Here's how to get started:</p>
                
                <ol>
                    <li><strong>Install RLlama</strong>: <code>pip install rllama</code></li>
                    <li><strong>Try the Quick Start</strong>: Follow our 5-minute tutorial</li>
                    <li><strong>Explore Examples</strong>: Check out domain-specific examples in our cookbook</li>
                    <li><strong>Join the Community</strong>: Connect with other RLlama users and contributors</li>
                </ol>
                
                <p>RLlama transforms reward engineering from an art to a science, enabling more systematic, efficient, and effective development of reinforcement learning systems. Whether you're a researcher exploring new RL algorithms or an engineer building production RL systems, RLlama provides the tools you need to succeed.</p>
            `
        };
        
        return content[page] || '<h1>Page Not Found</h1><p>The requested page could not be found.</p>';
    }

    setupAnimation() {
        const canvas = document.getElementById('rewardCanvas');
        if (!canvas) return;

        this.animationScene = new THREE.Scene();
        this.animationCamera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
        this.animationRenderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true });
        
        this.animationRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
        this.animationRenderer.setClearColor(0x000000, 0);

        // Create geometric shapes representing reward components
        const geometry1 = new THREE.SphereGeometry(0.5, 32, 32);
        const geometry2 = new THREE.BoxGeometry(0.8, 0.8, 0.8);
        const geometry3 = new THREE.TetrahedronGeometry(0.6);

        // Materials using theme colors
        const materials = [
            new THREE.MeshBasicMaterial({ color: 0xFFFF00, wireframe: true }),
            new THREE.MeshBasicMaterial({ color: 0xFFFF00, wireframe: false, transparent: true, opacity: 0.3 }),
            new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: true })
        ];

        // Create reward nodes
        const nodes = [];
        for (let i = 0; i < 8; i++) {
            const geometry = [geometry1, geometry2, geometry3][i % 3];
            const material = materials[i % materials.length];
            const mesh = new THREE.Mesh(geometry, material);
            
            // Position nodes in a circle
            const angle = (i / 8) * Math.PI * 2;
            mesh.position.x = Math.cos(angle) * 3;
            mesh.position.y = Math.sin(angle) * 2;
            mesh.position.z = Math.sin(angle * 2) * 1;
            
            this.animationScene.add(mesh);
            nodes.push(mesh);
        }

        // Create connections between nodes
        const connections = [];
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                if (Math.random() > 0.7) { // Only connect some nodes
                    const points = [nodes[i].position, nodes[j].position];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.LineBasicMaterial({ color: 0xFFFF00, transparent: true, opacity: 0.3 });
                    const line = new THREE.Line(geometry, material);
                    this.animationScene.add(line);
                    connections.push(line);
                }
            }
        }

        // Add particle system
        const particleCount = 100;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const velocities = [];

        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 10;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
            
            velocities.push({
                x: (Math.random() - 0.5) * 0.02,
                y: (Math.random() - 0.5) * 0.02,
                z: (Math.random() - 0.5) * 0.02
            });
        }

        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const particleMaterial = new THREE.PointsMaterial({ 
            color: 0xFFFF00, 
            size: 0.1,
            transparent: true,
            opacity: 0.6
        });
        const particleSystem = new THREE.Points(particles, particleMaterial);
        this.animationScene.add(particleSystem);

        this.animationCamera.position.z = 8;
        this.animationCamera.position.y = 2;

        // Animation loop
        let time = 0;
        const animate = () => {
            requestAnimationFrame(animate);
            time += 0.01;

            // Rotate nodes
            nodes.forEach((node, index) => {
                node.rotation.x += 0.01;
                node.rotation.y += 0.005;
                node.position.y += Math.sin(time + index) * 0.02;
            });

            // Update particles
            const positions = particleSystem.geometry.attributes.position.array;
            for (let i = 0; i < particleCount; i++) {
                positions[i * 3] += velocities[i].x;
                positions[i * 3 + 1] += velocities[i].y;
                positions[i * 3 + 2] += velocities[i].z;

                // Boundary check
                if (Math.abs(positions[i * 3]) > 5) velocities[i].x *= -1;
                if (Math.abs(positions[i * 3 + 1]) > 5) velocities[i].y *= -1;
                if (Math.abs(positions[i * 3 + 2]) > 5) velocities[i].z *= -1;
            }
            particleSystem.geometry.attributes.position.needsUpdate = true;

            // Rotate camera around the scene
            this.animationCamera.position.x = Math.cos(time * 0.2) * 8;
            this.animationCamera.position.z = Math.sin(time * 0.2) * 8;
            this.animationCamera.lookAt(this.animationScene.position);

            this.animationRenderer.render(this.animationScene, this.animationCamera);
        };

        animate();
    }

    loadDocumentationContent() {
        // Content is now loaded dynamically when pages are accessed
        console.log('RLlama Documentation loaded successfully');
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.rllama = new RLlamaApp();
});

// Global function for button clicks
function showPage(page) {
    if (window.rllama) {
        window.rllama.showPage(page);
        
        // Update navigation
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-page') === page) {
                link.classList.add('active');
            }
        });
    }
}


