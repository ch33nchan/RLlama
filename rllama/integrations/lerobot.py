import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union

class LeRobotEnv(gym.Env):
    """
    LeRobot simulation environment for reinforcement learning.
    
    This environment simulates a simple robot with:
    - Positional sensors
    - Velocity sensors
    - Force sensors
    - Target-following task
    
    The robot must learn to efficiently navigate to targets.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 max_steps: int = 1000,
                 difficulty: str = "easy",
                 render_mode: Optional[str] = None):
        """
        Initialize the LeRobot environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.render_mode = render_mode
        
        # Define action space (motor commands)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Robot state
            "position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            "velocity": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),
            "orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
            "angular_velocity": spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),
            
            # Sensor readings
            "force_sensors": spaces.Box(low=-50.0, high=50.0, shape=(6,), dtype=np.float32),
            
            # Target information
            "target_position": spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32),
            "target_relative": spaces.Box(low=-20.0, high=20.0, shape=(3,), dtype=np.float32),
        })
        
        # Set up difficulty parameters
        if difficulty == "easy":
            self.target_spawn_radius = 3.0
            self.obstacle_count = 0
        elif difficulty == "medium":
            self.target_spawn_radius = 5.0
            self.obstacle_count = 3
        elif difficulty == "hard":
            self.target_spawn_radius = 8.0
            self.obstacle_count = 10
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
            
        # Reset the environment
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.steps = 0
        
        # Reset robot state
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        
        # Generate random target
        self.target_position = self.np_random.uniform(
            low=-self.target_spawn_radius,
            high=self.target_spawn_radius,
            size=3
        ).astype(np.float32)
        
        # Generate obstacles if needed
        self.obstacles = []
        for _ in range(self.obstacle_count):
            obstacle_pos = self.np_random.uniform(
                low=-self.target_spawn_radius,
                high=self.target_spawn_radius,
                size=3
            ).astype(np.float32)
            obstacle_radius = self.np_random.uniform(0.2, 0.5)
            self.obstacles.append((obstacle_pos, obstacle_radius))
            
        # Simulate force sensor readings (placeholder)
        self.force_sensors = np.zeros(6, dtype=np.float32)
        
        # Return initial observation
        return self._get_observation(), {}
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Motor commands for the robot
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Increment step counter
        self.steps += 1
        
        # Apply action to update robot state (simplified physics)
        # In a real implementation, this would use proper physics simulation
        self.velocity += action[:3] * 0.1
        self.velocity = np.clip(self.velocity, -5.0, 5.0)
        
        self.angular_velocity += action[3:] * 0.1 if len(action) > 3 else 0
        self.angular_velocity = np.clip(self.angular_velocity, -5.0, 5.0)
        
        # Update position and orientation
        self.position += self.velocity * 0.1
        self.orientation += self.angular_velocity * 0.1
        
        # Normalize orientation angles
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        # Check collisions with obstacles (simplified)
        for obs_pos, obs_radius in self.obstacles:
            dist = np.linalg.norm(self.position - obs_pos)
            if dist < obs_radius + 0.5:  # Robot radius = 0.5
                # Collision response (simplified)
                self.velocity *= -0.5  # Bounce with damping
                
                # Apply forces to sensors (simplified)
                direction = (self.position - obs_pos) / (dist + 1e-6)
                force_magnitude = 10.0 * (obs_radius + 0.5 - dist)
                
                # Distribute force to sensors based on direction
                for i in range(6):
                    sensor_dir = np.array([
                        np.cos(i * np.pi/3), 
                        np.sin(i * np.pi/3), 
                        0
                    ])
                    self.force_sensors[i] = np.dot(direction, sensor_dir) * force_magnitude
        
        # Calculate distance to target
        target_vector = self.target_position - self.position
        distance = np.linalg.norm(target_vector)
        
        # Check if target reached
        target_reached = distance < 0.5
        
        # Check if episode is done
        terminated = target_reached
        truncated = self.steps >= self.max_steps
        
        # Calculate reward
        if target_reached:
            reward = 10.0  # Bonus for reaching target
        else:
            # Negative reward based on distance and velocity
            reward = -0.1 * distance - 0.01 * np.linalg.norm(self.velocity)
            
            # Penalty for high forces (collisions)
            reward -= 0.05 * np.sum(np.abs(self.force_sensors))
        
        # Return results
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self) -> Dict:
        """
        Get the current observation.
        
        Returns:
            Dictionary of observations
        """
        target_relative = self.target_position - self.position
        
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "force_sensors": self.force_sensors.copy(),
            "target_position": self.target_position.copy(),
            "target_relative": target_relative
        }
        
    def render(self):
        """
        Render the environment.
        
        Returns:
            Rendering depending on render_mode
        """
        if self.render_mode is None:
            return
            
        if self.render_mode == "human":
            # Print state information
            print(f"Position: {self.position}")
            print(f"Target: {self.target_position}")
            print(f"Distance: {np.linalg.norm(self.target_position - self.position):.2f}")
            print(f"Step: {self.steps}/{self.max_steps}")
            print("-" * 30)
            
        elif self.render_mode == "rgb_array":
            # Create a simple 2D visualization (top-down view)
            # This is a simplified placeholder
            grid_size = 100
            scale = grid_size / (2 * self.target_spawn_radius)
            
            # Create empty grid
            grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
            
            # Function to convert coordinates to grid indices
            def coord_to_grid(coord):
                x = int((coord[0] + self.target_spawn_radius) * scale)
                y = int((coord[1] + self.target_spawn_radius) * scale)
                return np.clip(x, 0, grid_size-1), np.clip(y, 0, grid_size-1)
            
            # Draw obstacles
            for obs_pos, obs_radius in self.obstacles:
                x, y = coord_to_grid(obs_pos)
                radius = int(obs_radius * scale)
                
                # Draw a circle for each obstacle
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx**2 + dy**2 <= radius**2:
                            gx, gy = x + dx, y + dy
                            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                                grid[gy, gx] = [100, 100, 100]  # Gray
            
            # Draw target
            tx, ty = coord_to_grid(self.target_position)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 0), (0, 1)]:
                gx, gy = tx + dx, ty + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    grid[gy, gx] = [0, 255, 0]  # Green
            
            # Draw robot
            rx, ry = coord_to_grid(self.position)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
                gx, gy = rx + dx, ry + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    grid[gy, gx] = [255, 0, 0]  # Red
                    
            return grid
        
    def close(self):
        """Clean up resources."""
        pass