#!/usr/bin/env python3
"""
Robotics-specific reward components for robotic control and manipulation tasks.
These components implement sophisticated reward functions for common robotics scenarios.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import deque
import warnings

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class CollisionAvoidanceReward(BaseReward):
    """
    Reward component that penalizes collisions and encourages safe navigation.
    Supports multiple collision detection methods and safety margins.
    """
    
    def __init__(self,
                 obstacle_key: str = "obstacles",
                 robot_position_key: str = "robot_position",
                 robot_radius: float = 0.5,
                 safety_margin: float = 0.2,
                 penalty_strength: float = -10.0,
                 distance_reward_scaling: float = 0.1,
                 collision_threshold: float = 0.01,
                 **kwargs):
        """
        Initialize collision avoidance reward component.
        
        Args:
            obstacle_key: Key in context containing obstacle positions/geometries
            robot_position_key: Key in context containing robot position
            robot_radius: Radius of the robot for collision detection
            safety_margin: Additional safety margin around obstacles
            penalty_strength: Penalty for collisions (should be negative)
            distance_reward_scaling: Scaling for distance-based rewards
            collision_threshold: Distance threshold for collision detection
        """
        super().__init__(**kwargs)
        self.obstacle_key = obstacle_key
        self.robot_position_key = robot_position_key
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.penalty_strength = penalty_strength
        self.distance_reward_scaling = distance_reward_scaling
        self.collision_threshold = collision_threshold
        
        # Track collision history for adaptive penalties
        self.collision_history = deque(maxlen=100)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate collision avoidance reward."""
        # Extract robot position and obstacles
        robot_pos = context.get(self.robot_position_key)
        obstacles = context.get(self.obstacle_key, [])
        
        if robot_pos is None:
            return 0.0
            
        # Convert to numpy array
        try:
            robot_pos = np.array(robot_pos)
        except (ValueError, TypeError):
            return 0.0
            
        if len(obstacles) == 0:
            return 0.0  # No obstacles, no collision risk
            
        # Calculate distances to all obstacles
        min_distance = float('inf')
        collision_detected = False
        
        for obstacle in obstacles:
            distance = self._calculate_obstacle_distance(robot_pos, obstacle)
            if distance is not None:
                min_distance = min(min_distance, distance)
                
                # Check for collision
                if distance <= self.collision_threshold:
                    collision_detected = True
                    
        # Track collision in history
        self.collision_history.append(collision_detected)
        
        # Calculate reward
        if collision_detected:
            # Strong penalty for collision
            collision_penalty = self.penalty_strength
            
            # Adaptive penalty based on recent collision frequency
            recent_collisions = sum(self.collision_history[-10:])
            if recent_collisions > 3:  # Frequent collisions
                collision_penalty *= 1.5
                
            return collision_penalty
            
        elif min_distance < self.safety_margin + self.robot_radius:
            # Penalty for being too close to obstacles
            safety_violation = (self.safety_margin + self.robot_radius) - min_distance
            safety_penalty = -self.distance_reward_scaling * safety_violation ** 2
            return safety_penalty
            
        else:
            # Small positive reward for maintaining safe distance
            safe_distance_bonus = self.distance_reward_scaling * min(1.0, min_distance / 2.0)
            return safe_distance_bonus
            
    def _calculate_obstacle_distance(self, robot_pos: np.ndarray, obstacle: Any) -> Optional[float]:
        """Calculate distance from robot to obstacle."""
        try:
            if isinstance(obstacle, dict):
                # Obstacle with position and geometry
                obs_pos = np.array(obstacle.get('position', [0, 0]))
                obs_radius = obstacle.get('radius', 0.1)
                
                # Distance between centers minus obstacle radius
                center_distance = np.linalg.norm(robot_pos[:len(obs_pos)] - obs_pos)
                return max(0, center_distance - obs_radius - self.robot_radius)
                
            elif isinstance(obstacle, (list, tuple, np.ndarray)):
                # Simple point obstacle
                obs_pos = np.array(obstacle)
                center_distance = np.linalg.norm(robot_pos[:len(obs_pos)] - obs_pos)
                return max(0, center_distance - self.robot_radius)
                
            else:
                return None
                
        except (ValueError, TypeError, IndexError):
            return None
            
    def get_collision_statistics(self) -> Dict[str, float]:
        """Get collision statistics."""
        if not self.collision_history:
            return {"collision_rate": 0.0, "recent_collisions": 0}
            
        total_collisions = sum(self.collision_history)
        collision_rate = total_collisions / len(self.collision_history)
        recent_collisions = sum(list(self.collision_history)[-10:])
        
        return {
            "collision_rate": collision_rate,
            "recent_collisions": recent_collisions,
            "total_samples": len(self.collision_history)
        }

@register_reward_component
class EnergyEfficiencyReward(BaseReward):
    """
    Reward component that encourages energy-efficient robot behavior.
    Penalizes high energy consumption and rewards smooth, efficient movements.
    """
    
    def __init__(self,
                 action_key: str = "action",
                 velocity_key: str = "velocity",
                 torque_key: str = "torque",
                 power_key: str = "power",
                 efficiency_weight: float = 1.0,
                 smoothness_weight: float = 0.5,
                 power_penalty_weight: float = 0.1,
                 max_power_threshold: float = 100.0,
                 **kwargs):
        """
        Initialize energy efficiency reward component.
        
        Args:
            action_key: Key in context containing robot actions
            velocity_key: Key in context containing joint/end-effector velocities
            torque_key: Key in context containing joint torques
            power_key: Key in context containing power consumption
            efficiency_weight: Weight for overall efficiency reward
            smoothness_weight: Weight for movement smoothness
            power_penalty_weight: Weight for power consumption penalty
            max_power_threshold: Maximum acceptable power consumption
        """
        super().__init__(**kwargs)
        self.action_key = action_key
        self.velocity_key = velocity_key
        self.torque_key = torque_key
        self.power_key = power_key
        self.efficiency_weight = efficiency_weight
        self.smoothness_weight = smoothness_weight
        self.power_penalty_weight = power_penalty_weight
        self.max_power_threshold = max_power_threshold
        
        # Track previous actions for smoothness calculation
        self.previous_actions = deque(maxlen=5)
        self.power_history = deque(maxlen=100)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate energy efficiency reward."""
        # Extract relevant data
        action = context.get(self.action_key)
        velocity = context.get(self.velocity_key)
        torque = context.get(self.torque_key)
        power = context.get(self.power_key)
        
        total_reward = 0.0
        
        # Power consumption penalty
        if power is not None:
            power_penalty = self._calculate_power_penalty(power)
            total_reward += self.power_penalty_weight * power_penalty
            
        # Movement smoothness reward
        if action is not None:
            smoothness_reward = self._calculate_smoothness_reward(action)
            total_reward += self.smoothness_weight * smoothness_reward
            
        # Energy efficiency based on velocity and torque
        if velocity is not None and torque is not None:
            efficiency_reward = self._calculate_efficiency_reward(velocity, torque)
            total_reward += self.efficiency_weight * efficiency_reward
            
        return total_reward
        
    def _calculate_power_penalty(self, power: Union[float, List[float], np.ndarray]) -> float:
        """Calculate penalty based on power consumption."""
        try:
            if isinstance(power, (list, np.ndarray)):
                total_power = np.sum(np.abs(power))
            else:
                total_power = abs(float(power))
                
            # Track power history
            self.power_history.append(total_power)
            
            # Penalty increases exponentially with power consumption
            if total_power > self.max_power_threshold:
                excess_power = total_power - self.max_power_threshold
                penalty = -excess_power ** 1.5  # Superlinear penalty
            else:
                # Small penalty for any power consumption
                penalty = -0.01 * total_power
                
            return penalty
            
        except (ValueError, TypeError):
            return 0.0
            
    def _calculate_smoothness_reward(self, action: Union[List[float], np.ndarray]) -> float:
        """Calculate reward for smooth movements."""
        try:
            action_array = np.array(action)
            
            # Add to history
            self.previous_actions.append(action_array)
            
            if len(self.previous_actions) < 2:
                return 0.0  # Need at least 2 actions for smoothness
                
            # Calculate action differences (jerk)
            action_diffs = []
            for i in range(1, len(self.previous_actions)):
                diff = np.linalg.norm(self.previous_actions[i] - self.previous_actions[i-1])
                action_diffs.append(diff)
                
            # Smoothness reward is inverse of action variation
            avg_diff = np.mean(action_diffs)
            smoothness_reward = 1.0 / (1.0 + avg_diff)
            
            return smoothness_reward
            
        except (ValueError, TypeError):
            return 0.0
            
    def _calculate_efficiency_reward(self, 
                                   velocity: Union[List[float], np.ndarray],
                                   torque: Union[List[float], np.ndarray]) -> float:
        """Calculate efficiency reward based on velocity-torque relationship."""
        try:
            vel_array = np.array(velocity)
            torque_array = np.array(torque)
            
            if vel_array.shape != torque_array.shape:
                return 0.0
                
            # Calculate mechanical power (P = τ * ω)
            mechanical_power = np.abs(vel_array * torque_array)
            total_mechanical_power = np.sum(mechanical_power)
            
            # Efficiency reward: encourage achieving motion with minimal torque
            if total_mechanical_power > 0:
                # Reward is higher when velocity is achieved with lower torque
                velocity_magnitude = np.linalg.norm(vel_array)
                torque_magnitude = np.linalg.norm(torque_array)
                
                if torque_magnitude > 0:
                    efficiency_ratio = velocity_magnitude / (torque_magnitude + 1e-6)
                    efficiency_reward = np.tanh(efficiency_ratio)  # Bounded between 0 and 1
                else:
                    efficiency_reward = 1.0  # Perfect efficiency (no torque needed)
            else:
                efficiency_reward = 0.0
                
            return efficiency_reward
            
        except (ValueError, TypeError):
            return 0.0
            
    def get_energy_statistics(self) -> Dict[str, float]:
        """Get energy consumption statistics."""
        if not self.power_history:
            return {"avg_power": 0.0, "max_power": 0.0, "efficiency_score": 0.0}
            
        avg_power = np.mean(list(self.power_history))
        max_power = np.max(list(self.power_history))
        
        # Efficiency score based on staying below threshold
        efficient_samples = sum(1 for p in self.power_history if p <= self.max_power_threshold)
        efficiency_score = efficient_samples / len(self.power_history)
        
        return {
            "avg_power": avg_power,
            "max_power": max_power,
            "efficiency_score": efficiency_score,
            "total_samples": len(self.power_history)
        }

@register_reward_component
class TaskCompletionReward(BaseReward):
    """
    Reward component for task completion in robotics.
    Supports various task types and completion criteria.
    """
    
    def __init__(self,
                 task_type: str = "reach",
                 target_key: str = "target",
                 current_position_key: str = "end_effector_position",
                 completion_threshold: float = 0.05,
                 progress_reward_scaling: float = 1.0,
                 completion_bonus: float = 10.0,
                 time_penalty_weight: float = 0.01,
                 **kwargs):
        """
        Initialize task completion reward component.
        
        Args:
            task_type: Type of task ("reach", "grasp", "place", "follow")
            target_key: Key in context containing target position/state
            current_position_key: Key in context containing current position
            completion_threshold: Distance threshold for task completion
            progress_reward_scaling: Scaling for progress-based rewards
            completion_bonus: Bonus reward for task completion
            time_penalty_weight: Weight for time-based penalty
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.target_key = target_key
        self.current_position_key = current_position_key
        self.completion_threshold = completion_threshold
        self.progress_reward_scaling = progress_reward_scaling
        self.completion_bonus = completion_bonus
        self.time_penalty_weight = time_penalty_weight
        
        # Track task progress
        self.previous_distance = None
        self.task_start_time = None
        self.task_completed = False
        self.step_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate task completion reward."""
        # Extract positions
        target = context.get(self.target_key)
        current_pos = context.get(self.current_position_key)
        
        if target is None or current_pos is None:
            return 0.0
            
        try:
            target_array = np.array(target)
            current_array = np.array(current_pos)
        except (ValueError, TypeError):
            return 0.0
            
        # Calculate current distance to target
        current_distance = np.linalg.norm(current_array - target_array)
        
        # Initialize tracking on first call
        if self.previous_distance is None:
            self.previous_distance = current_distance
            self.task_start_time = self.step_count
            
        self.step_count += 1
        
        # Check for task completion
        if current_distance <= self.completion_threshold and not self.task_completed:
            self.task_completed = True
            completion_reward = self.completion_bonus
            
            # Time bonus for quick completion
            time_taken = self.step_count - self.task_start_time
            time_bonus = max(0, self.completion_bonus * 0.5 * (1.0 - time_taken / 1000.0))
            
            return completion_reward + time_bonus
            
        elif self.task_completed:
            # Maintain completion bonus but with small penalty for moving away
            if current_distance > self.completion_threshold:
                return -0.1 * (current_distance - self.completion_threshold)
            else:
                return 0.1  # Small reward for staying at target
                
        else:
            # Progress-based reward
            progress_reward = self._calculate_progress_reward(current_distance)
            
            # Time penalty for taking too long
            time_penalty = -self.time_penalty_weight * (self.step_count - self.task_start_time)
            
            return progress_reward + time_penalty
            
    def _calculate_progress_reward(self, current_distance: float) -> float:
        """Calculate reward based on progress toward target."""
        if self.previous_distance is None:
            return 0.0
            
        # Progress is positive when getting closer to target
        progress = self.previous_distance - current_distance
        
        # Update previous distance
        self.previous_distance = current_distance
        
        # Scale progress reward
        progress_reward = self.progress_reward_scaling * progress
        
        # Add distance-based component (closer is better)
        distance_reward = self.progress_reward_scaling * (1.0 / (1.0 + current_distance))
        
        return progress_reward + distance_reward * 0.1
        
    def reset_task(self) -> None:
        """Reset task tracking for a new episode."""
        self.previous_distance = None
        self.task_start_time = None
        self.task_completed = False
        self.step_count = 0
        
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task completion statistics."""
        return {
            "task_completed": self.task_completed,
            "step_count": self.step_count,
            "time_since_start": self.step_count - (self.task_start_time or 0),
            "current_distance": self.previous_distance
        }

@register_reward_component
class SmoothTrajectoryReward(BaseReward):
    """
    Reward component that encourages smooth, natural robot trajectories.
    Penalizes jerky movements and rewards fluid motion.
    """
    
    def __init__(self,
                 position_key: str = "end_effector_position",
                 velocity_key: str = "end_effector_velocity",
                 acceleration_key: str = "end_effector_acceleration",
                 jerk_penalty_weight: float = 1.0,
                 acceleration_penalty_weight: float = 0.5,
                 velocity_smoothness_weight: float = 0.3,
                 trajectory_length: int = 10,
                 **kwargs):
        """
        Initialize smooth trajectory reward component.
        
        Args:
            position_key: Key in context containing position data
            velocity_key: Key in context containing velocity data
            acceleration_key: Key in context containing acceleration data
            jerk_penalty_weight: Weight for jerk (rate of acceleration change) penalty
            acceleration_penalty_weight: Weight for high acceleration penalty
            velocity_smoothness_weight: Weight for velocity smoothness reward
            trajectory_length: Length of trajectory history to maintain
        """
        super().__init__(**kwargs)
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.acceleration_key = acceleration_key
        self.jerk_penalty_weight = jerk_penalty_weight
        self.acceleration_penalty_weight = acceleration_penalty_weight
        self.velocity_smoothness_weight = velocity_smoothness_weight
        self.trajectory_length = trajectory_length
        
        # Trajectory history
        self.position_history = deque(maxlen=trajectory_length)
        self.velocity_history = deque(maxlen=trajectory_length)
        self.acceleration_history = deque(maxlen=trajectory_length)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate smooth trajectory reward."""
        # Extract motion data
        position = context.get(self.position_key)
        velocity = context.get(self.velocity_key)
        acceleration = context.get(self.acceleration_key)
        
        total_reward = 0.0
        
        # Add current data to history
        if position is not None:
            try:
                pos_array = np.array(position)
                self.position_history.append(pos_array)
            except (ValueError, TypeError):
                pass
                
        if velocity is not None:
            try:
                vel_array = np.array(velocity)
                self.velocity_history.append(vel_array)
                
                # Velocity smoothness reward
                smoothness_reward = self._calculate_velocity_smoothness()
                total_reward += self.velocity_smoothness_weight * smoothness_reward
                
            except (ValueError, TypeError):
                pass
                
        if acceleration is not None:
            try:
                acc_array = np.array(acceleration)
                self.acceleration_history.append(acc_array)
                
                # Acceleration penalty
                acc_penalty = self._calculate_acceleration_penalty(acc_array)
                total_reward += self.acceleration_penalty_weight * acc_penalty
                
                # Jerk penalty
                jerk_penalty = self._calculate_jerk_penalty()
                total_reward += self.jerk_penalty_weight * jerk_penalty
                
            except (ValueError, TypeError):
                pass
                
        return total_reward
        
    def _calculate_velocity_smoothness(self) -> float:
        """Calculate reward for smooth velocity changes."""
        if len(self.velocity_history) < 2:
            return 0.0
            
        # Calculate velocity changes
        velocity_changes = []
        for i in range(1, len(self.velocity_history)):
            vel_change = np.linalg.norm(self.velocity_history[i] - self.velocity_history[i-1])
            velocity_changes.append(vel_change)
            
        if not velocity_changes:
            return 0.0
            
        # Smoothness is inverse of velocity change variance
        change_variance = np.var(velocity_changes)
        smoothness_reward = 1.0 / (1.0 + change_variance)
        
        return smoothness_reward
        
    def _calculate_acceleration_penalty(self, acceleration: np.ndarray) -> float:
        """Calculate penalty for high accelerations."""
        acc_magnitude = np.linalg.norm(acceleration)
        
        # Penalty increases quadratically with acceleration
        penalty = -0.01 * acc_magnitude ** 2
        
        return penalty
        
    def _calculate_jerk_penalty(self) -> float:
        """Calculate penalty for high jerk (rate of acceleration change)."""
        if len(self.acceleration_history) < 2:
            return 0.0
            
        # Calculate jerk (derivative of acceleration)
        jerks = []
        for i in range(1, len(self.acceleration_history)):
            jerk = np.linalg.norm(self.acceleration_history[i] - self.acceleration_history[i-1])
            jerks.append(jerk)
            
        if not jerks:
            return 0.0
            
        # Penalty for high jerk
        avg_jerk = np.mean(jerks)
        jerk_penalty = -0.1 * avg_jerk ** 2
        
        return jerk_penalty
        
    def _calculate_trajectory_curvature(self) -> float:
        """Calculate trajectory curvature for smoothness assessment."""
        if len(self.position_history) < 3:
            return 0.0
            
        # Calculate curvature using three consecutive points
        curvatures = []
        
        for i in range(2, len(self.position_history)):
            p1 = self.position_history[i-2]
            p2 = self.position_history[i-1]
            p3 = self.position_history[i]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate curvature (simplified 2D/3D)
            if len(p1) >= 2:
                # Cross product magnitude for curvature
                if len(p1) == 2:
                    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                else:
                    cross_product = np.linalg.norm(np.cross(v1, v2))
                    
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    curvature = abs(cross_product) / (v1_norm * v2_norm)
                    curvatures.append(curvature)
                    
        if curvatures:
            avg_curvature = np.mean(curvatures)
            # Reward for low curvature (smooth paths)
            curvature_reward = 1.0 / (1.0 + avg_curvature)
            return curvature_reward
        else:
            return 0.0
            
    def get_trajectory_statistics(self) -> Dict[str, float]:
        """Get trajectory smoothness statistics."""
        stats = {
            "trajectory_length": len(self.position_history),
            "avg_velocity_change": 0.0,
            "avg_acceleration": 0.0,
            "avg_jerk": 0.0
        }
        
        # Calculate average velocity changes
        if len(self.velocity_history) >= 2:
            vel_changes = []
            for i in range(1, len(self.velocity_history)):
                change = np.linalg.norm(self.velocity_history[i] - self.velocity_history[i-1])
                vel_changes.append(change)
            stats["avg_velocity_change"] = np.mean(vel_changes)
            
        # Calculate average acceleration
        if self.acceleration_history:
            acc_magnitudes = [np.linalg.norm(acc) for acc in self.acceleration_history]
            stats["avg_acceleration"] = np.mean(acc_magnitudes)
            
        # Calculate average jerk
        if len(self.acceleration_history) >= 2:
            jerks = []
            for i in range(1, len(self.acceleration_history)):
                jerk = np.linalg.norm(self.acceleration_history[i] - self.acceleration_history[i-1])
                jerks.append(jerk)
            stats["avg_jerk"] = np.mean(jerks)
            
        return stats
