import numpy as np
from .base import RewardComponentBase
from typing import Optional, Tuple, Dict, Any

class TargetReachedReward(RewardComponentBase):
    """
    Rewards the agent for reaching a target position/configuration.
    """
    def __init__(self, name: str = "target_reached", target_key: str = "target_pose",
                 current_key: str = "current_pose", threshold: float = 0.05,
                 reward_on_reach: float = 1.0, penalty_per_unit_distance: Optional[float] = -0.01):
        super().__init__(name)
        self.target_key = target_key
        self.current_key = current_key
        self.threshold = threshold
        self.reward_on_reach = reward_on_reach
        self.penalty_per_unit_distance = penalty_per_unit_distance
        self._target_reached_this_episode = False

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        target_pose = context.get(self.target_key)
        current_pose = context.get(self.current_key)
        done = context.get("done", False)

        if target_pose is None or current_pose is None:
            # print(f"Warning: {self.name} - target_pose or current_pose not in context.")
            return 0.0

        # Assuming poses are numpy arrays for distance calculation
        distance = np.linalg.norm(np.array(current_pose) - np.array(target_pose))
        reward = 0.0

        if distance <= self.threshold and not self._target_reached_this_episode:
            reward += self.reward_on_reach
            self._target_reached_this_episode = True # Give reward once per episode or sustain?
                                                 # For now, once. Can be made configurable.
        
        if self.penalty_per_unit_distance is not None:
            reward += distance * self.penalty_per_unit_distance # penalty_per_unit_distance is negative

        if done: # Reset for next episode
            self._target_reached_this_episode = False
            
        return reward

    def reset(self):
        self._target_reached_this_episode = False


class CollisionAvoidanceReward(RewardComponentBase):
    """
    Penalizes the agent for collisions.
    """
    def __init__(self, name: str = "collision_avoidance", collision_key: str = "collision_detected",
                 penalty_on_collision: float = -1.0):
        super().__init__(name)
        self.collision_key = collision_key
        self.penalty_on_collision = penalty_on_collision

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        collision_detected = context.get(self.collision_key, False) # Expects a boolean
        
        if collision_detected:
            return self.penalty_on_collision
        return 0.0

class SmoothnessReward(RewardComponentBase):
    """
    Penalizes jerky movements by looking at joint accelerations or action differences.
    """
    def __init__(self, name: str = "smoothness", action_key: str = "action",
                 prev_action_key: str = "previous_action",
                 joint_accelerations_key: Optional[str] = None,
                 penalty_factor: float = -0.01):
        super().__init__(name)
        self.action_key = action_key
        self.prev_action_key = prev_action_key # Agent/wrapper needs to provide this in context
        self.joint_accelerations_key = joint_accelerations_key
        self.penalty_factor = penalty_factor # Should be negative

    def calculate_reward(self, context: Dict[str, Any]) -> float:
        if self.joint_accelerations_key:
            joint_accelerations = context.get(self.joint_accelerations_key)
            if joint_accelerations is not None:
                # Assuming joint_accelerations is a list/array of numbers
                return self.penalty_factor * np.sum(np.square(joint_accelerations))
        else: # Fallback to action differences
            action = context.get(self.action_key)
            prev_action = context.get(self.prev_action_key)
            if action is not None and prev_action is not None:
                action_diff = np.array(action) - np.array(prev_action)
                return self.penalty_factor * np.sum(np.square(action_diff))
        return 0.0

# Add to rllama/rewards/__init__.py
# from .robotics_components import TargetReachedReward, CollisionAvoidanceReward, SmoothnessReward
# __all__ = [..., "TargetReachedReward", "CollisionAvoidanceReward", "SmoothnessReward"]