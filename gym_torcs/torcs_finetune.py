#!/usr/bin/env python3
"""
TORCS Racing AI - Fine-Tuning Script
=====================================

Focused script for fine-tuning an already-trained network.
- Pure ES optimization (no SAC)
- Racing line optimization (not center-seeking)
- Configurable steering smoothness penalties
- OVERNIGHT MODE: Rotates through optimization phases automatically

Usage:
    # Manual single-config run
    python torcs_finetune.py --resume checkpoint.npz --smoothness 8.0
    
    # Overnight rotation mode (leave running)
    python torcs_finetune.py --resume checkpoint.npz --overnight
    
    # Custom overnight settings  
    python torcs_finetune.py --resume checkpoint.npz --overnight --gens_per_phase 15 --cycles 20
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import socket
import numpy as np
import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path
import argparse
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# PHASE CONFIGURATIONS FOR OVERNIGHT ROTATION
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    name: str
    description: str
    
    # Steering
    smoothness_penalty: float = 5.0
    steer_magnitude_penalty: float = 0.3
    steer_smoothing: float = 0.0
    
    # Racing line
    racing_line_reward: float = 0.5
    racing_line_penalty: float = 0.3
    
    # Track position
    edge_danger_threshold: float = 0.7
    edge_penalty: float = 0.5
    
    # Speed
    speed_reward_scale: float = 1.0
    high_speed_bonus: float = 0.15
    high_speed_threshold: float = 100.0
    
    # Sliding
    slide_penalty: float = 0.3
    
    # ES
    sigma: float = 0.008


# Rotation phases - each focuses on different aspects
OVERNIGHT_PHASES = [
    PhaseConfig(
        name="SMOOTH",
        description="Eliminate wobble with high smoothness penalties",
        smoothness_penalty=10.0,
        steer_magnitude_penalty=0.5,
        steer_smoothing=0.3,
        racing_line_reward=0.3,
        racing_line_penalty=0.2,
        speed_reward_scale=0.9,
        slide_penalty=0.4,
        sigma=0.003,  # Reduced from 0.008
    ),
    PhaseConfig(
        name="RACING_LINE", 
        description="Improve line adherence with smoother steering",
        smoothness_penalty=6.0,
        steer_magnitude_penalty=0.3,
        steer_smoothing=0.2,
        racing_line_reward=0.8,
        racing_line_penalty=0.5,
        speed_reward_scale=0.95,
        slide_penalty=0.3,
        sigma=0.003,  # Reduced from 0.008
    ),
    PhaseConfig(
        name="SPEED",
        description="Push speed up while maintaining quality",
        smoothness_penalty=5.0,
        steer_magnitude_penalty=0.2,
        steer_smoothing=0.1,
        racing_line_reward=0.5,
        racing_line_penalty=0.3,
        speed_reward_scale=1.2,
        high_speed_bonus=0.25,
        slide_penalty=0.2,
        sigma=0.004,  # Reduced from 0.010
    ),
    PhaseConfig(
        name="BALANCED",
        description="Refine all aspects together",
        smoothness_penalty=6.0,
        steer_magnitude_penalty=0.3,
        steer_smoothing=0.15,
        racing_line_reward=0.6,
        racing_line_penalty=0.4,
        speed_reward_scale=1.0,
        high_speed_bonus=0.2,
        slide_penalty=0.3,
        sigma=0.003,  # Reduced from 0.006
    ),
    PhaseConfig(
        name="PRECISION",
        description="Fine-grained refinement with low sigma",
        smoothness_penalty=5.0,
        steer_magnitude_penalty=0.25,
        steer_smoothing=0.1,
        racing_line_reward=0.7,
        racing_line_penalty=0.4,
        speed_reward_scale=1.0,
        slide_penalty=0.3,
        sigma=0.002,  # Reduced from 0.004
    ),
]


# ============================================================================
# FINE-TUNING CONFIGURATION - ACTIVE SETTINGS
# ============================================================================

class FineTuneConfig:
    """All the knobs you can turn for fine-tuning. Modified by phases in overnight mode."""
    
    # === STEERING SMOOTHNESS ===
    SMOOTHNESS_PENALTY = 5.0
    STEER_MAGNITUDE_PENALTY = 0.3
    STEER_SMOOTHING = 0.0
    
    # === RACING LINE ===
    RACING_LINE_REWARD = 0.5
    RACING_LINE_PENALTY = 0.3
    
    # === TRACK POSITION ===
    EDGE_DANGER_THRESHOLD = 0.7
    EDGE_PENALTY = 0.5
    
    # === SPEED ===
    SPEED_REWARD_SCALE = 1.0
    HIGH_SPEED_BONUS = 0.15
    HIGH_SPEED_THRESHOLD = 100.0
    
    # === SLIDING ===
    SLIDE_PENALTY = 0.3
    
    # === EVOLUTION STRATEGY ===
    SIGMA = 0.003                     # Reduced for fine-tuning
    SIGMA_MIN = 0.001
    SIGMA_MAX = 0.010
    LEARNING_RATE = 0.015
    POPULATION_SIZE = 24
    
    # === EXPLORATION ===
    ELITE_NOISE = 0.0
    MUTATION_NOISE = 0.02
    
    @classmethod
    def apply_phase(cls, phase: PhaseConfig):
        """Apply a phase configuration to the active settings."""
        cls.SMOOTHNESS_PENALTY = phase.smoothness_penalty
        cls.STEER_MAGNITUDE_PENALTY = phase.steer_magnitude_penalty
        cls.STEER_SMOOTHING = phase.steer_smoothing
        cls.RACING_LINE_REWARD = phase.racing_line_reward
        cls.RACING_LINE_PENALTY = phase.racing_line_penalty
        cls.EDGE_DANGER_THRESHOLD = phase.edge_danger_threshold
        cls.EDGE_PENALTY = phase.edge_penalty
        cls.SPEED_REWARD_SCALE = phase.speed_reward_scale
        cls.HIGH_SPEED_BONUS = phase.high_speed_bonus
        cls.SLIDE_PENALTY = phase.slide_penalty
        cls.SIGMA = phase.sigma
    
    @classmethod
    def print_current(cls):
        """Print current configuration."""
        print(f"    Smoothness penalty:    {cls.SMOOTHNESS_PENALTY}")
        print(f"    Steer magnitude pen:   {cls.STEER_MAGNITUDE_PENALTY}")
        print(f"    Steer EMA smoothing:   {cls.STEER_SMOOTHING}")
        print(f"    Racing line reward:    {cls.RACING_LINE_REWARD}")
        print(f"    Racing line penalty:   {cls.RACING_LINE_PENALTY}")
        print(f"    Speed reward scale:    {cls.SPEED_REWARD_SCALE}")
        print(f"    Slide penalty:         {cls.SLIDE_PENALTY}")
        print(f"    Sigma:                 {cls.SIGMA}")


class NetworkConfig:
    """Network architecture - must match your trained model."""
    HIDDEN_SIZES = [256, 256]
    STATE_DIM = 65
    ACTION_DIM = 2


class TrainingConfig:
    """General training settings."""
    MAX_EPISODE_STEPS = 10000
    EPISODE_TIMEOUT = 300
    MIN_SPEED_THRESHOLD = 5.0
    STUCK_STEPS_LIMIT = 60
    TRACK_LENGTH = 3602


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class RacingPolicy(nn.Module):
    """Policy network for racing."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, action_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        nn.init.uniform_(self.output.weight, -0.03, 0.03)
        self.output.bias.data[0] = 0.0
        self.output.bias.data[1] = 0.8
        
    def forward(self, state):
        x = self.hidden(state)
        return torch.tanh(self.output(x))
    
    def get_flat_params(self) -> np.ndarray:
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params: np.ndarray):
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            new_data = torch.FloatTensor(
                flat_params[idx:idx + param_size].reshape(param.shape)
            ).to(param.device)
            param.data = new_data
            idx += param_size
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# EVOLUTION STRATEGY
# ============================================================================

class ElitistES:
    """
    Elitist Evolution Strategy for fine-tuning.
    Population = elite (unchanged) + mutations around elite.
    """
    
    def __init__(self, num_params: int, initial_params: np.ndarray,
                 sigma: float, learning_rate: float, population_size: int):
        self.num_params = num_params
        self.sigma = sigma
        self.lr = learning_rate
        self.pop_size = population_size if population_size % 2 == 0 else population_size + 1
        
        self.mean = initial_params.copy()
        self.best_params = initial_params.copy()
        self.best_fitness = -np.inf
        
        self.generation = 0
        self.prev_mean_fitness = -np.inf
        
        self.sigma_min = FineTuneConfig.SIGMA_MIN
        self.sigma_max = FineTuneConfig.SIGMA_MAX
        
    def ask(self) -> List[np.ndarray]:
        """Generate population: elite + mirrored mutations."""
        population = []
        self.noise = []
        
        # Elite unchanged as first individual
        population.append(self.best_params.copy())
        
        # Mirrored mutations around elite
        pairs_needed = (self.pop_size - 1) // 2
        for _ in range(pairs_needed):
            eps = np.random.randn(self.num_params)
            self.noise.append(eps)
            population.append(self.best_params + self.sigma * eps)
            population.append(self.best_params - self.sigma * eps)
        
        return population
    
    def tell(self, population: List[np.ndarray], fitnesses: List[float]) -> Dict:
        """Update based on fitness results."""
        
        # Track best
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = population[best_idx].copy()
        
        # Rank-based fitness shaping
        ranks = np.argsort(np.argsort(fitnesses))
        normalized = (ranks - (len(population) - 1) / 2) / (len(population) - 1)
        
        # Gradient from mirrored pairs (skip elite at index 0)
        gradient = np.zeros(self.num_params)
        for i, eps in enumerate(self.noise):
            idx_pos = 1 + 2 * i
            idx_neg = 1 + 2 * i + 1
            if idx_neg < len(normalized):
                gradient += (normalized[idx_pos] - normalized[idx_neg]) * eps
        
        gradient /= (2 * len(self.noise) * self.sigma + 1e-8)
        
        # Update mean
        self.mean = self.best_params + self.lr * gradient
        
        # Adaptive sigma
        mean_fitness = np.mean(fitnesses)
        if self.generation > 0:
            if mean_fitness > self.prev_mean_fitness:
                self.sigma = min(self.sigma * 1.03, self.sigma_max)
            else:
                self.sigma = max(self.sigma * 0.97, self.sigma_min)
        self.prev_mean_fitness = mean_fitness
        
        self.generation += 1
        
        return {
            'best_fitness': self.best_fitness,
            'mean_fitness': mean_fitness,
            'sigma': self.sigma,
            'generation': self.generation,
        }


# ============================================================================
# RULE-BASED CONTROLLERS (for racing line reference)
# ============================================================================

class RacingLineController:
    """
    Computes optimal racing line position and steering.
    Used as REFERENCE for the network to follow, not as direct control.
    """
    
    def __init__(self):
        self.target_pos = 0.0  # Current racing line target position
        
    def reset(self):
        self.target_pos = 0.0
        
    def compute(self, sensors: Dict) -> Tuple[float, float]:
        """
        Returns (target_position, recommended_steer).
        target_position: where car SHOULD be on track (-1 to +1)
        recommended_steer: what steering would achieve that
        """
        trackPos = sensors.get('trackPos', 0.0)
        angle = sensors.get('angle', 0.0)
        speedX = max(sensors.get('speedX', 0.1), 0.1)
        
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [200.0] * 19
        track = list(track)
        while len(track) < 19:
            track.append(200.0)
        
        # Analyze track shape
        forward_dist = track[9]
        fwd_left = (track[7] + track[8]) / 2
        fwd_right = (track[10] + track[11]) / 2
        
        # Corner detection
        corner_bias = (fwd_right - fwd_left) / max(fwd_right + fwd_left, 1.0)
        is_corner = forward_dist < 100 and abs(corner_bias) > 0.15
        is_sharp_corner = forward_dist < 50 and abs(corner_bias) > 0.25
        
        # Calculate optimal position (racing line)
        if is_sharp_corner:
            # Sharp corner: apex inside
            target_pos = -0.6 * np.sign(corner_bias)
        elif is_corner:
            # Moderate corner: slight apex
            target_pos = -0.4 * np.sign(corner_bias)
        elif forward_dist > 150:
            # Straight: center or slight optimization
            target_pos = 0.0
        else:
            # Approaching corner: position outside for entry
            if abs(corner_bias) > 0.1:
                target_pos = 0.3 * np.sign(corner_bias)
            else:
                target_pos = 0.0
        
        # Smooth target transitions
        self.target_pos = 0.8 * self.target_pos + 0.2 * target_pos
        
        # Calculate steering to reach target
        position_error = trackPos - self.target_pos
        lookahead = 2.5
        predicted_error = position_error - lookahead * angle
        
        kp = 0.5
        recommended_steer = -kp * predicted_error
        
        # Speed scaling
        if speedX < 15:
            speed_scale = 0.4 + 0.6 * (speedX / 15.0)
        elif speedX > 60:
            speed_scale = min(1.5, 1.0 + (speedX - 60) / 100.0)
        else:
            speed_scale = 1.0
        
        recommended_steer *= speed_scale
        recommended_steer = np.clip(recommended_steer, -1.0, 1.0)
        
        return self.target_pos, recommended_steer


class ThrottleController:
    """Rule-based throttle for reference."""
    
    def __init__(self, target_speed=200.0):
        self.target_speed = target_speed
        
    def compute(self, sensors: Dict) -> float:
        speedX = sensors.get('speedX', 0.0)
        trackPos = sensors.get('trackPos', 0.0)
        angle = sensors.get('angle', 0.0)
        
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [200.0] * 19
        track = list(track)
        while len(track) < 19:
            track.append(200.0)
        
        forward_min = min(track[7:12])
        fwd_left = (track[6] + track[7] + track[8]) / 3
        fwd_right = (track[10] + track[11] + track[12]) / 3
        asymmetry = abs(fwd_left - fwd_right) / max(fwd_left + fwd_right, 1.0)
        
        is_real_corner = asymmetry > 0.2 and forward_min < 80
        is_sharp_corner = asymmetry > 0.35 and forward_min < 50
        is_very_sharp = asymmetry > 0.5 or forward_min < 25
        
        if is_very_sharp:
            safe_speed = 45 + forward_min * 0.5
        elif is_sharp_corner:
            safe_speed = 55 + forward_min * 0.6
        elif is_real_corner:
            safe_speed = 70 + forward_min * 0.5
        elif forward_min < 60:
            safe_speed = 75 + forward_min * 0.5 if asymmetry >= 0.1 else 90 + forward_min * 0.4
        elif forward_min < 100:
            safe_speed = 100 + forward_min * 0.3
        else:
            safe_speed = self.target_speed
        
        if abs(trackPos) > 0.7:
            safe_speed *= 0.75
        elif abs(trackPos) > 0.5:
            safe_speed *= 0.85
        
        if abs(angle) > 0.3:
            safe_speed *= 0.7
        elif abs(angle) > 0.15:
            safe_speed *= 0.85
        
        safe_speed = max(40, min(safe_speed, self.target_speed))
        speed_error = safe_speed - speedX
        
        if speed_error > 20:
            return min(1.0, 0.6 + speed_error / 60.0)
        elif speed_error > 5:
            return 0.4 + speed_error / 40.0
        elif speed_error > -5:
            return 0.2 + speed_error / 20.0
        elif speed_error > -20:
            return speed_error / 40.0
        else:
            return max(-1.0, speed_error / 25.0)


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """Extracts and normalizes all 65 TORCS sensors."""
    
    def __init__(self, track_length: float = 3602):
        self.track_length = track_length
        self.prev_speedX = 0.0
        self.prev_speedY = 0.0
        
    def extract(self, sensors: Dict) -> np.ndarray:
        features = []
        
        # 1. Angle (1)
        angle = sensors.get('angle', 0.0)
        features.append(angle / np.pi)
        
        # 2. Track sensors (19)
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        track = list(track)[:19]
        while len(track) < 19:
            track.append(200.0)
        features.extend([np.clip(t, 0, 200) / 200.0 for t in track])
        
        # 3. Track position (1)
        trackPos = sensors.get('trackPos', 0.0)
        features.append(np.clip(trackPos, -1.5, 1.5))
        
        # 4. Speed components (3)
        speedX = sensors.get('speedX', 0.0)
        speedY = sensors.get('speedY', 0.0)
        speedZ = sensors.get('speedZ', 0.0)
        features.append(speedX / 300.0)
        features.append(speedY / 50.0)
        features.append(speedZ / 10.0)
        
        # 5. Wheel spin velocities (4)
        wheelSpinVel = sensors.get('wheelSpinVel', [0.0] * 4)
        if isinstance(wheelSpinVel, (int, float)):
            wheelSpinVel = [wheelSpinVel] * 4
        wheelSpinVel = list(wheelSpinVel)[:4]
        while len(wheelSpinVel) < 4:
            wheelSpinVel.append(0.0)
        features.extend([np.clip(wsv, -100, 200) / 100.0 for wsv in wheelSpinVel])
        
        # 6. RPM (1)
        rpm = sensors.get('rpm', 0.0)
        features.append(rpm / 10000.0)
        
        # 7. Gear (1)
        gear = sensors.get('gear', 0)
        features.append(gear / 6.0)
        
        # 8. Distance from start (1)
        distFromStart = sensors.get('distFromStart', 0.0)
        features.append((distFromStart % self.track_length) / self.track_length)
        
        # 9. Distance raced (1)
        distRaced = sensors.get('distRaced', 0.0)
        features.append(np.log1p(distRaced) / 10.0)
        
        # 10. Damage (1)
        damage = sensors.get('damage', 0.0)
        features.append(np.clip(damage, 0, 10000) / 10000.0)
        
        # 11. Fuel (1)
        fuel = sensors.get('fuel', 0.0)
        features.append(np.clip(fuel, 0, 100) / 100.0)
        
        # 12. Z height (1)
        z = sensors.get('z', 0.0)
        features.append(np.clip(z, -5, 5) / 5.0)
        
        # 13. Focus sensors (5)
        focus = sensors.get('focus', [200.0] * 5)
        if isinstance(focus, (int, float)):
            focus = [focus] * 5
        focus = list(focus)[:5]
        while len(focus) < 5:
            focus.append(200.0)
        features.extend([np.clip(f, 0, 200) / 200.0 for f in focus])
        
        # 14. Opponents (19)
        opponents = sensors.get('opponents', [200.0] * 36)
        if isinstance(opponents, (int, float)):
            opponents = [opponents] * 36
        opponents = list(opponents)
        while len(opponents) < 36:
            opponents.append(200.0)
        sampled_opponents = [opponents[i * 2] for i in range(min(19, len(opponents) // 2))]
        while len(sampled_opponents) < 19:
            sampled_opponents.append(200.0)
        features.extend([np.clip(o, 0, 200) / 200.0 for o in sampled_opponents[:19]])
        
        # 15. Current lap time (1)
        curLapTime = sensors.get('curLapTime', 0.0)
        features.append(np.clip(curLapTime, 0, 300) / 300.0)
        
        # 16. Race position (1)
        racePos = sensors.get('racePos', 1)
        features.append(racePos / 20.0)
        
        # 17. Estimated acceleration (1)
        accelX = (speedX - self.prev_speedX) * 50
        features.append(np.clip(accelX, -50, 50) / 50.0)
        
        # 18. Lateral acceleration (1)
        accelY = (speedY - self.prev_speedY) * 50
        features.append(np.clip(accelY, -50, 50) / 50.0)
        
        # 19. Absolute speed (1)
        absSpeed = np.sqrt(speedX**2 + speedY**2)
        features.append(absSpeed / 300.0)
        
        # 20. Slip ratio (1)
        slip_ratio = abs(speedY) / max(abs(speedX), 1.0)
        features.append(np.clip(slip_ratio, 0, 2) / 2.0)
        
        self.prev_speedX = speedX
        self.prev_speedY = speedY
        
        assert len(features) == 65, f"Expected 65 features, got {len(features)}"
        return np.array(features, dtype=np.float32)


# ============================================================================
# REWARD FUNCTION - RACING LINE FOCUSED
# ============================================================================

class RacingLineReward:
    """
    Reward function optimized for racing line following.
    NOT center-seeking - rewards matching the optimal racing line.
    """
    
    def __init__(self, track_length: float = 3602):
        self.track_length = track_length
        self.reset()
        
    def reset(self):
        self.prev_dist = 0.0
        self.stuck_counter = 0
        self.step_count = 0
        self.total_speed = 0.0
        self.max_speed = 0.0
        self.lap_count = 0
        self.prev_steer = 0.0
        
    def calculate(self, sensors: Dict, action: Dict, 
                  target_pos: float, recommended_steer: float) -> Tuple[float, bool, Dict]:
        """
        Calculate reward based on racing line adherence.
        
        Args:
            sensors: TORCS sensor data
            action: Action taken (steer, accel, brake)
            target_pos: Optimal track position from racing line controller
            recommended_steer: What steering the racing line wants
        """
        speedX = sensors.get('speedX', 0.0)
        speedY = sensors.get('speedY', 0.0)
        angle = sensors.get('angle', 0.0)
        trackPos = sensors.get('trackPos', 0.0)
        distRaced = sensors.get('distRaced', 0.0)
        
        current_steer = action.get('steer', 0.0)
        
        self.step_count += 1
        self.total_speed += max(speedX, 0)
        self.max_speed = max(self.max_speed, speedX)
        
        reward = 0.0
        done = False
        info = {}
        
        cfg = FineTuneConfig
        
        # === SPEED REWARD (primary objective) ===
        speed_reward = speedX * np.cos(angle) / 80.0 * cfg.SPEED_REWARD_SCALE
        reward += speed_reward
        
        # Progress reward
        progress = distRaced - self.prev_dist
        if progress > 0:
            reward += progress * 0.015 * cfg.SPEED_REWARD_SCALE
        
        # High speed bonus
        if speedX > cfg.HIGH_SPEED_THRESHOLD:
            reward += cfg.HIGH_SPEED_BONUS
        
        # === RACING LINE REWARD (not center!) ===
        # Reward for being close to the optimal racing line position
        racing_line_error = abs(trackPos - target_pos)
        
        # Reward for good racing line adherence
        if racing_line_error < 0.1:
            reward += cfg.RACING_LINE_REWARD  # On the line!
        elif racing_line_error < 0.3:
            reward += cfg.RACING_LINE_REWARD * 0.5  # Close
        
        # Penalty for deviating from racing line
        reward -= racing_line_error * cfg.RACING_LINE_PENALTY
        
        # === EDGE PENALTY (safety, not centering) ===
        # Only penalize when actually near the edge
        if abs(trackPos) > cfg.EDGE_DANGER_THRESHOLD:
            edge_severity = (abs(trackPos) - cfg.EDGE_DANGER_THRESHOLD) / (1.0 - cfg.EDGE_DANGER_THRESHOLD)
            reward -= edge_severity * cfg.EDGE_PENALTY
        
        # === STEERING SMOOTHNESS ===
        steer_change = abs(current_steer - self.prev_steer)
        
        # Quadratic penalty for rapid steering changes
        smoothness_penalty = (steer_change ** 2) * cfg.SMOOTHNESS_PENALTY
        reward -= smoothness_penalty
        
        # Penalty for large steering magnitude
        steer_magnitude_penalty = (current_steer ** 2) * cfg.STEER_MAGNITUDE_PENALTY
        reward -= steer_magnitude_penalty
        
        self.prev_steer = current_steer
        
        # === SLIDING PENALTY ===
        slide_penalty = abs(speedY) / 50.0 * cfg.SLIDE_PENALTY
        reward -= slide_penalty
        
        # === TERMINATIONS ===
        
        # Slow penalty
        if speedX < TrainingConfig.MIN_SPEED_THRESHOLD:
            reward -= 0.2
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Off-track
        if abs(trackPos) > 1.0:
            reward -= 0.5
            if abs(trackPos) > 1.2:
                done = True
                reward -= 5.0
                info['termination'] = 'off_track'
        
        # Stuck
        if self.stuck_counter > TrainingConfig.STUCK_STEPS_LIMIT:
            done = True
            reward -= 5.0
            info['termination'] = 'stuck'
        
        # Lap bonus
        lap_dist = self.track_length * (self.lap_count + 1) - 50
        if distRaced >= lap_dist and self.prev_dist < lap_dist:
            self.lap_count += 1
            avg_speed = self.total_speed / max(self.step_count, 1)
            reward += 100.0 * (avg_speed / 100.0)
            info['lap_completed'] = True
        
        self.prev_dist = distRaced
        reward = np.clip(reward, -10.0, 15.0)
        
        info['speed'] = speedX
        info['max_speed'] = self.max_speed
        info['avg_speed'] = self.total_speed / max(self.step_count, 1)
        info['distance'] = distRaced
        info['lap'] = self.lap_count
        info['racing_line_error'] = racing_line_error
        
        return reward, done, info


# ============================================================================
# TORCS CLIENT
# ============================================================================

class TORCSClient:
    ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
    DATA_SIZE = 2**17
    
    def __init__(self, host='localhost', port=3001, sid='SCR', timeout=2.0):
        self.host = host
        self.port = port
        self.sid = sid
        self.timeout = timeout
        self.so = None
        self.S = {}
        self.R = self._default_action()
        
    def _default_action(self):
        return {'accel': 0.5, 'brake': 0, 'steer': 0, 'gear': 1, 'clutch': 0,
                'focus': [-90, -45, 0, 45, 90], 'meta': 0}
        
    def connect(self) -> bool:
        if self.so:
            try: self.so.close()
            except: pass
            self.so = None
            
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(self.timeout)
        except:
            return False
            
        init_msg = f'{self.sid}(init {self.ANGLES})'
        
        for _ in range(10):
            try:
                self.so.sendto(init_msg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(self.DATA_SIZE)
                if '***identified***' in data.decode():
                    return True
            except:
                pass
            time.sleep(0.3)
        return False
    
    def get_servers_input(self) -> bool:
        if not self.so:
            return False
        try:
            data, _ = self.so.recvfrom(self.DATA_SIZE)
            response = data.decode('utf-8')
            if '***shutdown***' in response or '***restart***' in response:
                return False
            if '***identified***' not in response:
                self._parse_sensors(response)
            return True
        except:
            return False
            
    def _parse_sensors(self, data: str):
        try:
            self.S = {}
            data = data.strip()
            if data.endswith(')'): data = data[:-1]
            for part in data.lstrip('(').split(')('):
                tokens = part.split(' ')
                if len(tokens) < 2: continue
                key, values = tokens[0], tokens[1:]
                if len(values) == 1:
                    try: self.S[key] = float(values[0])
                    except: self.S[key] = values[0]
                else:
                    self.S[key] = [float(v) for v in values]
        except:
            pass
            
    def respond_to_server(self):
        if not self.so:
            return
        msg = ''
        for k, v in self.R.items():
            msg += f'({k} '
            if isinstance(v, list):
                msg += ' '.join(str(x) for x in v)
            elif isinstance(v, float):
                msg += f'{v:.6f}'
            else:
                msg += str(v)
            msg += ')'
        try:
            self.so.sendto(msg.encode(), (self.host, self.port))
        except:
            pass
    
    def warmup(self) -> bool:
        for _ in range(200):
            self.R['accel'] = 1.0
            self.R['brake'] = 0.0
            self.R['gear'] = 1
            self.respond_to_server()
            if self.get_servers_input():
                if self.S.get('gear', 0) > 0 or self.S.get('speedX', 0) > 1:
                    return True
            time.sleep(0.02)
        print("TIMEOUT", end=" ")
        return False
            
    def request_restart(self) -> bool:
        self.R['meta'] = 1
        self.respond_to_server()
        time.sleep(0.1)
        self.respond_to_server()
        
        if self.so:
            try: self.so.close()
            except: pass
            self.so = None
            
        self.S = {}
        self.R = self._default_action()
        time.sleep(1.5)
        
        if not self.connect():
            return False
            
        for _ in range(30):
            if self.get_servers_input() and 'speedX' in self.S:
                return True
            self.respond_to_server()
            time.sleep(0.1)
        return False
    
    def close(self):
        if self.so:
            try: self.so.close()
            except: pass
            self.so = None


# ============================================================================
# FINE-TUNE TRAINER
# ============================================================================

class FineTuneTrainer:
    """
    Focused trainer for fine-tuning.
    Pure ES, no SAC, racing line optimization.
    """
    
    def __init__(self, host='localhost', port=3001, save_dir='./checkpoints_finetune',
                 track_length=3602, device='cpu'):
        
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        
        TrainingConfig.TRACK_LENGTH = track_length
        
        self.feature_extractor = FeatureExtractor(track_length=track_length)
        self.racing_line = RacingLineController()
        self.throttle_ctrl = ThrottleController(target_speed=200.0)
        self.reward_calculator = RacingLineReward(track_length)
        
        # These will be initialized when loading params
        self.es = None
        self.client = None
        self.best_fitness = -np.inf
        self.best_distance = 0
        self.best_speed = 0
        self.training_log = []
        self.evaluating_elite = False
        
        # Steering smoothing state
        self.prev_steer_smooth = 0.0
        
    def load_params(self, path: str):
        """Load parameters from checkpoint or params file."""
        print(f"Loading from {path}...")
        
        if path.endswith('.npz'):
            checkpoint = np.load(path, allow_pickle=True)
            params = checkpoint['best_params'] if checkpoint['best_params'] is not None else checkpoint['mean']
            self.best_fitness = float(checkpoint.get('best_fitness', -np.inf))
            self.best_distance = float(checkpoint.get('best_distance', 0))
            print(f"  Loaded checkpoint: fitness={self.best_fitness:.1f}, dist={self.best_distance:.0f}m")
        else:
            params = np.load(path)
            print(f"  Loaded params file")
        
        # Initialize ES with loaded params
        self.es = ElitistES(
            num_params=len(params),
            initial_params=params,
            sigma=FineTuneConfig.SIGMA,
            learning_rate=FineTuneConfig.LEARNING_RATE,
            population_size=FineTuneConfig.POPULATION_SIZE,
        )
        
        print(f"  {len(params)} parameters, sigma={self.es.sigma:.4f}")
        
    def evaluate_policy(self, policy: RacingPolicy) -> Tuple[float, Dict]:
        """Evaluate a single policy."""
        
        self.reward_calculator.reset()
        self.racing_line.reset()
        self.prev_steer_smooth = 0.0
        
        if not self.client.warmup():
            return -100.0, {'reward': -100, 'steps': 0, 'avg_speed': 0,
                           'max_speed': 0, 'distance': 0, 'laps': 0}
        
        state = self.feature_extractor.extract(self.client.S)
        
        total_reward = 0
        steps = 0
        speeds = []
        racing_line_errors = []
        
        start_time = time.time()
        
        policy.eval()
        with torch.no_grad():
            while steps < TrainingConfig.MAX_EPISODE_STEPS:
                if time.time() - start_time > TrainingConfig.EPISODE_TIMEOUT:
                    break
                
                # Get network action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = policy(state_tensor)
                raw_action = action_tensor.cpu().numpy().flatten()
                
                # Add exploration noise
                if self.evaluating_elite:
                    noise_scale = FineTuneConfig.ELITE_NOISE
                else:
                    noise_scale = FineTuneConfig.MUTATION_NOISE
                
                if noise_scale > 0:
                    noise = np.random.normal(0, noise_scale, size=NetworkConfig.ACTION_DIM)
                    raw_action = np.clip(raw_action + noise, -1, 1)
                
                # Get racing line reference
                target_pos, recommended_steer = self.racing_line.compute(self.client.S)
                
                # Network outputs
                network_steer = raw_action[0]
                network_throttle = raw_action[1]
                
                # Apply optional EMA smoothing
                if FineTuneConfig.STEER_SMOOTHING > 0:
                    network_steer = (self.prev_steer_smooth * FineTuneConfig.STEER_SMOOTHING + 
                                    network_steer * (1 - FineTuneConfig.STEER_SMOOTHING))
                    self.prev_steer_smooth = network_steer
                
                steer = float(np.clip(network_steer, -1, 1))
                
                # Convert throttle to accel/brake
                if network_throttle >= 0:
                    accel = float(network_throttle)
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = float(-network_throttle * 0.8)
                
                # Debug for first step
                if steps == 0:
                    action_str = f"ac={accel:.2f}" if accel > 0 else f"br={brake:.2f}"
                    print(f"st={steer:+.2f} {action_str}", end="", flush=True)
                
                # Auto gear
                rpm = self.client.S.get('rpm', 0)
                gear = int(self.client.S.get('gear', 1))
                speedX = self.client.S.get('speedX', 0)
                
                if gear < 1:
                    gear = 1
                elif rpm > 8000 and gear < 6:
                    gear += 1
                elif rpm < 5000 and gear > 1:
                    gear -= 1
                
                min_speed_for_gear = {2: 20, 3: 40, 4: 60, 5: 80, 6: 100}
                if gear > 1 and speedX < min_speed_for_gear.get(gear, 0):
                    gear -= 1
                
                gear = max(1, min(6, gear))
                
                action = {'steer': steer, 'accel': accel, 'brake': brake, 'gear': gear}
                
                # Send to TORCS
                self.client.R['steer'] = steer
                self.client.R['accel'] = accel
                self.client.R['brake'] = brake
                self.client.R['gear'] = gear
                self.client.R['meta'] = 0
                self.client.respond_to_server()
                
                if not self.client.get_servers_input():
                    break
                
                # Calculate reward
                next_state = self.feature_extractor.extract(self.client.S)
                reward, done, info = self.reward_calculator.calculate(
                    self.client.S, action, target_pos, recommended_steer
                )
                
                total_reward += reward
                steps += 1
                speeds.append(info.get('speed', 0))
                racing_line_errors.append(info.get('racing_line_error', 0))
                
                state = next_state
                
                if done:
                    term = info.get('termination', 'done')
                    if term == 'off_track':
                        print(" ✗", end="")
                    elif term == 'stuck':
                        print(" ⊘", end="")
                    break
        
        result = {
            'reward': total_reward,
            'steps': steps,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'distance': self.client.S.get('distRaced', 0),
            'laps': self.reward_calculator.lap_count,
            'avg_line_error': np.mean(racing_line_errors) if racing_line_errors else 0,
        }
        
        return total_reward, result
    
    def train(self, num_generations: int = 100, start_generation: int = 1):
        """Run fine-tuning."""
        
        print("\n" + "=" * 70)
        print("TORCS Racing AI - FINE-TUNING")
        print("=" * 70)
        print(f"Population: {self.es.pop_size}, σ: {self.es.sigma:.4f}")
        print(f"Device: {self.device}")
        print("-" * 70)
        print("FINE-TUNE PARAMETERS:")
        print(f"  Smoothness penalty:    {FineTuneConfig.SMOOTHNESS_PENALTY}")
        print(f"  Steer magnitude pen:   {FineTuneConfig.STEER_MAGNITUDE_PENALTY}")
        print(f"  Steer EMA smoothing:   {FineTuneConfig.STEER_SMOOTHING}")
        print(f"  Racing line reward:    {FineTuneConfig.RACING_LINE_REWARD}")
        print(f"  Racing line penalty:   {FineTuneConfig.RACING_LINE_PENALTY}")
        print(f"  Edge threshold:        {FineTuneConfig.EDGE_DANGER_THRESHOLD}")
        print(f"  Slide penalty:         {FineTuneConfig.SLIDE_PENALTY}")
        print("=" * 70)
        
        print("\nConnecting to TORCS...")
        self.client = TORCSClient(host=self.host, port=self.port)
        if not self.client.connect():
            print("Failed to connect!")
            return
        print("Connected!")
        
        input("\nStart a race in TORCS, then press Enter...")
        
        gen = start_generation
        try:
            for gen in range(start_generation, start_generation + num_generations):
                gen_start = time.time()
                
                print(f"\n{'='*70}")
                print(f"GEN {gen:3d} | σ: {self.es.sigma:.4f} | Best: fit={self.best_fitness:.0f}, dist={self.best_distance:.0f}m")
                print(f"{'='*70}")
                
                population = self.es.ask()
                fitnesses = []
                gen_speeds = []
                gen_distances = []
                gen_line_errors = []
                
                for i, params in enumerate(population):
                    policy = RacingPolicy(NetworkConfig.STATE_DIM, NetworkConfig.ACTION_DIM, 
                                         NetworkConfig.HIDDEN_SIZES)
                    policy.set_flat_params(params)
                    policy.to(self.device)
                    
                    is_elite = (i == 0)
                    self.evaluating_elite = is_elite
                    elite_marker = "E" if is_elite else " "
                    
                    # Print with checksum for elite to verify params consistency
                    if is_elite:
                        checksum = params.sum()
                        print(f"  [{i+1:2d}/{len(population)}]{elite_marker}[chk={checksum:.2f}]", end="", flush=True)
                    else:
                        print(f"  [{i+1:2d}/{len(population)}]{elite_marker}", end="", flush=True)
                    
                    fitness, result = self.evaluate_policy(policy)
                    self.evaluating_elite = False
                    
                    fitnesses.append(fitness)
                    gen_speeds.append(result['avg_speed'])
                    gen_distances.append(result['distance'])
                    gen_line_errors.append(result['avg_line_error'])
                    
                    # Check for new best
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_distance = max(self.best_distance, result['distance'])
                        # Save immediately with copies to preserve exact tested params
                        best_params_copy = params.copy()
                        torch.save(policy.state_dict(), self.save_dir / 'best.pt')
                        np.save(self.save_dir / 'best_params.npy', best_params_copy)
                        self.es.best_params = best_params_copy
                        self.es.best_fitness = fitness
                        checksum = best_params_copy.sum()
                        print(f" → fit={fitness:8.1f}  spd={result['avg_speed']:5.1f}  "
                              f"dist={result['distance']:5.0f}m  line={result['avg_line_error']:.2f} ⭐ NEW BEST [chk={checksum:.2f}]")
                    else:
                        print(f" → fit={fitness:8.1f}  spd={result['avg_speed']:5.1f}  "
                              f"dist={result['distance']:5.0f}m  line={result['avg_line_error']:.2f}")
                    
                    if i < len(population) - 1:
                        if not self.client.request_restart():
                            self.client.close()
                            time.sleep(2)
                            self.client.connect()
                
                # Update ES
                info = self.es.tell(population, fitnesses)
                
                # Generation summary
                gen_time = time.time() - gen_start
                best_idx = np.argmax(fitnesses)
                print(f"\n  ── GEN {gen} SUMMARY ({gen_time:.0f}s) ──")
                print(f"  Best:    fit={fitnesses[best_idx]:8.1f}  spd={gen_speeds[best_idx]:5.1f}  "
                      f"dist={gen_distances[best_idx]:5.0f}m  line={gen_line_errors[best_idx]:.2f}")
                print(f"  Average: fit={info['mean_fitness']:8.1f}  spd={np.mean(gen_speeds):5.1f}  "
                      f"dist={np.mean(gen_distances):5.0f}m  line={np.mean(gen_line_errors):.2f}")
                print(f"  All-time: fit={self.best_fitness:8.1f}  dist={self.best_distance:5.0f}m")
                
                # Log
                self.training_log.append({
                    'generation': gen,
                    'best_fitness': info['best_fitness'],
                    'mean_fitness': info['mean_fitness'],
                    'sigma': info['sigma'],
                    'avg_speed': np.mean(gen_speeds),
                    'max_speed': max(gen_speeds),
                    'avg_distance': np.mean(gen_distances),
                    'avg_line_error': np.mean(gen_line_errors),
                })
                
                if gen % 5 == 0:
                    with open(self.save_dir / 'training_log.json', 'w') as f:
                        json.dump(self.training_log, f, indent=2)
                    np.save(self.save_dir / f'params_gen{gen}.npy', self.es.best_params)
                
                if not self.client.request_restart():
                    self.client.close()
                    time.sleep(2)
                    self.client.connect()
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted - saving checkpoint...")
        finally:
            checkpoint = {
                'generation': gen,
                'mean': self.es.mean,
                'sigma': self.es.sigma,
                'best_params': self.es.best_params,
                'best_fitness': self.best_fitness,
                'best_distance': self.best_distance,
            }
            np.savez(self.save_dir / 'checkpoint.npz', **checkpoint)
            
            with open(self.save_dir / 'training_log.json', 'w') as f:
                json.dump(self.training_log, f, indent=2)
            if self.es.best_params is not None:
                np.save(self.save_dir / 'final_best.npy', self.es.best_params)
            print(f"\nSaved checkpoint to {self.save_dir}")
            if self.client:
                self.client.close()

    def train_overnight(self, gens_per_phase: int = 10, num_cycles: int = 10):
        """
        Overnight training with rotating phases.
        
        Each cycle runs through all phases, with each phase running for gens_per_phase generations.
        Best model is carried forward between phases.
        Fitness is reset at each phase change since reward function changes.
        """
        
        total_phases = len(OVERNIGHT_PHASES)
        total_gens = gens_per_phase * total_phases * num_cycles
        
        print("\n" + "=" * 70)
        print("TORCS Racing AI - OVERNIGHT FINE-TUNING")
        print("=" * 70)
        print(f"Cycles: {num_cycles}")
        print(f"Phases per cycle: {total_phases}")
        print(f"Generations per phase: {gens_per_phase}")
        print(f"Total generations: {total_gens}")
        print("-" * 70)
        print("PHASE ROTATION:")
        for i, phase in enumerate(OVERNIGHT_PHASES):
            print(f"  {i+1}. {phase.name:12s} - {phase.description}")
        print("=" * 70)
        
        print("\nConnecting to TORCS...")
        self.client = TORCSClient(host=self.host, port=self.port)
        if not self.client.connect():
            print("Failed to connect!")
            return
        print("Connected!")
        
        input("\nStart a race in TORCS, then press Enter to begin overnight training...")
        
        start_time = datetime.now()
        overall_gen = 0
        cycle_stats = []
        
        try:
            for cycle in range(num_cycles):
                cycle_start = time.time()
                print(f"\n{'#'*70}")
                print(f"# CYCLE {cycle+1}/{num_cycles} - Started at {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'#'*70}")
                
                cycle_best_speed = 0
                cycle_best_distance = 0
                
                for phase_idx, phase in enumerate(OVERNIGHT_PHASES):
                    phase_start = time.time()
                    
                    # Apply phase configuration
                    FineTuneConfig.apply_phase(phase)
                    self.es.sigma = phase.sigma  # Update ES sigma
                    
                    # Reset fitness for new reward function
                    phase_best_fitness = -np.inf
                    
                    print(f"\n{'='*70}")
                    print(f"PHASE: {phase.name} ({phase_idx+1}/{total_phases})")
                    print(f"  {phase.description}")
                    print("-" * 70)
                    FineTuneConfig.print_current()
                    print("=" * 70)
                    
                    phase_speeds = []
                    phase_distances = []
                    phase_line_errors = []
                    
                    for phase_gen in range(gens_per_phase):
                        overall_gen += 1
                        gen_start = time.time()
                        
                        # Same format as regular train()
                        print(f"\n{'='*70}")
                        print(f"GEN {overall_gen:3d} | C{cycle+1} {phase.name} {phase_gen+1}/{gens_per_phase} | "
                              f"σ: {self.es.sigma:.4f} | Phase best: {phase_best_fitness:.0f}")
                        print(f"{'='*70}")
                        
                        population = self.es.ask()
                        fitnesses = []
                        gen_speeds = []
                        gen_distances = []
                        gen_line_errors = []
                        
                        for i, params in enumerate(population):
                            policy = RacingPolicy(NetworkConfig.STATE_DIM, NetworkConfig.ACTION_DIM,
                                                 NetworkConfig.HIDDEN_SIZES)
                            policy.set_flat_params(params)
                            policy.to(self.device)
                            
                            is_elite = (i == 0)
                            self.evaluating_elite = is_elite
                            elite_marker = "E" if is_elite else " "
                            
                            # Print with checksum for elite to verify params consistency
                            if is_elite:
                                checksum = params.sum()
                                print(f"  [{i+1:2d}/{len(population)}]{elite_marker}[chk={checksum:.2f}]", end="", flush=True)
                            else:
                                print(f"  [{i+1:2d}/{len(population)}]{elite_marker}", end="", flush=True)
                            
                            fitness, result = self.evaluate_policy(policy)
                            self.evaluating_elite = False
                            
                            fitnesses.append(fitness)
                            gen_speeds.append(result['avg_speed'])
                            gen_distances.append(result['distance'])
                            gen_line_errors.append(result['avg_line_error'])
                            
                            # Check for new best (phase-local fitness comparison)
                            if fitness > phase_best_fitness:
                                phase_best_fitness = fitness
                                # Save immediately with copies to preserve exact tested params
                                best_params_copy = params.copy()
                                torch.save(policy.state_dict(), self.save_dir / 'best.pt')
                                np.save(self.save_dir / 'best_params.npy', best_params_copy)
                                self.es.best_params = best_params_copy
                                self.es.best_fitness = fitness
                                checksum = best_params_copy.sum()
                                print(f" → fit={fitness:8.1f}  spd={result['avg_speed']:5.1f}  "
                                      f"dist={result['distance']:5.0f}m  line={result['avg_line_error']:.2f} ⭐ NEW BEST [chk={checksum:.2f}]")
                            else:
                                print(f" → fit={fitness:8.1f}  spd={result['avg_speed']:5.1f}  "
                                      f"dist={result['distance']:5.0f}m  line={result['avg_line_error']:.2f}")
                            
                            # Track overall bests (speed/distance are comparable across phases)
                            if result['avg_speed'] > self.best_speed:
                                self.best_speed = result['avg_speed']
                            if result['distance'] > self.best_distance:
                                self.best_distance = result['distance']
                            
                            if i < len(population) - 1:
                                if not self.client.request_restart():
                                    self.client.close()
                                    time.sleep(2)
                                    self.client.connect()
                        
                        # Update ES
                        info = self.es.tell(population, fitnesses)
                        
                        # Generation summary (same format as regular train)
                        best_idx = np.argmax(fitnesses)
                        gen_time = time.time() - gen_start
                        print(f"\n  ── GEN {overall_gen} SUMMARY ({gen_time:.0f}s) ──")
                        print(f"  Best:    fit={fitnesses[best_idx]:8.1f}  spd={gen_speeds[best_idx]:5.1f}  "
                              f"dist={gen_distances[best_idx]:5.0f}m  line={gen_line_errors[best_idx]:.2f}")
                        print(f"  Average: fit={info['mean_fitness']:8.1f}  spd={np.mean(gen_speeds):5.1f}  "
                              f"dist={np.mean(gen_distances):5.0f}m  line={np.mean(gen_line_errors):.2f}")
                        print(f"  All-time: spd={self.best_speed:5.1f}  dist={self.best_distance:5.0f}m")
                        
                        phase_speeds.extend(gen_speeds)
                        phase_distances.extend(gen_distances)
                        phase_line_errors.extend(gen_line_errors)
                        
                        # Log
                        self.training_log.append({
                            'overall_gen': overall_gen,
                            'cycle': cycle + 1,
                            'phase': phase.name,
                            'phase_gen': phase_gen + 1,
                            'best_fitness': float(info['best_fitness']),
                            'mean_fitness': float(info['mean_fitness']),
                            'sigma': float(info['sigma']),
                            'avg_speed': float(np.mean(gen_speeds)),
                            'max_speed': float(max(gen_speeds)),
                            'avg_distance': float(np.mean(gen_distances)),
                            'avg_line_error': float(np.mean(gen_line_errors)),
                        })
                        
                        # Save every 5 generations (like regular train)
                        if overall_gen % 5 == 0:
                            with open(self.save_dir / 'training_log.json', 'w') as f:
                                json.dump(self.training_log, f, indent=2)
                            np.save(self.save_dir / f'params_gen{overall_gen}.npy', self.es.best_params)
                        
                        if not self.client.request_restart():
                            self.client.close()
                            time.sleep(2)
                            self.client.connect()
                    
                    # Phase summary
                    phase_time = time.time() - phase_start
                    print(f"\n  ── {phase.name} COMPLETE ({phase_time/60:.1f}min) ──")
                    print(f"  Avg speed: {np.mean(phase_speeds):.1f} km/h")
                    print(f"  Avg distance: {np.mean(phase_distances):.0f}m")
                    print(f"  Avg line error: {np.mean(phase_line_errors):.3f}")
                    
                    cycle_best_speed = max(cycle_best_speed, max(phase_speeds))
                    cycle_best_distance = max(cycle_best_distance, max(phase_distances))
                    
                    # Save checkpoint after each phase
                    checkpoint = {
                        'overall_gen': overall_gen,
                        'cycle': cycle + 1,
                        'phase': phase.name,
                        'mean': self.es.mean,
                        'sigma': self.es.sigma,
                        'best_params': self.es.best_params,
                        'best_speed': self.best_speed,
                        'best_distance': self.best_distance,
                    }
                    np.savez(self.save_dir / 'checkpoint.npz', **checkpoint)
                    np.save(self.save_dir / f'params_cycle{cycle+1}_{phase.name}.npy', self.es.best_params)
                
                # Cycle summary
                cycle_time = time.time() - cycle_start
                elapsed = datetime.now() - start_time
                
                cycle_stats.append({
                    'cycle': cycle + 1,
                    'best_speed': cycle_best_speed,
                    'best_distance': cycle_best_distance,
                    'time_minutes': cycle_time / 60,
                })
                
                print(f"\n{'#'*70}")
                print(f"# CYCLE {cycle+1} COMPLETE")
                print(f"# Time: {cycle_time/60:.1f} min | Total elapsed: {elapsed}")
                print(f"# Best speed: {cycle_best_speed:.1f} km/h | Best distance: {cycle_best_distance:.0f}m")
                print(f"# Overall best: speed={self.best_speed:.1f} km/h, dist={self.best_distance:.0f}m")
                print(f"{'#'*70}")
                
                # Save log
                with open(self.save_dir / 'training_log.json', 'w') as f:
                    json.dump(self.training_log, f, indent=2)
                with open(self.save_dir / 'cycle_stats.json', 'w') as f:
                    json.dump(cycle_stats, f, indent=2)
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted - saving checkpoint...")
        finally:
            elapsed = datetime.now() - start_time
            print(f"\n{'='*70}")
            print(f"OVERNIGHT TRAINING COMPLETE")
            print(f"Total time: {elapsed}")
            print(f"Total generations: {overall_gen}")
            print(f"Best speed achieved: {self.best_speed:.1f} km/h")
            print(f"Best distance achieved: {self.best_distance:.0f}m")
            print(f"{'='*70}")
            
            # Final save
            checkpoint = {
                'overall_gen': overall_gen,
                'mean': self.es.mean,
                'sigma': self.es.sigma,
                'best_params': self.es.best_params,
                'best_speed': self.best_speed,
                'best_distance': self.best_distance,
            }
            np.savez(self.save_dir / 'checkpoint.npz', **checkpoint)
            
            with open(self.save_dir / 'training_log.json', 'w') as f:
                json.dump(self.training_log, f, indent=2)
            if self.es.best_params is not None:
                np.save(self.save_dir / 'final_best.npy', self.es.best_params)
            print(f"Saved to {self.save_dir}")
            
            if self.client:
                self.client.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TORCS Racing AI - Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning (manual mode)
  python torcs_finetune.py --resume checkpoint.npz
  
  # Increase smoothness penalty (reduce wobble)
  python torcs_finetune.py --resume checkpoint.npz --smoothness 8.0
  
  # OVERNIGHT MODE - rotates through optimization phases
  python torcs_finetune.py --resume checkpoint.npz --overnight
  
  # Custom overnight settings
  python torcs_finetune.py --resume checkpoint.npz --overnight --gens_per_phase 15 --cycles 20
        """
    )
    
    # Required
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint.npz or params.npy file')
    
    # Connection
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--save_dir', default='./checkpoints_finetune')
    parser.add_argument('--track_length', type=float, default=3602)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    # === OVERNIGHT MODE ===
    parser.add_argument('--overnight', action='store_true',
                        help='Enable overnight mode with rotating phases')
    parser.add_argument('--gens_per_phase', type=int, default=10,
                        help='Generations per phase in overnight mode (default: 10)')
    parser.add_argument('--cycles', type=int, default=10,
                        help='Number of full cycles through all phases (default: 10)')
    
    # === FINE-TUNE PARAMETERS (for manual mode) ===
    
    # Steering smoothness
    parser.add_argument('--smoothness', type=float, default=None,
                        help=f'Steering change penalty (default: {FineTuneConfig.SMOOTHNESS_PENALTY})')
    parser.add_argument('--steer_mag', type=float, default=None,
                        help=f'Steering magnitude penalty (default: {FineTuneConfig.STEER_MAGNITUDE_PENALTY})')
    parser.add_argument('--steer_smooth', type=float, default=None,
                        help=f'Steering EMA smoothing 0-1 (default: {FineTuneConfig.STEER_SMOOTHING})')
    
    # Racing line
    parser.add_argument('--racing_line', type=float, default=None,
                        help=f'Racing line reward (default: {FineTuneConfig.RACING_LINE_REWARD})')
    parser.add_argument('--line_penalty', type=float, default=None,
                        help=f'Racing line deviation penalty (default: {FineTuneConfig.RACING_LINE_PENALTY})')
    
    # Track position
    parser.add_argument('--edge_threshold', type=float, default=None,
                        help=f'Edge danger threshold (default: {FineTuneConfig.EDGE_DANGER_THRESHOLD})')
    parser.add_argument('--edge_penalty', type=float, default=None,
                        help=f'Edge penalty multiplier (default: {FineTuneConfig.EDGE_PENALTY})')
    
    # Speed
    parser.add_argument('--speed_scale', type=float, default=None,
                        help=f'Speed reward scale (default: {FineTuneConfig.SPEED_REWARD_SCALE})')
    parser.add_argument('--slide_penalty', type=float, default=None,
                        help=f'Lateral slide penalty (default: {FineTuneConfig.SLIDE_PENALTY})')
    
    # ES
    parser.add_argument('--sigma', type=float, default=None,
                        help=f'ES sigma/mutation scale (default: {FineTuneConfig.SIGMA})')
    parser.add_argument('--pop_size', type=int, default=None,
                        help=f'Population size (default: {FineTuneConfig.POPULATION_SIZE})')
    
    # Utility
    parser.add_argument('--reset_best', action='store_true',
                        help='Reset best fitness (use when reward function changed)')
    
    args = parser.parse_args()
    
    # Apply command-line overrides to config (only for manual mode)
    if not args.overnight:
        if args.smoothness is not None:
            FineTuneConfig.SMOOTHNESS_PENALTY = args.smoothness
        if args.steer_mag is not None:
            FineTuneConfig.STEER_MAGNITUDE_PENALTY = args.steer_mag
        if args.steer_smooth is not None:
            FineTuneConfig.STEER_SMOOTHING = args.steer_smooth
        if args.racing_line is not None:
            FineTuneConfig.RACING_LINE_REWARD = args.racing_line
        if args.line_penalty is not None:
            FineTuneConfig.RACING_LINE_PENALTY = args.line_penalty
        if args.edge_threshold is not None:
            FineTuneConfig.EDGE_DANGER_THRESHOLD = args.edge_threshold
        if args.edge_penalty is not None:
            FineTuneConfig.EDGE_PENALTY = args.edge_penalty
        if args.speed_scale is not None:
            FineTuneConfig.SPEED_REWARD_SCALE = args.speed_scale
        if args.slide_penalty is not None:
            FineTuneConfig.SLIDE_PENALTY = args.slide_penalty
        if args.sigma is not None:
            FineTuneConfig.SIGMA = args.sigma
        if args.pop_size is not None:
            FineTuneConfig.POPULATION_SIZE = args.pop_size
    
    # Create trainer
    trainer = FineTuneTrainer(
        host=args.host,
        port=args.port,
        save_dir=args.save_dir,
        track_length=args.track_length,
        device=args.device,
    )
    
    # Load parameters
    trainer.load_params(args.resume)
    
    # Reset best if requested
    if args.reset_best:
        trainer.best_fitness = -np.inf
        trainer.es.best_fitness = -np.inf
        trainer.best_distance = 0
        trainer.best_speed = 0
        print("Best fitness/speed/distance RESET")
    
    # Train
    if args.overnight:
        trainer.train_overnight(
            gens_per_phase=args.gens_per_phase,
            num_cycles=args.cycles
        )
    else:
        trainer.train(num_generations=args.generations)


if __name__ == "__main__":
    main()