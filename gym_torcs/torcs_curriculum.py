#!/usr/bin/env python3
"""
TORCS Racing AI - Elitist Evolution + Imitation Learning
=========================================================

APPROACH:
Rule-based controllers (steering + throttle) handle driving initially.
Network gradually takes over BOTH controls simultaneously.
SAC trains with imitation loss to copy rule-based behavior.
Elitist ES: population = mutations of best individual.

CURRICULUM:
- Gen 1-4:     100% rule-based control (network observes)
- Gen 5-33:    IMITATION - rules→net blend, strong imitation loss
- Gen 34-71:   HANDOVER - imitation loss decays
- Gen 72-100:  FINE-TUNE - pure RL, no imitation
- Gen 101+:    100% network control

Rule-based controllers:
- Steering: Racing line with apex targeting
- Throttle: 200 km/h target, corner detection via asymmetry

CONFIGURATION:
- Network: [256, 256] = 83k params
- Population: 24
- Sigma: 0.02 (low for large network)
- Uses ALL 65 available sensors

SENSORS USED (65 total):
- angle (1): car angle relative to track
- track (19): distance to track edges at various angles  
- trackPos (1): lateral position on track
- speedX/Y/Z (3): velocity components
- wheelSpinVel (4): wheel angular velocities
- rpm (1): engine RPM
- gear (1): current gear
- distFromStart (1): position on track
- distRaced (1): total distance (log scale)
- damage (1): accumulated damage
- fuel (1): remaining fuel
- z (1): car height
- focus (5): high-precision forward distance
- opponents (19): distance to other cars
- curLapTime (1): current lap time
- racePos (1): race position
- accelX (1): estimated longitudinal acceleration
- accelY (1): estimated lateral acceleration
- absSpeed (1): total speed magnitude
- slipRatio (1): lateral/forward speed ratio (sliding indicator)

Author: AI Racing Research
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Network - BIGGER for more sensors
    HIDDEN_SIZES = [256, 256]
    STATE_DIM = 65  # ALL sensors!
    ACTION_DIM = 2  # [steer, accel_brake]
    
    # Evolution Strategy (OpenAI-ES style) - Memory efficient!
    POPULATION_SIZE = 24
    ES_SIGMA = 0.02         # Noise scale - lower for large networks (83k params)
    ES_LEARNING_RATE = 0.02 # Gradient step size
    
    # Fine-tuning mode settings
    FINE_TUNE_SIGMA = 0.005       # Very small mutations for fine-tuning
    FINE_TUNE_SIGMA_MIN = 0.002   # Minimum sigma
    FINE_TUNE_SIGMA_MAX = 0.02    # Maximum sigma
    STEER_SMOOTHING = 0.7         # EMA factor (0=no smoothing, 1=no change)
    
    # SAC
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.3
    MIN_ALPHA = 0.1
    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4
    LR_ALPHA = 3e-4
    
    # Buffer
    BUFFER_SIZE = 1_000_000    # Bigger buffer
    BATCH_SIZE = 256
    MIN_BUFFER_SIZE = 2000
    
    # Training
    SAC_UPDATES_PER_GEN = 200  # More SAC updates
    MAX_EPISODE_STEPS = 10000  # Long episodes for full laps
    EPISODE_TIMEOUT = 300      # 5 minutes max
    
    # Curriculum - unified handover from rules to network
    # Start blending early so CMA-ES can optimize, but strong imitation keeps network safe
    HANDOVER_START = 5    # Start blending at generation 5
    HANDOVER_END = 100    # Full network control by generation 100 (slow transition)
    
    # Racing
    MIN_SPEED_THRESHOLD = 5.0
    TARGET_SPEED = 120.0
    STUCK_STEPS_LIMIT = 60
    TRACK_LENGTH = 3602


# ============================================================================
# RULE-BASED CONTROLLERS (baseline for learning)
# ============================================================================

class RuleBasedSteering:
    """
    Racing line steering controller with apex targeting.
    - On straights: stay centered
    - Approaching corner: position to outside
    - In corner: cut toward apex
    - Exiting corner: track out
    """
    
    def __init__(self, kp=0.5, lookahead=2.5):
        self.kp = kp
        self.lookahead = lookahead
        
    def reset(self):
        pass
        
    def compute(self, sensors: Dict) -> float:
        trackPos = sensors.get('trackPos', 0.0)
        angle = sensors.get('angle', 0.0)
        speedX = max(sensors.get('speedX', 0.1), 0.1)
        
        # Get track sensors
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [200.0] * 19
        track = list(track)
        while len(track) < 19:
            track.append(200.0)
        
        # Analyze track shape
        # Sensors: 0=far left (-90°), 9=center (0°), 18=far right (+90°)
        left_dist = track[0]    # Far left
        right_dist = track[18]  # Far right
        forward_dist = track[9]  # Straight ahead
        
        # Forward-left and forward-right (about 20-30 degrees)
        fwd_left = (track[7] + track[8]) / 2   # ~20-30° left
        fwd_right = (track[10] + track[11]) / 2  # ~20-30° right
        
        # Detect corner direction and severity
        # Positive = left turn coming, Negative = right turn coming
        corner_bias = (fwd_right - fwd_left) / max(fwd_right + fwd_left, 1.0)
        
        # Is there actually a corner? Compare forward to sides
        is_corner = forward_dist < 100 and abs(corner_bias) > 0.15
        is_sharp_corner = forward_dist < 50 and abs(corner_bias) > 0.25
        
        # Calculate target position based on situation
        if is_sharp_corner:
            # Sharp corner: aim for apex (inside of corner)
            # Left turn (corner_bias > 0): target right side (negative trackPos)
            # Right turn (corner_bias < 0): target left side (positive trackPos)
            apex_target = -0.6 * np.sign(corner_bias)  # Cut toward inside
            target_pos = apex_target
        elif is_corner:
            # Moderate corner: slight apex cut
            apex_target = -0.4 * np.sign(corner_bias)
            target_pos = apex_target
        elif forward_dist > 150:
            # Long straight: stay centered
            target_pos = 0.0
        else:
            # Approaching corner: position toward outside for better entry
            # (opposite of apex - set up for the turn)
            if abs(corner_bias) > 0.1:
                target_pos = 0.3 * np.sign(corner_bias)  # Outside positioning
            else:
                target_pos = 0.0
        
        # Blend between racing line target and immediate correction needs
        # If we're badly positioned or angled, prioritize recovery
        position_urgency = abs(trackPos) + abs(angle) * 2
        
        if position_urgency > 0.8:
            # Emergency: just try to center and straighten
            target_pos = 0.0
            blend = 0.0  # Full correction mode
        else:
            blend = max(0, 1.0 - position_urgency)  # How much to trust racing line
        
        # Final target is blend of racing line and center
        effective_target = blend * target_pos + (1 - blend) * 0.0
        
        # Calculate error from target (not from center)
        position_error = trackPos - effective_target
        
        # Predict where car will be based on angle
        predicted_error = position_error - self.lookahead * angle
        
        # Steer to correct
        steer = -self.kp * predicted_error
        
        # Speed-based scaling
        if speedX < 15:
            speed_scale = 0.4 + 0.6 * (speedX / 15.0)
        elif speedX > 60:
            # More aggressive at high speed
            speed_scale = 1.0 + (speedX - 60) / 100.0
            speed_scale = min(1.5, speed_scale)
        else:
            speed_scale = 1.0
        
        steer *= speed_scale
        
        return np.clip(steer, -1.0, 1.0)


class RuleBasedThrottle:
    """
    Racing throttle controller - maximizes speed while respecting corners.
    
    Key improvements:
    - Detects ACTUAL corners vs walls/track edges
    - Higher target speed on straights (150 km/h)
    - Late braking into corners
    - Quick acceleration out of corners
    """
    
    def __init__(self, target_speed=200.0):
        self.target_speed = target_speed  # Higher straight-line target
        
    def reset(self):
        pass
        
    def compute(self, sensors: Dict) -> float:
        """
        Returns value in [-1, 1] where:
        - Positive = throttle
        - Negative = brake
        """
        speedX = sensors.get('speedX', 0.0)
        trackPos = sensors.get('trackPos', 0.0)
        angle = sensors.get('angle', 0.0)
        
        # Get track sensors
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [200.0] * 19
        track = list(track)
        while len(track) < 19:
            track.append(200.0)
        
        # Analyze track geometry
        forward_dist = track[9]  # Straight ahead
        
        # Forward cone (sensors 7-11, roughly ±20°)
        forward_cone = track[7:12]
        forward_min = min(forward_cone)
        forward_avg = sum(forward_cone) / len(forward_cone)
        
        # Side sensors for corner detection
        fwd_left = (track[6] + track[7] + track[8]) / 3   # 20-40° left
        fwd_right = (track[10] + track[11] + track[12]) / 3  # 20-40° right
        
        # Corner detection: is there asymmetry in forward sensors?
        # This distinguishes "corner ahead" from "narrow section"
        asymmetry = abs(fwd_left - fwd_right) / max(fwd_left + fwd_right, 1.0)
        
        # Track width indicator: can we see both sides?
        left_side = track[2]   # ~60° left
        right_side = track[16]  # ~60° right
        track_width_visible = min(left_side, right_side) > 5  # Can see track edges
        
        # Determine corner severity
        # Real corner: asymmetric + short forward distance
        # Straight with walls: symmetric or just one side blocked
        
        is_real_corner = asymmetry > 0.2 and forward_min < 80
        is_sharp_corner = asymmetry > 0.35 and forward_min < 50
        is_very_sharp = asymmetry > 0.5 or forward_min < 25
        
        # Calculate safe speed based on corner analysis
        if is_very_sharp:
            # Hairpin or very sharp - need to be slow
            safe_speed = 45 + forward_min * 0.5
        elif is_sharp_corner:
            # Sharp corner - brake hard but not emergency
            safe_speed = 55 + forward_min * 0.6
        elif is_real_corner:
            # Moderate corner - controlled braking
            safe_speed = 70 + forward_min * 0.5
        elif forward_min < 60:
            # Something ahead but not clearly a corner
            # Could be narrow section - moderate caution
            if asymmetry < 0.1:
                # Symmetric = probably straight, just narrow
                safe_speed = 90 + forward_min * 0.4
            else:
                # Some asymmetry - mild corner
                safe_speed = 75 + forward_min * 0.5
        elif forward_min < 100:
            # Gentle curve or distant corner
            safe_speed = 100 + forward_min * 0.3
        else:
            # Open road - FULL SPEED
            safe_speed = self.target_speed
        
        # Adjust for car state
        # Off-center: reduce speed proportionally
        if abs(trackPos) > 0.7:
            safe_speed *= 0.75  # Danger zone
        elif abs(trackPos) > 0.5:
            safe_speed *= 0.85
        elif abs(trackPos) > 0.3:
            safe_speed *= 0.95
        
        # Bad angle: reduce speed
        if abs(angle) > 0.3:
            safe_speed *= 0.7  # Very sideways
        elif abs(angle) > 0.15:
            safe_speed *= 0.85
        elif abs(angle) > 0.08:
            safe_speed *= 0.95
        
        # Minimum speed (don't crawl)
        safe_speed = max(40, min(safe_speed, self.target_speed))
        
        # Control logic - more aggressive
        speed_error = safe_speed - speedX
        
        if speed_error > 20:
            # Need significant acceleration
            return min(1.0, 0.6 + speed_error / 60.0)
        elif speed_error > 5:
            # Moderate acceleration
            return 0.4 + speed_error / 40.0
        elif speed_error > -5:
            # Close to target - maintain
            return 0.2 + speed_error / 20.0
        elif speed_error > -20:
            # Slightly too fast - light braking
            return speed_error / 40.0  # -0.5 to 0
        else:
            # Way too fast - HARD BRAKE
            return max(-1.0, speed_error / 25.0)


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class RacingPolicy(nn.Module):
    """
    Policy network for racing.
    Outputs: [steer, accel_brake]
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU for hidden layers
            prev_size = hidden_size
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, action_dim)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Output layer - small weights, biased toward forward
        nn.init.uniform_(self.output.weight, -0.03, 0.03)
        self.output.bias.data[0] = 0.0   # steer: center
        self.output.bias.data[1] = 0.8   # accel: forward (tanh(0.8) ≈ 0.66)
        
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


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC - also bigger."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        layers1 = []
        prev = input_dim
        for h in hidden_sizes:
            layers1.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers1.append(nn.Linear(prev, 1))
        self.q1 = nn.Sequential(*layers1)
        
        layers2 = []
        prev = input_dim
        for h in hidden_sizes:
            layers2.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers2.append(nn.Linear(prev, 1))
        self.q2 = nn.Sequential(*layers2)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================================
# EVOLUTION STRATEGY (Memory Efficient - OpenAI ES style)
# ============================================================================

class EvolutionStrategy:
    """
    OpenAI-style Evolution Strategy.
    
    Memory efficient: O(n) instead of O(n²) for CMA-ES.
    Works well with large networks (83k+ params).
    
    Key idea: Use shared random seeds to reconstruct perturbations
    instead of storing full covariance matrix.
    """
    
    def __init__(self, num_params: int, initial_mean: np.ndarray = None,
                 sigma: float = 0.02, learning_rate: float = 0.01,
                 population_size: int = 24):
        self.num_params = num_params
        self.sigma = sigma
        self.lr = learning_rate
        self.pop_size = population_size
        
        # Ensure even population for mirrored sampling
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        
        # Mean (the current best solution)
        if initial_mean is not None:
            self.mean = initial_mean.copy()
        else:
            self.mean = np.zeros(num_params)
        
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_params = None
        
        # Adaptive sigma bounds (conservative for 83k params)
        self.sigma_min = 0.005
        self.sigma_max = 0.1
        
    def ask(self, elitist: bool = True) -> List[np.ndarray]:
        """Generate population using mirrored sampling.
        
        If elitist=True, include best solution unchanged + mutations around it.
        Otherwise, mutations around the evolving mean.
        """
        population = []
        self.noise = []  # Store noise for gradient computation
        
        # Elitist: use best params as center if available
        if elitist and self.best_params is not None:
            center = self.best_params
            # ALWAYS include the elite unchanged as first individual
            population.append(self.best_params.copy())
        else:
            center = self.mean
        
        # Fill remaining population with mirrored mutations
        pairs_needed = (self.pop_size - len(population)) // 2
        for _ in range(pairs_needed):
            # Random perturbation
            eps = np.random.randn(self.num_params)
            self.noise.append(eps)
            
            # Mirrored sampling: +eps and -eps
            population.append(center + self.sigma * eps)
            population.append(center - self.sigma * eps)
        
        return population
    
    def tell(self, population: List[np.ndarray], fitnesses: List[float]) -> Dict:
        """Update mean using fitness-weighted gradient."""
        
        # Track best
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = population[best_idx].copy()
        
        # If elitist mode, first individual is the elite (unchanged)
        # Mirrored pairs start at index 1
        has_elite = len(population) > len(self.noise) * 2
        offset = 1 if has_elite else 0
        
        # Normalize fitnesses (rank-based)
        ranks = np.argsort(np.argsort(fitnesses))  # Rank transformation
        normalized = (ranks - (len(population) - 1) / 2) / (len(population) - 1)
        
        # Compute gradient estimate from mirrored pairs only
        gradient = np.zeros(self.num_params)
        for i, eps in enumerate(self.noise):
            # Mirrored sampling: offset+2*i uses +eps, offset+2*i+1 uses -eps
            idx_pos = offset + 2 * i
            idx_neg = offset + 2 * i + 1
            if idx_neg < len(normalized):
                fit_pos = normalized[idx_pos]
                fit_neg = normalized[idx_neg]
                gradient += (fit_pos - fit_neg) * eps
        
        gradient /= (2 * len(self.noise) * self.sigma + 1e-8)
        
        # Update mean with gradient
        self.mean += self.lr * gradient
        
        # ELITIST: Also pull mean toward best_params (50% blend)
        if self.best_params is not None:
            self.mean = 0.5 * self.mean + 0.5 * self.best_params
        
        # Adaptive sigma (increase if improving, decrease if stagnating)
        mean_fitness = np.mean(fitnesses)
        if self.generation > 0:
            if mean_fitness > self.prev_mean_fitness:
                self.sigma = min(self.sigma * 1.02, self.sigma_max)
            else:
                self.sigma = max(self.sigma * 0.98, self.sigma_min)
        self.prev_mean_fitness = mean_fitness
        
        self.generation += 1
        
        return {
            'best_fitness': self.best_fitness,
            'mean_fitness': mean_fitness,
            'sigma': self.sigma,
            'generation': self.generation,
        }


# Alias for compatibility
CMAES = EvolutionStrategy


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def add_trajectory(self, trajectory):
        for s, a, r, ns, d in trajectory:
            self.add(s, a, r, ns, d)
        
    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.FloatTensor(self.states[idx]),
            'actions': torch.FloatTensor(self.actions[idx]),
            'rewards': torch.FloatTensor(self.rewards[idx]),
            'next_states': torch.FloatTensor(self.next_states[idx]),
            'dones': torch.FloatTensor(self.dones[idx]),
        }
    
    def __len__(self):
        return self.size


# ============================================================================
# FEATURE EXTRACTOR - ALL 65 SENSORS
# ============================================================================

class FeatureExtractor:
    """
    Extracts and normalizes ALL available TORCS sensors.
    
    Total: 65 features
    - angle (1): car angle relative to track
    - track (19): distance sensors to track edges
    - trackPos (1): lateral position on track
    - speedX/Y/Z (3): velocity components
    - wheelSpinVel (4): wheel angular velocities
    - rpm (1): engine RPM
    - gear (1): current gear
    - distFromStart (1): distance from start line (normalized by track length)
    - distRaced (1): total distance (normalized)
    - damage (1): accumulated damage
    - fuel (1): remaining fuel
    - z (1): car height
    - focus (5): focused distance sensors
    - opponents (19): distance to opponents (sampled/compressed)
    - curLapTime (1): current lap time (normalized)
    - racePos (1): race position
    - accel (1): estimated longitudinal acceleration
    - lateralAccel (1): estimated lateral acceleration
    - absSpeedX (1): absolute forward speed
    """
    
    def __init__(self, track_length: float = 3602):
        self.track_length = track_length
        self.prev_speedX = 0.0
        self.prev_speedY = 0.0
        
    def extract(self, sensors: Dict) -> np.ndarray:
        features = []
        
        # 1. Angle (1)
        angle = sensors.get('angle', 0.0)
        features.append(angle / np.pi)
        
        # 2. Track sensors (19) - distance to track edge at various angles
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        track = list(track)[:19]
        while len(track) < 19:
            track.append(200.0)
        features.extend([np.clip(t, 0, 200) / 200.0 for t in track])
        
        # 3. Track position (1) - lateral position, -1 to +1
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
        
        # 7. Gear (1) - normalized 0-6 to 0-1
        gear = sensors.get('gear', 0)
        features.append(gear / 6.0)
        
        # 8. Distance from start (1)
        distFromStart = sensors.get('distFromStart', 0.0)
        features.append((distFromStart % self.track_length) / self.track_length)
        
        # 9. Distance raced (1) - log scale for large distances
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
        
        # 13. Focus sensors (5) - high precision forward distance
        focus = sensors.get('focus', [200.0] * 5)
        if isinstance(focus, (int, float)):
            focus = [focus] * 5
        focus = list(focus)[:5]
        while len(focus) < 5:
            focus.append(200.0)
        features.extend([np.clip(f, 0, 200) / 200.0 for f in focus])
        
        # 14. Opponents (19) - distance to nearest opponents at angles
        # These are 36 sensors, we sample 19 evenly
        opponents = sensors.get('opponents', [200.0] * 36)
        if isinstance(opponents, (int, float)):
            opponents = [opponents] * 36
        opponents = list(opponents)
        while len(opponents) < 36:
            opponents.append(200.0)
        # Sample 19 from 36
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
        
        # 17. Estimated acceleration (1) - from speed change
        accelX = (speedX - self.prev_speedX) * 50  # Assuming 50Hz
        features.append(np.clip(accelX, -50, 50) / 50.0)
        
        # 18. Lateral acceleration (1)
        accelY = (speedY - self.prev_speedY) * 50
        features.append(np.clip(accelY, -50, 50) / 50.0)
        
        # 19. Absolute speed (1)
        absSpeed = np.sqrt(speedX**2 + speedY**2)
        features.append(absSpeed / 300.0)
        
        # 20. Slip ratio - lateral/forward speed ratio (1)
        # Indicates sliding/drifting - useful for learning control
        slip_ratio = abs(speedY) / max(abs(speedX), 1.0)
        features.append(np.clip(slip_ratio, 0, 2) / 2.0)
        
        # Update prev values for acceleration calc
        self.prev_speedX = speedX
        self.prev_speedY = speedY
        
        # Verify we have exactly 65 features
        assert len(features) == 65, f"Expected 65 features, got {len(features)}"
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# REWARD FUNCTION
# ============================================================================

class RacingReward:
    """
    Reward function that adapts to network_weight (0-1).
    - Low network_weight: Focus on speed and progress (rule-based period)
    - High network_weight: Add steering quality penalties (network control period)
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
        self.prev_steer = 0.0  # For smoothness calculation
        self.steer_changes = []  # Track steering changes for stats
        
    def calculate(self, sensors: Dict, action: Dict, network_weight: float) -> Tuple[float, bool, Dict]:
        """
        Calculate reward based on network_weight (0 = all rules, 1 = all network).
        As network takes more control, quality metrics become more important.
        """
        speedX = sensors.get('speedX', 0.0)
        speedY = sensors.get('speedY', 0.0)
        angle = sensors.get('angle', 0.0)
        trackPos = sensors.get('trackPos', 0.0)
        distRaced = sensors.get('distRaced', 0.0)
        
        # Get current steering from action
        current_steer = action.get('steer', 0.0)
        
        self.step_count += 1
        self.total_speed += max(speedX, 0)
        self.max_speed = max(self.max_speed, speedX)
        
        reward = 0.0
        done = False
        info = {}
        
        # === CORE: Speed and Progress (always important) ===
        
        # Speed in forward direction
        reward += speedX * np.cos(angle) / 80.0
        
        # Progress reward
        progress = distRaced - self.prev_dist
        if progress > 0:
            reward += progress * 0.015
        
        # Speed bonus (quadratic, encourages going fast)
        if speedX > 40:
            reward += ((speedX - 40) / 80.0) ** 2 * 0.3
        
        # Top speed bonus
        if speedX > 100:
            reward += 0.1
        
        # === QUALITY METRICS: Scale with network_weight ===
        # As network takes control, penalize poor steering/control
        
        # Penalize being off-center (scales with network control)
        center_penalty = abs(trackPos) * 0.2 * network_weight
        reward -= center_penalty
        
        # Penalize lateral velocity / sliding (scales with network control)
        slide_penalty = abs(speedY) / 60.0 * network_weight
        reward -= slide_penalty
        
        # Strong centering when network has significant control
        if network_weight > 0.5:
            reward -= (trackPos ** 2) * 0.15 * network_weight
        
        # === STEERING SMOOTHNESS (scales with network_weight) ===
        # Penalize rapid steering changes - encourages smooth driving
        steer_change = abs(current_steer - self.prev_steer)
        self.steer_changes.append(steer_change)
        
        # Only penalize when network has control
        if network_weight > 0.3:
            # Quadratic penalty - small changes OK, large changes bad
            smoothness_penalty = (steer_change ** 2) * 2.0 * network_weight
            reward -= smoothness_penalty
        
        # Update previous steering
        self.prev_steer = current_steer
        
        # === TERMINATIONS (always apply) ===
        
        # Slow penalty
        if speedX < Config.MIN_SPEED_THRESHOLD:
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
        if self.stuck_counter > Config.STUCK_STEPS_LIMIT:
            done = True
            reward -= 5.0
            info['termination'] = 'stuck'
        
        # Lap bonus (big!)
        lap_dist = self.track_length * (self.lap_count + 1) - 50
        if distRaced >= lap_dist and self.prev_dist < lap_dist:
            self.lap_count += 1
            avg_speed = self.total_speed / max(self.step_count, 1)
            reward += 100.0 * (avg_speed / 100.0)  # Big lap bonus
            info['lap_completed'] = True
        
        self.prev_dist = distRaced
        reward = np.clip(reward, -10.0, 15.0)
        
        info['speed'] = speedX
        info['max_speed'] = self.max_speed
        info['avg_speed'] = self.total_speed / max(self.step_count, 1)
        info['distance'] = distRaced
        info['lap'] = self.lap_count
        
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
# CURRICULUM TRAINER
# ============================================================================

class CurriculumTrainer:
    """
    Unified Handover Trainer.
    
    Rule-based controllers (steering + throttle) handle driving initially.
    Network gradually takes over BOTH controls simultaneously using
    a smooth cosine blend from generation HANDOVER_START to HANDOVER_END.
    """
    
    def __init__(self, host='localhost', port=3001, save_dir='./checkpoints_curriculum',
                 track_length=3602, device='cpu', fine_tune=False):
        
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.fine_tune = fine_tune
        self.evaluating_elite = False  # Flag to disable noise for elite evaluation
        Config.TRACK_LENGTH = track_length
        
        self.feature_extractor = FeatureExtractor(track_length=track_length)
        self.rule_steering = RuleBasedSteering()
        self.rule_throttle = RuleBasedThrottle(target_speed=120.0)
        
        state_dim = Config.STATE_DIM
        action_dim = Config.ACTION_DIM
        
        # Create initial policy
        init_policy = RacingPolicy(state_dim, action_dim, Config.HIDDEN_SIZES)
        num_params = init_policy.get_num_params()
        initial_params = init_policy.get_flat_params()
        
        print(f"Network: {Config.HIDDEN_SIZES} hidden, {num_params} parameters")
        
        # Adjust sigma for fine-tuning
        sigma = Config.FINE_TUNE_SIGMA if fine_tune else Config.ES_SIGMA
        if fine_tune:
            print(f"FINE-TUNE MODE: sigma={sigma}, SAC disabled, pure ES")
        
        # Evolution Strategy (memory efficient)
        self.cmaes = CMAES(
            num_params=num_params,
            initial_mean=initial_params,
            sigma=sigma,
            learning_rate=Config.ES_LEARNING_RATE,
            population_size=Config.POPULATION_SIZE,
        )
        
        # Adjust sigma bounds for fine-tuning
        if fine_tune:
            self.cmaes.sigma_min = Config.FINE_TUNE_SIGMA_MIN
            self.cmaes.sigma_max = Config.FINE_TUNE_SIGMA_MAX
        
        # SAC
        self.sac_actor = RacingPolicy(state_dim, action_dim, Config.HIDDEN_SIZES).to(self.device)
        self.sac_critic = TwinQNetwork(state_dim, action_dim, Config.HIDDEN_SIZES).to(self.device)
        self.sac_critic_target = TwinQNetwork(state_dim, action_dim, Config.HIDDEN_SIZES).to(self.device)
        self.sac_critic_target.load_state_dict(self.sac_critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.sac_actor.parameters(), lr=Config.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.sac_critic.parameters(), lr=Config.LR_CRITIC)
        
        self.log_alpha = torch.tensor([np.log(Config.ALPHA)], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=Config.LR_ALPHA)
        self.alpha = Config.ALPHA
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, Config.BUFFER_SIZE)
        self.reward_calculator = RacingReward(track_length)
        
        self.client = None
        self.best_fitness = -np.inf
        self.best_distance = 0
        self.training_log = []
        
    def get_network_weight(self, generation: int) -> float:
        """
        How much network control vs rule-based control.
        Returns network_weight: 0 = all rules, 1 = all network
        
        Applies to BOTH steering and throttle simultaneously.
        """
        if generation < Config.HANDOVER_START:
            return 0.0  # All rule-based
        elif generation >= Config.HANDOVER_END:
            return 1.0  # All network
        else:
            # Smooth cosine blend
            progress = (generation - Config.HANDOVER_START) / (Config.HANDOVER_END - Config.HANDOVER_START)
            # Use cosine for smoother transition
            return 0.5 * (1 - np.cos(np.pi * progress))
    
    def evaluate_policy(self, policy: RacingPolicy, generation: int) -> Tuple[float, Dict, List]:
        """Evaluate policy with unified rule-to-network handover."""
        
        self.reward_calculator.reset()
        self.rule_steering.reset()
        self.rule_throttle.reset()
        self.prev_steer_smooth = 0.0  # For steering smoothing
        
        network_weight = self.get_network_weight(generation)
        
        if not self.client.warmup():
            return -100.0, {'reward': -100, 'steps': 0, 'avg_speed': 0, 
                           'max_speed': 0, 'distance': 0, 'laps': 0}, []
        
        state = self.feature_extractor.extract(self.client.S)
        
        total_reward = 0
        steps = 0
        speeds = []
        trajectory = []
        
        start_time = time.time()
        
        policy.eval()
        with torch.no_grad():
            while steps < Config.MAX_EPISODE_STEPS:
                if time.time() - start_time > Config.EPISODE_TIMEOUT:
                    break
                
                # Get network action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = policy(state_tensor)
                raw_action = action_tensor.cpu().numpy().flatten()
                
                # Add exploration noise (but NOT to elite individual)
                if self.evaluating_elite:
                    noise_scale = 0.0  # No noise for elite - pure evaluation
                elif self.fine_tune:
                    noise_scale = 0.01  # Very small noise for fine-tuning
                else:
                    noise_scale = 0.15 * (1 - network_weight) + 0.05
                
                if noise_scale > 0:
                    noise = np.random.normal(0, noise_scale, size=Config.ACTION_DIM)
                    raw_action = np.clip(raw_action + noise, -1, 1)
                
                # Get rule-based actions
                rule_steer = self.rule_steering.compute(self.client.S)
                rule_throttle = self.rule_throttle.compute(self.client.S)
                
                # Network outputs
                network_steer = raw_action[0]
                network_throttle = raw_action[1]  # In [-1, 1]
                
                # Blend based on network_weight
                steer = (1 - network_weight) * rule_steer + network_weight * network_steer
                throttle_blend = (1 - network_weight) * rule_throttle + network_weight * network_throttle
                
                # NO smoothing during training - let network learn raw control
                # Smoothness is enforced via reward penalty instead
                
                steer = float(np.clip(steer, -1, 1))
                
                # Convert blended throttle to accel/brake
                if throttle_blend >= 0:
                    accel = float(throttle_blend)
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = float(-throttle_blend * 0.8)
                
                # Debug for first steps only (minimal)
                if steps == 0:
                    pre_tpos = self.client.S.get('trackPos', 0)
                    pre_spd = self.client.S.get('speedX', 0)
                    action_str = f"ac={accel:.2f}" if accel > 0 else f"br={brake:.2f}"
                    print(f"st={steer:+.2f} {action_str}", end="", flush=True)
                
                # Auto gear - more aggressive downshifting for acceleration
                rpm = self.client.S.get('rpm', 0)
                gear = int(self.client.S.get('gear', 1))
                speedX = self.client.S.get('speedX', 0)
                
                if gear < 1: 
                    gear = 1
                elif rpm > 8000 and gear < 6: 
                    gear += 1  # Upshift at redline
                elif rpm < 5000 and gear > 1:
                    gear -= 1  # Downshift earlier to stay in power band
                
                # Also force downshift if we're way too slow for the gear
                # Rough speed thresholds per gear (km/h)
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
                
                # Reward (use network_weight instead of phase)
                next_state = self.feature_extractor.extract(self.client.S)
                reward, done, info = self.reward_calculator.calculate(self.client.S, action, network_weight)
                
                # Store ACTUAL actions sent to TORCS
                accel_brake = accel if accel > 0 else -brake
                actual_action = np.array([steer, accel_brake], dtype=np.float32)
                trajectory.append((state, actual_action, reward, next_state, done))
                
                total_reward += reward
                steps += 1
                speeds.append(info.get('speed', 0))
                
                state = next_state
                
                if done:
                    term = info.get('termination', 'done')
                    # Short termination indicator
                    if term == 'off_track':
                        print(" ✗", end="")
                    elif term == 'stuck':
                        print(" ⊘", end="")
                    elif term == 'timeout':
                        print(" ⏱", end="")
                    break
        
        result = {
            'reward': total_reward,
            'steps': steps,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'distance': self.client.S.get('distRaced', 0),
            'laps': self.reward_calculator.lap_count,
            'network_weight': network_weight,
        }
        
        return total_reward, result, trajectory
    
    def sac_update(self, batch_size: int = 256, imitation_weight: float = 0.0) -> Dict:
        """
        SAC update with optional imitation learning.
        imitation_weight: How much to penalize deviation from rule-based actions.
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)  # These are the ACTUAL actions taken (blended)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute rule-based actions for imitation learning
        if imitation_weight > 0:
            with torch.no_grad():
                rule_actions = []
                for i in range(states.shape[0]):
                    state = states[i].cpu().numpy()
                    # Reconstruct sensor dict from state
                    # State order: angle, track[19], trackPos, speedX, speedY, speedZ, ...
                    sensors = {
                        'angle': state[0] * np.pi if len(state) > 0 else 0,  # Denormalize
                        'trackPos': state[20] if len(state) > 20 else 0,
                        'speedX': state[21] * 300 if len(state) > 21 else 0,  # Denormalize
                        'speedY': state[22] * 50 if len(state) > 22 else 0,
                        'track': [state[j] * 200 for j in range(1, 20)] if len(state) > 19 else [200]*19,
                    }
                    rule_steer = self.rule_steering.compute(sensors)
                    rule_throttle = self.rule_throttle.compute(sensors)
                    rule_actions.append([rule_steer, rule_throttle])
                rule_actions = torch.FloatTensor(rule_actions).to(self.device)
        
        with torch.no_grad():
            next_actions = self.sac_actor(next_states)
            q1_target, q2_target = self.sac_critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_value = rewards + (1 - dones) * Config.GAMMA * q_target
        
        q1, q2 = self.sac_critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        new_actions = self.sac_actor(states)
        q1_new, q2_new = self.sac_critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Standard RL loss: maximize Q-value
        rl_loss = -q_new.mean()
        
        # Imitation loss: match rule-based actions
        if imitation_weight > 0:
            imitation_loss = F.mse_loss(new_actions, rule_actions)
            actor_loss = rl_loss + imitation_weight * imitation_loss
        else:
            actor_loss = rl_loss
            imitation_loss = torch.tensor(0.0)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.sac_critic.parameters(), self.sac_critic_target.parameters()):
            target_param.data.copy_(Config.TAU * param.data + (1 - Config.TAU) * target_param.data)
        
        result = {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
        if imitation_weight > 0:
            result['imitation_loss'] = imitation_loss.item()
        return result
    
    def train(self, num_generations: int = 500, start_generation: int = 1):
        print("\n" + "=" * 70)
        if self.fine_tune:
            print("TORCS Racing AI - FINE-TUNE MODE")
        else:
            print("TORCS Racing AI - ELITIST EVOLUTION + IMITATION LEARNING")
        print("=" * 70)
        print(f"Network: {Config.HIDDEN_SIZES} ({self.cmaes.num_params} params)")
        print(f"Population: {self.cmaes.pop_size}, Initial σ: {self.cmaes.sigma:.4f}")
        print(f"Device: {self.device}")
        if self.fine_tune:
            print("-" * 70)
            print("FINE-TUNE SETTINGS:")
            print(f"  • Sigma: {self.cmaes.sigma:.4f} (min={self.cmaes.sigma_min}, max={self.cmaes.sigma_max})")
            print(f"  • SAC: DISABLED (pure ES)")
            print(f"  • Exploration noise: 0 for elite, 0.01 for mutations")
            print(f"  • Steering smoothness penalty: enabled in reward")
        else:
            print("-" * 70)
            print("ALGORITHM:")
            print("  • Elitist ES: Population = mutations of best individual")
            print("  • SAC: Trains actor with imitation loss → blends back to ES")
            print("  • Imitation: Network learns to copy rule-based controller")
            print("-" * 70)
            print("CURRICULUM PHASES:")
            print(f"  Gen 1-{Config.HANDOVER_START-1}:       RULE-BASED (100% rules, imitation=5.0)")
            print(f"  Gen {Config.HANDOVER_START}-{int(Config.HANDOVER_START + (Config.HANDOVER_END-Config.HANDOVER_START)*0.3)}:      IMITATION   (rules→net, imitation=5.0)")
            print(f"  Gen {int(Config.HANDOVER_START + (Config.HANDOVER_END-Config.HANDOVER_START)*0.3)+1}-{int(Config.HANDOVER_START + (Config.HANDOVER_END-Config.HANDOVER_START)*0.7)}:     HANDOVER    (imitation decays 5.0→0)")
            print(f"  Gen {int(Config.HANDOVER_START + (Config.HANDOVER_END-Config.HANDOVER_START)*0.7)+1}-{Config.HANDOVER_END}:     FINE-TUNE   (imitation=0, network learning)")
            print(f"  Gen {Config.HANDOVER_END+1}+:       NETWORK     (100% network control)")
        if start_generation > 1:
            print("-" * 70)
            print(f"  *** RESUMING from generation {start_generation} ***")
            print(f"  *** Best fitness so far: {self.best_fitness:.1f} ***")
        print("=" * 70)
        
        print("\nConnecting to TORCS...")
        self.client = TORCSClient(host=self.host, port=self.port)
        if not self.client.connect():
            print("Failed to connect!")
            return
        print("Connected!")
        
        input("\nStart a race in TORCS, then press Enter...")
        
        gen = start_generation  # For checkpoint saving if interrupted early
        try:
            for gen in range(start_generation, num_generations + 1):
                gen_start = time.time()
                net_weight = self.get_network_weight(gen)
                
                # Determine imitation weight for display
                if net_weight < 0.3:
                    imitation_weight = 5.0
                elif net_weight < 0.7:
                    imitation_weight = 5.0 * (0.7 - net_weight) / 0.4
                else:
                    imitation_weight = 0.0
                
                # Phase description
                if net_weight == 0:
                    phase = "RULE-BASED"
                elif net_weight < 0.3:
                    phase = "IMITATION"
                elif net_weight < 0.7:
                    phase = "HANDOVER"
                elif net_weight < 1.0:
                    phase = "FINE-TUNE"
                else:
                    phase = "NETWORK"
                
                print(f"\n{'='*70}")
                print(f"GEN {gen:3d} | {phase:10s} | Net: {net_weight*100:5.1f}% | "
                      f"Imit: {imitation_weight:.1f} | σ: {self.cmaes.sigma:.4f}")
                print(f"        | Best so far: fit={self.best_fitness:.0f}, dist={self.best_distance:.0f}m")
                print(f"{'='*70}")
                
                population = self.cmaes.ask(elitist=True)
                fitnesses = []
                gen_speeds = []
                gen_distances = []
                
                for i, params in enumerate(population):
                    policy = RacingPolicy(Config.STATE_DIM, Config.ACTION_DIM, Config.HIDDEN_SIZES)
                    policy.set_flat_params(params)
                    policy.to(self.device)
                    
                    # Mark elite (first individual when best_params exists)
                    is_elite = (i == 0 and self.cmaes.best_params is not None)
                    self.evaluating_elite = is_elite  # Flag for no-noise evaluation
                    elite_marker = "E" if is_elite else " "
                    
                    # Print diagnostic for elite
                    if is_elite:
                        test_input = torch.zeros(1, Config.STATE_DIM)
                        with torch.no_grad():
                            test_out = policy(test_input).squeeze().numpy()
                        print(f"  [{i+1:2d}/{len(population)}]{elite_marker} [chk={params.sum():.2f}] ", end="", flush=True)
                    else:
                        print(f"  [{i+1:2d}/{len(population)}]{elite_marker}", end="", flush=True)
                    
                    fitness, result, trajectory = self.evaluate_policy(policy, gen)
                    self.evaluating_elite = False  # Reset flag
                    self.replay_buffer.add_trajectory(trajectory)
                    
                    fitnesses.append(fitness)
                    gen_speeds.append(result['avg_speed'])
                    gen_distances.append(result['distance'])
                    
                    # Compact result line
                    dist_str = f"{result['distance']:4.0f}m"
                    spd_str = f"{result['avg_speed']:5.1f}/{result['max_speed']:5.1f}"
                    fit_str = f"{fitness:8.1f}"
                    
                    # Check for new best and SAVE IMMEDIATELY
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_distance = max(self.best_distance, result['distance'])
                        # Save immediately AND sync to ES
                        torch.save(policy.state_dict(), self.save_dir / 'best.pt')
                        np.save(self.save_dir / 'best_params.npy', params)
                        # Also update ES best_params so elite is correct next gen
                        self.cmaes.best_params = params.copy()
                        self.cmaes.best_fitness = fitness
                        print(f" → fit={fit_str}  spd={spd_str}  dist={dist_str} ⭐ NEW BEST - SAVED!")
                    elif fitness > self.best_fitness * 0.9:
                        print(f" → fit={fit_str}  spd={spd_str}  dist={dist_str} ★")
                    else:
                        print(f" → fit={fit_str}  spd={spd_str}  dist={dist_str}")
                    
                    # Silent restart
                    if i < len(population) - 1:
                        if not self.client.request_restart():
                            self.client.close()
                            time.sleep(2)
                            self.client.connect()
                
                # Update Evolution Strategy
                info = self.cmaes.tell(population, fitnesses)
                
                # Generation summary
                gen_time = time.time() - gen_start
                best_idx = np.argmax(fitnesses)
                print(f"\n  ── GEN {gen} SUMMARY ({gen_time:.0f}s) ──")
                print(f"  Best:    fit={fitnesses[best_idx]:8.1f}  spd={gen_speeds[best_idx]:5.1f}  dist={gen_distances[best_idx]:5.0f}m")
                print(f"  Average: fit={info['mean_fitness']:8.1f}  spd={np.mean(gen_speeds):5.1f}  dist={np.mean(gen_distances):5.0f}m")
                print(f"  All-time: fit={self.best_fitness:8.1f}  dist={self.best_distance:5.0f}m")
                
                # SAC updates with imitation learning
                # (imitation_weight already calculated above)
                
                # Skip SAC in fine-tune mode - pure ES optimization
                if self.fine_tune:
                    print(f"SAC: DISABLED (fine-tune mode)")
                elif len(self.replay_buffer) >= Config.MIN_BUFFER_SIZE:
                    # Sync SAC actor FROM CMA-ES mean before training
                    self.sac_actor.set_flat_params(self.cmaes.mean)
                    
                    print(f"SAC: {Config.SAC_UPDATES_PER_GEN} updates (imitation={imitation_weight:.1f})...", end=" ", flush=True)
                    for _ in range(Config.SAC_UPDATES_PER_GEN):
                        self.sac_update(Config.BATCH_SIZE, imitation_weight=imitation_weight)
                    
                    # Copy SAC actor back TO CMA-ES mean (imitation-improved policy)
                    sac_params = self.sac_actor.get_flat_params()
                    blend = 0.5 if imitation_weight > 0 else 0.1
                    self.cmaes.mean = (1 - blend) * self.cmaes.mean + blend * sac_params
                    print(f"Done (blend={blend:.0%})")
                
                # Track best distance (fitness already tracked above)
                if np.mean(gen_distances) > self.best_distance:
                    self.best_distance = np.mean(gen_distances)
                
                # Log
                self.training_log.append({
                    'generation': gen,
                    'network_weight': net_weight,
                    'best_fitness': info['best_fitness'],
                    'mean_fitness': info['mean_fitness'],
                    'sigma': info['sigma'],
                    'avg_speed': np.mean(gen_speeds),
                    'max_speed': max(gen_speeds),
                    'avg_distance': np.mean(gen_distances),
                    'buffer_size': len(self.replay_buffer),
                })
                
                if gen % 5 == 0:
                    with open(self.save_dir / 'training_log.json', 'w') as f:
                        json.dump(self.training_log, f, indent=2)
                    np.save(self.save_dir / f'params_gen{gen}.npy', self.cmaes.mean)
                
                if not self.client.request_restart():
                    self.client.close()
                    time.sleep(2)
                    self.client.connect()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted - saving checkpoint...")
        finally:
            # Save full checkpoint for resume
            checkpoint = {
                'generation': gen,
                'mean': self.cmaes.mean,
                'sigma': self.cmaes.sigma,
                'best_params': self.cmaes.best_params,
                'best_fitness': self.best_fitness,
                'best_distance': self.best_distance,
            }
            np.savez(self.save_dir / 'checkpoint.npz', **checkpoint)
            
            with open(self.save_dir / 'training_log.json', 'w') as f:
                json.dump(self.training_log, f, indent=2)
            if self.cmaes.best_params is not None:
                np.save(self.save_dir / 'final_best.npy', self.cmaes.best_params)
            np.save(self.save_dir / 'final_mean.npy', self.cmaes.mean)
            print(f"\nSaved checkpoint to {self.save_dir}")
            print(f"  Resume with: --resume {self.save_dir / 'checkpoint.npz'}")
            if self.client:
                self.client.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TORCS Racing AI - Curriculum Learning")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--save_dir', default='./checkpoints_curriculum')
    parser.add_argument('--track_length', type=float, default=3602)
    parser.add_argument('--generations', type=int, default=500)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint or params file')
    parser.add_argument('--start_gen', type=int, default=None,
                        help='Override starting generation (use with --resume for old .npy files)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override ES sigma (noise scale). Lower for large networks (e.g. 0.02)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune mode: very low sigma, SAC disabled, smoothness reward')
    parser.add_argument('--reset_best', action='store_true',
                        help='Reset best_fitness to 0 (use when reward function changed)')
    
    args = parser.parse_args()
    
    start_generation = 1
    
    trainer = CurriculumTrainer(
        host=args.host,
        port=args.port,
        save_dir=args.save_dir,
        track_length=args.track_length,
        device=args.device,
        fine_tune=args.fine_tune,
    )
    
    if args.resume:
        print(f"\nLoading from {args.resume}...")
        if args.resume.endswith('.npz'):
            # Full checkpoint
            checkpoint = np.load(args.resume, allow_pickle=True)
            trainer.cmaes.mean = checkpoint['mean']
            trainer.cmaes.sigma = float(checkpoint['sigma'])
            if checkpoint['best_params'] is not None:
                trainer.cmaes.best_params = checkpoint['best_params']
                trainer.cmaes.best_fitness = float(checkpoint['best_fitness'])  # CRITICAL: sync ES best_fitness
                trainer.best_fitness = float(checkpoint['best_fitness'])
                trainer.best_distance = float(checkpoint['best_distance'])
            start_generation = int(checkpoint['generation']) + 1
            print(f"Resuming from generation {start_generation}")
        else:
            # Just params file (.npy)
            params = np.load(args.resume)
            trainer.cmaes.mean = params.copy()
            trainer.cmaes.best_params = params.copy()  # Use loaded as best
            trainer.cmaes.best_fitness = 0  # Sync ES best_fitness
            trainer.best_fitness = 0  # Will be updated on first good run
        
        trainer.sac_actor.set_flat_params(trainer.cmaes.mean)
        trainer.sac_actor = trainer.sac_actor.to(trainer.device)
        print(f"Loaded {len(trainer.cmaes.mean)} parameters")
    
    # Reset best fitness (use when reward function changed)
    if args.reset_best:
        trainer.best_fitness = -np.inf
        trainer.cmaes.best_fitness = -np.inf
        trainer.best_distance = 0
        print("Best fitness RESET (new reward function baseline)")
    
    # Override sigma if specified
    if args.sigma is not None:
        trainer.cmaes.sigma = args.sigma
        print(f"Sigma set to {args.sigma}")
    
    # Apply fine-tune settings AFTER loading checkpoint
    if args.fine_tune:
        trainer.cmaes.sigma = Config.FINE_TUNE_SIGMA
        trainer.cmaes.sigma_min = Config.FINE_TUNE_SIGMA_MIN
        trainer.cmaes.sigma_max = Config.FINE_TUNE_SIGMA_MAX
        print(f"Fine-tune sigma applied: {Config.FINE_TUNE_SIGMA}")
    
    # Manual override for starting generation
    if args.start_gen is not None:
        start_generation = args.start_gen
        print(f"Starting from generation {start_generation} (manual override)")
    
    trainer.train(num_generations=args.generations, start_generation=start_generation)