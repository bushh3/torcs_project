#!/usr/bin/env python3
"""
TORCS CMA-ES Racing Driver Trainer
==================================

Works with VANILLA TORCS + scr_server (no modifications needed).

CRITICAL UNDERSTANDING:
-----------------------
Vanilla TORCS + scr_server does NOT properly support race restarts via meta=1.
When meta=1 is sent:
1. The server disconnects the client  
2. TORCS goes back to "waiting for client" screen
3. A NEW init message must be sent to reconnect

The gym_torcs library works around this by:
1. Creating a completely NEW Client object (new socket)
2. OR killing TORCS and using xautomation to restart via GUI

This trainer implements BOTH approaches:
- Mode 1 (default): Fresh client reconnection after meta=1
- Mode 2: Continuous driving without restarts (simpler, no reconnection issues)

Author: Enhanced for IBM AI Racing League
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import socket
import numpy as np
import torch
import torch.nn as nn
import cma
import time
import json
import traceback
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass


# ============================================================================
# CONTROL PARAMETERS
# ============================================================================

@dataclass
class ControlParams:
    """Tunable control parameters optimized by CMA-ES."""
    target_speed: float = 200.0
    corner_speed_factor: float = 0.6
    steer_angle_gain: float = 0.8
    steer_pos_gain: float = 0.15
    steer_lookahead: float = 0.5
    accel_gain: float = 1.0
    brake_gain: float = 1.0
    brake_angle_thresh: float = 0.4
    brake_dist_thresh: float = 80.0
    coast_zone: float = 0.1
    tcs_threshold: float = 5.0
    tcs_reduction: float = 0.2
    nn_steer_blend: float = 0.5
    nn_accel_blend: float = 0.3
    nn_brake_blend: float = 0.3
    gear_up_rpm: float = 8000.0
    gear_down_rpm: float = 3000.0
    gear1_max: float = 60.0
    gear2_max: float = 100.0
    gear3_max: float = 140.0
    gear4_max: float = 180.0
    gear5_max: float = 220.0
    stuck_speed: float = 5.0
    stuck_steps: int = 50
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.target_speed / 300.0,
            self.corner_speed_factor,
            self.steer_angle_gain,
            self.steer_pos_gain,
            self.steer_lookahead,
            self.accel_gain,
            self.brake_gain,
            self.brake_angle_thresh,
            self.brake_dist_thresh / 200.0,
            self.coast_zone,
            self.tcs_threshold / 20.0,
            self.tcs_reduction,
            self.nn_steer_blend,
            self.nn_accel_blend,
            self.nn_brake_blend,
            self.gear_up_rpm / 10000.0,
            self.gear_down_rpm / 10000.0,
            self.gear1_max / 100.0,
            self.gear2_max / 150.0,
            self.gear3_max / 200.0,
            self.gear4_max / 250.0,
            self.gear5_max / 300.0,
            self.stuck_speed / 20.0,
            self.stuck_steps / 100.0,
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlParams':
        return cls(
            target_speed=float(np.clip(arr[0] * 300.0, 80, 350)),
            corner_speed_factor=float(np.clip(arr[1], 0.3, 0.9)),
            steer_angle_gain=float(np.clip(arr[2], 0.2, 2.0)),
            steer_pos_gain=float(np.clip(arr[3], 0.05, 0.5)),
            steer_lookahead=float(np.clip(arr[4], 0.1, 0.9)),
            accel_gain=float(np.clip(arr[5], 0.5, 2.0)),
            brake_gain=float(np.clip(arr[6], 0.5, 2.0)),
            brake_angle_thresh=float(np.clip(arr[7], 0.1, 1.0)),
            brake_dist_thresh=float(np.clip(arr[8] * 200.0, 30, 150)),
            coast_zone=float(np.clip(arr[9], 0.0, 0.3)),
            tcs_threshold=float(np.clip(arr[10] * 20.0, 1.0, 15.0)),
            tcs_reduction=float(np.clip(arr[11], 0.05, 0.5)),
            nn_steer_blend=float(np.clip(arr[12], 0.0, 0.9)),
            nn_accel_blend=float(np.clip(arr[13], 0.0, 0.7)),
            nn_brake_blend=float(np.clip(arr[14], 0.0, 0.7)),
            gear_up_rpm=float(np.clip(arr[15] * 10000.0, 6000, 9500)),
            gear_down_rpm=float(np.clip(arr[16] * 10000.0, 2000, 5000)),
            gear1_max=float(np.clip(arr[17] * 100.0, 30, 80)),
            gear2_max=float(np.clip(arr[18] * 150.0, 60, 130)),
            gear3_max=float(np.clip(arr[19] * 200.0, 100, 170)),
            gear4_max=float(np.clip(arr[20] * 250.0, 140, 220)),
            gear5_max=float(np.clip(arr[21] * 300.0, 180, 280)),
            stuck_speed=float(np.clip(arr[22] * 20.0, 2.0, 15.0)),
            stuck_steps=int(np.clip(arr[23] * 100.0, 20, 100)),
        )
    
    @staticmethod
    def num_params() -> int:
        return 24


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class NeuralDriver(nn.Module):
    """Neural network for driving adjustments."""
    
    def __init__(self, n_inputs: int = 85, n_hidden1: int = 128, 
                 n_hidden2: int = 64, n_hidden3: int = 32):
        super().__init__()
        self.input_layer = nn.Linear(n_inputs, n_hidden1)
        self.hidden1 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden2 = nn.Linear(n_hidden2, n_hidden3)
        self.output_layer = nn.Linear(n_hidden3, 3)
        self.act = nn.Tanh()
        self._init_weights()
        
    def _init_weights(self):
        for layer in [self.input_layer, self.hidden1, self.hidden2, self.output_layer]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.input_layer(x))
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        return self.output_layer(x)
    
    def get_params(self) -> np.ndarray:
        return torch.cat([p.data.view(-1) for p in self.parameters()]).numpy()
    
    def set_params(self, params: np.ndarray):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.tensor(params[idx:idx+size], dtype=torch.float32).view(p.shape)
            idx += size
            
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# HYBRID DRIVER
# ============================================================================

class HybridDriver:
    """Combines neural network with tunable control parameters."""
    
    def __init__(self):
        self.net = NeuralDriver(n_inputs=85)
        self.params = ControlParams()
        self.nn_params = self.net.num_params()
        self.ctrl_params = ControlParams.num_params()
        self.total_params = self.nn_params + self.ctrl_params
        
        # State for temporal features
        self.prev_speed = 0.0
        self.prev_angle = 0.0
        self.prev_trackPos = 0.0
        self.stuck_counter = 0
        
    def reset_state(self):
        self.prev_speed = 0.0
        self.prev_angle = 0.0
        self.prev_trackPos = 0.0
        self.stuck_counter = 0
        
    def get_all_params(self) -> np.ndarray:
        return np.concatenate([self.net.get_params(), self.params.to_array()])
    
    def set_all_params(self, params: np.ndarray):
        self.net.set_params(params[:self.nn_params])
        self.params = ControlParams.from_array(params[self.nn_params:])
        
    def get_action(self, sensors: Dict) -> Dict:
        """Compute driving action from sensors."""
        x = self._encode_sensors(sensors)
        
        with torch.no_grad():
            out = self.net.forward(x)
        
        nn_steer = torch.tanh(out[0]).item()
        nn_accel = torch.sigmoid(out[1]).item()
        nn_brake = torch.sigmoid(out[2]).item()
        
        # Extract sensor values
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)
        trackPos = sensors.get('trackPos', 0)
        rpm = sensors.get('rpm', 0)
        gear = int(sensors.get('gear', 1))
        
        track = sensors.get('track', [200] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        while len(track) < 19:
            track.append(200)
            
        wheelSpinVel = sensors.get('wheelSpinVel', [0, 0, 0, 0])
        if isinstance(wheelSpinVel, (int, float)):
            wheelSpinVel = [wheelSpinVel] * 4
        while len(wheelSpinVel) < 4:
            wheelSpinVel.append(0)
        
        front_dist = track[9]
        left_dists = track[:9]
        right_dists = track[10:]
        
        # Stuck detection
        if abs(speed) < self.params.stuck_speed:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        is_stuck = self.stuck_counter > self.params.stuck_steps
        
        # === STEERING ===
        reactive_steer = (angle * self.params.steer_angle_gain) - (trackPos * self.params.steer_pos_gain)
        
        if len(left_dists) >= 4 and len(right_dists) >= 4:
            left_look = np.mean(left_dists[:4])
            right_look = np.mean(right_dists[-4:])
            look_diff = (right_look - left_look) / 200.0
            reactive_steer += look_diff * self.params.steer_lookahead
        
        reactive_steer = np.clip(reactive_steer, -1, 1)
        steer = (1 - self.params.nn_steer_blend) * reactive_steer + self.params.nn_steer_blend * nn_steer
        
        # === THROTTLE/BRAKE ===
        in_corner = abs(angle) > self.params.brake_angle_thresh or front_dist < self.params.brake_dist_thresh
        target = self.params.target_speed * (self.params.corner_speed_factor if in_corner else 1.0)
        
        if is_stuck:
            # Recovery
            if trackPos > 0.5:
                steer = -0.5
            elif trackPos < -0.5:
                steer = 0.5
            accel = 0.8
            brake = 0.0
        elif speed < 30:
            accel = 1.0 * self.params.accel_gain
            brake = 0.0
        elif speed < target - 10:
            deficit = (target - speed) / target
            accel = (0.7 + 0.3 * deficit) * self.params.accel_gain
            brake = 0.0
        elif speed > target + 10:
            excess = (speed - target) / target
            accel = 0.0
            brake = (0.3 + 0.3 * excess) * self.params.brake_gain
        else:
            accel = self.params.coast_zone
            brake = 0.0
            
        # Blend with NN
        accel = accel * (1 - self.params.nn_accel_blend) + nn_accel * self.params.nn_accel_blend
        brake = brake * (1 - self.params.nn_brake_blend) + nn_brake * self.params.nn_brake_blend
        
        # Traction control
        rear_spin = wheelSpinVel[2] + wheelSpinVel[3]
        front_spin = wheelSpinVel[0] + wheelSpinVel[1]
        if rear_spin - front_spin > self.params.tcs_threshold:
            accel -= self.params.tcs_reduction
            
        # Gear selection
        if rpm > self.params.gear_up_rpm and gear < 6:
            gear = min(gear + 1, 6)
        elif rpm < self.params.gear_down_rpm and gear > 1:
            gear = max(gear - 1, 1)
        else:
            if speed < self.params.gear1_max:
                gear = max(1, min(gear, 2))
            elif speed < self.params.gear2_max:
                gear = max(1, min(gear, 3))
            elif speed < self.params.gear3_max:
                gear = max(2, min(gear, 4))
            elif speed < self.params.gear4_max:
                gear = max(3, min(gear, 5))
            elif speed < self.params.gear5_max:
                gear = max(4, min(gear, 6))
            else:
                gear = max(5, 6)
                
        gear = max(1, min(6, gear))
        
        # Prevent pedal fighting
        if accel > 0.3:
            brake = 0.0
        if brake > 0.3:
            accel = 0.0
        
        steer = float(np.clip(steer, -1, 1))
        accel = float(np.clip(accel, 0, 1))
        brake = float(np.clip(brake, 0, 1))
        
        self.prev_speed = speed
        self.prev_angle = angle
        self.prev_trackPos = trackPos
        
        return {'steer': steer, 'accel': accel, 'brake': brake, 'gear': int(gear), 'clutch': 0.0}
    
    def _encode_sensors(self, S: Dict) -> torch.Tensor:
        """Encode all 85 sensor features."""
        features = []
        
        # Basic (10)
        features.append(S.get('speedX', 0) / 300.0)
        features.append(S.get('speedY', 0) / 50.0)
        features.append(S.get('speedZ', 0) / 50.0)
        features.append(S.get('angle', 0) / np.pi)
        features.append(S.get('trackPos', 0))
        features.append(S.get('z', 0.35) / 1.0)
        features.append(S.get('rpm', 0) / 10000.0)
        features.append(S.get('gear', 1) / 6.0)
        features.append(min(S.get('damage', 0), 10000) / 10000.0)
        features.append(S.get('fuel', 100) / 100.0)
        
        # Track (19)
        track = S.get('track', [200] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        for t in track[:19]:
            features.append(min(max(t, -1), 200) / 200.0)
        while len(features) < 29:
            features.append(1.0)
            
        # Wheels (4)
        wheelSpinVel = S.get('wheelSpinVel', [0, 0, 0, 0])
        if isinstance(wheelSpinVel, (int, float)):
            wheelSpinVel = [wheelSpinVel] * 4
        for w in wheelSpinVel[:4]:
            features.append(np.clip(w / 100.0, -1, 1))
        while len(features) < 33:
            features.append(0.0)
            
        # Derived (4)
        speedX = S.get('speedX', 0)
        speedY = S.get('speedY', 0)
        features.append(np.sqrt(speedX**2 + speedY**2) / 300.0)
        features.append(np.arctan2(speedY, max(speedX, 0.1)) / np.pi)
        if len(wheelSpinVel) >= 4:
            features.append(np.clip((wheelSpinVel[2] + wheelSpinVel[3] - wheelSpinVel[0] - wheelSpinVel[1]) / 50.0, -1, 1))
            features.append(np.clip((wheelSpinVel[0] + wheelSpinVel[2] - wheelSpinVel[1] - wheelSpinVel[3]) / 50.0, -1, 1))
        else:
            features.extend([0.0, 0.0])
            
        # Opponents (36)
        opponents = S.get('opponents', [200] * 36)
        if isinstance(opponents, (int, float)):
            opponents = [opponents] * 36
        for o in opponents[:36]:
            features.append(min(max(o, 0), 200) / 200.0)
        while len(features) < 73:
            features.append(1.0)
            
        # Focus (5)
        focus = S.get('focus', [200] * 5)
        if isinstance(focus, (int, float)):
            focus = [focus] * 5
        for f in focus[:5]:
            features.append(min(max(f, -1), 200) / 200.0)
        while len(features) < 78:
            features.append(1.0)
            
        # Temporal (7)
        features.append((S.get('speedX', 0) - self.prev_speed) / 50.0)
        features.append((S.get('angle', 0) - self.prev_angle) / 0.5)
        features.append((S.get('trackPos', 0) - self.prev_trackPos) / 0.5)
        features.append(min(S.get('curLapTime', 0), 300) / 300.0)
        features.append(min(S.get('lastLapTime', 0), 300) / 300.0)
        features.append(S.get('distFromStart', 0) / 5000.0)
        features.append(S.get('distRaced', 0) / 10000.0)
            
        while len(features) < 85:
            features.append(0.0)
            
        return torch.tensor(features[:85], dtype=torch.float32)
    
    def save(self, path: str):
        torch.save({
            'nn_state': self.net.state_dict(),
            'ctrl_params': self.params.__dict__,
            'version': 3,
        }, path)
        
    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.net.load_state_dict(data['nn_state'])
        if 'ctrl_params' in data:
            self.params = ControlParams(**data['ctrl_params'])


# ============================================================================
# TORCS CLIENT - Proper Implementation Based on gym_torcs
# ============================================================================

class TORCSClient:
    """
    TORCS SCR client based on snakeoil/gym_torcs patterns.
    
    Key insight: After meta=1, we need a COMPLETELY NEW socket connection.
    gym_torcs does this by creating a new Client() object each time.
    """
    
    ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
    DATA_SIZE = 2**17
    
    def __init__(self, host: str = 'localhost', port: int = 3001, 
                 sid: str = 'SCR', timeout: float = 1.0):
        self.host = host
        self.port = port
        self.sid = sid
        self.timeout = timeout
        self.so: Optional[socket.socket] = None
        self.S: Dict = {}  # Server state (sensors)
        self.R: Dict = self._default_action()  # Response (action)
        
    def _default_action(self) -> Dict:
        return {
            'accel': 0.2,
            'brake': 0,
            'steer': 0,
            'gear': 1,
            'clutch': 0,
            'focus': [-90, -45, 0, 45, 90],
            'meta': 0
        }
        
    def connect(self) -> bool:
        """
        Establish fresh connection to TORCS.
        Based on snakeoil setup_connection().
        """
        # Close existing socket if any
        if self.so:
            try:
                self.so.close()
            except:
                pass
            self.so = None
            
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.so.settimeout(self.timeout)
        except socket.error:
            print("    [Error: Could not create socket]")
            return False
            
        # Send init message and wait for identification
        init_msg = f'{self.sid}(init {self.ANGLES})'
        
        for attempt in range(10):
            try:
                self.so.sendto(init_msg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(self.DATA_SIZE)
                response = data.decode('utf-8')
                
                if '***identified***' in response:
                    print(f"    [Connected to TORCS on port {self.port}]")
                    return True
                    
            except socket.timeout:
                pass
            except socket.error:
                pass
            time.sleep(0.3)
            
        print("    [Failed to connect after 10 attempts]")
        return False
    
    def get_servers_input(self) -> bool:
        """
        Receive sensor data from TORCS.
        Based on snakeoil get_servers_input().
        Returns True if valid data received, False if disconnected.
        """
        if not self.so:
            return False
            
        try:
            data, _ = self.so.recvfrom(self.DATA_SIZE)
            response = data.decode('utf-8')
            
            if '***identified***' in response:
                # Re-identified, continue loop
                return True
                
            if '***shutdown***' in response:
                print("    [Server shutdown]")
                return False
                
            if '***restart***' in response:
                print("    [Server restart signal]")
                return False
                
            # Parse sensor data
            self._parse_sensors(response)
            return True
            
        except socket.timeout:
            return True  # Timeout is OK, just no data yet
        except socket.error:
            return False
            
    def _parse_sensors(self, data: str):
        """Parse TORCS sensor string format: (key value)(key value)..."""
        try:
            self.S = {}
            data = data.strip()
            if data.endswith(')'):
                data = data[:-1]
                
            for part in data.lstrip('(').split(')('):
                tokens = part.split(' ')
                key = tokens[0]
                values = tokens[1:]
                
                if len(values) == 1:
                    try:
                        self.S[key] = float(values[0])
                    except ValueError:
                        self.S[key] = values[0]
                elif len(values) > 1:
                    self.S[key] = [float(v) for v in values]
        except:
            pass
            
    def respond_to_server(self):
        """
        Send action to TORCS.
        Based on snakeoil respond_to_server().
        """
        if not self.so:
            return
            
        # Build message in TORCS format
        msg = ''
        for k, v in self.R.items():
            msg += f'({k} '
            if isinstance(v, list):
                msg += ' '.join(str(x) for x in v)
            elif isinstance(v, float):
                msg += f'{v:.3f}'
            else:
                msg += str(v)
            msg += ')'
            
        try:
            self.so.sendto(msg.encode(), (self.host, self.port))
        except:
            pass
            
    def request_restart(self) -> bool:
        """
        Request race restart via meta=1.
        
        IMPORTANT: After this, TORCS disconnects us!
        We must create a completely new connection.
        This is exactly what gym_torcs does.
        """
        print("    [Requesting restart via meta=1...]")
        
        # Send meta=1 to request restart
        self.R['meta'] = 1
        self.respond_to_server()
        time.sleep(0.1)
        self.respond_to_server()  # Send twice to be sure
        time.sleep(0.1)
        
        # Close our socket - TORCS is going to disconnect us anyway
        if self.so:
            try:
                self.so.close()
            except:
                pass
            self.so = None
            
        # Reset our state
        self.S = {}
        self.R = self._default_action()
        
        # Wait for TORCS to reset the race
        time.sleep(1.0)
        
        # Create fresh connection (this is what gym_torcs does!)
        print("    [Creating new connection...]")
        if not self.connect():
            print("    [Reconnection failed]")
            return False
            
        # Try to get first sensor reading to confirm race restarted
        for _ in range(20):
            if self.get_servers_input():
                if 'speedX' in self.S:
                    damage = self.S.get('damage', 0)
                    dist = self.S.get('distRaced', 0)
                    print(f"    [Restart OK - damage={damage:.0f}, dist={dist:.0f}]")
                    return True
            self.respond_to_server()
            time.sleep(0.1)
            
        print("    [Restart - no sensor data received]")
        return False
    
    def close(self):
        """Clean shutdown."""
        if self.so:
            try:
                self.so.close()
            except:
                pass
            self.so = None


# ============================================================================
# FITNESS EVALUATION
# ============================================================================

def evaluate(client: TORCSClient, driver: HybridDriver, 
             max_steps: int = 8000, track_length: float = 3600,
             start_distance: float = 0) -> Dict:
    """
    Evaluate a driver configuration.
    
    start_distance: If continuing from previous eval, the starting distance.
                   Used for continuous mode without restarts.
    """
    driver.reset_state()
    
    # Make sure we're ready
    client.R['meta'] = 0
    client.respond_to_server()
    
    # Wait for valid sensor data
    for _ in range(10):
        if client.get_servers_input() and 'speedX' in client.S:
            break
        client.respond_to_server()
        time.sleep(0.1)
    else:
        return {'fitness': -1000, 'distance': 0, 'avg_speed': 0, 'max_speed': 0,
                'laps': 0, 'steps': 0, 'damage': 0, 'skipped': True}
    
    # Get initial state
    initial_dist = client.S.get('distRaced', 0)
    initial_damage = client.S.get('damage', 0)
    
    # Tracking
    max_speed = 0
    steps = 0
    speed_samples = []
    laps_completed = 0
    last_dist_from_start = client.S.get('distFromStart', 0)
    
    total_steer_change = 0
    prev_steer = 0
    
    on_track_steps = 0
    off_track_steps = 0
    severe_off_steps = 0
    consecutive_severe = 0
    
    last_progress_dist = initial_dist
    no_progress_steps = 0
    
    corner_speeds = []
    
    # Main loop
    for step in range(max_steps):
        if not client.get_servers_input():
            print("    [Connection lost]")
            break
            
        S = client.S
        if not S or 'speedX' not in S:
            client.respond_to_server()
            continue
        
        speed = S.get('speedX', 0)
        trackPos = S.get('trackPos', 0)
        angle = S.get('angle', 0)
        dist = S.get('distRaced', 0)
        distFromStart = S.get('distFromStart', 0)
        damage = S.get('damage', 0)
        
        track = S.get('track', [200] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        front_dist = track[9] if len(track) > 9 else 200
        
        # Check for terminal damage
        if damage > 10000:
            print(f"    [Car totaled at {dist - initial_dist:.0f}m]")
            break
        
        # Lap detection
        if distFromStart < 100 and last_dist_from_start > track_length * 0.8:
            laps_completed += 1
            print(f"    LAP {laps_completed}!")
        last_dist_from_start = distFromStart
        
        # Progress check
        if dist > last_progress_dist + 10:
            last_progress_dist = dist
            no_progress_steps = 0
        else:
            no_progress_steps += 1
            
        if no_progress_steps > 400 and step > 200:
            print(f"    [No progress - stuck at {dist - initial_dist:.0f}m]")
            break
        
        # Track position scoring
        abs_pos = abs(trackPos)
        if abs_pos <= 1.0:
            on_track_steps += 1
            consecutive_severe = 0
        elif abs_pos <= 1.5:
            off_track_steps += 1
            consecutive_severe = 0
        else:
            severe_off_steps += 1
            consecutive_severe += 1
            if consecutive_severe > 200:
                print(f"    [Lost track at {dist - initial_dist:.0f}m]")
                break
        
        # Corner detection
        is_corner = abs(angle) > 0.15 or front_dist < 100
        if is_corner:
            corner_speeds.append(speed)
        
        speed_samples.append(speed)
        
        # Get action
        action = driver.get_action(S)
        
        # Track smoothness
        total_steer_change += abs(action['steer'] - prev_steer)
        prev_steer = action['steer']
        
        # Send action
        client.R['steer'] = action['steer']
        client.R['accel'] = action['accel']
        client.R['brake'] = action['brake']
        client.R['gear'] = action['gear']
        client.R['clutch'] = action.get('clutch', 0)
        client.R['meta'] = 0
        client.respond_to_server()
        
        max_speed = max(max_speed, speed)
        steps += 1
    
    # Calculate fitness
    distance = (client.S.get('distRaced', initial_dist) - initial_dist) if client.S else 0
    total_damage = (client.S.get('damage', initial_damage) - initial_damage) if client.S else 0
    
    if steps < 20:
        return {'fitness': -500, 'distance': 0, 'avg_speed': 0, 'max_speed': 0,
                'laps': 0, 'steps': 0, 'damage': 0}
    
    avg_speed = np.mean(speed_samples) if speed_samples else 0
    wobble = total_steer_change / steps
    avg_corner_speed = np.mean(corner_speeds) if corner_speeds else avg_speed * 0.6
    
    total_pos_steps = on_track_steps + off_track_steps + severe_off_steps
    on_track_pct = on_track_steps / total_pos_steps if total_pos_steps > 0 else 0
    
    # Fitness components
    distance_score = distance * 1.0
    speed_score = avg_speed * 1 + max_speed * 0.5 + avg_corner_speed * 0.5
    lap_score = laps_completed * 200
    
    clean_bonus = 300 if total_damage == 0 else (200 if total_damage < 100 else (100 if total_damage < 500 else 0))
    smooth_bonus = 100 if wobble < 0.1 else (50 if wobble < 0.2 else 0)
    
    off_track_penalty = off_track_steps * 0.3 + severe_off_steps * 2.0
    damage_penalty = total_damage * 0.1
    wobble_penalty = wobble * 50
    
    fitness = (distance_score + speed_score + lap_score + 
               clean_bonus + smooth_bonus - 
               off_track_penalty - damage_penalty - wobble_penalty)
    
    return {
        'fitness': fitness,
        'distance': distance,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'corner_speed': avg_corner_speed,
        'laps': laps_completed,
        'steps': steps,
        'wobble': wobble,
        'on_track_pct': on_track_pct * 100,
        'damage': total_damage,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(host: str = 'localhost', port: int = 3001,
          max_generations: int = 100, checkpoint: str = None,
          save_dir: str = "./checkpoints", popsize: int = 20,
          track_length: float = 3600, use_restart: bool = True):
    """
    Main CMA-ES training loop.
    
    use_restart: If True, restart race between evaluations (requires working meta=1).
                 If False, run continuous without restarts.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    driver = HybridDriver()
    
    if checkpoint and os.path.exists(checkpoint):
        driver.load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")
        
    print(f"\nTotal parameters: {driver.total_params}")
    print(f"  Neural network: {driver.nn_params}")
    print(f"  Control params: {driver.ctrl_params}")
    
    x0 = driver.get_all_params()
    
    print("\n" + "=" * 70)
    print("TORCS CMA-ES Racing Driver Trainer")
    print("=" * 70)
    print("\nSetup:")
    print("1. Start TORCS")
    print("2. Race -> Quick Race -> Configure Race")
    print("3. Select track, set MANY laps (e.g., 1000)")
    print("4. In Drivers, add 'scr_server 1' bot")
    print("5. Click 'New Race' - wait for 'Waiting for request on 3001'")
    print("")
    print(f"Server: {host}:{port}")
    print(f"Track length: {track_length}m")
    print(f"Restart mode: {'ON' if use_restart else 'OFF (continuous)'}")
    print("=" * 70)
    input("\nPress Enter to start training...")
    
    # Connect
    client = TORCSClient(host=host, port=port)
    
    print("\nConnecting to TORCS...")
    if not client.connect():
        print("Failed to connect!")
        return None
    
    # Verify connection
    for _ in range(10):
        if client.get_servers_input() and 'speedX' in client.S:
            print("Connection verified - receiving sensor data")
            break
        client.respond_to_server()
        time.sleep(0.1)
    
    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(x0, 0.3, {
        'popsize': popsize,
        'maxfevals': max_generations * popsize,
        'verb_disp': 0,
    })
    
    best_fitness = -np.inf
    best_params = x0.copy()
    gen = 0
    
    log_path = f"{save_dir}/training_log.json"
    training_log = []
    
    print(f"\nStarting training - {popsize} evaluations per generation")
    print("-" * 70)
    
    try:
        while not es.stop() and gen < max_generations:
            gen += 1
            gen_start = time.time()
            
            solutions = es.ask()
            fitnesses = []
            results = []
            
            for i, params in enumerate(solutions):
                driver.set_all_params(params)
                driver.reset_state()
                
                # Evaluate
                result = None
                for attempt in range(3):
                    try:
                        result = evaluate(client, driver, max_steps=8000, 
                                        track_length=track_length)
                        
                        if result.get('skipped') or result['fitness'] <= -900:
                            print(f"    [Retry {attempt+1}/3...]")
                            time.sleep(0.5)
                            if use_restart:
                                client.request_restart()
                            else:
                                client.connect()
                            continue
                        break
                    except Exception as e:
                        print(f"    [Error: {e}]")
                        time.sleep(0.5)
                        client.connect()
                
                if result is None or result.get('skipped') or result['fitness'] <= -900:
                    result = {'fitness': 0, 'distance': 0, 'avg_speed': 0,
                              'max_speed': 0, 'laps': 0, 'steps': 0, 'damage': 0}
                
                fitnesses.append(-result['fitness'])
                results.append(result)
                
                # Progress output
                laps_str = f"L{result.get('laps', 0)}" if result.get('laps', 0) > 0 else "  "
                dmg_str = f"dmg={result.get('damage', 0):4.0f}" if result.get('damage', 0) > 0 else "        "
                print(f"  [{i+1:2d}/{popsize}] "
                      f"dist={result['distance']:5.0f}m "
                      f"avg={result['avg_speed']:4.0f}km/h "
                      f"max={result['max_speed']:4.0f}km/h "
                      f"{dmg_str} {laps_str} "
                      f"fit={result['fitness']:7.0f}")
                
                # Request restart for next evaluation
                if use_restart and i < len(solutions) - 1:
                    if not client.request_restart():
                        print("    [Restart failed, reconnecting...]")
                        time.sleep(1.0)
                        client.connect()
            
            es.tell(solutions, fitnesses)
            
            # Track best
            best_idx = np.argmin(fitnesses)
            gen_best = -fitnesses[best_idx]
            
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_params = solutions[best_idx].copy()
                print(f"    *** NEW BEST: {best_fitness:.0f} ***")
                
            gen_time = time.time() - gen_start
            best_result = results[best_idx]
            
            # Log
            log_entry = {
                'gen': gen,
                'best_fitness': float(gen_best),
                'best_ever': float(best_fitness),
                'mean_fitness': float(-np.mean(fitnesses)),
                'best_distance': float(best_result['distance']),
                'best_speed': float(best_result['avg_speed']),
                'max_speed': float(best_result['max_speed']),
                'laps': int(best_result.get('laps', 0)),
                'damage': float(best_result.get('damage', 0)),
                'sigma': float(es.sigma),
                'time': gen_time,
            }
            training_log.append(log_entry)
            
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
            
            # Summary
            print(f"\n{'='*70}")
            print(f"Generation {gen}")
            print(f"  Best fitness:    {gen_best:.0f} (best ever: {best_fitness:.0f})")
            print(f"  Mean fitness:    {-np.mean(fitnesses):.0f}")
            print(f"  Best distance:   {best_result['distance']:.0f}m")
            print(f"  Speed avg/max:   {best_result['avg_speed']:.1f} / {best_result['max_speed']:.1f}")
            print(f"  Laps:            {best_result.get('laps', 0)}")
            print(f"  Damage:          {best_result.get('damage', 0):.0f}")
            print(f"  Sigma:           {es.sigma:.4f}")
            print(f"  Time:            {gen_time:.1f}s")
            print(f"{'='*70}\n")
            
            # Save
            driver.set_all_params(best_params)
            driver.save(f"{save_dir}/best.pt")
            
            if gen % 10 == 0:
                driver.save(f"{save_dir}/gen{gen}.pt")
                
            # Restart for next generation
            if use_restart:
                client.request_restart()
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        
    finally:
        driver.set_all_params(best_params)
        driver.save(f"{save_dir}/best.pt")
        driver.save(f"{save_dir}/final.pt")
        print(f"\nSaved: {save_dir}/best.pt")
        client.close()
    
    return driver


# ============================================================================
# RACE MODE
# ============================================================================

def race(checkpoint: str, host: str = 'localhost', port: int = 3001):
    """Run trained driver."""
    driver = HybridDriver()
    driver.load(checkpoint)
    print(f"Loaded: {checkpoint}")
    
    print("\nStart TORCS with scr_server, then press Enter...")
    input()
    
    client = TORCSClient(host=host, port=port)
    if not client.connect():
        print("Connection failed!")
        return
        
    print("Racing! Ctrl+C to stop.\n")
    
    step = 0
    driver.reset_state()
    
    try:
        while True:
            if not client.get_servers_input():
                time.sleep(0.1)
                continue
                
            S = client.S
            if not S or 'speedX' not in S:
                client.respond_to_server()
                continue
                
            action = driver.get_action(S)
            
            client.R['steer'] = action['steer']
            client.R['accel'] = action['accel']
            client.R['brake'] = action['brake']
            client.R['gear'] = action['gear']
            client.R['meta'] = 0
            client.respond_to_server()
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step}: speed={S.get('speedX', 0):.0f} "
                      f"dist={S.get('distRaced', 0):.0f} "
                      f"damage={S.get('damage', 0):.0f}")
                      
    except KeyboardInterrupt:
        print("\nRace ended")
    finally:
        client.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TORCS CMA-ES Trainer")
    parser.add_argument('--mode', choices=['train', 'race'], default='train')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--track_length', type=float, default=3600)
    parser.add_argument('--no_restart', action='store_true',
                        help='Disable restart between evaluations (continuous mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            host=args.host,
            port=args.port,
            max_generations=args.generations,
            checkpoint=args.checkpoint,
            save_dir=args.save_dir,
            popsize=args.popsize,
            track_length=args.track_length,
            use_restart=not args.no_restart
        )
    else:
        if not args.checkpoint:
            args.checkpoint = "./checkpoints/best.pt"
        race(args.checkpoint, args.host, args.port)