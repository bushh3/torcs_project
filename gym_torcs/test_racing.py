#!/usr/bin/env python3
"""
Test/Demo script - Watch your trained model race!

Usage:
    python test_racing.py --params ./checkpoints_hybrid/best_params.npy
    python test_racing.py --params ./checkpoints_hybrid/best_params.npy --laps 3
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import socket
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List


# ============================================================================
# NETWORK (must match training)
# ============================================================================

class RacingPolicy(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 2, 
                 hidden_sizes: List[int] = [64, 64]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, action_dim)
        
    def forward(self, state):
        x = self.hidden(state)
        return torch.tanh(self.output(x))
    
    def set_flat_params(self, flat_params: np.ndarray):
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = torch.FloatTensor(
                flat_params[idx:idx + param_size].reshape(param.shape)
            )
            idx += param_size


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    def extract(self, sensors: Dict) -> np.ndarray:
        angle = sensors.get('angle', 0.0)
        trackPos = sensors.get('trackPos', 0.0)
        speedX = sensors.get('speedX', 0.0)
        speedY = sensors.get('speedY', 0.0)
        speedZ = sensors.get('speedZ', 0.0)
        rpm = sensors.get('rpm', 0.0)
        
        track = sensors.get('track', [200.0] * 19)
        if isinstance(track, (int, float)):
            track = [track] * 19
        track = list(track)[:19]
        while len(track) < 19:
            track.append(200.0)
        
        wheelSpinVel = sensors.get('wheelSpinVel', [0.0] * 4)
        if isinstance(wheelSpinVel, (int, float)):
            wheelSpinVel = [wheelSpinVel] * 4
        wheelSpinVel = list(wheelSpinVel)[:4]
        while len(wheelSpinVel) < 4:
            wheelSpinVel.append(0.0)
        
        features = [angle / np.pi]
        features.extend([min(max(t, 0), 200) / 200.0 for t in track])
        features.append(np.clip(trackPos, -1.5, 1.5))
        features.extend([speedX / 300.0, speedY / 50.0, speedZ / 10.0])
        features.extend([wsv / 100.0 for wsv in wheelSpinVel])
        features.append(rpm / 10000.0)
        
        return np.array(features, dtype=np.float32)


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
    
    def close(self):
        if self.so:
            try: self.so.close()
            except: pass
            self.so = None


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_test(params_path: str, host: str = 'localhost', port: int = 3001,
             track_length: float = 3602, max_laps: int = 3):
    """Run the trained model and display live stats."""
    
    print("\n" + "=" * 60)
    print("TORCS Racing AI - Test Mode")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {params_path}...")
    params = np.load(params_path)
    policy = RacingPolicy()
    policy.set_flat_params(params)
    policy.eval()
    print(f"Loaded {len(params)} parameters")
    
    # Connect
    print(f"\nConnecting to TORCS at {host}:{port}...")
    client = TORCSClient(host=host, port=port)
    if not client.connect():
        print("Failed to connect! Make sure TORCS is running.")
        return
    print("Connected!")
    
    feature_extractor = FeatureExtractor()
    
    input("\nStart a race in TORCS, then press Enter to begin...")
    
    # Warmup
    print("Warming up...", end=" ", flush=True)
    for _ in range(100):
        client.R['accel'] = 1.0
        client.R['brake'] = 0.0
        client.R['gear'] = 1
        client.respond_to_server()
        if client.get_servers_input():
            if client.S.get('gear', 0) > 0:
                break
        time.sleep(0.02)
    print("Ready!")
    
    # Stats
    lap_count = 0
    prev_dist = 0
    total_speed = 0
    steps = 0
    lap_start_time = time.time()
    lap_times = []
    max_speed = 0
    
    print("\n" + "-" * 60)
    print("Racing! Press Ctrl+C to stop.")
    print("-" * 60)
    print(f"{'Lap':<6}{'Speed':<12}{'Max Spd':<12}{'Distance':<12}{'Lap Time':<12}")
    print("-" * 60)
    
    try:
        with torch.no_grad():
            while lap_count < max_laps:
                if not client.get_servers_input():
                    print("\nConnection lost!")
                    break
                
                # Extract state
                state = feature_extractor.extract(client.S)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action (no noise for testing)
                action = policy(state_tensor).numpy().flatten()
                
                steer = float(np.clip(action[0], -1, 1))
                if action[1] >= 0:
                    accel = float(action[1])
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = float(-action[1] * 0.8)
                
                # Auto gear
                rpm = client.S.get('rpm', 0)
                gear = int(client.S.get('gear', 1))
                if gear < 1: gear = 1
                elif rpm > 8000 and gear < 6: gear += 1
                elif rpm < 3000 and gear > 1: gear -= 1
                gear = max(1, min(6, gear))
                
                # Send action
                client.R['steer'] = steer
                client.R['accel'] = accel
                client.R['brake'] = brake
                client.R['gear'] = gear
                client.R['meta'] = 0
                client.respond_to_server()
                
                # Track stats
                speedX = client.S.get('speedX', 0)
                distRaced = client.S.get('distRaced', 0)
                trackPos = client.S.get('trackPos', 0)
                
                total_speed += speedX
                steps += 1
                max_speed = max(max_speed, speedX)
                
                # Check lap completion
                if distRaced >= track_length * (lap_count + 1) - 50:
                    if prev_dist < track_length * (lap_count + 1) - 50:
                        lap_time = time.time() - lap_start_time
                        lap_times.append(lap_time)
                        avg_speed = total_speed / max(steps, 1)
                        
                        print(f"{lap_count+1:<6}{avg_speed:<12.1f}{max_speed:<12.1f}{distRaced:<12.0f}{lap_time:<12.1f}")
                        
                        lap_count += 1
                        lap_start_time = time.time()
                        total_speed = 0
                        steps = 0
                        max_speed = 0
                
                prev_dist = distRaced
                
                # Live display every 50 steps
                if steps % 50 == 0:
                    avg_spd = total_speed / max(steps, 1)
                    print(f"\r  Current: {speedX:.1f} km/h | Avg: {avg_spd:.1f} km/h | "
                          f"Dist: {distRaced:.0f}m | Pos: {trackPos:.2f}    ", end="", flush=True)
                
                # Check off-track
                if abs(trackPos) > 1.15:
                    print(f"\n\n*** OFF TRACK at {distRaced:.0f}m! ***")
                    break
                
                time.sleep(0.02)  # ~50Hz
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        client.close()
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Laps completed: {lap_count}")
        if lap_times:
            print(f"Best lap time: {min(lap_times):.2f}s")
            print(f"Avg lap time: {np.mean(lap_times):.2f}s")
        print(f"Total distance: {prev_dist:.0f}m")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained TORCS racing model")
    parser.add_argument('--params', type=str, required=True,
                        help='Path to params .npy file (e.g., best_params.npy)')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--track_length', type=float, default=3602)
    parser.add_argument('--laps', type=int, default=3, help='Number of laps to attempt')
    
    args = parser.parse_args()
    
    run_test(
        params_path=args.params,
        host=args.host,
        port=args.port,
        track_length=args.track_length,
        max_laps=args.laps,
    )