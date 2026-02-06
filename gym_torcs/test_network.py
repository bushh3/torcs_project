#!/usr/bin/env python3
"""
Test a trained TORCS racing network.

Usage:
    python test_network.py                           # Uses best.pt
    python test_network.py --params best_params.npy  # Uses .npy file
    python test_network.py --laps 3                  # Run 3 laps
    python test_network.py --manual                  # Manual gear control
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import socket
import time
from typing import Dict, List

# ============================================================================
# NETWORK (must match training)
# ============================================================================

class RacingPolicy(nn.Module):
    def __init__(self, state_dim: int = 65, action_dim: int = 2, 
                 hidden_sizes: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
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
            new_data = torch.FloatTensor(
                flat_params[idx:idx + param_size].reshape(param.shape)
            )
            param.data = new_data
            idx += param_size

# ============================================================================
# FEATURE EXTRACTION (must match training)
# ============================================================================

class FeatureExtractor:
    """
    Extracts and normalizes ALL available TORCS sensors.
    Total: 65 features (must match training!)
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
        
        # 14. Opponents (19) - sampled from 36
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
        
        # Update prev values
        self.prev_speedX = speedX
        self.prev_speedY = speedY
        
        return np.array(features, dtype=np.float32)

# ============================================================================
# TORCS CLIENT
# ============================================================================

class TORCSClient:
    def __init__(self, host='localhost', port=3001):
        self.host = host
        self.port = port
        self.so = None
        self.S = {}  # Sensors
        self.R = {}  # Response
        
    def connect(self) -> bool:
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1.0)
        
        # Initialize response
        self.R = {'accel': 0, 'brake': 0, 'gear': 1, 'steer': 0, 'meta': 0, 'focus': 0}
        
        # Send init
        angles = ' '.join([str(a) for a in range(-90, 91, 10)])
        init_msg = f"SCR(init -90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90)"
        
        for attempt in range(20):
            try:
                self.so.sendto(init_msg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(4096)
                if data:
                    self._parse_sensors(data.decode())
                    return True
            except socket.timeout:
                continue
        return False
    
    def _parse_sensors(self, msg: str):
        try:
            self.S = {}
            # Remove null bytes and clean
            msg = msg.replace('\x00', '').strip()
            if msg.endswith(')'): 
                msg = msg[:-1]
            for part in msg.lstrip('(').split(')('):
                tokens = part.split(' ')
                if len(tokens) < 2: 
                    continue
                key, values = tokens[0], tokens[1:]
                # Clean values - remove trailing parens
                clean_values = [v.rstrip(')') for v in values]
                if len(clean_values) == 1:
                    try: 
                        self.S[key] = float(clean_values[0])
                    except: 
                        self.S[key] = clean_values[0]
                else:
                    try:
                        self.S[key] = [float(v) for v in clean_values]
                    except:
                        pass
        except:
            pass
    
    def get_servers_input(self) -> bool:
        try:
            data, _ = self.so.recvfrom(4096)
            if data:
                msg = data.decode()
                if '***shutdown***' in msg or '***restart***' in msg:
                    return False
                self._parse_sensors(msg)
                return True
        except socket.timeout:
            pass
        return False
    
    def respond_to_server(self):
        msg = f"(accel {self.R['accel']:.3f})(brake {self.R['brake']:.3f})" \
              f"(gear {int(self.R['gear'])})(steer {self.R['steer']:.3f})" \
              f"(meta {int(self.R['meta'])})(focus 0)"
        try:
            self.so.sendto(msg.encode(), (self.host, self.port))
        except:
            pass
    
    def close(self):
        if self.so:
            self.so.close()

# ============================================================================
# MAIN TEST LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test trained TORCS network")
    parser.add_argument('--model', default='checkpoints_curriculum/best.pt',
                        help='Path to .pt model file')
    parser.add_argument('--params', default=None,
                        help='Path to .npy params file (alternative to --model)')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--track_length', type=float, default=3602)
    parser.add_argument('--laps', type=int, default=1, help='Number of laps to run')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer sizes')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='Steering smoothing (EMA factor, 0=off, 0.7=smooth)')
    args = parser.parse_args()
    
    # Load network
    policy = RacingPolicy(state_dim=65, action_dim=2, hidden_sizes=args.hidden)
    
    if args.params:
        print(f"Loading params from {args.params}...")
        params = np.load(args.params)
        policy.set_flat_params(params)
        print(f"  Params checksum: {params.sum():.6f}, mean: {params.mean():.6f}")
    else:
        print(f"Loading model from {args.model}...")
        policy.load_state_dict(torch.load(args.model, map_location='cpu'))
        # Compute checksum of loaded weights
        total = sum(p.sum().item() for p in policy.parameters())
        print(f"  Weights checksum: {total:.6f}")
    
    # Cross-verify: if both files exist, compare them
    import os
    npy_path = args.model.replace('best.pt', 'best_params.npy')
    if os.path.exists(npy_path) and os.path.exists(args.model):
        print(f"\nCross-verification:")
        npy_params = np.load(npy_path)
        pt_policy = RacingPolicy(state_dim=65, action_dim=2, hidden_sizes=args.hidden)
        pt_policy.load_state_dict(torch.load(args.model, map_location='cpu'))
        
        # Extract params from .pt
        pt_params = []
        for p in pt_policy.parameters():
            pt_params.extend(p.detach().numpy().flatten())
        pt_params = np.array(pt_params)
        
        diff = np.abs(npy_params - pt_params).max()
        print(f"  .npy sum: {npy_params.sum():.6f}")
        print(f"  .pt  sum: {pt_params.sum():.6f}")
        print(f"  Max diff: {diff:.10f}")
        if diff > 1e-6:
            print(f"  ⚠️  WARNING: Files don't match!")
    
    policy.eval()
    print(f"Network: {args.hidden}, {sum(p.numel() for p in policy.parameters())} params")
    
    # Connect
    print(f"\nConnecting to TORCS at {args.host}:{args.port}...")
    client = TORCSClient(host=args.host, port=args.port)
    if not client.connect():
        print("Failed to connect!")
        return
    print("Connected!")
    
    input("\nStart a race in TORCS, then press Enter...")
    
    # Feature extractor
    extractor = FeatureExtractor(track_length=args.track_length)
    
    # Run
    print(f"\nRunning for {args.laps} lap(s)...\n")
    
    # DIAGNOSTIC: Print first few network outputs to verify model
    print("Model verification (sample outputs):")
    test_inputs = [
        np.zeros(65, dtype=np.float32),  # All zeros
        np.ones(65, dtype=np.float32) * 0.5,  # All 0.5
    ]
    for i, inp in enumerate(test_inputs):
        with torch.no_grad():
            out = policy(torch.FloatTensor(inp).unsqueeze(0)).squeeze().numpy()
        print(f"  Input {i}: steer={out[0]:+.4f}, throttle={out[1]:+.4f}")
    print()
    
    # Warmup - match training exactly
    print("Warming up...", end=" ", flush=True)
    for _ in range(200):
        client.R['steer'] = 0.0
        client.R['accel'] = 1.0
        client.R['brake'] = 0.0
        client.R['gear'] = 1
        client.R['meta'] = 0
        client.respond_to_server()
        if client.get_servers_input():
            if client.S.get('gear', 0) > 0 or client.S.get('speedX', 0) > 1:
                break
        time.sleep(0.02)
    print(f"OK (speed={client.S.get('speedX', 0):.1f})")
    
    step = 0
    laps_completed = 0
    last_dist = 0
    speeds = []
    start_time = time.time()
    prev_steer = 0.0  # For smoothing
    
    try:
        while laps_completed < args.laps:
            state = extractor.extract(client.S)
            
            # Debug: print raw sensors at step 0
            if step == 0:
                print(f"\n  === STEP 0 DEBUG ===")
                print(f"  Raw sensors: angle={client.S.get('angle', 0):.4f}, trackPos={client.S.get('trackPos', 0):.4f}")
                print(f"  Speed: X={client.S.get('speedX', 0):.1f}, Y={client.S.get('speedY', 0):.1f}")
                print(f"  Feature vector first 10: {state[:10]}")
                print(f"  Feature vector sum: {state.sum():.4f}")
                print(f"  ===================\n")
            
            # Get action from network
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy(state_t).squeeze().numpy()
            
            steer = float(action[0])
            throttle = float(action[1])
            
            # Apply steering smoothing if enabled
            if args.smooth > 0:
                steer = args.smooth * prev_steer + (1 - args.smooth) * steer
                prev_steer = steer
            
            if throttle >= 0:
                accel = throttle
                brake = 0.0
            else:
                accel = 0.0
                brake = -throttle * 0.8
            
            # Auto gear
            rpm = client.S.get('rpm', 0)
            gear = int(client.S.get('gear', 1))
            speedX = client.S.get('speedX', 0)
            
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
            
            # Send to TORCS
            client.R['steer'] = steer
            client.R['accel'] = accel
            client.R['brake'] = brake
            client.R['gear'] = gear
            client.R['meta'] = 0
            client.respond_to_server()
            
            if not client.get_servers_input():
                print("\nConnection lost!")
                break
            
            # Track progress
            dist = client.S.get('distRaced', 0)
            speeds.append(speedX)
            
            # Lap detection
            if dist < last_dist - 1000:  # Crossed start line
                laps_completed += 1
                lap_time = time.time() - start_time
                print(f"  Lap {laps_completed}: {lap_time:.1f}s, avg speed: {np.mean(speeds):.1f} km/h")
                speeds = []
                start_time = time.time()
            last_dist = dist
            
            # Status every 100 steps (more frequent for debugging)
            if step % 100 == 0 or step < 5:
                trackPos = client.S.get('trackPos', 0)
                angle = client.S.get('angle', 0)
                print(f"  Step {step:5d}: dist={dist:6.0f}m  spd={speedX:5.1f}  "
                      f"pos={trackPos:+.2f}  ang={angle:+.3f}  st={steer:+.2f}  thr={throttle:+.2f}")
            
            step += 1
            
            # Safety check
            trackPos = client.S.get('trackPos', 0)
            if abs(trackPos) > 1.0:
                angle = client.S.get('angle', 0)
                print(f"\n  OFF TRACK at step {step}, dist={dist:.0f}m, pos={trackPos:+.2f}, ang={angle:+.3f}")
                print(f"  Last action: steer={steer:+.3f}, throttle={throttle:+.3f}")
                break
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        client.close()
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Steps: {step}")
    print(f"  Distance: {last_dist:.0f}m")
    print(f"  Laps completed: {laps_completed}")
    if speeds:
        print(f"  Avg speed: {np.mean(speeds):.1f} km/h")
        print(f"  Max speed: {max(speeds):.1f} km/h")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()