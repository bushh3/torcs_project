#!/usr/bin/env python3
"""
Simple TORCS Race Client

A minimal script to run a trained driver in TORCS.
Can also run with default reactive-only control (no checkpoint needed).

Usage:
    python race_torcs.py                      # Run with default reactive controller
    python race_torcs.py --checkpoint best.pt # Run with trained model
"""

import socket
import numpy as np
import time
import argparse
from typing import Dict, Optional


class SimpleTORCSClient:
    """Minimal TORCS SCR client"""
    
    ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
    
    def __init__(self, host='localhost', port=3001, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.so = None
        self.S = {}  # Sensors
        self.R = {   # Response
            'accel': 0.2, 'brake': 0, 'steer': 0, 'gear': 1,
            'clutch': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0
        }
        
    def connect(self) -> bool:
        """Connect to TORCS server"""
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(self.timeout)
        
        init_msg = f'SCR(init {self.ANGLES})'
        
        for _ in range(10):
            try:
                self.so.sendto(init_msg.encode(), (self.host, self.port))
                data, _ = self.so.recvfrom(2**17)
                if '***identified***' in data.decode():
                    return True
            except socket.timeout:
                time.sleep(0.5)
        return False
    
    def step(self) -> bool:
        """Receive sensor data"""
        try:
            data, _ = self.so.recvfrom(2**17)
            response = data.decode()
            
            if '***shutdown***' in response or '***restart***' in response:
                return False
            if '***identified***' in response:
                return True
                
            # Parse sensors
            self.S = {}
            for part in response.strip().rstrip(')').lstrip('(').split(')('):
                tokens = part.split(' ')
                key = tokens[0]
                values = tokens[1:]
                if len(values) == 1:
                    try:
                        self.S[key] = float(values[0])
                    except:
                        self.S[key] = values[0]
                elif len(values) > 1:
                    self.S[key] = [float(v) for v in values]
            return True
            
        except socket.timeout:
            return True
        except:
            return False
    
    def send(self):
        """Send response to TORCS"""
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
    
    def close(self):
        if self.so:
            self.so.close()


def reactive_drive(S: Dict) -> Dict:
    """
    Simple reactive driving controller.
    No neural network, just hand-tuned rules.
    """
    speed = S.get('speedX', 0)
    angle = S.get('angle', 0)
    trackPos = S.get('trackPos', 0)
    rpm = S.get('rpm', 0)
    gear = S.get('gear', 1)
    
    # Track sensors
    track = S.get('track', [200] * 19)
    if isinstance(track, (int, float)):
        track = [track] * 19
    front = track[9] if len(track) > 9 else 200
    
    # Wheel spin for traction control
    wheelSpinVel = S.get('wheelSpinVel', [0, 0, 0, 0])
    if isinstance(wheelSpinVel, (int, float)):
        wheelSpinVel = [wheelSpinVel] * 4
    
    # === STEERING ===
    # Steer to correct angle and center on track
    steer = (angle * 10 / 3.14159) - (trackPos * 0.15)
    steer = np.clip(steer, -1, 1)
    
    # === TARGET SPEED ===
    # Reduce speed in corners
    in_corner = abs(angle) > 0.3 or front < 80
    target_speed = 120 if in_corner else 250
    
    # === THROTTLE/BRAKE ===
    if speed < target_speed - 20:
        accel = 0.8
        brake = 0.0
    elif speed > target_speed + 10:
        accel = 0.0
        brake = 0.3
    else:
        accel = 0.3
        brake = 0.0
        
    # Low speed boost
    if speed < 30:
        accel = 1.0
        brake = 0.0
        
    # Traction control
    if len(wheelSpinVel) >= 4:
        spin_diff = (wheelSpinVel[2] + wheelSpinVel[3]) - (wheelSpinVel[0] + wheelSpinVel[1])
        if spin_diff > 5:
            accel -= 0.2
            
    accel = np.clip(accel, 0, 1)
    brake = np.clip(brake, 0, 1)
    
    # Prevent pedal fighting
    if accel > 0.3:
        brake = 0.0
    
    # === GEAR ===
    if rpm > 8000 and gear < 6:
        gear += 1
    elif rpm < 3000 and gear > 1:
        gear -= 1
        
    return {
        'steer': float(steer),
        'accel': float(accel),
        'brake': float(brake),
        'gear': int(gear)
    }


def main():
    parser = argparse.ArgumentParser(description="TORCS Race Client")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    # Load trained driver if checkpoint provided
    driver = None
    if args.checkpoint:
        try:
            import torch
            from torcs_cmaes_trainer import HybridDriver
            driver = HybridDriver()
            driver.load(args.checkpoint)
            print(f"Loaded trained model: {args.checkpoint}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Falling back to reactive control")
            driver = None
    
    print("\n" + "="*50)
    print("TORCS Race Client")
    print("="*50)
    print("Setup:")
    print("1. Start TORCS")
    print("2. Race -> Quick Race -> Configure Race")
    print("3. Add 'scr_server 1' bot")
    print("4. Click 'New Race'")
    print("5. Press Enter here when ready")
    print("="*50)
    input()
    
    client = SimpleTORCSClient(host=args.host, port=args.port)
    
    print("Connecting...")
    if not client.connect():
        print("Failed to connect!")
        return
        
    print("Connected! Racing... (Ctrl+C to stop)\n")
    
    step = 0
    if driver:
        driver.reset_state()
    
    try:
        while True:
            if not client.step():
                time.sleep(0.1)
                continue
            
            S = client.S
            if 'speedX' not in S:
                continue
            
            # Get action
            if driver:
                action = driver.get_action(S)
            else:
                action = reactive_drive(S)
            
            # Send to TORCS
            client.R['steer'] = action['steer']
            client.R['accel'] = action['accel']
            client.R['brake'] = action['brake']
            client.R['gear'] = action['gear']
            client.R['meta'] = 0
            client.send()
            
            # Status output
            step += 1
            if step % 100 == 0:
                print(f"Step {step:5d} | "
                      f"Speed: {S.get('speedX', 0):5.0f} km/h | "
                      f"Dist: {S.get('distRaced', 0):6.0f}m | "
                      f"Pos: {S.get('trackPos', 0):+5.2f} | "
                      f"Gear: {S.get('gear', 1)} | "
                      f"Dmg: {S.get('damage', 0):.0f}")
                
    except KeyboardInterrupt:
        print("\n\nRace ended!")
    finally:
        client.close()


if __name__ == "__main__":
    main()
