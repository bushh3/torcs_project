# TORCS CMA-ES Racing Driver Trainer

A fully automated trainer for TORCS (The Open Racing Car Simulator) that uses CMA-ES to optimize both a neural network and hand-tunable control parameters for autonomous driving.

## ⚠️ Critical: Race Restart Behavior

**Vanilla TORCS + scr_server does NOT properly support race restarts via the `meta=1` flag.**

When you send `meta=1`:
1. The scr_server disconnects your client
2. TORCS goes back to "waiting for client to connect" screen  
3. The race does NOT actually restart automatically

This is documented in gym_torcs README:
> "Because torcs has memory leak bug at race reset. As an ad-hoc solution, relaunch and automate the gui setting in torcs."

### Solution

This trainer properly handles restarts by:
1. Sending `meta=1` to signal restart request
2. **Closing the socket completely**
3. **Creating a fresh new socket connection** (like gym_torcs does)
4. Sending a new `init` message
5. Waiting for `***identified***` confirmation

If restarts still cause issues, use `--no_restart` flag for continuous mode (recommended for vanilla TORCS).

## Features

- **Hybrid Architecture**: Combines neural network learning with interpretable control parameters
- **Comprehensive Sensor Usage**: Uses ALL 85+ available TORCS sensors
  - 19 track edge sensors
  - 4 wheel spin velocities  
  - 36 opponent sensors (useful even in single-car mode)
  - RPM, speed, angle, position, and more
- **Full Parameter Optimization**: CMA-ES optimizes 25+ control parameters including:
  - Speed targets and corner factors
  - Steering gains and lookahead
  - Throttle/brake thresholds
  - Traction control settings
  - Gear shift RPM thresholds
  - Neural network blending weights
- **Proper Restart Handling**: Creates fresh socket connection after meta=1 (mimics gym_torcs)
- **Comprehensive Fitness Function**: Evaluates distance, speed, smoothness, damage, lap times

## Requirements

- Python 3.7+
- Vanilla TORCS with scr_server bot (comes with standard TORCS installation)
- PyTorch, NumPy, CMA-ES library

```bash
pip install -r requirements.txt
```

## TORCS Setup

### 1. Install TORCS

**Ubuntu/Debian:**
```bash
sudo apt-get install torcs
```

**Arch Linux:**
```bash
sudo pacman -S torcs
```

**macOS (via Homebrew):**
```bash
brew install torcs
```

### 2. Configure TORCS for Training

1. Start TORCS:
   ```bash
   torcs
   ```

2. Go to: **Race → Quick Race → Configure Race**

3. Select your track (e.g., "e-track-1", "wheel-1", "forza")
   - Note the track length for the `--track_length` parameter

4. Set number of laps (e.g., 100 for training)

5. Click **"Drivers"**:
   - Find **"scr_server 1"** in the list (usually under "Robots")
   - Click **">> (Accept)"** to add it to selected drivers
   - Remove any human players

6. Click **"New Race"** to start

7. The screen will show "scr_server: Waiting for request on 3001" - this is normal!

8. Now run the trainer (see below)

## Usage

### Training Mode

```bash
# Basic training (with restart between evaluations)
python torcs_cmaes_trainer.py --mode train

# Continuous mode (no restart - recommended for vanilla TORCS)
python torcs_cmaes_trainer.py --mode train --no_restart

# With custom settings
python torcs_cmaes_trainer.py --mode train \
    --generations 200 \
    --popsize 30 \
    --track_length 3185 \
    --save_dir ./my_checkpoints

# Resume from checkpoint
python torcs_cmaes_trainer.py --mode train \
    --checkpoint ./checkpoints/best.pt \
    --generations 100
```

**Note**: If restarts aren't working (TORCS disconnects after each evaluation), use `--no_restart` for continuous mode.

### Racing Mode (Test Trained Model)

```bash
# Use best checkpoint
python torcs_cmaes_trainer.py --mode race --checkpoint ./checkpoints/best.pt

# Or just (uses default checkpoint location)
python torcs_cmaes_trainer.py --mode race
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | train | 'train' or 'race' |
| `--host` | localhost | TORCS server host |
| `--port` | 3001 | TORCS SCR server port |
| `--generations` | 100 | Maximum training generations |
| `--checkpoint` | None | Path to checkpoint to load/resume |
| `--save_dir` | ./checkpoints | Directory for saving checkpoints |
| `--popsize` | 20 | CMA-ES population size |
| `--track_length` | 3600 | Track length in meters |

## Track Lengths Reference

| Track | Approximate Length (m) |
|-------|----------------------|
| e-track-1 | 3185 |
| e-track-2 | 3260 |
| e-track-3 | 3260 |
| e-track-4 | 4010 |
| wheel-1 | 1490 |
| wheel-2 | 2090 |
| forza | 5784 |
| cg-track-1 | 2000 |
| alpine-1 | 3774 |

## Output Files

The trainer creates several files in `save_dir`:

- `best.pt` - Best model found so far (updated whenever a new best is found)
- `final.pt` - Final model when training completes
- `gen{N}.pt` - Checkpoint every 10 generations
- `training_log.json` - Complete training history with fitness, speed, distance metrics

## Sensor Information

The neural network receives 85 input features:

### Basic State (10 features)
- `speedX`, `speedY`, `speedZ` - 3D velocity
- `angle` - Car angle relative to track
- `trackPos` - Position on track (-1 to 1)
- `z` - Height above track
- `rpm` - Engine RPM
- `gear` - Current gear
- `damage` - Cumulative damage
- `fuel` - Fuel level

### Track Sensors (19 features)
Distance to track edge at angles from -45° to +45°

### Wheel Spin (4 features)
Rotational velocity of each wheel (rad/s)

### Opponents (36 features)
Distance to opponents at angles around the car

### Focus (5 features)
Configurable distance sensors

### Derived/Temporal (11 features)
Speed magnitude, drift angle, wheel slip, acceleration, etc.

## Control Parameters Optimized

The CMA-ES algorithm optimizes 25 control parameters including:

- **Speed Control**: Target speed, corner speed factor
- **Steering**: Angle gain, position gain, lookahead
- **Throttle/Brake**: Gains, thresholds, coast zone
- **Traction Control**: Threshold, reduction amount
- **Neural Blending**: How much the NN influences each output
- **Gear Shifting**: RPM thresholds, speed limits per gear
- **Recovery**: Stuck detection thresholds

## Troubleshooting

### "Connection failed" or timeout
- Make sure TORCS is running with scr_server bot active
- Check that port 3001 is not blocked by firewall
- Try increasing timeout in the code

### "No progress" during evaluation
- Car might be stuck - the trainer will restart automatically
- If persistent, try different initial parameters or smaller sigma

### Training not improving
- Try larger population size (`--popsize 30`)
- Run for more generations
- Try different track (some are harder)

### Memory issues
- Reduce population size
- Reduce network size (modify n_hidden parameters)

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    HybridDriver                               │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │  NeuralDriver   │    │    ControlParams             │    │
│  │  (85 → 128 →   │    │    (25 tunable params)       │    │
│  │   64 → 32 → 3) │    │    - Speed targets           │    │
│  │                 │    │    - Steering gains          │    │
│  │  Outputs:       │    │    - Throttle/brake          │    │
│  │  - steer_adj   │    │    - Traction control        │    │
│  │  - accel_adj   │    │    - Gear shifting           │    │
│  │  - brake_adj   │    │    - Recovery params         │    │
│  └─────────────────┘    └──────────────────────────────┘    │
│           │                        │                         │
│           └────────┬───────────────┘                         │
│                    ▼                                         │
│           ┌─────────────────┐                               │
│           │  Blended Output  │                               │
│           │  steer, accel,   │                               │
│           │  brake, gear     │                               │
│           └─────────────────┘                               │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   CMA-ES Optimizer     │
            │   Optimizes ALL params │
            │   (NN weights + ctrl)  │
            └────────────────────────┘
```

## Tips for Best Results

1. **Start with shorter tracks** - wheel-1 and wheel-2 are good for initial training
2. **Use appropriate track_length** - This affects lap detection and progress tracking
3. **Watch the fitness trend** - Should generally increase over generations
4. **Save checkpoints often** - Training can be interrupted and resumed
5. **Try different population sizes** - Larger (30-50) can find better solutions but is slower

## License

MIT License - Feel free to use and modify for your own projects.

## Acknowledgments

- TORCS developers and community
- CMA-ES library by Nikolaus Hansen
- Original snakeoil.py client by Chris X Edwards
- gym-torcs project for reference implementation

python torcs_cmaes_trainer.py --mode train --track_length 3608.45 --checkpoint ./checkpoints/best.pt