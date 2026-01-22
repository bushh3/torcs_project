#!/usr/bin/env python3
"""
Training Progress Visualizer

Plots training metrics from the JSON log file.

Usage:
    python visualize_training.py                           # Default log location
    python visualize_training.py --log ./checkpoints/training_log.json
"""

import json
import argparse
import sys

def plot_training(log_path: str):
    """Plot training progress from log file"""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("\nAlternatively, here's a text summary:")
        text_summary(log_path)
        return
    
    # Load log
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return
    
    if not log:
        print("Log file is empty")
        return
    
    # Extract data
    generations = [entry['gen'] for entry in log]
    best_fitness = [entry['best_fitness'] for entry in log]
    best_ever = [entry['best_ever'] for entry in log]
    mean_fitness = [entry['mean_fitness'] for entry in log]
    distances = [entry['best_distance'] for entry in log]
    speeds = [entry['best_speed'] for entry in log]
    max_speeds = [entry.get('max_speed', 0) for entry in log]
    laps = [entry.get('laps', 0) for entry in log]
    damages = [entry.get('damage', 0) for entry in log]
    sigmas = [entry.get('sigma', 0) for entry in log]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('TORCS CMA-ES Training Progress', fontsize=14, fontweight='bold')
    
    # 1. Fitness over generations
    ax = axes[0, 0]
    ax.plot(generations, best_fitness, 'b-', label='Best (gen)', alpha=0.7)
    ax.plot(generations, best_ever, 'g-', label='Best ever', linewidth=2)
    ax.plot(generations, mean_fitness, 'r--', label='Mean', alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distance
    ax = axes[0, 1]
    ax.plot(generations, distances, 'b-', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Best Distance per Generation')
    ax.grid(True, alpha=0.3)
    
    # 3. Speed
    ax = axes[1, 0]
    ax.plot(generations, speeds, 'b-', label='Avg Speed', linewidth=2)
    ax.plot(generations, max_speeds, 'r--', label='Max Speed', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Speed Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Laps completed
    ax = axes[1, 1]
    ax.bar(generations, laps, color='green', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Laps')
    ax.set_title('Laps Completed')
    ax.grid(True, alpha=0.3)
    
    # 5. Damage
    ax = axes[2, 0]
    ax.plot(generations, damages, 'r-', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Damage')
    ax.set_title('Damage per Generation')
    ax.grid(True, alpha=0.3)
    
    # 6. CMA-ES Sigma
    ax = axes[2, 1]
    ax.plot(generations, sigmas, 'purple', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Sigma')
    ax.set_title('CMA-ES Step Size (Sigma)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and show
    output_path = log_path.replace('.json', '_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def text_summary(log_path: str):
    """Print text summary of training"""
    
    try:
        with open(log_path, 'r') as f:
            log = json.load(f)
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return
    
    if not log:
        print("Log file is empty")
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\nTotal generations: {len(log)}")
    
    # Best results
    best_entry = max(log, key=lambda x: x['best_ever'])
    print(f"\nBest Results (Generation {best_entry['gen']}):")
    print(f"  Fitness:    {best_entry['best_ever']:.0f}")
    print(f"  Distance:   {best_entry['best_distance']:.0f}m")
    print(f"  Avg Speed:  {best_entry['best_speed']:.1f} km/h")
    print(f"  Max Speed:  {best_entry.get('max_speed', 0):.1f} km/h")
    print(f"  Laps:       {best_entry.get('laps', 0)}")
    print(f"  Damage:     {best_entry.get('damage', 0):.0f}")
    
    # Progress
    first = log[0]
    last = log[-1]
    
    print(f"\nProgress (Gen 1 → Gen {last['gen']}):")
    print(f"  Fitness:    {first['best_fitness']:.0f} → {last['best_ever']:.0f}")
    print(f"  Distance:   {first['best_distance']:.0f}m → {last['best_distance']:.0f}m")
    print(f"  Avg Speed:  {first['best_speed']:.1f} → {last['best_speed']:.1f} km/h")
    
    # Stats
    all_fitness = [e['best_ever'] for e in log]
    all_distances = [e['best_distance'] for e in log]
    
    print(f"\nStatistics:")
    print(f"  Mean Best Fitness: {sum(all_fitness)/len(all_fitness):.0f}")
    print(f"  Mean Distance:     {sum(all_distances)/len(all_distances):.0f}m")
    print(f"  Final Sigma:       {last.get('sigma', 0):.4f}")
    
    # Time
    total_time = sum(e.get('time', 0) for e in log)
    print(f"\nTotal Training Time: {total_time/60:.1f} minutes")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize TORCS training progress")
    parser.add_argument('--log', type=str, default='./checkpoints/training_log.json',
                        help='Path to training log JSON file')
    parser.add_argument('--text', action='store_true',
                        help='Show text summary only (no plots)')
    args = parser.parse_args()
    
    if args.text:
        text_summary(args.log)
    else:
        plot_training(args.log)


if __name__ == "__main__":
    main()
