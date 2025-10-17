#!/usr/bin/env python3
"""
Visualize Face Consistency Results

This script creates visualizations of face consistency metrics across training steps.

Usage:
    python visualize_face_consistency.py \
        --metrics_dir path/to/output \
        --output_dir visualizations
"""

import argparse
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_metrics_files(metrics_dir: str) -> Dict[int, dict]:
    """
    Load all face consistency metric files from a directory.
    
    Returns:
        Dictionary mapping step number to metrics
    """
    pattern = os.path.join(metrics_dir, "face_metrics_*.json")
    files = glob.glob(pattern)
    
    if not files:
        # Also try the pattern from evaluate script
        pattern = os.path.join(metrics_dir, "face_consistency_*.json")
        files = glob.glob(pattern)
    
    logger.info(f"Found {len(files)} metric files")
    
    metrics_by_step = {}
    
    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Extract metrics per step
            if 'by_step' in data:
                for step_str, metrics in data['by_step'].items():
                    if step_str.isdigit():
                        step = int(step_str)
                        if step not in metrics_by_step:
                            metrics_by_step[step] = []
                        metrics_by_step[step].append(metrics)
            
            # Also store overall metrics
            if 'overall_mean' in data:
                # Try to infer step from filename
                filename = Path(file_path).stem
                for part in filename.split('_'):
                    if part.isdigit():
                        step = int(part)
                        if step not in metrics_by_step:
                            metrics_by_step[step] = []
                        metrics_by_step[step].append({
                            'mean': data['overall_mean'],
                            'std': data.get('overall_std', 0),
                            'min': data.get('overall_min', 0),
                            'max': data.get('overall_max', 1),
                            'count': data.get('valid_pairs', 0),
                        })
                        break
                        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Average metrics if multiple files for same step
    aggregated = {}
    for step, metrics_list in metrics_by_step.items():
        if len(metrics_list) == 1:
            aggregated[step] = metrics_list[0]
        else:
            # Average across multiple measurements
            import numpy as np
            aggregated[step] = {
                'mean': np.mean([m['mean'] for m in metrics_list]),
                'std': np.mean([m['std'] for m in metrics_list]),
                'min': np.min([m['min'] for m in metrics_list]),
                'max': np.max([m['max'] for m in metrics_list]),
                'count': int(np.mean([m['count'] for m in metrics_list])),
            }
    
    return aggregated


def plot_face_similarity_over_time(
    metrics_by_step: Dict[int, dict],
    output_path: str,
):
    """Plot face similarity progression over training steps."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    steps = sorted(metrics_by_step.keys())
    means = [metrics_by_step[s]['mean'] for s in steps]
    stds = [metrics_by_step[s]['std'] for s in steps]
    
    plt.figure(figsize=(12, 6))
    
    # Plot mean with error bars
    plt.errorbar(steps, means, yerr=stds, marker='o', capsize=5, 
                 label='Mean ± Std', linewidth=2, markersize=6)
    
    # Add trend line
    if len(steps) > 2:
        z = np.polyfit(steps, means, 2)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "--", alpha=0.5, label='Trend', linewidth=2)
    
    # Add reference lines
    plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.3, label='Target (0.85)')
    plt.axhline(y=0.75, color='orange', linestyle='--', alpha=0.3, label='Good (0.75)')
    plt.axhline(y=0.65, color='r', linestyle='--', alpha=0.3, label='Needs Improvement (0.65)')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Face Similarity', fontsize=12)
    plt.title('Face Consistency Over Training', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {output_path}")
    plt.close()


def plot_similarity_distribution(
    metrics_by_step: Dict[int, dict],
    output_path: str,
):
    """Plot distribution of similarities across training."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    
    steps = sorted(metrics_by_step.keys())
    
    # Select key steps to show (first, middle, last)
    if len(steps) > 6:
        selected_steps = [
            steps[0],
            steps[len(steps)//4],
            steps[len(steps)//2],
            steps[3*len(steps)//4],
            steps[-1]
        ]
    else:
        selected_steps = steps
    
    fig, axes = plt.subplots(1, len(selected_steps), figsize=(4*len(selected_steps), 4))
    if len(selected_steps) == 1:
        axes = [axes]
    
    for idx, step in enumerate(selected_steps):
        metrics = metrics_by_step[step]
        mean = metrics['mean']
        std = metrics['std']
        
        # Create approximate distribution
        x = np.linspace(max(0, mean - 3*std), min(1, mean + 3*std), 100)
        y = np.exp(-0.5 * ((x - mean) / (std + 1e-6))**2)
        
        axes[idx].fill_between(x, y, alpha=0.6)
        axes[idx].axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
        axes[idx].set_xlabel('Face Similarity')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'Step {step}')
        axes[idx].set_xlim(0, 1)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Face Similarity Distribution Across Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot to: {output_path}")
    plt.close()


def plot_min_max_range(
    metrics_by_step: Dict[int, dict],
    output_path: str,
):
    """Plot min/max range of similarities over training."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    steps = sorted(metrics_by_step.keys())
    means = [metrics_by_step[s]['mean'] for s in steps]
    mins = [metrics_by_step[s]['min'] for s in steps]
    maxs = [metrics_by_step[s]['max'] for s in steps]
    
    plt.figure(figsize=(12, 6))
    
    # Plot range
    plt.fill_between(steps, mins, maxs, alpha=0.3, label='Min-Max Range')
    plt.plot(steps, means, 'o-', linewidth=2, markersize=6, label='Mean', color='blue')
    plt.plot(steps, mins, 's--', linewidth=1, markersize=4, label='Min', alpha=0.7, color='red')
    plt.plot(steps, maxs, '^--', linewidth=1, markersize=4, label='Max', alpha=0.7, color='green')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Face Similarity', fontsize=12)
    plt.title('Face Similarity Range Over Training', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved range plot to: {output_path}")
    plt.close()


def create_summary_table(
    metrics_by_step: Dict[int, dict],
    output_path: str,
):
    """Create a summary table of metrics."""
    steps = sorted(metrics_by_step.keys())
    
    # Create markdown table
    lines = [
        "# Face Consistency Summary\n",
        "| Step | Mean | Std | Min | Max | Count |",
        "|------|------|-----|-----|-----|-------|",
    ]
    
    for step in steps:
        m = metrics_by_step[step]
        lines.append(
            f"| {step} | {m['mean']:.4f} | {m['std']:.4f} | "
            f"{m['min']:.4f} | {m['max']:.4f} | {m['count']} |"
        )
    
    # Add improvement metrics
    if len(steps) > 1:
        first_mean = metrics_by_step[steps[0]]['mean']
        last_mean = metrics_by_step[steps[-1]]['mean']
        improvement = last_mean - first_mean
        improvement_pct = (improvement / (first_mean + 1e-6)) * 100
        
        lines.extend([
            "",
            "## Improvement",
            f"- Initial similarity: {first_mean:.4f}",
            f"- Final similarity: {last_mean:.4f}",
            f"- Absolute improvement: {improvement:+.4f}",
            f"- Relative improvement: {improvement_pct:+.2f}%",
        ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved summary table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize face consistency metrics from training"
    )
    
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing face metrics JSON files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="face_consistency_plots",
        help="Output directory for visualizations (default: face_consistency_plots)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    logger.info(f"Loading metrics from: {args.metrics_dir}")
    metrics_by_step = load_metrics_files(args.metrics_dir)
    
    if not metrics_by_step:
        logger.error("No metrics found!")
        return
    
    logger.info(f"Loaded metrics for {len(metrics_by_step)} steps")
    
    # Create visualizations
    plot_face_similarity_over_time(
        metrics_by_step,
        os.path.join(args.output_dir, "face_similarity_over_time.png")
    )
    
    plot_similarity_distribution(
        metrics_by_step,
        os.path.join(args.output_dir, "similarity_distribution.png")
    )
    
    plot_min_max_range(
        metrics_by_step,
        os.path.join(args.output_dir, "similarity_range.png")
    )
    
    create_summary_table(
        metrics_by_step,
        os.path.join(args.output_dir, "summary.md")
    )
    
    logger.info(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

