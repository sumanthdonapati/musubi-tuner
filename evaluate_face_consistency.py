#!/usr/bin/env python3
"""
Evaluate Face Consistency for Generated Images

This script evaluates face embedding similarity between control and generated images,
useful for tracking LoRA training progress.

Usage:
    python evaluate_face_consistency.py \
        --control_dir path/to/control_images \
        --generated_dir path/to/generated_images \
        --output_json results.json

Or evaluate from sample directory structure:
    python evaluate_face_consistency.py \
        --sample_dir path/to/output_dir/sample \
        --output_json results.json
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.musubi_tuner.utils.face_embedding_loss import FaceEmbeddingLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_image_pairs(control_dir: str, generated_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching pairs of control and generated images.
    
    Assumes images have matching filenames.
    """
    control_path = Path(control_dir)
    generated_path = Path(generated_dir)
    
    pairs = []
    
    for control_img in control_path.glob("*"):
        if control_img.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        # Look for matching generated image
        gen_img = generated_path / control_img.name
        if gen_img.exists():
            pairs.append((str(control_img), str(gen_img)))
    
    return pairs


def find_sample_pairs(sample_dir: str) -> List[Tuple[str, str]]:
    """
    Find control-generated pairs from sample directory structure.
    
    Expected structure:
        sample/
            control_image_0.png
            generated_step_1000_0.png
            generated_step_1000_1.png
            ...
    """
    sample_path = Path(sample_dir)
    pairs = []
    
    # Find all control images
    control_images = sorted(sample_path.glob("control_image_*.png"))
    
    for control_img in control_images:
        # Extract index from filename (e.g., control_image_0.png -> 0)
        try:
            idx = control_img.stem.split('_')[-1]
            
            # Find all generated images with this index
            gen_pattern = f"*_{idx}.png"
            generated_images = sorted(sample_path.glob(gen_pattern))
            
            # Add pairs for all generated versions
            for gen_img in generated_images:
                if "control" not in gen_img.name:
                    pairs.append((str(control_img), str(gen_img)))
                    
        except Exception as e:
            logger.warning(f"Could not parse index from {control_img.name}: {e}")
            continue
    
    return pairs


def evaluate_pairs(
    pairs: List[Tuple[str, str]],
    model_name: str = "buffalo_l",
    device: str = "cuda",
) -> dict:
    """Evaluate face consistency for all image pairs."""
    
    logger.info(f"Evaluating {len(pairs)} image pairs")
    
    # Initialize face embedding module
    face_module = FaceEmbeddingLoss(
        model_name=model_name,
        device=device,
        use_as_loss=False,
    )
    
    # Separate pairs by whether they're from same control or different steps
    results_by_step = {}
    all_similarities = []
    
    for ctrl_path, gen_path in pairs:
        # Extract step number from generated image name
        gen_name = Path(gen_path).stem
        step = "unknown"
        
        # Try to extract step number (e.g., generated_step_1000_0 -> 1000)
        try:
            parts = gen_name.split('_')
            if 'step' in parts:
                step_idx = parts.index('step') + 1
                step = int(parts[step_idx])
        except:
            pass
        
        # Load and evaluate
        try:
            metrics = face_module.evaluate_on_image_files([ctrl_path], [gen_path])
            
            if metrics['valid_samples'] > 0:
                similarity = metrics['face_similarity_mean']
                all_similarities.append(similarity)
                
                if step not in results_by_step:
                    results_by_step[step] = []
                results_by_step[step].append({
                    'control': ctrl_path,
                    'generated': gen_path,
                    'similarity': similarity,
                })
                
                logger.info(f"Step {step}: {Path(gen_path).name} -> similarity: {similarity:.4f}")
            else:
                logger.warning(f"No faces detected in pair: {ctrl_path} <-> {gen_path}")
                
        except Exception as e:
            logger.error(f"Error evaluating pair {ctrl_path} <-> {gen_path}: {e}")
    
    # Compute statistics
    if len(all_similarities) > 0:
        summary = {
            'total_pairs': len(pairs),
            'valid_pairs': len(all_similarities),
            'overall_mean': float(np.mean(all_similarities)),
            'overall_std': float(np.std(all_similarities)),
            'overall_min': float(np.min(all_similarities)),
            'overall_max': float(np.max(all_similarities)),
            'by_step': {},
        }
        
        # Compute per-step statistics
        for step, results in results_by_step.items():
            sims = [r['similarity'] for r in results]
            summary['by_step'][str(step)] = {
                'count': len(sims),
                'mean': float(np.mean(sims)),
                'std': float(np.std(sims)),
                'min': float(np.min(sims)),
                'max': float(np.max(sims)),
            }
        
        # Store detailed results
        summary['detailed_results'] = results_by_step
        
        return summary
    else:
        logger.error("No valid face pairs found")
        return {
            'total_pairs': len(pairs),
            'valid_pairs': 0,
            'error': 'No valid face pairs detected',
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face consistency between control and generated images"
    )
    
    parser.add_argument(
        "--control_dir",
        type=str,
        help="Directory containing control images"
    )
    
    parser.add_argument(
        "--generated_dir",
        type=str,
        help="Directory containing generated images"
    )
    
    parser.add_argument(
        "--sample_dir",
        type=str,
        help="Sample directory from training (alternative to control_dir/generated_dir)"
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        default="face_consistency_results.json",
        help="Output JSON file for results (default: face_consistency_results.json)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_s", "antelopev2"],
        help="InsightFace model name (default: buffalo_l)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for face detection (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Find image pairs
    if args.sample_dir:
        pairs = find_sample_pairs(args.sample_dir)
        logger.info(f"Found {len(pairs)} pairs in sample directory: {args.sample_dir}")
    elif args.control_dir and args.generated_dir:
        pairs = find_image_pairs(args.control_dir, args.generated_dir)
        logger.info(f"Found {len(pairs)} pairs between directories")
    else:
        parser.error("Must provide either --sample_dir or both --control_dir and --generated_dir")
    
    if len(pairs) == 0:
        logger.error("No image pairs found!")
        return
    
    # Evaluate
    results = evaluate_pairs(pairs, args.model_name, args.device)
    
    # Save results
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {args.output_json}")
    logger.info(f"\nSummary:")
    logger.info(f"  Total pairs: {results['total_pairs']}")
    logger.info(f"  Valid pairs: {results['valid_pairs']}")
    if results['valid_pairs'] > 0:
        logger.info(f"  Mean similarity: {results['overall_mean']:.4f} ± {results['overall_std']:.4f}")
        logger.info(f"  Range: [{results['overall_min']:.4f}, {results['overall_max']:.4f}]")
        
        # Print per-step summary
        if 'by_step' in results and len(results['by_step']) > 1:
            logger.info(f"\nPer-step summary:")
            for step in sorted(results['by_step'].keys(), key=lambda x: int(x) if x.isdigit() else 0):
                step_data = results['by_step'][step]
                logger.info(f"  Step {step}: {step_data['mean']:.4f} ± {step_data['std']:.4f} "
                          f"(n={step_data['count']})")


if __name__ == "__main__":
    main()

