# Face Consistency Training Guide for Qwen Image Edit 2509

This guide explains how to use face embedding similarity (via InsightFace) to track and improve face consistency during LoRA training.

## Overview

When training a face consistency LoRA, you want to ensure that the generated images preserve facial identity from the control images. This implementation provides:

1. **Face Embedding Loss**: Optional training loss component based on InsightFace embeddings
2. **Face Consistency Metrics**: Validation metrics logged during training
3. **Post-training Evaluation**: Standalone script to evaluate face consistency

## Installation

### Install InsightFace

```bash
pip install insightface onnxruntime
```

For GPU support:
```bash
pip install insightface onnxruntime-gpu
```

### Download Face Recognition Models

The first time you run the code, InsightFace will automatically download the face recognition models (buffalo_l, ~600MB).

## Usage Options

### Option 1: Validation Metrics Only (Recommended)

Track face similarity during training without affecting the loss:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/qwen_image_edit_2509_bf16.safetensors \
    --vae path/to/vae_model \
    --text_encoder path/to/qwen_2.5_vl_7b.safetensors \
    --dataset_config path/to/dataset.toml \
    --edit_plus \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 5e-5 \
    --gradient_checkpointing \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --sample_prompts sample_prompts.txt \
    --sample_every_n_steps 500 \
    --output_dir path/to/output --output_name face_lora
```

Then periodically evaluate face consistency on sample images:

```bash
python evaluate_face_consistency.py \
    --sample_dir path/to/output/sample \
    --output_json face_metrics_step_1000.json \
    --model_name buffalo_l \
    --device cuda
```

### Option 2: Direct Integration (Advanced)

Use face embedding as part of the training loss:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network_face_loss.py \
    --dit path/to/qwen_image_edit_2509_bf16.safetensors \
    --vae path/to/vae_model \
    --text_encoder path/to/qwen_2.5_vl_7b.safetensors \
    --dataset_config path/to/dataset.toml \
    --edit_plus \
    --enable_face_loss \
    --use_face_as_loss \
    --face_loss_weight 0.1 \
    --face_model_name buffalo_l \
    --face_eval_frequency 100 \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --discrete_flow_shift 2.2 \
    --optimizer_type adamw8bit --learning_rate 5e-5 \
    --gradient_checkpointing \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name face_lora
```

## Face Loss Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_face_loss` | Enable face consistency tracking | False |
| `--use_face_as_loss` | Use as training loss (vs. metric only) | False |
| `--face_loss_weight` | Weight for face loss component | 0.1 |
| `--face_model_name` | InsightFace model (buffalo_l/buffalo_s/antelopev2) | buffalo_l |
| `--face_device` | Device for face detection | cuda |
| `--face_det_size` | Detection resolution | 640 |
| `--face_eval_frequency` | Evaluate every N steps | 100 |

## Understanding Face Similarity Scores

Face similarity is measured using cosine similarity of normalized embeddings:

- **1.0**: Perfect match (same face)
- **0.8-1.0**: Very high similarity (likely same person)
- **0.6-0.8**: High similarity (possibly same person)
- **0.4-0.6**: Moderate similarity (different person, similar features)
- **< 0.4**: Low similarity (different person)

For face consistency LoRA training, you typically want:
- **Training target**: > 0.85 similarity
- **Good results**: > 0.75 similarity
- **Needs improvement**: < 0.65 similarity

## Evaluation Script Usage

### Basic Evaluation

Evaluate all samples in a training run:

```bash
python evaluate_face_consistency.py \
    --sample_dir path/to/output/sample \
    --output_json results.json
```

### Custom Directory Structure

If you have separate directories:

```bash
python evaluate_face_consistency.py \
    --control_dir path/to/control_images \
    --generated_dir path/to/generated_images \
    --output_json results.json
```

### Output Format

The script generates a JSON file with:

```json
{
  "total_pairs": 50,
  "valid_pairs": 48,
  "overall_mean": 0.8234,
  "overall_std": 0.0456,
  "overall_min": 0.7123,
  "overall_max": 0.9345,
  "by_step": {
    "1000": {
      "count": 10,
      "mean": 0.7856,
      "std": 0.0523,
      "min": 0.7123,
      "max": 0.8567
    },
    "2000": {
      "count": 10,
      "mean": 0.8456,
      "std": 0.0389,
      "min": 0.7890,
      "max": 0.9123
    }
  }
}
```

## Integration with Training Loop

### Monitoring Face Similarity

Create a monitoring script that runs periodically:

```bash
#!/bin/bash
# monitor_training.sh

OUTPUT_DIR="path/to/output"
SAMPLE_DIR="$OUTPUT_DIR/sample"

while true; do
    if [ -d "$SAMPLE_DIR" ]; then
        echo "Evaluating face consistency..."
        python evaluate_face_consistency.py \
            --sample_dir "$SAMPLE_DIR" \
            --output_json "$OUTPUT_DIR/face_metrics_$(date +%s).json"
    fi
    sleep 300  # Check every 5 minutes
done
```

### Tracking Progress

Plot face similarity over training steps:

```python
import json
import glob
import matplotlib.pyplot as plt

# Load all metric files
metric_files = sorted(glob.glob("path/to/output/face_metrics_*.json"))
steps = []
similarities = []

for file in metric_files:
    with open(file) as f:
        data = json.load(f)
        for step, metrics in data.get('by_step', {}).items():
            if step.isdigit():
                steps.append(int(step))
                similarities.append(metrics['mean'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, similarities, marker='o')
plt.xlabel('Training Step')
plt.ylabel('Face Similarity')
plt.title('Face Consistency Over Training')
plt.grid(True)
plt.savefig('face_consistency_plot.png')
```

## Best Practices

### 1. Start with Metrics Only

Begin training with face metrics logging (not as loss) to understand baseline:

```bash
--enable_face_loss --face_eval_frequency 100
```

### 2. Gradually Add Face Loss

If metrics show poor face consistency, add face loss with low weight:

```bash
--use_face_as_loss --face_loss_weight 0.05
```

### 3. Balance Loss Weights

If face loss weight is too high, it may hurt other aspects:
- **Too high (>0.3)**: May preserve face but lose other editing capabilities
- **Too low (<0.01)**: May not have significant effect
- **Recommended**: 0.05-0.15

### 4. Monitor Both Losses

Track both reconstruction loss and face similarity:
- Reconstruction loss should decrease normally
- Face similarity should increase over training
- If face similarity plateaus early, consider adjusting face_loss_weight

### 5. Use Appropriate Models

InsightFace model selection:
- **buffalo_l**: Best accuracy, larger model (~600MB)
- **buffalo_s**: Faster, smaller model (~200MB)
- **antelopev2**: Good balance

## Advanced: Custom Face Loss Implementation

To implement custom face loss logic, modify `FaceEmbeddingLoss` class:

```python
# src/musubi_tuner/utils/face_embedding_loss.py

class CustomFaceLoss(FaceEmbeddingLoss):
    def compute_weighted_loss(self, similarities):
        """
        Custom loss weighting based on similarity scores.
        Penalize low similarities more heavily.
        """
        # Convert similarity to distance
        distances = 1.0 - torch.tensor(similarities)
        
        # Apply squared loss for low similarities
        weights = torch.where(
            distances > 0.3,  # If similarity < 0.7
            distances ** 2,   # Square the loss
            distances         # Linear loss otherwise
        )
        
        return weights.mean()
```

## Troubleshooting

### No Faces Detected

**Problem**: `valid_pairs: 0` in metrics

**Solutions**:
1. Check image resolution (faces should be at least 50x50 pixels)
2. Increase `--face_det_size` to 1024
3. Verify control images actually contain faces
4. Check image preprocessing (ensure RGB format)

### High Memory Usage

**Problem**: OOM errors when face loss is enabled

**Solutions**:
1. Reduce `--face_eval_frequency` (e.g., 500 instead of 100)
2. Use smaller face model: `--face_model_name buffalo_s`
3. Set `--face_device cpu` if GPU memory is limited
4. Don't use `--use_face_as_loss` (use post-training evaluation instead)

### Slow Training

**Problem**: Training significantly slower with face loss

**Solutions**:
1. Use evaluation script periodically instead of inline loss
2. Increase `--face_eval_frequency`
3. Use `buffalo_s` model instead of `buffalo_l`

### Face Loss Not Decreasing

**Problem**: Face similarity not improving

**Solutions**:
1. Increase `--face_loss_weight` (e.g., from 0.1 to 0.2)
2. Check if faces are being detected (`valid_pairs` should be > 0)
3. Verify control images have clear, frontal faces
4. Ensure dataset has sufficient face diversity

## Example Workflow

### 1. Initial Training (No Face Loss)

```bash
accelerate launch ... qwen_image_train_network.py \
    --sample_every_n_steps 500 \
    ... [other params]
```

### 2. Evaluate Initial Results

```bash
python evaluate_face_consistency.py \
    --sample_dir output/sample \
    --output_json initial_metrics.json
```

Check results:
```bash
cat initial_metrics.json | jq '.overall_mean'
# Output: 0.6234  (needs improvement!)
```

### 3. Retrain with Face Loss

```bash
accelerate launch ... qwen_image_train_network_face_loss.py \
    --enable_face_loss \
    --use_face_as_loss \
    --face_loss_weight 0.1 \
    --face_eval_frequency 100 \
    ... [other params]
```

### 4. Compare Results

```bash
python evaluate_face_consistency.py \
    --sample_dir output_v2/sample \
    --output_json improved_metrics.json

cat improved_metrics.json | jq '.overall_mean'
# Output: 0.8456  (much better!)
```

## References

- **InsightFace GitHub**: https://github.com/deepinsight/insightface
- **ArcFace Paper**: [Deng et al., 2019](https://arxiv.org/abs/1801.07698)
- **Face Recognition Benchmarks**: [NIST FRVT](https://pages.nist.gov/frvt/)

## Citation

If you use this face consistency tracking in your work:

```bibtex
@inproceedings{deng2018arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```

