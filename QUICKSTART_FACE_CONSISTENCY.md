# Quick Start: Face Consistency Training

This guide gets you started with face consistency tracking in under 5 minutes.

## Installation

```bash
# Install InsightFace
pip install insightface onnxruntime-gpu

# Models will auto-download on first run (~600MB for buffalo_l)
```

## Method 1: Post-Training Evaluation (Simplest)

Train your LoRA normally with sample generation enabled:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit models/qwen_image_edit_2509_bf16.safetensors \
    --vae models/vae_model.safetensors \
    --text_encoder models/qwen_2.5_vl_7b.safetensors \
    --dataset_config dataset.toml \
    --edit_plus \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 5e-5 \
    --gradient_checkpointing \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --max_train_epochs 16 \
    --sample_prompts sample_prompts.txt \
    --sample_every_n_steps 500 \
    --output_dir output --output_name face_lora
```

After training, evaluate face consistency:

```bash
./evaluate_face_consistency.py \
    --sample_dir output/sample \
    --output_json face_results.json
```

View results:

```bash
cat face_results.json | jq '.'
```

## Method 2: Continuous Monitoring During Training

In one terminal, start training:

```bash
accelerate launch ... (same as above)
```

In another terminal, monitor face consistency:

```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    if [ -d "output/sample" ]; then
        echo "=== Checking face consistency at $(date) ==="
        ./evaluate_face_consistency.py \
            --sample_dir output/sample \
            --output_json "output/face_metrics_$(date +%s).json"
    fi
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x monitor.sh
./monitor.sh
```

## Viewing Results

### Quick Check

```bash
# Show overall similarity
cat face_results.json | jq '.overall_mean'
# Output: 0.8456

# Show per-step progression
cat face_results.json | jq '.by_step | to_entries[] | "\(.key): \(.value.mean)"'
```

### Visualize Progress

```bash
./visualize_face_consistency.py \
    --metrics_dir output \
    --output_dir plots
```

This creates:
- `plots/face_similarity_over_time.png` - Main progress chart
- `plots/similarity_distribution.png` - Distribution evolution
- `plots/similarity_range.png` - Min/max ranges
- `plots/summary.md` - Markdown table

## Interpreting Results

**Face Similarity Scores:**
- **> 0.85** ✅ Excellent face consistency
- **0.75-0.85** ✓ Good face consistency
- **0.65-0.75** ⚠️ Moderate consistency (may need tuning)
- **< 0.65** ❌ Poor consistency (needs improvement)

**Example Output:**
```json
{
  "overall_mean": 0.8456,  ← Good!
  "overall_std": 0.0389,   ← Low variance is good
  "valid_pairs": 48,       ← 48/50 faces detected
  "by_step": {
    "1000": {"mean": 0.7856},  ← Starting point
    "2000": {"mean": 0.8234},  ← Improving
    "3000": {"mean": 0.8456}   ← Final (good!)
  }
}
```

## Next Steps

### If Results Are Good (> 0.80)

Your LoRA is working well! Continue with:
- Testing on more diverse faces
- Adjusting other hyperparameters (learning rate, network_dim)
- Training longer if still improving

### If Results Need Improvement (< 0.75)

Try these adjustments:

**Option A: Use Face Loss (Recommended)**
```bash
accelerate launch ... \
    src/musubi_tuner/qwen_image_train_network_face_loss.py \
    --enable_face_loss \
    --use_face_as_loss \
    --face_loss_weight 0.1 \
    --face_eval_frequency 100 \
    ... (other params)
```

**Option B: Training Adjustments**
- Lower learning rate: `--learning_rate 3e-5`
- Increase network dimension: `--network_dim 32`
- Train longer: `--max_train_epochs 24`
- Adjust shift: `--discrete_flow_shift 2.5`

**Option C: Data Quality**
- Ensure control images have clear, frontal faces
- Check that captions accurately describe changes
- Verify image pairs are properly aligned

## Common Issues

### "No faces detected"

**Solution:** Check your images
```bash
# Test face detection on a sample
python -c "
from src.musubi_tuner.utils.face_embedding_loss import FaceEmbeddingLoss
import numpy as np
from PIL import Image

face_module = FaceEmbeddingLoss()
img = np.array(Image.open('your_image.jpg'))
emb = face_module.extract_face_embedding(img)
print('Face detected!' if emb is not None else 'No face found')
"
```

### Training is slow

**Solution:** Use evaluation script instead of inline loss
- Don't use `--use_face_as_loss`
- Run `evaluate_face_consistency.py` periodically instead

### Face similarity not improving

**Solutions:**
1. Increase `--face_loss_weight` from 0.1 to 0.2
2. Check dataset quality (faces clear and frontal?)
3. Verify correct control-target image pairing
4. Try different `--discrete_flow_shift` values (1.8-2.5)

## Example Workflow

```bash
# 1. Initial training
accelerate launch ... --output_name v1

# 2. Evaluate
./evaluate_face_consistency.py --sample_dir output_v1/sample --output_json v1.json

# 3. Check results
cat v1.json | jq '.overall_mean'
# → 0.6723 (needs improvement)

# 4. Retrain with face loss
accelerate launch ... \
    src/musubi_tuner/qwen_image_train_network_face_loss.py \
    --enable_face_loss --use_face_as_loss --face_loss_weight 0.15 \
    --output_name v2

# 5. Compare
./evaluate_face_consistency.py --sample_dir output_v2/sample --output_json v2.json
cat v2.json | jq '.overall_mean'
# → 0.8456 (much better!)

# 6. Visualize improvement
./visualize_face_consistency.py --metrics_dir output_v2 --output_dir plots_v2
```

## Tips

1. **Start without face loss** - Get baseline metrics first
2. **Monitor regularly** - Set up automated monitoring during training  
3. **Compare versions** - Keep metrics for each training run
4. **Balance losses** - Don't set face_loss_weight too high (>0.3)
5. **Quality over quantity** - Better to have fewer high-quality face pairs

## Full Documentation

For complete details, see [FACE_CONSISTENCY_GUIDE.md](FACE_CONSISTENCY_GUIDE.md)

## Citation

Uses InsightFace for face recognition:
- GitHub: https://github.com/deepinsight/insightface
- Paper: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019

