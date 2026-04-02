# Training Scripts for CS336 Assignment 1

This document describes the training scripts for the Transformer Language Model assignment.

## Quick Start

### 1. Prepare Data

First, download the TinyStories dataset and tokenize it:

```bash
# Create data directory
mkdir -p data

# Download TinyStories data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..

# Tokenize data (train BPE tokenizer and tokenize files)
uv run python tokenize_data.py \
    --train_text data/TinyStoriesV2-GPT4-train.txt \
    --valid_text data/TinyStoriesV2-GPT4-valid.txt \
    --output_dir data \
    --vocab_size 10000 \
    --special_tokens "<|endoftext|>" \
    --num_workers 16
```

```
Training BPE tokenizer on /nfs/ofs-llm-ssd/user/gogongxt/datasets/data/TinyStoriesV2-GPT4-train.txt...
  Target vocab size: 10000
  Special tokens: ['<|endoftext|>']
  Total lines in training file: 15,600,057
  Parallel pretokenization...
  Processing chunks: 100%|████████████████████████████████████████████████████████| 8189/8189 [01:05<00:00, 125.88it/s]
  Unique words: 59,933
  Building initial pair counts...
  Starting BPE training loop (9743 merges)...
Training BPE: 100%|███████████████████████████████████████████████████████████████| 9743/9743 [00:44<00:00, 220.97it/s]
  Final vocab size: 10000
  Number of merges: 9743
  Training time: 3.9 minutes
  Saved vocab to data/vocab.json
  Saved merges to data/merges.txt
  Vocab size: 10000, Merges: 9743
  Creating tokenizer...
  Tokenizer created in 0.00s
Tokenizing /nfs/ofs-llm-ssd/user/gogongxt/datasets/data/TinyStoriesV2-GPT4-train.txt...
  Total lines: 15,600,057
  Tokenizing: 100%|███████████████████████████████████████████████████| 15600057/15600057 [48:32<00:00, 5357.05lines/s]
  Total: 540,796,778 tokens
  Tokenization time: 48.5 minutes
  Average speed: 185,698 tokens/sec
  Saved to data/train.npy
Tokenizing /nfs/ofs-llm-ssd/user/gogongxt/datasets/data/TinyStoriesV2-GPT4-valid.txt...
  Total lines: 157,832
  Tokenizing: 100%|███████████████████████████████████████████████████████| 157832/157832 [00:29<00:00, 5364.14lines/s]
  Total: 5,461,210 tokens
  Tokenization time: 0.5 minutes
  Average speed: 185,584 tokens/sec
  Saved to data/valid.npy

Data preprocessing complete!
  Training data: data/train.npy
  Validation data: data/valid.npy
```

This will create:

- `data/vocab.json` - BPE vocabulary
- `data/merges.txt` - BPE merges
- `data/train.npy` - Tokenized training data
- `data/valid.npy` - Tokenized validation data

### 2. Train Model

Basic training:

```bash
uv run python train.py \
    --train_data data/train.npy \
    --valid_data data/valid.npy \
    --vocab_size 10000 \
    --batch_size 64 \
    --total_steps 5000 \
    --learning_rate 3e-3 \
    --device cuda
```

```
============================================================
CS336 Assignment 1 - Transformer Language Model Training
============================================================

Configuration:
  train_data_path: data/train.npy
  valid_data_path: data/valid.npy
  vocab_size: 10000
  context_length: 256
  d_model: 512
  num_layers: 4
  num_heads: 16
  d_ff: 1344
  rope_theta: 10000.0
  batch_size: 64
  total_steps: 5000
  learning_rate: 0.003
  min_learning_rate: 3e-05
  warmup_steps: 500
  weight_decay: 0.01
  grad_clip: 1.0
  beta1: 0.9
  beta2: 0.999
  adam_eps: 1e-08
  checkpoint_dir: checkpoints
  checkpoint_interval: 500
  resume_from: None
  log_interval: 10
  eval_interval: 100
  eval_batches: 10
  device: cuda
  use_wandb: False
  wandb_project: cs336-assignment1
  wandb_run_name: None

Using device: cuda

Loading datasets...
  Train tokens: 540,796,778
  Valid tokens: 5,461,210
  Saved config to checkpoints/config.json

Creating model...
  Parameters: 22,696,448

Starting training...
  Total steps: 5000
  Warmup steps: 500
  Batch size: 64
  Context length: 256

  Total tokens to process: 81,920,000

Step 10/5000 | Loss: 78.9231 | PPL: 18874432598690562099299061466136576.00 | LR: 5.40e-05 | Tokens/sec: 45,142
Step 20/5000 | Loss: 58.0627 | PPL: 16455251860736024648876032.00 | LR: 1.14e-04 | Tokens/sec: 61,528
Step 30/5000 | Loss: 41.5259 | PPL: 1082643763164612992.00 | LR: 1.74e-04 | Tokens/sec: 63,923
Step 40/5000 | Loss: 27.4835 | PPL: 862862736490.68 | LR: 2.34e-04 | Tokens/sec: 65,745
Step 50/5000 | Loss: 21.2151 | PPL: 1635268146.52 | LR: 2.94e-04 | Tokens/sec: 67,240
Step 60/5000 | Loss: 20.8432 | PPL: 1127416512.55 | LR: 3.54e-04 | Tokens/sec: 68,521
Step 70/5000 | Loss: 19.3678 | PPL: 257827787.37 | LR: 4.14e-04 | Tokens/sec: 55,998
Step 80/5000 | Loss: 16.6268 | PPL: 16631758.19 | LR: 4.74e-04 | Tokens/sec: 70,215
Step 90/5000 | Loss: 14.6692 | PPL: 2348388.49 | LR: 5.34e-04 | Tokens/sec: 71,416
Step 100/5000 | Loss: 12.4192 | PPL: 247505.46 | LR: 5.94e-04 | Tokens/sec: 72,726

  Validation Loss: 11.6214 | PPL: 111462.29

  Saved best checkpoint to checkpoints/best.pt
Step 110/5000 | Loss: 10.8907 | PPL: 53677.31 | LR: 6.54e-04 | Tokens/sec: 46,672
Step 120/5000 | Loss: 9.7009 | PPL: 16331.51 | LR: 7.14e-04 | Tokens/sec: 74,240
Step 130/5000 | Loss: 8.5528 | PPL: 5181.14 | LR: 7.74e-04 | Tokens/sec: 74,782

...
...
...

Step 4890/5000 | Loss: 1.8507 | PPL: 6.36 | LR: 3.45e-05 | Tokens/sec: 89,829
Step 4900/5000 | Loss: 1.8594 | PPL: 6.42 | LR: 3.37e-05 | Tokens/sec: 89,879

  Validation Loss: 1.8523 | PPL: 6.37

  Saved best checkpoint to checkpoints/best.pt
Step 4910/5000 | Loss: 1.8634 | PPL: 6.45 | LR: 3.30e-05 | Tokens/sec: 55,800
Step 4920/5000 | Loss: 1.8621 | PPL: 6.44 | LR: 3.24e-05 | Tokens/sec: 89,690
Step 4930/5000 | Loss: 1.8813 | PPL: 6.56 | LR: 3.18e-05 | Tokens/sec: 89,864
Step 4940/5000 | Loss: 1.8593 | PPL: 6.42 | LR: 3.13e-05 | Tokens/sec: 89,832
Step 4950/5000 | Loss: 1.8548 | PPL: 6.39 | LR: 3.09e-05 | Tokens/sec: 89,845
Step 4960/5000 | Loss: 1.8678 | PPL: 6.47 | LR: 3.06e-05 | Tokens/sec: 89,868
Step 4970/5000 | Loss: 1.8529 | PPL: 6.38 | LR: 3.03e-05 | Tokens/sec: 89,810
Step 4980/5000 | Loss: 1.8733 | PPL: 6.51 | LR: 3.02e-05 | Tokens/sec: 89,829
Step 4990/5000 | Loss: 1.8771 | PPL: 6.53 | LR: 3.00e-05 | Tokens/sec: 89,784
Step 5000/5000 | Loss: 1.8639 | PPL: 6.45 | LR: 3.00e-05 | Tokens/sec: 89,866

  Validation Loss: 1.8796 | PPL: 6.55

  Saved checkpoint to checkpoints/step_5000.pt

Saved final checkpoint to checkpoints/final.pt

============================================================
Training Complete!
============================================================
  Final Validation Loss: 1.8773
  Final Validation Perplexity: 6.54
  Total Training Time: 0.27 hours
  Total Steps: 5000
  Total Tokens: 81,920,000
```

### 3. Generate Text

After training, generate text from the model:

```bash
uv run python generate.py \
    --checkpoint checkpoints/final.pt \
    --config_path checkpoints/config.json \
    --vocab_path data/vocab.json \
    --merges_path data/merges.txt \
    --prompt "Once upon a time" \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --top_p 0.3
```

```
Loading tokenizer from data/vocab.json and data/merges.txt...
Loading model from checkpoints/final.pt...
Loading config from checkpoints/config.json...
Model config: {'vocab_size': 10000, 'context_length': 256, 'd_model': 512, 'num_layers': 4, 'num_heads': 16, 'd_ff': 1344, 'rope_theta': 10000.0}

Prompt: Once upon a time
------------------------------------------------------------

[Sample 1]
Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big, red apple on the ground. Lily was very happy and wanted to eat it all.
Lily took the apple home and showed it to her mom. Her mom said, "Good job, Lily! You did a great job!" Lily was very proud of her work. She knew that she could do anything if she tried hard.
<|endoftext|>
------------------------------------------------------------
```

## Experiments

### Learning Rate Sweep

Run learning rate tuning experiments:

```bash
uv run python run_experiments.py \
    --experiment lr_sweep \
    --learning_rates 1e-4 3e-4 1e-3 3e-3 \
    --train_data data/train.npy \
    --valid_data data/valid.npy \
    --total_steps 5000
```

### Batch Size Sweep

Run batch size experiments:

```bash
uv run python run_experiments.py \
    --experiment batch_sweep \
    --batch_sizes 16 32 64 128 \
    --train_data data/train.npy \
    --valid_data data/valid.npy
```

### Single Run

Run a single training with specific parameters:

```bash
uv run python run_experiments.py \
    --experiment single \
    --learning_rate 3e-4 \
    --batch_size 64 \
    --total_steps 5000
```

## Low Resource Configuration (CPU/MPS)

For training on CPU or Apple Silicon:

```bash
uv run python train.py \
    --train_data data/train.npy \
    --valid_data data/valid.npy \
    --batch_size 32 \
    --total_steps 5000 \
    --learning_rate 3e-4 \
    --device cpu
```

Expected runtime:

- M3 Max (36GB RAM): ~36 minutes on MPS, ~1h 22min on CPU
- Target validation loss: ~1.80

## Checkpoints

The training script saves model checkpoints in HuggingFace style:

```
checkpoints/
├── config.json      # Model architecture config (vocab_size, d_model, num_heads, etc.)
├── best.pt          # Best validation loss checkpoint
├── final.pt         # Final checkpoint
└── step_N.pt        # Periodic checkpoints
```

**Config format** (`config.json`):

```json
{
  "vocab_size": 10000,
  "context_length": 256,
  "d_model": 512,
  "num_layers": 4,
  "num_heads": 16,
  "d_ff": 1344,
  "rope_theta": 10000.0
}
```

To resume training:

```bash
uv run python train.py --resume checkpoints/step_1000.pt
```

## Hyperparameter Guidelines

Based on the assignment guidelines:

### TinyStories (H100)

- vocab_size: 10000
- context_length: 256
- d_model: 512
- d_ff: 1344
- num_layers: 4
- num_heads: 16
- batch_size: 64
- total_steps: 5000
- Expected: val_loss ≤ 1.45, ~30-40 minutes

### TinyStories (CPU/MPS)

- batch_size: 32
- total_steps: 5000
- total_tokens: ~40M
- Expected: val_loss ≤ 2.00, ~1 hour

### OpenWebText

- vocab_size: 32000
- Same model architecture
- Expected: higher loss than TinyStories

## Tips

1. **Overfit a single batch first** to verify implementation:

   ```bash
   uv run python train.py --total_steps 100 --eval_interval 10
   ```

2. **Monitor gradients** - if gradients explode/vanish, check:
   - Learning rate
   - Gradient clipping
   - Weight initialization

3. **Learning rate tuning** - start with:
   - 1e-4, 3e-4, 1e-3, 3e-3
   - Look for "edge stability" (just before divergence)

4. **Cosine schedule** - ensure final step reaches min learning rate exactly

5. **MPS acceleration** on Apple Silicon:
   ```bash
   uv run python train.py --device mps
   ```
