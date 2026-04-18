# LLM - Transformer Language Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Pure%20Python-orange?style=flat&logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License">
</p>

An educational transformer-based Large Language Model built from scratch in pure Python/NumPy. **Built by AI** as a learning resource to understand, study, and rebuild in C++. No frameworks, no black boxes - just mathematical foundations.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [About This Project](#about-this-project)
- [Why This Project?](#why-this-project)
- [How to Learn From This Code](#how-to-learn-from-this-code)
- [Commands](#commands)
- [Training Examples](#training-examples)
- [Architecture](#architecture)
- [Files](#files)
- [Known Issues](#known-issues)

---

## About This Project

This repository was **built by AI** as a learning tool to understand how Large Language Models work under the hood.

**Purpose:**
- Understand every component of an LLM (embeddings, attention, RoPE, optimizers, etc.)
- Serve as a reference for porting to C++ or other languages
- Help anyone learn about transformer architecture

**This is not a production model** - it's an educational resource designed for study and understanding.

---

## Why This Project?

**Built by AI to learn, understand, and rebuild.**

This repository serves as a learning resource to understand how LLMs work:
- **Understand every component**: Each part of the transformer is its own file with documentation
- **Port to C++**: The clean code is designed to be translated to C++ or other languages
- **No black boxes**: Pure implementation - see the actual math
- **For anyone**: Anyone wanting to learn how LLMs work can study this code

---

## How to Learn From This Code

Recommended reading order to understand transformers:

1. **llm/tokenization.py** - How text becomes numbers (tokenizer)
2. **llm/embeddings.py** - Token representations
3. **llm/positional_encoding.py** / **llm/rope.py** - Position awareness
4. **llm/attention.py** - Multi-head self-attention
5. **llm/transformer_block.py** - Transformer layer
6. **llm/llm_model.py** - Full model integration
7. **llm/optimizer.py** - Model learning (AdamW)
8. **llm/simple_training.py** - Training loop

---

## Features

| Feature | Description |
|---------|-------------|
| **Transformer Architecture** | 8-layer decoder with multi-head self-attention |
| **RoPE Positional Embeddings** | Rotary Position Embeddings for better position awareness |
| **AdamW Optimizer** | Weight decay regularization with gradient clipping |
| **BPE Tokenizer** | Byte-Pair Encoding subword tokenization |
| **Dropout** | Regularization during training |
| **Beam Search** | Improved text generation |
| **Multiple Sampling Methods** | Greedy, top-k, temperature sampling |
| **Auto-save Checkpoints** | Save model during training |
| **Web Learning** | Learn from Wikipedia API |

---

## Quick Start

```bash
# Install dependencies
pip install numpy

# Train the model (web learning)
python main.py web --general --num-topics 50 --epochs 10 --save-model mymodel

# Chat with the model
python main.py chat --load-model models/mymodel
```

---

## Commands

```bash
# Help command - shows usage examples
python main.py help

# Train from text file
python main.py train --file data.txt --epochs 10 --save-model mymodel

# Train from folder
python main.py train --folder ./texts --epochs 5

# Interactive chat
python main.py chat --load-model models/mymodel

# Generate text
python main.py generate --prompt "Hello world" --max-length 50

# Evaluate model
python main.py eval --file test.txt --block-size 128
```

### Web Learning

```bash
# Learn from specific topic
python main.py web "artificial intelligence"

# Learn from 500 random topics (recommended)
python main.py web --general --num-topics 500 --epochs 30 --save-model trained_model

# With auto-save every 500 steps
python main.py web --general --num-topics 500 --epochs 30 --save-interval 500 --save-model trained_model

# Larger Model
python main.py train --file data.txt --vocab-size 10000 --d-model 768 --num-layers 12 --epochs 20 --save-interval 500 --save-model trained_model
```

### Using Checkpoints

You can use checkpoint files while training is in progress:

```bash
# Generate text with checkpoint (e.g., step 2000)
python main.py generate --prompt "Hello" --load-model models/trained_model_step2000

# Chat with checkpoint (run in terminal)
python main.py chat --load-model models/trained_model_step2000
```

**Note:** Checkpoint files are saved as `models/<name>_step<step>_embedding.npy` and `models/<name>_step<step>_linear.npy`. The system automatically detects and loads them.


---

## Training Examples

### Basic Training

```bash
# Learn from web (500 random topics, 30 epochs)
python main.py web --general --num-topics 500 --epochs 30 --save-model trained_model

# Learn from specific topic
python main.py web "quantum physics"

# Train from local file
python main.py train --file mytext.txt --epochs 10 --save-model mymodel

# Train from folder with multiple files
python main.py train --folder ./data --epochs 5 --save-model mymodel
```

### Generation Methods

```bash
# Greedy generation
python main.py generate --prompt "The" --method greedy

# Temperature sampling
python main.py generate --prompt "Hello" --temperature 0.8

# Top-k sampling
python main.py generate --prompt "Hi" --method top_k --top-k 40

# Beam search
python main.py generate --prompt "Start" --method beam --beam-width 3
```

### Model Options

```bash
# Custom model size
python main.py train --file data.txt --vocab-size 10000 --d-model 768 --num-layers 12

# Smaller model (faster)
python main.py train --file data.txt --vocab-size 2000 --d-model 256 --num-layers 4
```

---

## Architecture

### Model Specifications

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 5000 | Vocabulary size |
| `d_model` | 512 | Model dimension |
| `num_layers` | 8 | Transformer layers |
| `num_heads` | 8 | Attention heads |
| `max_seq_length` | 128 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |

### Components

- **Embedding Layer**: Token embeddings with learned positional encodings
- **Multi-Head Attention**: 8-head self-attention with dropout
- **RoPE (Rotary Position Embedding)**: Rotary positional embeddings for better context understanding
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Layer Normalization**: Pre-norm architecture
- **Output Linear**: Project to vocabulary for next-token prediction

### Training

- **Optimizer**: AdamW with weight decay (default: 0.01)
- **Gradient Clipping**: Default max norm 1.0
- **Learning Rate**: 0.001 with warmup (500 steps)
- **Loss**: Cross-entropy with perplexity tracking

---

## Files

```
LLM_made_by_AI/
├── main.py                 # CLI entry point
├── README.md               # This file
├── .gitignore
├── llm/
│   ├── __init__.py         # Package init
│   ├── llm_model.py       # Main model (LLM class)
│   ├── tokenization.py     # Tokenizer
│   ├── training.py        # Legacy training
│   ├── simple_training.py  # Working trainer with auto-save
│   ├── enhanced_training.py # Enhanced trainer with LR scheduling
│   ├── attention.py       # Multi-head attention
│   ├── transformer_block.py # Transformer layers
│   ├── embeddings.py     # Token embeddings
│   ├── positional_encoding.py # Sinusoidal positional encoding
│   ├── layer_norm.py    # Layer normalization
│   ├── linear.py        # Linear layer
│   ├── softmax.py       # Softmax activation
│   ├── qkv.py          # Query-Key-Value projection
│   ├── rope.py         # RoPE positional embeddings
│   ├── bpe_tokenizer.py # BPE subword tokenizer
│   ├── optimizer.py     # Adam/AdamW optimizer
│   ├── dataloader.py    # Data loading utilities
│   ├── utils.py        # Helper functions
│   └── inference.py    # Inference utilities
└── models/             # Saved models (auto-created)
```

---

## Known Issues

### Wikipedia 429 Rate Limit

When fetching many topics, Wikipedia API may return HTTP 429 (Too Many Requests).

**Workarounds:**
- Use fewer topics: `--num-topics 50` instead of 500
- Run multiple times with different topics
- The model still trains on successfully fetched topics

### GPU Acceleration

Currently runs on CPU only. CUDA/PyTorch integration planned for future release.

---

## License

MIT License - Educational project to understand how LLMs work under the hood.

---

## Credits

**Built by AI** as a learning resource to understand, study, and rebuild in C++.

For anyone who wants to learn how LLMs work - this code is for you.
