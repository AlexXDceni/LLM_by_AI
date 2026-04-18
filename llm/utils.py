"""
Utilities Module
Common helper functions and utilities for the LLM.
"""

import numpy as np
import json
import os


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def count_parameters(model):
    """Count total trainable parameters in model."""
    total = 0
    for layer in [model.embedding, model.transformer_stack, model.output_linear]:
        if hasattr(layer, 'weights'):
            for key, val in layer.weights.items():
                if val is not None:
                    total += val.size
    return total


def get_model_size_mb(model):
    """Get model size in megabytes."""
    total_bytes = 0
    for layer in [model.embedding, model.transformer_stack, model.output_linear]:
        if hasattr(layer, 'weights'):
            for key, val in layer.weights.items():
                if val is not None:
                    total_bytes += val.nbytes
    return total_bytes / (1024 * 1024)


def print_model_info(model):
    """Print model architecture information."""
    print(f"\n{'='*50}")
    print(f"Model: LLM")
    print(f"{'='*50}")
    print(f"Vocab Size: {model.vocab_size}")
    print(f"d_model: {model.d_model}")
    print(f"Num Layers: {model.num_layers}")
    print(f"Num Heads: {model.num_heads}")
    print(f"d_ff: {model.d_ff}")
    print(f"Max Seq Length: {model.max_seq_length}")
    print(f"Dropout: {model.dropout}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Size: {get_model_size_mb(model):.2f} MB")
    print(f"{'='*50}\n")


def softmax_temperature(logits, temperature):
    """Apply temperature scaling to logits."""
    if temperature == 0:
        return np.zeros_like(logits)
    return logits / temperature


def top_k_logits(logits, k):
    """Keep only top-k logits."""
    if k <= 0:
        return logits
    
    top_k = min(k, len(logits))
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    
    masked_logits = np.full_like(logits, -1e9)
    masked_logits[top_indices] = logits[top_indices]
    
    return masked_logits


def top_p_logits(logits, p):
    """Nucleus/Top-p sampling."""
    if p <= 0 or p >= 1:
        return logits
    
    sorted_indices = np.argsort(logits)[::-1]
    sorted_probs = np.exp(logits[sorted_indices] - np.max(logits))
    sorted_probs = sorted_probs / np.sum(sorted_probs)
    
    cumsum = np.cumsum(sorted_probs)
    
    cutoff_idx = np.searchsorted(cumsum, p)
    kept_indices = sorted_indices[:cutoff_idx + 1]
    
    masked_logits = np.full_like(logits, -1e9)
    masked_logits[kept_indices] = logits[kept_indices]
    
    return masked_logits


def repeat_interleave(x, repeats):
    """Repeat elements of array."""
    return np.repeat(x, repeats)


def sliding_window_view(arr, window_size, stride=1):
    """Create sliding window view of array."""
    n = len(arr)
    num_windows = (n - window_size) // stride + 1
    
    windows = []
    for i in range(0, n - window_size + 1, stride):
        windows.append(arr[i:i+window_size])
    
    return np.array(windows)


def pad_sequences(sequences, max_length, pad_token=0):
    """Pad sequences to same length."""
    padded = []
    masks = []
    
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[:max_length])
            masks.append([1] * max_length)
        else:
            padding = [pad_token] * (max_length - len(seq))
            padded.append(seq + padding)
            masks.append([1] * len(seq) + [0] * (max_length - len(seq)))
    
    return np.array(padded), np.array(masks)


def create_causal_mask(seq_len, device='cpu'):
    """Create causal mask for autoregressive attention."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def create_padding_mask(token_ids, pad_token=0):
    """Create padding mask."""
    return token_ids != pad_token


def merge_masks(*masks):
    """Merge multiple masks."""
    result = masks[0]
    for mask in masks[1:]:
        result = result * mask
    return result


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)


class Timer:
    """Simple timer for profiling."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name):
        self.start_times[name] = __import__('time').time()
    
    def stop(self, name):
        if name in self.start_times:
            elapsed = __import__('time').time() - self.start_times[name]
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0
    
    def get_average(self, name):
        if name in self.times and self.times[name]:
            return sum(self.times[name]) / len(self.times[name])
        return 0
    
    def reset(self):
        self.times = {}
        self.start_times = {}
    
    def print_stats(self):
        print("\nTiming Statistics:")
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            print(f"  {name}: {avg:.4f}s (avg over {len(times)} calls)")


class MetricsTracker:
    """Track training/evaluation metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
            self.history[name] = []
        
        self.metrics[name].append(value)
    
    def get_avg(self, name, last_n=None):
        if name not in self.metrics:
            return 0
        
        values = self.metrics[name]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else 0
    
    def history_step(self):
        for name, values in self.metrics.items():
            if values:
                self.history[name].append(values[-1])
    
    def reset(self):
        self.metrics = {}
    
    def save(self, filepath):
        save_json({'metrics': self.metrics, 'history': self.history}, filepath)
    
    def load(self, filepath):
        data = load_json(filepath)
        self.metrics = data.get('metrics', {})
        self.history = data.get('history', {})