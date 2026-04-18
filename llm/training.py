"""
Training Module - Legacy Support
Basic training utilities kept for backward compatibility.
Use enhanced_training.py for full-featured training.
"""

import numpy as np
import os
from llm.llm_model import LLM
from llm.tokenization import Tokenizer


def cross_entropy_loss(logits, target_tokens):
    """Compute cross-entropy loss."""
    if len(logits.shape) != 3:
        raise ValueError("Logits must be 3D (batch, seq_len, vocab_size)")

    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_tokens.reshape(-1)

    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    losses = -np.log(probs[np.arange(len(targets_flat)), targets_flat])
    return np.mean(losses)


def compute_loss_backward(logits, target_tokens):
    """Compute gradient of loss w.r.t. logits."""
    batch_size, seq_len, vocab_size = logits.shape

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    targets_onehot = np.zeros_like(probs)
    targets_onehot[np.arange(batch_size)[:, None], np.arange(seq_len), target_tokens] = 1

    grad_logits = probs - targets_onehot
    grad_logits = grad_logits / batch_size

    return grad_logits


class Trainer:
    """Legacy Trainer - use EnhancedTrainer instead."""

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate

    def train_step(self, input_ids, target_ids):
        """Single training step."""
        logits = self.model.forward(input_ids, training=True)
        loss = cross_entropy_loss(logits, target_ids)
        return loss

    def train_on_texts(self, texts, epochs=10):
        """Train on texts."""
        tokenizer = Tokenizer(texts, vocab_size=self.model.vocab_size)
        
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        print(f"Training on {len(all_tokens)} tokens...")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")


def train_step(model, input_ids, target_ids, optimizer_config):
    """Single training step."""
    logits = model.forward(input_ids, training=True)
    loss = cross_entropy_loss(logits, target_ids)
    return loss


def train_epoch(model, train_data, batch_size, learning_rate, optimizer_config):
    """Train for one epoch."""
    total_loss = 0
    num_batches = 0

    np.random.shuffle(train_data)

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        if len(batch) < batch_size:
            continue

        loss = train_step(model, batch, batch, optimizer_config)
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def load_text_file(filepath):
    """Load text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_text_files(directory):
    """Load all text files from a directory."""
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(('.txt', '.md', '.py', '.json')):
            filepath = os.path.join(directory, filename)
            try:
                texts.append(load_text_file(filepath))
            except:
                pass
    return texts


def fetch_from_internet(urls):
    """Fetch text from URLs."""
    texts = []
    for url in urls:
        try:
            import urllib.request
            response = urllib.request.urlopen(url, timeout=10)
            texts.append(response.read().decode('utf-8'))
        except:
            pass
    return texts


def prepare_training_data(texts, tokenizer, block_size=128):
    """Prepare training data from texts."""
    train_data = []

    for text in texts:
        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - block_size, block_size // 2):
            input_tokens = tokens[i:i+block_size]
            target_tokens = tokens[i+1:i+block_size+1]

            if len(input_tokens) < block_size:
                continue

            train_data.append((input_tokens, target_tokens))

    return train_data


class SimpleTrainer:
    """Simplified trainer for demonstration."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.model = LLM(vocab_size)

    def quick_train(self, text, iterations=100):
        """Quick training on text."""
        tokenizer = Tokenizer([text], vocab_size=self.vocab_size)
        tokens = tokenizer.encode(text)

        print(f"Training on {len(tokens)} tokens...")

        for i in range(iterations):
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations}")

        print("Training complete!")


class DataLoader:
    """Data loading utility."""

    def __init__(self, data, batch_size=32, block_size=128):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size

    def __iter__(self):
        for i in range(0, len(self.data) - self.block_size, self.block_size // 2):
            batch_data = self.data[i:i+self.batch_size]

            if len(batch_data) < self.batch_size:
                break

            yield batch_data

    def __len__(self):
        return (len(self.data) - self.block_size) // (self.block_size // 2)


def create_synthetic_data(vocab_size=1000, num_samples=10000, avg_length=50):
    """Create synthetic training data."""
    data = []

    for _ in range(num_samples):
        length = int(np.random.normal(avg_length, 10))
        length = max(10, min(length, avg_length * 2))

        tokens = np.random.randint(1, vocab_size, size=length).tolist()
        data.append(tokens)

    return data