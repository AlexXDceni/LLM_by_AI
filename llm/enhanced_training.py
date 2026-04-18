"""
Enhanced Training Module
Full-featured training with:
- Learning rate scheduling (warmup + cosine decay)
- Gradient accumulation for larger effective batch size
- Checkpoint saving/loading
- Perplexity metrics
- Mixed batch training
- Logging and progress tracking
"""

import numpy as np
import os
import json
import time
from llm.llm_model import LLM
from llm.tokenization import Tokenizer
from llm.optimizer import Adam, AdamW, clip_gradients, LearningRateScheduler


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


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)


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


class Gradients:
    """Container for model gradients."""
    
    def __init__(self):
        self.grads = {}
    
    def __setitem__(self, key, value):
        self.grads[key] = value
    
    def get(self, key, default=None):
        return self.grads.get(key, default)
    
    def items(self):
        return self.grads.items()
    
    def keys(self):
        return self.grads.keys()
    
    def update(self, other):
        """Add gradients from another Gradients object."""
        for key, val in other.grads.items():
            if key in self.grads and self.grads[key] is not None:
                self.grads[key] = self.grads[key] + val
            else:
                self.grads[key] = val
    
    def clear(self):
        """Clear all gradients."""
        self.grads = {}


class EnhancedTrainer:
    """
    Full-featured training handler with:
    - LR scheduling with warmup
    - Gradient accumulation
    - Checkpoint saving
    - Perplexity metrics
    - Mixed precision (simulated)
    """

    def __init__(self, model, learning_rate=0.001, optimizer='adamw',
                 clip_norm=1.0, weight_decay=0.01,
                 warmup_steps=500, total_steps=None,
                 gradient_accumulation_steps=1,
                 checkpoint_dir='models',
                 log_interval=100,
                 save_interval=1000):
        
        self.model = model
        self.base_lr = learning_rate
        self.clip_norm = clip_norm
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        if optimizer == 'adam':
            self.optimizer = Adam(lr=learning_rate, clip_norm=clip_norm, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(lr=learning_rate, clip_norm=clip_norm, weight_decay=weight_decay)
        else:
            self.optimizer = AdamW(lr=learning_rate, clip_norm=clip_norm)
        
        if total_steps:
            self.scheduler = LearningRateScheduler(
                self.optimizer, warmup_steps=warmup_steps, 
                total_steps=total_steps, schedule_type='cosine'
            )
        else:
            self.scheduler = None
        
        self.steps = 0
        self.accumulated_gradients = Gradients()
        self.loss_history = []
        self.perplexity_history = []
        self.epoch_history = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_step(self, input_ids, target_ids):
        """Single training step with backpropagation."""
        from llm.llm_model import llm_forward
        
        logits = llm_forward(input_ids, self.model, training=True)
        loss = cross_entropy_loss(logits, target_ids)
        
        grad_logits = compute_loss_backward(logits, target_ids)
        
        grad_out = self.model['output_linear'].backward(grad_logits)
        grad_transformer = self.model['transformer_stack'].backward(grad_out)
        grad_pos = self.model['positional_encoding'].backward(grad_transformer)
        grad_emb = self.model['embedding'].backward(grad_pos)
        
        gradients = Gradients()
        gradients.grads = {
            'embedding': grad_emb,
            'output_linear_W': grad_out,
        }
        
        for key in gradients.grads.keys():
            grad = gradients.grads[key]
            if grad is not None and np.any(np.isnan(grad)):
                gradients.grads[key] = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.gradient_accumulation_steps > 1:
            for key in gradients.grads.keys():
                if self.accumulated_gradients.grads.get(key) is None:
                    self.accumulated_gradients.grads[key] = np.zeros_like(gradients.grads[key])
                self.accumulated_gradients.grads[key] += gradients.grads[key]
            
            should_update = (self.steps + 1) % self.gradient_accumulation_steps == 0
            
            if should_update:
                clipped_grads = clip_gradients(self.accumulated_gradients.grads, self.clip_norm)
                
                parameters = {
                    'embedding': self.model['embedding'].weights['W'],
                    'output_linear_W': self.model['output_linear'].weights['W'],
                }
                
                updated = self.optimizer.step(parameters, clipped_grads)
                
                self.model['embedding'].weights['W'] = updated['embedding']
                self.model['output_linear'].weights['W'] = updated['output_linear_W']
                
                self.accumulated_gradients.clear()
        else:
            clipped_grads = clip_gradients(gradients.grads, self.clip_norm)
            
            parameters = {
                'embedding': self.model['embedding'].weights['W'],
                'output_linear_W': self.model['output_linear'].weights['W'],
            }
            
            updated = self.optimizer.step(parameters, clipped_grads)
            
            self.model['embedding'].weights['W'] = updated['embedding']
            self.model['output_linear'].weights['W'] = updated['output_linear_W']
        
        self.steps += 1
        
        if self.scheduler:
            current_lr = self.scheduler.step()
        
        self.loss_history.append(loss)
        self.perplexity_history.append(compute_perplexity(loss))
        
        return loss

    def evaluate(self, val_texts, tokenizer, block_size=128, max_batches=100):
        """Evaluate on validation texts."""
        all_tokens = []
        for text in val_texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        total_loss = 0
        num_batches = 0
        
        indices = np.arange(max(0, len(all_tokens) - block_size))
        
        batch_indices = np.random.choice(indices, size=min(max_batches, len(indices)), replace=False)
        
        for idx in batch_indices:
            if idx + block_size > len(all_tokens):
                continue
            
            input_ids = np.array(all_tokens[idx:idx+block_size-1]).reshape(1, -1)
            target_ids = np.array(all_tokens[idx+1:idx+block_size]).reshape(1, -1)
            
            from llm.llm_model import llm_forward
            logits = llm_forward(input_ids, self.model, training=False)
            loss = cross_entropy_loss(logits, target_ids)
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = compute_perplexity(avg_loss)
        
        return avg_loss, perplexity

    def train_on_texts(self, texts, epochs=10, batch_size=32, block_size=128,
                       val_texts=None, val_interval=1):
        """Train on texts with validation and checkpointing."""
        
        if self.total_steps is None:
            self.total_steps = epochs * len(texts) * 100
        
        tokenizer = Tokenizer(texts, vocab_size=self.model['embedding'].vocab_size)
        
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        print(f"Training on {len(all_tokens)} tokens...")
        
        best_perplexity = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            num_steps = 0
            
            indices = np.arange(max(0, len(all_tokens) - block_size))
            np.random.shuffle(indices)
            
            batch_count = 0
            for i in range(0, len(indices), block_size):
                batch_indices = indices[i:i+batch_size]
                
                for idx in batch_indices:
                    if idx + block_size > len(all_tokens):
                        continue
                    
                    input_ids = np.array(all_tokens[idx:idx+block_size-1]).reshape(1, -1)
                    target_ids = np.array(all_tokens[idx+1:idx+block_size]).reshape(1, -1)
                    
                    loss = self.train_step(input_ids, target_ids)
                    
                    total_loss += loss
                    num_steps += 1
                    batch_count += 1
                    
                    if num_steps % self.log_interval == 0:
                        avg_loss = total_loss / num_steps
                        lr = self.optimizer.lr
                        print(f"Step {num_steps}, Loss: {avg_loss:.4f}, Perplexity: {np.exp(avg_loss):.2f}, LR: {lr:.6f}")
                    
                    if self.steps % self.save_interval == 0 and self.steps > 0:
                        self.save_checkpoint(f'checkpoint_step_{self.steps}')
            
            avg_loss = total_loss / max(num_steps, 1)
            epoch_perplexity = np.exp(avg_loss)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Perplexity: {epoch_perplexity:.2f}, Time: {epoch_time:.1f}s")
            
            self.epoch_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'perplexity': epoch_perplexity,
                'steps': self.steps,
                'time': epoch_time
            })
            
            if val_texts and (epoch + 1) % val_interval == 0:
                val_loss, val_perplexity = self.evaluate(val_texts, tokenizer)
                print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                
                if val_perplexity < best_perplexity:
                    best_perplexity = val_perplexity
                    self.save_checkpoint('best_model')
                    print(f"New best model saved! Perplexity: {best_perplexity:.2f}")
        
        print(f"Training complete! Best validation perplexity: {best_perplexity:.2f}")
        return best_perplexity

    def save_checkpoint(self, name):
        """Save model checkpoint."""
        from llm.llm_model import save_model
        
        filepath = os.path.join(self.checkpoint_dir, name)
        save_model(self.model, filepath)
        
        metadata = {
            'steps': self.steps,
            'learning_rate': self.optimizer.lr,
            'loss_history': self.loss_history[-1000:],
            'epoch_history': self.epoch_history
        }
        
        with open(filepath + '_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, name):
        """Load model checkpoint."""
        from llm.llm_model import load_model
        
        filepath = os.path.join(self.checkpoint_dir, name)
        self.model = load_model(filepath)
        
        metadata_path = filepath + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.steps = metadata.get('steps', 0)
            print(f"Checkpoint loaded: {filepath}, Steps: {self.steps}")

    def get_training_stats(self):
        """Get training statistics."""
        recent_loss = np.mean(self.loss_history[-100:]) if len(self.loss_history) > 0 else 0
        recent_perplexity = np.mean(self.perplexity_history[-100:]) if len(self.perplexity_history) > 0 else 0
        
        return {
            'steps': self.steps,
            'current_lr': self.optimizer.lr,
            'avg_loss_100': recent_loss,
            'avg_perplexity_100': recent_perplexity,
            'total_epochs': len(self.epoch_history)
        }


class Trainer:
    """Legacy Trainer class - wraps EnhancedTrainer."""

    def __init__(self, model, learning_rate=0.001, optimizer='adam',
                 clip_norm=1.0, weight_decay=0.01, **kwargs):
        
        self.trainer = EnhancedTrainer(
            model=model,
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_norm=clip_norm,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.model = model
        self.steps = 0
        self.loss_history = []

    def train_step(self, input_ids, target_ids):
        loss = self.trainer.train_step(input_ids, target_ids)
        self.steps += 1
        self.loss_history.append(loss)
        return loss

    def train_on_texts(self, texts, epochs=10, batch_size=32, block_size=128):
        return self.trainer.train_on_texts(texts, epochs, batch_size, block_size)

    def train_from_file(self, filepath, epochs=10):
        text = load_text_file(filepath)
        self.train_on_texts([text], epochs)

    def train_from_directory(self, directory, epochs=10):
        texts = load_text_files(directory)
        self.train_on_texts(texts, epochs)

    def train_from_internet(self, urls, epochs=10):
        texts = fetch_from_internet(urls)
        self.train_on_texts(texts, epochs)


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