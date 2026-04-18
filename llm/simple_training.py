"""
Simple Working Training Module
Focuses on getting training to work rather than perfect gradients.
"""

import numpy as np
import os
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


def compute_perplexity(loss):
    return np.exp(loss)


class SimpleTrainer:
    """
    Simplified trainer that works - updates only key weights.
    With auto-save capability.
    """

    def __init__(self, model, learning_rate=0.01, save_interval=500, save_path=None):
        self.model = model
        self.lr = learning_rate
        self.loss_history = []
        self.save_interval = save_interval
        self.save_path = save_path
        self.total_steps = 0
        print(f"SimpleTrainer initialized (auto-save every {save_interval} steps)")

    def train_step(self, input_ids, target_ids):
        """Simple training step with approximated gradients."""
        # Forward pass
        embeddings = self.model['embedding'].forward(input_ids)
        pos_encoded = self.model['positional_encoding'].forward(embeddings)
        
        # Transformer blocks (no backward for now)
        transformer_out = self.model['transformer_stack'].forward(pos_encoded, training=True)
        
        # Output projection
        logits = self.model['output_linear'].forward(transformer_out)
        
        # Compute loss
        loss = cross_entropy_loss(logits, target_ids)
        
        # Simplified gradient - just update output layer
        batch_size, seq_len, vocab_size = logits.shape
        
        # Get vocab_size from model
        vocab_size = self.model['output_linear'].weights['W'].shape[1]
        
        # Compute gradient for output layer only
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        targets_onehot = np.zeros_like(probs)
        targets_onehot[np.arange(batch_size)[:, None], np.arange(seq_len), target_ids] = 1
        
        grad_logits = (probs - targets_onehot) / batch_size
        
        # Backward through output linear
        W = self.model['output_linear'].weights['W']
        
        # Gradient w.r.t. input to linear (d_model dimension)
        grad_input = np.matmul(grad_logits, W.T)  # (1, seq_len, d_model)
        
        # Gradient w.r.t. weights
        grad_logits_2d = grad_logits.reshape(-1, vocab_size)
        input_2d = transformer_out.reshape(-1, transformer_out.shape[-1])
        
        grad_W = np.matmul(input_2d.T, grad_logits_2d)
        
        # Update output weights
        self.model['output_linear'].weights['W'] -= self.lr * grad_W
        
        # Update embeddings - simplified: use gradient from transformer input
        # Get gradient at transformer input and propagate back to embeddings
        grad_pos = grad_input  # Same shape as pos_encoded (batch, seq, d_model)
        
        # Average gradient across sequence for embedding update
        avg_grad = np.mean(grad_pos, axis=(0, 1))  # (d_model,)
        
        # Update embeddings based on input tokens
        input_tokens = input_ids.flatten()
        for token_id in input_tokens:
            if token_id < self.model['embedding'].vocab_size:
                self.model['embedding'].weights['W'][token_id] -= self.lr * 0.01 * avg_grad
        
        self.loss_history.append(loss)
        self.total_steps += 1
        return loss
    
    def _save_checkpoint(self, step):
        """Save model checkpoint"""
        if not self.save_path:
            return
        
        from llm.llm_model import save_model
        
        # Create a model-like dict from the training model's components
        # For now, save using the LLM class with current params
        try:
            # Get the embedding and linear weights from training
            checkpoint_path = f"{self.save_path}_step{step}"
            
            # Save embedding weights
            emb_path = f"models/{checkpoint_path}_embedding.npy"
            np.save(emb_path, self.model['embedding'].weights['W'])
            
            # Save linear weights
            linear_path = f"models/{checkpoint_path}_linear.npy"
            np.save(linear_path, self.model['output_linear'].weights['W'])
            
            print(f"\n>>> Checkpoint saved at step {step}: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def train_on_texts(self, texts, epochs=10, block_size=64):
        """Train on texts."""
        tokenizer = Tokenizer(texts, vocab_size=self.model['embedding'].vocab_size)
        
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        print(f"Training on {len(all_tokens)} tokens...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_steps = 0
            
            indices = np.arange(max(0, len(all_tokens) - block_size))
            np.random.shuffle(indices)
            
            for i, idx in enumerate(indices):
                if idx + block_size + 1 > len(all_tokens):
                    continue
                
                input_ids = np.array(all_tokens[idx:idx+block_size]).reshape(1, -1)
                target_ids = np.array(all_tokens[idx+1:idx+block_size+1]).reshape(1, -1)
                
                loss = self.train_step(input_ids, target_ids)
                
                total_loss += loss
                num_steps += 1
                
                if num_steps % 100 == 0:
                    print(f"  Step {num_steps}, Loss: {loss:.4f}")
                
                # Auto-save every save_interval steps
                if self.save_interval > 0 and self.save_path and num_steps % self.save_interval == 0:
                    self._save_checkpoint(num_steps)
            
            avg_loss = total_loss / max(num_steps, 1)
            perplexity = compute_perplexity(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        print("Training complete!")


def train_model(model, texts, epochs=10, learning_rate=0.01, block_size=64, save_interval=500, save_path=None):
    """Train the model with auto-save capability."""
    import os
    
    # Create save directory
    if save_path:
        os.makedirs('models', exist_ok=True)
    
    # Pass save_interval and save_path to trainer
    trainer = SimpleTrainer(model, learning_rate=learning_rate, save_interval=save_interval, save_path=save_path)
    trainer.train_on_texts(texts, epochs=epochs, block_size=block_size)
    return trainer