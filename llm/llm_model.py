"""
LLM Model Module
The complete transformer-based language model putting all components together:
1. Token Embeddings
2. Positional Encoding
3. Stack of Transformer Blocks
4. Final Linear Layer (to vocabulary)

This is the core model that processes tokens and generates predictions.
"""

import numpy as np
from llm.embeddings import TokenEmbedding, lookup_embeddings
from llm.positional_encoding import PositionalEncoding
from llm.transformer_block import TransformerBlock, TransformerStack
from llm.linear import Linear


def build_llm(vocab_size, d_model, num_layers, num_heads, max_seq_length, d_ff=None):
    """
    Build the complete LLM.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_length: Maximum sequence length
        d_ff: Feed-forward dimension

    Returns:
        model: Dictionary with all model components
    """
    if d_ff is None:
        d_ff = d_model * 4

    model = {
        'embedding': TokenEmbedding(vocab_size, d_model),
        'positional_encoding': PositionalEncoding(max_seq_length, d_model),
        'transformer_stack': TransformerStack(num_layers, d_model, num_heads, d_ff),
        'output_linear': Linear(d_model, vocab_size, bias=False),
    }

    return model


def llm_forward(token_ids, model, mask=None, training=True):
    """
    Forward pass through the complete LLM.

    Args:
        token_ids: Input token IDs (batch, seq_len)
        model: LLM model dictionary
        mask: Optional attention mask
        training: Training mode

    Returns:
        logits: Output logits (batch, seq_len, vocab_size)
    """
    # Ensure 2D input
    if len(token_ids.shape) == 1:
        token_ids = token_ids.reshape(1, -1)

    # Embed tokens
    embeddings = model['embedding'].forward(token_ids)

    # Add positional encoding
    pos_encoding = model['positional_encoding'].forward(embeddings)

    # Pass through transformer stack
    transformer_output = model['transformer_stack'].forward(pos_encoding, mask, training)

    # Project to vocabulary
    logits = model['output_linear'].forward(transformer_output)

    return logits


def generate_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate next token from logits.

    Args:
        logits: Output logits (vocab_size,)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling threshold (0 = disabled)

    Returns:
        token_id: Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities via softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Top-k sampling
    if top_k > 0:
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)
        probs = np.zeros_like(probs)
        probs[top_indices] = top_probs

    # Nucleus sampling
    if top_p > 0:
        sorted_indices = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_indices])
        mask = cumsum <= top_p
        probs_filtered = np.zeros_like(probs)
        probs_filtered[sorted_indices[mask]] = probs[sorted_indices[mask]]
        probs_filtered = probs_filtered / np.sum(probs_filtered)
        probs = probs_filtered

    # Sample from distribution
    token_id = np.random.choice(len(probs), p=probs)

    return token_id


def generate_text(model, start_tokens, max_length, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate text autoregressively with KV cache support.

    Args:
        model: LLM model (dict or LLM object)
        start_tokens: Starting token IDs
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling

    Returns:
        generated_tokens: Generated token IDs
    """
    if hasattr(model, 'vocab_size'):
        model_dict = {
            'embedding': model.embedding,
            'positional_encoding': model.positional_encoding,
            'transformer_stack': model.transformer_stack,
            'output_linear': model.output_linear,
            'vocab_size': model.vocab_size,
        }
    else:
        model_dict = model

    current_tokens = list(start_tokens)
    kv_cache = None

    for _ in range(max_length - len(start_tokens)):
        input_ids = np.array(current_tokens).reshape(1, -1)

        max_seq = 64
        if input_ids.shape[1] > max_seq:
            break

        logits, kv_cache = llm_forward_with_cache(
            input_ids, model_dict, training=False, kv_cache=kv_cache
        )

        last_logits = logits[0, -1, :]
        next_token = generate_next_token(last_logits, temperature, top_k, top_p)

        current_tokens.append(next_token)

        if next_token == 0:
            break

    return current_tokens


def llm_forward_with_cache(token_ids, model, mask=None, training=True, kv_cache=None, use_cache=True):
    """
    Forward pass with KV cache for faster autoregressive generation.
    
    Args:
        token_ids: Input token IDs
        model: LLM model dictionary
        mask: Optional attention mask
        training: Training mode
        kv_cache: Cached K and V from previous forward passes
        use_cache: Whether to use/update cache
    
    Returns:
        logits: Output logits
        kv_cache: Updated KV cache
    """
    if len(token_ids.shape) == 1:
        token_ids = token_ids.reshape(1, -1)

    embeddings = model['embedding'].forward(token_ids)
    pos_encoding = model['positional_encoding'].forward(embeddings)

    transformer_output, new_cache = model['transformer_stack'].forward(
        pos_encoding, mask, training, use_cache=use_cache, kv_cache=kv_cache
    )

    logits = model['output_linear'].forward(transformer_output)

    return logits, new_cache


def beam_search(model, start_tokens, max_length, beam_width=3, temperature=1.0):
    """
    Beam search for better text generation.
    
    Args:
        model: LLM model
        start_tokens: Starting token IDs
        max_length: Maximum length
        beam_width: Number of beams to maintain
        temperature: Sampling temperature
    
    Returns:
        best_sequence: Best generated token sequence
    """
    if hasattr(model, 'vocab_size'):
        model_dict = {
            'embedding': model.embedding,
            'positional_encoding': model.positional_encoding,
            'transformer_stack': model.transformer_stack,
            'output_linear': model.output_linear,
            'vocab_size': model.vocab_size,
        }
    else:
        model_dict = model

    beams = [(list(start_tokens), 0.0)]
    completed = []

    for step in range(max_length - len(start_tokens)):
        all_candidates = []

        for seq, score in beams:
            if len(seq) > 0 and seq[-1] == 0:
                completed.append((seq, score))
                continue

            input_ids = np.array(seq).reshape(1, -1)
            
            try:
                logits = llm_forward(input_ids, model_dict, training=False)
                last_logits = logits[0, -1, :] / temperature
            except:
                continue

            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)

            top_k = min(beam_width * 2, len(probs))
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]

            for idx, prob in zip(top_indices, top_probs):
                new_seq = seq + [int(idx)]
                new_score = score - np.log(prob + 1e-10)
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x[1])
        beams = all_candidates[:beam_width]

        if all(seq[-1] == 0 for seq, _ in beams):
            completed.extend(beams)
            break

    all_candidates = beams + completed
    all_candidates.sort(key=lambda x: x[1])
    
    return all_candidates[0][0] if all_candidates else start_tokens


def generate_text_advanced(model, start_tokens, max_length, method='greedy',
                           temperature=1.0, top_k=0, top_p=0.0, beam_width=3):
    """
    Advanced generation with multiple methods.
    
    Args:
        model: LLM model
        start_tokens: Starting tokens
        max_length: Maximum length
        method: 'greedy', 'sampling', 'top_k', 'top_p', 'beam'
        temperature: Temperature
        top_k: Top-k
        top_p: Top-p (nucleus)
        beam_width: Beam width for beam search
    
    Returns:
        Generated tokens
    """
    if method == 'beam':
        return beam_search(model, start_tokens, max_length, beam_width, temperature)
    elif method == 'greedy':
        return generate_text(model, start_tokens, max_length, temperature=0.0, top_k=0, top_p=0)
    else:
        return generate_text(model, start_tokens, max_length, temperature, top_k, top_p)


class LLM:
    """
    Complete Language Model.
    """

    def __init__(self, vocab_size, d_model=512, num_layers=8, num_heads=8, 
                 max_seq_length=128, d_ff=None, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.d_ff = d_ff if d_ff else d_model * 4
        self.dropout = dropout

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model)
        self.transformer_stack = TransformerStack(num_layers, d_model, num_heads, self.d_ff, dropout)
        self.output_linear = Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, mask=None, training=True):
        """
        Forward pass.

        Args:
            token_ids: Input tokens (batch, seq_len)
            mask: Optional mask
            training: Training mode

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        # Ensure 2D
        if len(token_ids.shape) == 1:
            token_ids = token_ids.reshape(1, -1)

        # Embed
        embeddings = self.embedding.forward(token_ids)

        # Add positional encoding
        pos_encoded = self.positional_encoding.forward(embeddings)

        # Transformer
        output = self.transformer_stack.forward(pos_encoded, mask, training)

        # Project to vocab
        logits = self.output_linear.forward(output)

        return logits

    def generate(self, start_tokens, max_length, temperature=1.0, top_k=0, method='sampling'):
        """
        Generate text with various methods.

        Args:
            start_tokens: Starting token IDs
            max_length: Maximum length
            temperature: Sampling temperature
            top_k: Top-k sampling
            method: 'sampling', 'greedy', 'beam'

        Returns:
            Generated tokens
        """
        if method == 'greedy':
            return generate_text(self, start_tokens, max_length, temperature=0.0, top_k=0, top_p=0)
        elif method == 'beam':
            return beam_search(self, start_tokens, max_length, beam_width=3, temperature=temperature)
        else:
            return generate_text(self, start_tokens, max_length, temperature, top_k, top_p=0)

    def get_parameters(self):
        """Get all model parameters."""
        params = {
            'embedding': self.embedding.get_embeddings(),
            'output_linear': self.output_linear.get_weights(),
        }

        for i, block in enumerate(self.transformer_stack.get_blocks()):
            block_params = {
                'attention': block.attention.get_weights(),
                'ffn': block.ffn.get_weights(),
            }
            params[f'block_{i}'] = block_params

        return params


def save_model(model, filepath):
    """
    Save model parameters to file.

    Args:
        model: LLM model
        filepath: Path to save file (without extension)
    """
    params = model.get_parameters()

    filepath_np = filepath + '.npz'
    np.savez(filepath_np, **params)

    config_path = filepath + '_config.txt'
    with open(config_path, 'w') as f:
        f.write(f"vocab_size={model.vocab_size}\n")
        f.write(f"d_model={model.d_model}\n")
        f.write(f"num_layers={model.num_layers}\n")
        f.write(f"num_heads={model.num_heads}\n")
        f.write(f"max_seq_length={model.max_seq_length}\n")

    print(f"Model saved to {filepath}.npz and {filepath}_config.txt")


def load_model(filepath):
    """
    Load model from file.

    Args:
        filepath: Path to model file (without extension)

    Returns:
        model: Loaded model
    """
    config_path = filepath + '_config.txt'
    with open(config_path, 'r') as f:
        config = {}
        for line in f:
            key, value = line.strip().split('=')
            config[key] = int(value)

    model = LLM(**config)

    filepath_np = filepath + '.npz'
    data = np.load(filepath_np, allow_pickle=True)

    model.embedding.set_embeddings(data['embedding'])
    model.output_linear.set_weights(data['output_linear'].item())

    print(f"Model loaded from {filepath}")
    return model


def load_checkpoint(checkpoint_path):
    """
    Load model checkpoint from .npy files.

    Args:
        checkpoint_path: Path like "models/trained_model_step2000" (without _embedding.npy/_linear.npy)

    Returns:
        model: Loaded model
    """
    import os
    
    emb_path = checkpoint_path + "_embedding.npy"
    linear_path = checkpoint_path + "_linear.npy"
    
    if not os.path.exists(emb_path) or not os.path.exists(linear_path):
        raise FileNotFoundError(f"Checkpoint files not found: {emb_path}, {linear_path}")
    
    emb = np.load(emb_path)
    linear = np.load(linear_path)
    
    vocab_size = emb.shape[0]
    d_model = emb.shape[1]
    
    model = LLM(vocab_size=vocab_size, d_model=d_model)
    model.embedding.set_embeddings(emb)
    model.output_linear.set_weights({'W': linear, 'b': np.zeros(vocab_size)})
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model


class AutoTrainer:
    """
    Trainer care invata automat de pe internet.
    """

    def __init__(self, vocab_size=5000, d_model=256, num_layers=4, num_heads=4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = 128

        self.model = LLM(vocab_size, d_model, num_layers, num_heads, self.max_seq_length)
        self.tokenizer = None

    def learn_from_urls(self, urls, epochs=3):
        """
        Learn from internet URLs.

        Args:
            urls: List of URLs to fetch text from
            epochs: Number of training epochs
        """
        from training import fetch_from_internet

        print(f"Fetching text from {len(urls)} URLs...")
        texts = fetch_from_internet(urls)

        if not texts:
            print("No text fetched!")
            return

        print(f"Fetched {len(texts)} text samples")

        self.train(texts, epochs)


    def learn_from_file(self, filepath, epochs=3):
        """
        Learn from a file.

        Args:
            filepath: Path to text file
            epochs: Number of epochs
        """
        from training import load_text_file

        text = load_text_file(filepath)
        self.train([text], epochs)

    def learn_from_directory(self, directory, epochs=3):
        """
        Learn from all files in a directory.

        Args:
            directory: Directory path
            epochs: Number of epochs
        """
        from training import load_text_files

        texts = load_text_files(directory)
        self.train(texts, epochs)

    def train(self, texts, epochs=3):
        """
        Antreneaza modelul pe textele date.

        Args:
            texts: Lista de texte
            epochs: Numar de epoci
        """
        global Tokenizer

        print(f"Training on {len(texts)} text samples for {epochs} epochs...")

        all_text = ' '.join(texts[:10])
        self.tokenizer = Tokenizer([all_text], vocab_size=self.vocab_size)

        total_loss = 0

        for epoch in range(epochs):
            for text in texts[:5]:
                tokens = self.tokenizer.encode(text)

                if len(tokens) < 10:
                    continue

                block_size = min(32, len(tokens) - 1)

                for i in range(0, len(tokens) - block_size, block_size // 2):
                    input_ids = tokens[i:i+block_size]
                    target_ids = tokens[i+1:i+block_size+1]

                    if len(input_ids) < block_size:
                        continue

                    logits = self.model.forward(
                        np.array(input_ids).reshape(1, -1),
                        training=True
                    )

                    from training import cross_entropy_loss
                    loss = cross_entropy_loss(logits, np.array([target_ids]))
                    total_loss += loss

            avg_loss = total_loss / (len(texts[:5]) * 10)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            total_loss = 0

        print("Training complete!")

    def generate(self, prompt, max_length=50):
        """
        Genereaza text.

        Args:
            prompt: Prompt de start
            max_length: Lungime maxima

        Returns:
            text: Text generat
        """
        if self.tokenizer is None:
            print("Model not trained yet!")
            return ""

        tokens = self.tokenizer.encode(prompt)

        generated = self.model.generate(tokens, max_length)

        return self.tokenizer.decode(generated)

    def save(self, filepath):
        """Salveaza modelul."""
        from llm_model import save_model as save_llm
        save_llm(self.model, filepath)

        if self.tokenizer:
            import pickle
            with open(filepath + '_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)

    def load(self, filepath):
        """Incarca modelul."""
        from llm_model import load_model as load_llm
        self.model = load_llm(filepath)

        import pickle
        try:
            with open(filepath + '_tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
        except:
            pass


class SmallerLLM:
    """
    Compact version of LLM for demonstration.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = 128
        self.num_layers = 2
        self.num_heads = 4
        self.max_seq_length = 64

        self.llm = LLM(vocab_size, self.d_model, self.num_layers, self.num_heads, self.max_seq_length)

    def forward(self, token_ids):
        """Forward pass."""
        return self.llm.forward(token_ids)

    def generate(self, start_tokens, max_length):
        """Generate text."""
        return self.llm.generate(start_tokens, max_length)