"""
Inference Module
Handles generating text (inference/prediction) with the trained LLM.
Supports various decoding strategies: greedy, random sampling, beam search.
"""

import numpy as np
from llm.llm_model import generate_text
from llm.tokenization import Tokenizer


def greedy_decode(logits):
    """
    Greedy decoding: always pick the highest probability token.

    Args:
        logits: Output logits (vocab_size,)

    Returns:
        token_id: Selected token ID
    """
    return np.argmax(logits)


def random_sample(logits, temperature=1.0):
    """
    Random sampling with temperature.

    Args:
        logits: Output logits
        temperature: Sampling temperature (higher = more random)

    Returns:
        token_id: Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    token_id = np.random.choice(len(probs), p=probs)

    return token_id


def top_k_sample(logits, top_k=40):
    """
    Top-k sampling: only consider top k tokens.

    Args:
        logits: Output logits
        top_k: Number of top tokens to consider

    Returns:
        token_id: Sampled token ID
    """
    # Get top k indices
    top_indices = np.argsort(logits)[-top_k:]
    top_logits = logits[top_indices]

    # Convert to probabilities
    exp_logits = np.exp(top_logits - np.max(top_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sample from top k
    chosen_idx = np.random.choice(top_k, p=probs)

    return top_indices[chosen_idx]


def top_p_sample(logits, top_p=0.9):
    """
    Nucleus/top-p sampling: sample from smallest set with cumulative probability >= p.

    Args:
        logits: Output logits
        top_p: Cumulative probability threshold

    Returns:
        token_id: Sampled token ID
    """
    # Convert to probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find smallest set with cumulative probability >= top_p
    cumsum = np.cumsum(sorted_probs)
    mask = cumsum <= top_p

    # Include at least one token
    if np.any(mask):
        mask[np.argmax(mask)] = True
    else:
        mask = np.ones_like(mask, dtype=bool)

    # Filter to selected tokens
    selected_indices = sorted_indices[mask]
    selected_probs = sorted_probs[mask]
    selected_probs = selected_probs / np.sum(selected_probs)

    # Sample
    chosen_idx = np.random.choice(len(selected_indices), p=selected_probs)

    return selected_indices[chosen_idx]


def beam_search_decode(model, start_tokens, max_length, beam_width=3):
    """
    Beam search: keep beam_width most likely sequences.

    Args:
        model: LLM model
        start_tokens: Starting tokens
        max_length: Maximum length
        beam_width: Number of beams to maintain

    Returns:
        best_sequence: Best generated sequence
    """
    # Initialize beams with start tokens
    beams = [list(start_tokens)]

    for _ in range(max_length - len(start_tokens)):
        candidates = []

        for beam in beams:
            # Get logits
            logits = model.forward(np.array(beam).reshape(1, -1), training=False)
            last_logits = logits[0, -1, :]

            # Get top beam_width tokens
            top_indices = np.argsort(last_logits)[-beam_width:]

            for idx in top_indices:
                new_beam = beam + [int(idx)]
                score = last_logits[idx]
                candidates.append((new_beam, score))

        # Keep best beams
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        beams = [c[0] for c in candidates[:beam_width]]

    # Return best sequence
    return beams[0]


class Decoder:
    """
    Text decoder with multiple strategies.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt, max_length=100, method='greedy', temperature=1.0, top_k=40, top_p=0.9):
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length to generate
            method: Decoding method ('greedy', 'random', 'top_k', 'top_p', 'beam')
            temperature: Temperature for random sampling
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            generated_text: Generated text
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)

        if len(tokens) == 0:
            return ""

        # Convert to array
        current_tokens = list(tokens)

        for _ in range(max_length - len(tokens)):
            # Get model output
            input_ids = np.array(current_tokens).reshape(1, -1)

            # Forward pass
            logits = self.model.forward(input_ids, training=False)
            last_logits = logits[0, -1, :]

            # Select next token based on method
            if method == 'greedy':
                next_token = greedy_decode(last_logits)
            elif method == 'random':
                next_token = random_sample(last_logits, temperature)
            elif method == 'top_k':
                next_token = top_k_sample(last_logits, top_k)
            elif method == 'top_p':
                next_token = top_p_sample(last_logits, top_p)
            else:
                next_token = greedy_decode(last_logits)

            current_tokens.append(next_token)

            # Stop if EOS
            if next_token == 0:
                break

            # Check sequence length
            if len(current_tokens) >= max_length:
                break

        # Decode tokens back to text
        generated_text = self.tokenizer.decode(current_tokens)

        return generated_text

    def generate_with_prefix(self, prompt, max_length=100):
        """Generate with the prompt as prefix."""
        return self.generate(prompt, max_length)


class InferenceEngine:
    """
    Complete inference engine.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = Decoder(model, tokenizer)

    def complete(self, text, max_length=100, method='greedy'):
        """
        Complete text.

        Args:
            text: Input text
            max_length: Maximum tokens to generate
            method: Decoding method

        Returns:
            completion: Text completion
        """
        return self.decoder.generate(text, max_length, method)

    def chat(self, message, max_length=100):
        """
        Simple chat response.

        Args:
            message: User message
            max_length: Maximum response length

        Returns:
            response: Model response
        """
        return self.decoder.generate(message, max_length)


def create_inference_engine(vocab_size=5000):
    """
    Create inference engine for quick testing.

    Args:
        vocab_size: Vocabulary size

    Returns:
        engine: Inference engine
    """
    from llm_model import LLM

    # Create model
    model = LLM(vocab_size)

    # Create tokenizer
    tokenizer = Tokenizer(vocab_size=vocab_size)

    return InferenceEngine(model, tokenizer)


def interactive_mode(model, tokenizer):
    """
    Interactive text generation mode.

    Args:
        model: LLM model
        tokenizer: Tokenizer
    """
    decoder = Decoder(model, tokenizer)

    print("Interactive mode. Type 'quit' to exit.")

    while True:
        prompt = input("\nEnter text: ")

        if prompt.lower() == 'quit':
            break

        method = input("Method (greedy/random/top_k/top_p): ") or 'greedy'

        completion = decoder.generate(prompt, method=method)

        print(f"\nGenerated: {completion}\n")


def batch_generate(model, prompts, max_lengths=None):
    """
    Generate text for multiple prompts.

    Args:
        model: LLM model
        prompts: List of prompts
        max_lengths: List of max lengths (or single value)

    Returns:
        completions: List of generated texts
    """
    if max_lengths is None:
        max_lengths = [100] * len(prompts)
    elif isinstance(max_lengths, int):
        max_lengths = [max_lengths] * len(prompts)

    completions = []
    decoder = Decoder(model, None)  # We'd need tokenizer here

    for prompt, max_len in zip(prompts, max_lengths):
        completion = decoder.generate(prompt, max_length=max_len)
        completions.append(completion)

    return completions


class StreamingGenerator:
    """
    Generator that yields tokens one at a time.
    """

    def __init__(self, model):
        self.model = model

    def generate_stream(self, prompt, max_length=100):
        """
        Generate tokens as a stream.

        Args:
            prompt: Input prompt
            max_length: Maximum length

        Yields:
            token_id: Each generated token
        """
        tokens = list(prompt)

        for _ in range(max_length - len(tokens)):
            input_ids = np.array(tokens).reshape(1, -1)

            logits = self.model.forward(input_ids, training=False)
            last_logits = logits[0, -1, :]

            next_token = np.argmax(last_logits)

            yield next_token

            tokens.append(next_token)

            if next_token == 0:
                break


def evaluate_perplexity(model, tokens):
    """
    Calculate perplexity of a token sequence.

    Perplexity = exp(average cross-entropy loss)

    Lower is better.

    Args:
        model: LLM model
        tokens: Token sequence

    Returns:
        perplexity: Perplexity score
    """
    from training import cross_entropy_loss

    total_loss = 0
    num_tokens = 0

    for i in range(1, len(tokens)):
        input_tokens = tokens[:i]
        target_token = tokens[i]

        logits = model.forward(np.array(input_tokens).reshape(1, -1), training=False)

        loss = cross_entropy_loss(logits, np.array([target_token]))
        total_loss += loss
        num_tokens += 1

    if num_tokens == 0:
        return float('inf')

    avg_loss = total_loss / num_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def measure_quality(model, test_data):
    """
    Measure model quality on test data.

    Args:
        model: LLM model
        test_data: List of token sequences

    Returns:
        metrics: Dictionary of quality metrics
    """
    total_loss = 0
    total_perplexity = 0

    for tokens in test_data:
        perplexity = evaluate_perplexity(model, tokens)
        total_perplexity += perplexity

    avg_perplexity = total_perplexity / len(test_data)

    return {
        'perplexity': avg_perplexity,
    }