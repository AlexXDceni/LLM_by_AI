"""
inference.py — Text generation with KV cache and multiple strategies.

C++ port:
  - generate(): one-step-ahead loop with cached K/V
  - beam_search(): maintain top-K beams, each with own KVCache
"""

import numpy as np
from typing import Optional
from optimized_llm.config import ModelConfig
from optimized_llm.model import LLMModel, generate_next_token, greedy_token
from optimized_llm.attention import KVCache


def generate_text(
    model: LLMModel,
    start_tokens: list[int],
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    use_cache: bool = True,
) -> list[int]:
    """
    Autoregressive generation with optional KV cache.
    C++: while (generated < max_len) { forward(token); sample(); shift cache; }
    """
    cfg = model.cfg
    current = list(start_tokens)
    kv_cache = KVCache(cfg.max_seq_len, cfg.num_layers, 1, cfg.num_heads, cfg.head_dim) \
        if use_cache else None

    for step in range(max_length - len(start_tokens)):
        # Only feed the last token when using cache
        if use_cache and kv_cache is not None and len(current) > len(start_tokens):
            input_ids = np.array([current[-1:]], dtype=np.int32)  # (1, 1)
        else:
            input_ids = np.array([current], dtype=np.int32)      # (1, seq)

        logits = model.forward(input_ids, use_cache=use_cache, kv_cache=kv_cache)
        last_logits = logits[0, -1, :]

        if temperature == 0.0:
            next_token = greedy_token(last_logits)
        else:
            next_token = generate_next_token(last_logits, temperature, top_k)

        current.append(next_token)

        if next_token == 0:  # EOS
            break

        # Safety: prevent runaway
        if len(current) >= max_length:
            break

    return current


def generate_text_advanced(
    model: LLMModel,
    start_tokens: list[int],
    max_length: int = 100,
    method: str = 'sampling',
    temperature: float = 1.0,
    top_k: int = 40,
    beam_width: int = 3,
) -> list[int]:
    """Generate with multiple strategies."""
    if method == 'beam':
        return beam_search(model, start_tokens, max_length, beam_width, temperature)
    elif method == 'greedy':
        return generate_text(model, start_tokens, max_length, temperature=0.0)
    else:
        return generate_text(model, start_tokens, max_length, temperature, top_k)


def beam_search(
    model: LLMModel,
    start_tokens: list[int],
    max_length: int = 100,
    beam_width: int = 3,
    temperature: float = 1.0,
) -> list[int]:
    """
    Beam search with cumulative log-probability scoring.
    C++: vector<pair<vector<int>, float>> beams; expand, prune, repeat.
    """
    beams: list[tuple[list[int], float]] = [(list(start_tokens), 0.0)]
    completed: list[tuple[list[int], float]] = []

    for step in range(max_length - len(start_tokens)):
        candidates: list[tuple[list[int], float]] = []

        for seq, score in beams:
            if seq[-1] == 0:  # EOS
                completed.append((seq, score))
                continue

            input_ids = np.array([seq], dtype=np.int32)
            logits = model.forward(input_ids)
            last_logits = logits[0, -1, :] / max(temperature, 1e-8)

            exp_l = np.exp(last_logits - np.max(last_logits))
            probs = exp_l / np.sum(exp_l)

            top_probs_idx = np.argsort(probs)[-beam_width * 2:]
            for idx in top_probs_idx:
                token_id = int(idx)
                prob = float(probs[idx])
                new_seq = seq + [token_id]
                new_score = score - np.log(prob + 1e-10)
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1])
        beams = candidates[:beam_width]

        if all(seq[-1] == 0 for seq, _ in beams):
            completed.extend(beams)
            break

    all_candidates = beams + completed
    all_candidates.sort(key=lambda x: x[1])
    return all_candidates[0][0] if all_candidates else start_tokens
