"""
model.py — Complete LLM model.

Assembles all components into a single forward pass:
  token_ids → Embedding → PositionalEncoding → [TransformerBlock × N] → Linear → logits

C++ port:
  - struct LLMModel { Embedding emb; PositionalEncoding pos; vector<TransformerBlock> blocks; Linear head; }
  - model.forward() runs the entire pipeline
  - save/load via flat binary (memcpy entire weight buffer)
"""

import numpy as np
import os
import json
from typing import Optional
from optimized_llm.config import ModelConfig
from optimized_llm.embedding import TokenEmbedding
from optimized_llm.positional import build_positional_encoding, SinusoidalEncoding, RoPE
from optimized_llm.transformer_block import TransformerBlock
from optimized_llm.linear import Linear
from optimized_llm.attention import KVCache


class LLMModel:
    """
    Complete transformer language model.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.embedding = TokenEmbedding(cfg)
        self.pos_enc = build_positional_encoding(cfg)
        self.blocks = [TransformerBlock(cfg, i) for i in range(cfg.num_layers)]
        self.output_linear = Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: np.ndarray, mask: np.ndarray = None,
                training: bool = False,
                use_cache: bool = False, kv_cache: KVCache = None) -> np.ndarray:
        """
        Full forward pass.
        token_ids: (batch, seq) integer token IDs
        training: if True, enables dropout
        Returns: logits (batch, seq, vocab_size)
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        x = self.embedding.forward(token_ids)

        if isinstance(self.pos_enc, SinusoidalEncoding):
            x = self.pos_enc.forward(x)

        for block in self.blocks:
            x = block.forward(x, mask, training=training,
                              use_cache=use_cache, kv_cache=kv_cache)

        logits = self.output_linear.forward(x)
        return logits

    def get_parameters(self) -> dict:
        """Collect all trainable parameters into a flat dict."""
        params = {}
        params['embedding_W'] = self.embedding.W
        params['output_W'] = self.output_linear.weights['W']

        for i, block in enumerate(self.blocks):
            params[f'block_{i}_attn_W_qkv'] = block.attention.qkv_weights['W_qkv']
            if block.attention.qkv_weights.get('b_qkv') is not None:
                params[f'block_{i}_attn_b_qkv'] = block.attention.qkv_weights['b_qkv']
            params[f'block_{i}_ffn_W1_W'] = block.ffn.W1['W']
            params[f'block_{i}_ffn_W1_b'] = block.ffn.W1['b']
            params[f'block_{i}_ffn_W2_W'] = block.ffn.W2['W']
            params[f'block_{i}_ffn_W2_b'] = block.ffn.W2['b']
            params[f'block_{i}_gamma1'] = block.gamma1
            params[f'block_{i}_beta1'] = block.beta1
            params[f'block_{i}_gamma2'] = block.gamma2
            params[f'block_{i}_beta2'] = block.beta2

        return params

    def set_parameters(self, params: dict) -> None:
        """Restore all parameters from a dict."""
        self.embedding.W = params['embedding_W']
        self.output_linear.weights['W'] = params['output_W']

        for i, block in enumerate(self.blocks):
            block.attention.qkv_weights['W_qkv'] = params[f'block_{i}_attn_W_qkv']
            if f'block_{i}_attn_b_qkv' in params:
                block.attention.qkv_weights['b_qkv'] = params[f'block_{i}_attn_b_qkv']
            block.ffn.W1['W'] = params[f'block_{i}_ffn_W1_W']
            block.ffn.W1['b'] = params[f'block_{i}_ffn_W1_b']
            block.ffn.W2['W'] = params[f'block_{i}_ffn_W2_W']
            block.ffn.W2['b'] = params[f'block_{i}_ffn_W2_b']
            block.gamma1 = params[f'block_{i}_gamma1']
            block.beta1 = params[f'block_{i}_beta1']
            block.gamma2 = params[f'block_{i}_gamma2']
            block.beta2 = params[f'block_{i}_beta2']


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model: LLMModel, filepath: str,
               tokenizer=None) -> None:
    """Save model to .npz + config.json (+ tokenizer.json)."""
    params = model.get_parameters()
    np.savez_compressed(filepath + '.npz', **params)

    cfg_dict = {
        'vocab_size': model.cfg.vocab_size,
        'd_model': model.cfg.d_model,
        'num_layers': model.cfg.num_layers,
        'num_heads': model.cfg.num_heads,
        'd_ff': model.cfg.d_ff,
        'max_seq_len': model.cfg.max_seq_len,
        'dropout': model.cfg.dropout,
        'use_rope': model.cfg.use_rope,
        'use_bias': model.cfg.use_bias,
    }
    with open(filepath + '_config.json', 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    if tokenizer is not None:
        tdata = {
            'vocab_size': tokenizer.vocab_size,
            'vocab': tokenizer.vocab,
        }
        with open(filepath + '_tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(tdata, f, indent=2, ensure_ascii=False)

    print(f"Model saved to {filepath}.npz")


def load_tokenizer(filepath: str):
    """Load tokenizer from model's _tokenizer.json."""
    from optimized_llm.tokenizer import Tokenizer
    path = filepath + '_tokenizer.json'
    with open(path, 'r', encoding='utf-8') as f:
        tdata = json.load(f)
    tok = object.__new__(Tokenizer)
    tok.vocab_size = tdata['vocab_size']
    tok.vocab = tdata['vocab']
    tok.reverse_vocab = {v: k for k, v in tok.vocab.items()}
    tok.pad_token = '<pad>'
    tok.unk_token = '<unk>'
    tok.bos_token = '<s>'
    tok.eos_token = '</s>'
    # Rebuild trie
    tok.trie_root = None
    from optimized_llm.tokenizer import TrieNode
    tok.trie_root = TrieNode()
    for token, tid in tok.vocab.items():
        tok._add_to_trie(token, tid)
    return tok


def load_model(filepath: str) -> LLMModel:
    """Load model from .npz + config.json."""
    with open(filepath + '_config.json', 'r') as f:
        cfg_dict = json.load(f)

    cfg = ModelConfig(**cfg_dict)
    model = LLMModel(cfg)

    data = np.load(filepath + '.npz', allow_pickle=True)
    # Convert back to regular dict
    params = {k: data[k] for k in data.keys()}
    model.set_parameters(params)

    print(f"Model loaded from {filepath}")
    return model


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_next_token(logits: np.ndarray, temperature: float = 1.0,
                        top_k: int = 0) -> int:
    """Sample next token from logits vector."""
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    if top_k > 0:
        top_indices = np.argsort(probs)[-top_k:]
        probs = probs[top_indices]
        probs = probs / np.sum(probs)
        return int(np.random.choice(top_indices, p=probs))

    return int(np.random.choice(len(probs), p=probs))


def greedy_token(logits: np.ndarray) -> int:
    """Greedy: pick argmax."""
    return int(np.argmax(logits))
