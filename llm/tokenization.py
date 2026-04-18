"""
Tokenization Module
This module converts raw text into integer token IDs.
Implements BPE (Byte Pair Encoding) tokenizer with subword support.
"""

import numpy as np
import re
from collections import Counter


def build_vocab(texts, vocab_size=5000):
    """
    Build vocabulary from a list of texts.
    Uses BPE-style subword tokenization.

    Args:
        texts: List of text strings to build vocab from
        vocab_size: Target vocabulary size

    Returns:
        vocab: Dictionary mapping token string to token ID
        reverse_vocab: Dictionary mapping token ID to token string
    """
    char_counts = {}
    for text in texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

    vocab = {char: idx for idx, char in enumerate(sorted(char_counts.keys()))}
    
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    max_iterations = min(vocab_size - len(vocab), 100)
    for _ in range(max_iterations):
        bigram_counts = {}
        for text in texts:
            for i in range(len(text) - 1):
                bigram = text[i:i+2]
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        if not bigram_counts:
            break

        most_common = max(bigram_counts, key=bigram_counts.get)
        if most_common not in vocab:
            vocab[most_common] = len(vocab)

        if len(vocab) >= vocab_size:
            break

    reverse_vocab = {idx: token for token, idx in vocab.items()}

    return vocab, reverse_vocab


def tokenize(text, vocab):
    """
    Convert text to sequence of token IDs using BPE-style tokenization.

    Args:
        text: Input text string
        vocab: Vocabulary dictionary

    Returns:
        tokens: List of token IDs
    """
    tokens = []
    unknown_token = vocab.get('<unk>', 0)
    
    text = text.lower().strip()
    
    i = 0
    while i < len(text):
        matched = False
        for length in range(min(len(text) - i, 12), 0, -1):
            substring = text[i:i+length]
            if substring in vocab:
                tokens.append(vocab[substring])
                i += length
                matched = True
                break
        
        if not matched:
            if text[i] in vocab:
                tokens.append(vocab[text[i]])
            else:
                tokens.append(unknown_token)
            i += 1

    return tokens


def detokenize(tokens, reverse_vocab):
    """
    Convert token IDs back to text.

    Args:
        tokens: List of token IDs
        reverse_vocab: Reverse vocabulary dictionary

    Returns:
        text: Reconstructed text string
    """
    text = ""
    for token_id in tokens:
        if token_id in reverse_vocab:
            text += reverse_vocab[token_id]
    return text


def encode(text, vocab):
    """Alias for tokenize function."""
    return tokenize(text, vocab)


def decode(tokens, reverse_vocab):
    """Alias for detokenize function."""
    return detokenize(tokens, reverse_vocab)


class Tokenizer:
    """
    Tokenizer class with BPE-style subword tokenization.
    """

    def __init__(self, texts=None, vocab_size=5000, add_special_tokens=True):
        self.vocab_size = vocab_size
        self.vocab = None
        self.reverse_vocab = None
        self.add_special_tokens = add_special_tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        if texts is not None:
            self.build_vocab(texts)

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        processed_texts = [self._preprocess(t) for t in texts]
        self.vocab, self.reverse_vocab = build_vocab(processed_texts, self.vocab_size)
        self.vocab_size = len(self.vocab)

    def _preprocess(self, text):
        """Preprocess text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def encode(self, text, max_length=None, padding='post', truncation=False):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum length
            padding: 'post' or 'pre'
            truncation: Whether to truncate
        
        Returns:
            tokens: List of token IDs
        """
        tokens = tokenize(text, self.vocab)
        
        if self.add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, 0)
            eos_id = self.vocab.get(self.eos_token, 0)
            tokens = [bos_id] + tokens + [eos_id]
        
        if max_length is not None:
            if len(tokens) > max_length:
                if truncation:
                    tokens = tokens[:max_length]
                else:
                    tokens = tokens + [self.vocab.get(self.pad_token, 0)] * (max_length - len(tokens))
            else:
                pad_id = self.vocab.get(self.pad_token, 0)
                if padding == 'post':
                    tokens = tokens + [pad_id] * (max_length - len(tokens))
                else:
                    tokens = [pad_id] * (max_length - len(tokens)) + tokens
        
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        """
        Decode tokens to text.
        
        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            text: Decoded text
        """
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        
        result = []
        for token_id in tokens:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if skip_special_tokens and token in special:
                    continue
                result.append(token)
        
        return ''.join(result)

    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.vocab_size