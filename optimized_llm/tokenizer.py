"""
tokenizer.py — BPE tokenizer with Trie for O(n) tokenization.

C++ port:
  - Trie: struct Node { unordered_map<char, Node*> children; int id = -1; }
  - Encode: walk trie char by char, longest match wins
  - Build: count bigrams, merge most frequent
"""

import numpy as np
import re
from collections import Counter, defaultdict


class TrieNode:
    """Single node in the tokenizer trie."""

    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.token_id: int = -1


class Tokenizer:
    """
    BPE tokenizer with Trie-based encoding.
    """

    def __init__(self, texts: list[str] = None, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.trie_root = TrieNode()

        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'

        if texts:
            self.build_vocab(texts)

    def _add_to_trie(self, token: str, token_id: int) -> None:
        """Insert a token into the trie."""
        node = self.trie_root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.token_id = token_id

    def _longest_match(self, text: str, start: int) -> tuple[int, int]:
        """Walk trie from start, return (end_pos, token_id).
        C++: node = root; while (ch = text[pos]) { if (node->children[ch]) ... }
        """
        node = self.trie_root
        last_id = self.vocab.get(self.unk_token, 0)
        last_pos = start
        pos = start

        while pos < len(text):
            ch = text[pos]
            if ch in node.children:
                node = node.children[ch]
                pos += 1
                if node.token_id >= 0:
                    last_id = node.token_id
                    last_pos = pos
            else:
                break

        if last_pos == start:
            return start + 1, last_id
        return last_pos, last_id

    def _preprocess(self, text: str) -> str:
        """Lowercase + normalize whitespace."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary using BPE."""
        processed = [self._preprocess(t) for t in texts]

        # Start with character-level vocab
        char_counts: Counter = Counter()
        for text in processed:
            for ch in text:
                char_counts[ch] += 1

        self.vocab = {ch: i for i, (ch, _) in enumerate(sorted(char_counts.items()))}

        # Add special tokens
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

        # BPE merges
        max_merges = min(self.vocab_size - len(self.vocab), 1000)
        for _ in range(max_merges):
            bigram_counts: Counter = Counter()
            for text in processed:
                for i in range(len(text) - 1):
                    bigram = text[i:i + 2]
                    if bigram not in self.vocab:
                        bigram_counts[bigram] += 1

            if not bigram_counts:
                break

            most_common = bigram_counts.most_common(1)[0][0]
            self.vocab[most_common] = len(self.vocab)
            if len(self.vocab) >= self.vocab_size:
                break

        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Build trie
        self.trie_root = TrieNode()
        for token, tid in self.vocab.items():
            self._add_to_trie(token, tid)

    def encode(self, text: str, max_length: int = None) -> list[int]:
        """
        Encode text to token IDs using Trie (O(n)).
        C++: while (pos < len(text)) { auto [next_pos, id] = trie.longest_match(text, pos); }
        """
        text = self._preprocess(text)
        tokens: list[int] = []

        # Add BOS token
        bos_id = self.vocab.get(self.bos_token, 0)
        tokens.append(bos_id)

        pos = 0
        while pos < len(text):
            next_pos, token_id = self._longest_match(text, pos)
            tokens.append(token_id)
            pos = next_pos

        # Add EOS token
        eos_id = self.vocab.get(self.eos_token, 0)
        tokens.append(eos_id)

        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, tokens: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        result: list[str] = []

        for tid in tokens:
            if tid in self.reverse_vocab:
                token = self.reverse_vocab[tid]
                if skip_special and token in special:
                    continue
                result.append(token)

        return ''.join(result)

    def get_vocab_size(self) -> int:
        return self.vocab_size
