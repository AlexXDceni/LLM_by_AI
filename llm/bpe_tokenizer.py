"""
BPE (Byte Pair Encoding) Tokenizer
More efficient tokenization that handles subword units.
"""

import re
import json
from collections import Counter, defaultdict


class BPETokenizer:
    """
    BPE Tokenizer implementation.
    """

    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.merges = []
        self.vocab = {}
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def train(self, texts, vocab_size=None):
        """
        Train BPE on texts.
        
        Args:
            texts: List of text strings
            vocab_size: Target vocabulary size
        """
        if vocab_size is None:
            vocab_size = self.vocab_size

        print(f"Training BPE on {len(texts)} texts...")

        word_freq = self._get_word_frequencies(texts)
        
        vocab = set()
        for word in word_freq:
            vocab.update(word)
        
        print(f"Initial vocabulary size: {len(vocab)}")

        self.merges = []
        
        while len(vocab) < vocab_size:
            pairs = self._get_pair_frequencies(word_freq)
            
            if not pairs:
                break
            
            most_common = max(pairs.items(), key=lambda x: x[1])
            best_pair = most_common[0]
            
            self.merges.append(best_pair)
            vocab.add(''.join(best_pair))
            
            word_freq = self._apply_merge(word_freq, best_pair)

        self._build_vocab(vocab)
        
        print(f"BPE training complete. Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")

    def _get_word_frequencies(self, texts):
        """Count word frequencies."""
        word_freq = Counter()
        
        for text in texts:
            words = self._preprocess(text).split()
            word_freq.update(words)
        
        return word_freq

    def _preprocess(self, text):
        """Preprocess text."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _get_pair_frequencies(self, word_freq):
        """Get frequencies of all adjacent pairs."""
        pairs = Counter()
        
        for word, freq in word_freq.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        
        return {k: v for k, v in pairs.items() if v >= self.min_freq}

    def _apply_merge(self, word_freq, pair):
        """Apply a merge to all words."""
        merged = ''.join(pair)
        
        new_freq = Counter()
        for word, freq in word_freq.items():
            new_word = word.replace(''.join(pair), merged)
            new_freq[new_word] = freq
        
        return new_freq

    def _build_vocab(self, vocab):
        """Build vocabulary and encoder."""
        self.vocab = set(vocab)
        
        for token in self.special_tokens:
            self.vocab.add(token)
        
        self.encoder = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.decoder = {idx: token for token, idx in self.encoder.items()}
        
        for idx, token in enumerate(sorted(vocab), start=len(self.special_tokens)):
            if token not in self.encoder:
                self.encoder[token] = idx
                self.decoder[idx] = token
        
        for rank, merge in enumerate(self.merges):
            self.bpe_ranks[merge] = rank

    def encode(self, text, max_length=None):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            tokens: List of token IDs
        """
        words = self._preprocess(text).split()
        
        tokens = []
        
        for word in words:
            subwords = self._encode_word(word)
            tokens.extend(subwords)
        
        tokens = [self.encoder.get(self.bos_token, 0)] + tokens + [self.encoder.get(self.eos_token, 0)]
        
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.encoder.get(self.pad_token, 0)] * (max_length - len(tokens))
        
        return tokens

    def _encode_word(self, word):
        """Encode a single word using BPE."""
        if word in self.encoder:
            return [self.encoder[word]]
        
        symbols = list(word)
        
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            
            pair_ranks = []
            for pair in pairs:
                if pair in self.bpe_ranks:
                    pair_ranks.append((self.bpe_ranks[pair], i))
                else:
                    pair_ranks.append((float('inf'), i))
            
            min_rank = min(pair_ranks)
            
            if min_rank[0] == float('inf'):
                break
            
            merge_idx = min_rank[1]
            merge_pair = (symbols[merge_idx], symbols[merge_idx + 1])
            
            symbols[merge_idx] = ''.join(merge_pair)
            del symbols[merge_idx + 1]
        
        result = []
        for symbol in symbols:
            if symbol in self.encoder:
                result.append(self.encoder[symbol])
            else:
                result.append(self.encoder.get(self.unk_token, 1))
        
        return result

    def decode(self, tokens):
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            text: Decoded text
        """
        words = []
        
        for token_id in tokens:
            token = self.decoder.get(token_id, self.unk_token)
            
            if token in self.special_tokens:
                continue
            
            words.append(token)
        
        return ' '.join(words)

    def save(self, filepath):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'merges': [list(m) for m in self.merges],
            'encoder': self.encoder,
            'special_tokens': self.special_tokens,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Load tokenizer from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        self.merges = [tuple(m) for m in data['merges']]
        self.encoder = data['encoder']
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.special_tokens = data['special_tokens']
        
        for rank, merge in enumerate(self.merges):
            self.bpe_ranks[merge] = rank
        
        print(f"Tokenizer loaded from {filepath}")


class SimpleTokenizer:
    """
    Simple character-level tokenizer with fallback.
    """

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from most common characters."""
        self.encoder = {t: i for i, t in enumerate(self.special_tokens)}
        self.decoder = {i: t for t, i in self.encoder.items()}

    def train(self, texts):
        """Build vocabulary from texts."""
        char_freq = Counter()
        for text in texts:
            char_freq.update(text)
        
        most_common = char_freq.most_common(self.vocab_size - len(self.special_tokens))
        
        for char, _ in most_common:
            idx = len(self.encoder)
            self.encoder[char] = idx
            self.decoder[idx] = char

    def encode(self, text, max_length=None):
        """Encode text to tokens."""
        tokens = [self.encoder.get(self.special_tokens[2], 0)]
        
        for char in text:
            tokens.append(self.encoder.get(char, self.encoder[self.special_tokens[1]]))
        
        tokens.append(self.encoder.get(self.special_tokens[3], 0))
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens += [self.encoder[self.special_tokens[0]]] * (max_length - len(tokens))
        
        return tokens

    def decode(self, tokens):
        """Decode tokens to text."""
        result = []
        for t in tokens:
            token = self.decoder.get(t, '')
            if token and not token.startswith('<'):
                result.append(token)
        return ''.join(result)


def create_tokenizer(tokenizer_type='bpe', vocab_size=5000, texts=None):
    """
    Factory function to create tokenizer.
    
    Args:
        tokenizer_type: 'bpe' or 'simple'
        vocab_size: Vocabulary size
        texts: Training texts (required for BPE)
    
    Returns:
        tokenizer: Tokenizer instance
    """
    if tokenizer_type == 'bpe':
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if texts:
            tokenizer.train(texts)
        return tokenizer
    elif tokenizer_type == 'simple':
        tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        if texts:
            tokenizer.train(texts)
        return tokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")