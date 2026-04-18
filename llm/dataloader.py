"""
Data Loading Module
Efficient data loading utilities for training with batching,
shuffling, and sequence packing.
"""

import numpy as np
import random


class DataLoader:
    """
    Efficient data loader with batching and shuffling.
    """

    def __init__(self, tokens, block_size=128, batch_size=32, shuffle=True):
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_batches = max(0, (len(tokens) - block_size) // (block_size // 2))
        
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        indices = np.arange(max(0, len(self.tokens) - self.block_size))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.block_size // 2):
            batch_indices = indices[i:i+self.batch_size]
            
            if len(batch_indices) < self.batch_size:
                continue
            
            batch_inputs = []
            batch_targets = []
            
            for idx in batch_indices:
                if idx + self.block_size + 1 > len(self.tokens):
                    continue
                
                input_seq = self.tokens[idx:idx+self.block_size]
                target_seq = self.tokens[idx+1:idx+self.block_size+1]
                
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
            
            if batch_inputs:
                yield np.array(batch_inputs), np.array(batch_targets)


class StreamingDataLoader:
    """
    Memory-efficient data loader for large datasets.
    """

    def __init__(self, texts, tokenizer, block_size=128, batch_size=32):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        
    def __iter__(self):
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            
            indices = range(0, len(tokens) - self.block_size, self.block_size // 2)
            
            batch_inputs = []
            batch_targets = []
            
            for idx in indices:
                input_seq = tokens[idx:idx+self.block_size]
                target_seq = tokens[idx+1:idx+self.block_size+1]
                
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
                
                if len(batch_inputs) >= self.batch_size:
                    yield np.array(batch_inputs), np.array(batch_targets)
                    batch_inputs = []
                    batch_targets = []
            
            if batch_inputs:
                yield np.array(batch_inputs), np.array(batch_targets)


class MixedBatchLoader:
    """
    Mix batches from different sources (train/val/test).
    """

    def __init__(self, batch_sources, mix_ratios=None):
        self.batch_sources = batch_sources
        self.mix_ratios = mix_ratios or [1.0] * len(batch_sources)
        
        self.iterators = [iter(source) for source in batch_sources]
        
    def __iter__(self):
        for source_idx in range(len(self.batch_sources)):
            source = self.batch_sources[source_idx]
            ratio = self.mix_ratios[source_idx]
            
            for batch in source:
                if random.random() < ratio:
                    yield batch


class RandomBatchSampler:
    """
    Sample random batches for contrastive learning.
    """

    def __init__(self, tokens, block_size=128, num_samples=100):
        self.tokens = tokens
        self.block_size = block_size
        self.num_samples = num_samples
        
    def __iter__(self):
        for _ in range(self.num_samples):
            idx = random.randint(0, len(self.tokens) - self.block_size - 1)
            
            input_seq = self.tokens[idx:idx+self.block_size]
            target_seq = self.tokens[idx+1:idx+self.block_size+1]
            
            yield np.array([input_seq]), np.array([target_seq])


def create_dataloader(tokens, block_size=128, batch_size=32, shuffle=True):
    """Create dataloader."""
    return DataLoader(tokens, block_size, batch_size, shuffle)


def create_bucketed_dataloader(tokens, block_sizes=[64, 128, 256], batch_size=32):
    """
    Create bucketed dataloader - groups sequences of similar length.
    """
    buckets = {bs: [] for bs in block_sizes}
    
    for i in range(0, len(tokens) - max(block_sizes), max(block_sizes) // 2):
        for bs in block_sizes:
            if i + bs < len(tokens):
                buckets[bs].append(tokens[i:i+bs])
    
    for bs in block_sizes:
        random.shuffle(buckets[bs])
    
    all_buckets = []
    for bs, sequences in buckets.items():
        loader = DataLoader(np.concatenate(sequences), bs, batch_size, shuffle=False)
        all_buckets.extend([loader])
    
    return all_buckets