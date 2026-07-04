"""Full end-to-end integration test."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_llm.config import ModelConfig
from optimized_llm.model import LLMModel, save_model, load_model
from optimized_llm.tokenizer import Tokenizer
from optimized_llm.training import Trainer
from optimized_llm.inference import generate_text, generate_text_advanced, beam_search

print("=== TEST 1: Model creation ===")
cfg = ModelConfig(vocab_size=200, d_model=32, num_layers=2, num_heads=4, max_seq_len=32)
model = LLMModel(cfg)
assert model.cfg.vocab_size == 200
assert len(model.blocks) == 2
print("  PASS")

print("=== TEST 2: Forward/Backward ===")
tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
logits = model.forward(tokens)
assert logits.shape == (1, 5, 200)
print(f"  Forward: {logits.shape}")

from optimized_llm.tensor_ops import cross_entropy_loss
targets = np.array([[2, 3, 4, 5, 6]], dtype=np.int32)
loss = cross_entropy_loss(logits, targets)
print(f"  Loss: {loss:.4f}")
print("  PASS")

print("=== TEST 3: Training (loss must decrease) ===")
texts = [
    "the quick brown fox jumps over the lazy dog near the river bank",
    "hello world this is a test of the transformer language model system",
    "transformers use attention mechanisms for sequence processing tasks",
    "deep learning models learn patterns from large amounts of text data",
    "the cat sat on the mat and watched the birds fly through the sky",
]
trainer = Trainer(model, cfg, lr=0.001, save_interval=0)
trainer.train_on_texts(texts, epochs=10, block_size=16)
first_avg = np.mean(trainer.loss_history[:50])
last_avg = np.mean(trainer.loss_history[-50:])
assert last_avg < first_avg, f"Loss must decrease! first_avg={first_avg:.4f} last_avg={last_avg:.4f}"
print(f"  First 50 avg: {first_avg:.4f} -> Last 50 avg: {last_avg:.4f}")
print("  PASS")

print("=== TEST 4: Tokenizer ===")
t = Tokenizer(texts, vocab_size=200)
enc = t.encode("hello world")
dec = t.decode(enc)
print(f"  Encode: {enc}")
print(f"  Decode: \"{dec}\"")
assert len(enc) > 0
print("  PASS")

print("=== TEST 5: Generation ===")
tokens = t.encode("the")
gen = generate_text(model, tokens, max_length=20, temperature=0.8)
gen_text = t.decode(gen)
print(f"  Generated: \"{gen_text}\"")
assert len(gen) >= len(tokens)
print("  PASS")

print("=== TEST 6: Generate methods ===")
for method in ['greedy', 'sampling', 'beam']:
    gen = generate_text_advanced(model, tokens, max_length=15, method=method, temperature=0.8)
    print(f"  {method}: \"{t.decode(gen)}\"")
print("  PASS")

print("=== TEST 7: Save/Load ===")
os.makedirs('models', exist_ok=True)
save_model(model, 'models/integration_test')
loaded = load_model('models/integration_test')
logits_orig = model.forward(np.array([[1, 2, 3]], dtype=np.int32))
logits_new = loaded.forward(np.array([[1, 2, 3]], dtype=np.int32))
assert np.max(np.abs(logits_orig - logits_new)) < 1e-6
print("  PASS")

print("=== TEST 8: KV Cache gen ===")
gen_cached = generate_text(model, tokens, max_length=15, temperature=0.8, use_cache=True)
print(f"  Cached gen: \"{t.decode(gen_cached)}\"")
print("  PASS")

print("\n=== ALL 8 INTEGRATION TESTS PASSED ===")
