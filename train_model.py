"""Fast training script — tokenizes once, trains with stride."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from optimized_llm.config import ModelConfig
from optimized_llm.model import LLMModel, save_model
from optimized_llm.training import Trainer
from optimized_llm.tokenizer import Tokenizer

# --- Config CPU-friendly ---
cfg = ModelConfig(
    vocab_size=2000,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
    max_seq_len=256,
    dropout=0.1,
    use_rope=True,
)
n_params = (cfg.vocab_size * cfg.d_model + cfg.d_model * cfg.vocab_size +
            cfg.num_layers * (cfg.d_model * 3 * cfg.d_model +
                              cfg.d_model * cfg.d_ff * 2 + cfg.d_ff * cfg.d_model * 2))
print(f"Model: vocab={cfg.vocab_size}, d_model={cfg.d_model}, layers={cfg.num_layers}, heads={cfg.num_heads}")
print(f"Params: ~{n_params:,}")

with open("data/antrenare.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"Data: {len(text):,} chars")

model = LLMModel(cfg)
trainer = Trainer(model, cfg, lr=0.001, weight_decay=0.1, max_norm=1.0, save_interval=0)

# Tokenize once + train with stride=block_size
trainer.train_on_texts([text], epochs=30, block_size=128, step_stride=128)

# Save final model with tokenizer
os.makedirs("models", exist_ok=True)
tokenizer = Tokenizer(texts=[text], vocab_size=cfg.vocab_size)
save_model(model, "models/model_final", tokenizer=tokenizer)
print("Model saved to models/model_final")

# Generate a sample
from optimized_llm.inference import generate_text_advanced
t = Tokenizer(texts=[text], vocab_size=cfg.vocab_size)
for prompt in ["Buna ziua", "Ce este", "In sistemul", "Planeta"]:
    tokens = t.encode(prompt)
    gen = generate_text_advanced(model, tokens, max_length=30, temperature=0.8)
    try:
        print(f'  "{prompt}" -> "{t.decode(gen)}"')
    except UnicodeEncodeError:
        print(f'  "{prompt}" -> ... (unicode error)')
