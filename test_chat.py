"""Test the trained model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimized_llm.model import load_model
from optimized_llm.tokenizer import Tokenizer
from optimized_llm.inference import generate_text_advanced

model = load_model("models/model_final")
with open("data/antrenare.txt", "r", encoding="utf-8") as f:
    text = f.read()
t = Tokenizer(texts=[text], vocab_size=2000)

for prompt in ["Buna ziua", "Ce este", "In sistemul", "Planeta", "Pamantul"]:
    tokens = t.encode(prompt)
    gen = generate_text_advanced(model, tokens, max_length=40, temperature=0.8)
    decoded = t.decode(gen)
    decoded_safe = decoded.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    print(f'  "{prompt}" -> "{decoded_safe}"')
