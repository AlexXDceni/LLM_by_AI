#!/usr/bin/env python3
# LLM - CLI Entry Point
# Updated to use optimized_llm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

from optimized_llm.config import ModelConfig
from optimized_llm.model import LLMModel, save_model, load_model, load_tokenizer
from optimized_llm.tokenizer import Tokenizer
from optimized_llm.training import Trainer
from optimized_llm.inference import generate_text, generate_text_advanced, beam_search
from optimized_llm.tensor_ops import cross_entropy_loss


def create_model(vocab_size=2000, d_model=64, num_layers=2, num_heads=4,
                 max_seq_len=256, dropout=0.1, use_rope=True):
    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_rope=use_rope
    )
    return LLMModel(cfg)


def train_command(args):
    model = create_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_length,
        dropout=args.dropout
    )

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    if args.folder:
        for fn in os.listdir(args.folder):
            if fn.endswith(('.txt', '.md')):
                with open(os.path.join(args.folder, fn), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog. " * 100]

    print(f"Loaded {len(texts)} text sources")

    trainer = Trainer(model, model.cfg, lr=args.learning_rate,
                      save_interval=args.save_interval)
    trainer.train_on_texts(texts, epochs=args.epochs,
                           block_size=args.block_size)

    if args.save_model:
        save_path = args.save_model
        if not save_path.startswith('models/'):
            save_path = 'models/' + save_path
        os.makedirs('models', exist_ok=True)
        save_model(model, save_path)


def chat_command(args):
    model = create_model(vocab_size=args.vocab_size)
    load_path = args.load_model
    if load_path:
        try:
            model = load_model(load_path)
            print(f"Loaded model from {load_path}")
        except:
            print(f"No model found at {load_path}, using fresh model")

    if load_path and os.path.exists(load_path + '_tokenizer.json'):
        tokenizer = load_tokenizer(load_path)
    else:
        tokenizer = Tokenizer(texts=["sample text for vocabulary " * 100],
                              vocab_size=args.vocab_size)

    print(f"\n{'='*50}\nChat Mode - Type 'quit' to exit\n{'='*50}\n")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if not prompt.strip():
                continue

            tokens = tokenizer.encode(prompt)
            generated = generate_text_advanced(
                model, tokens, max_length=args.max_length,
                method=args.method, temperature=args.temperature,
                top_k=args.top_k, beam_width=args.beam_width
            )
            response = tokenizer.decode(generated)
            safe_print(f"AI: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def web_command(args):
    topics = []
    if args.general:
        all_topics = [
            "physics", "chemistry", "biology", "astronomy", "geology",
            "quantum_mechanics", "thermodynamics", "evolution", "genetics",
            "neuroscience", "particle_physics", "astrophysics", "cosmology",
            "artificial_intelligence", "machine_learning", "deep_learning",
            "robotics", "blockchain", "quantum_computing", "cybersecurity",
            "ancient_history", "renaissance", "world_war_ii", "cold_war",
            "philosophy", "psychology", "economics", "mathematics",
            "literature", "music", "film", "architecture",
        ]
        import random
        topics = random.sample(all_topics, min(args.num_topics, len(all_topics)))
    elif args.topic:
        topics = [args.topic]

    texts = []
    for topic in topics:
        print(f"Fetching: {topic}")
        try:
            import urllib.request, json
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode('utf-8'))
            if 'extract' in data and data['extract']:
                texts.append(data['extract'])
                print(f"  - Fetched {len(data['extract'])} chars")
        except Exception as e:
            print(f"  - Error: {e}")

    if not texts:
        texts = ["This is a sample text for training. " * 100]

    model = create_model(vocab_size=args.vocab_size, d_model=args.d_model,
                         num_layers=args.num_layers, num_heads=args.num_heads)
    print(f"\nTraining on {len(texts)} text sources...")

    trainer = Trainer(model, model.cfg, lr=0.001, save_interval=args.save_interval)
    trainer.train_on_texts(texts, epochs=args.epochs, block_size=64)

    if args.save_model:
        save_path = 'models/' + args.save_model
        os.makedirs('models', exist_ok=True)
        save_model(model, save_path)
    print("\nLearning complete!")


def generate_command(args):
    model = create_model(vocab_size=args.vocab_size)
    load_path = args.load_model
    if load_path:
        try:
            model = load_model(load_path)
        except:
            pass

    if load_path and os.path.exists(load_path + '_tokenizer.json'):
        tokenizer = load_tokenizer(load_path)
    else:
        tokenizer = Tokenizer(texts=["sample text " * 100], vocab_size=args.vocab_size)
    tokens = tokenizer.encode(args.prompt)

    generated = generate_text_advanced(
        model, tokens, max_length=args.max_length,
        method=args.method, temperature=args.temperature,
        top_k=args.top_k, beam_width=args.beam_width
    )
    safe_print(tokenizer.decode(generated))


def eval_command(args):
    model = create_model(vocab_size=args.vocab_size)
    load_path = args.load_model
    if load_path:
        try:
            model = load_model(load_path)
        except:
            pass

    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()

    if load_path and os.path.exists(load_path + '_tokenizer.json'):
        tokenizer = load_tokenizer(load_path)
    else:
        tokenizer = Tokenizer(texts=[text], vocab_size=args.vocab_size)
    tokens = tokenizer.encode(text)

    total_loss = 0.0
    num_batches = 0
    block_size = args.block_size

    for i in range(0, len(tokens) - block_size, block_size // 2):
        input_ids = np.array(tokens[i:i+block_size-1]).reshape(1, -1)
        target_ids = np.array(tokens[i+1:i+block_size]).reshape(1, -1)
        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)
        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = np.exp(avg_loss)
    print(f"Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Optimized LLM')
    subparsers = parser.add_subparsers(dest='command')

    # train
    tp = subparsers.add_parser('train')
    tp.add_argument('--text', type=str)
    tp.add_argument('--file', type=str)
    tp.add_argument('--folder', type=str)
    tp.add_argument('--vocab-size', type=int, default=2000)
    tp.add_argument('--d-model', type=int, default=64)
    tp.add_argument('--num-layers', type=int, default=2)
    tp.add_argument('--num-heads', type=int, default=4)
    tp.add_argument('--max-seq-length', type=int, default=128)
    tp.add_argument('--dropout', type=float, default=0.1)
    tp.add_argument('--epochs', type=int, default=50)
    tp.add_argument('--block-size', type=int, default=64)
    tp.add_argument('--learning-rate', type=float, default=0.001)
    tp.add_argument('--save-interval', type=int, default=500)
    tp.add_argument('--save-model', type=str)

    # chat
    cp = subparsers.add_parser('chat')
    cp.add_argument('--vocab-size', type=int, default=2000)
    cp.add_argument('--load-model', type=str)
    cp.add_argument('--max-length', type=int, default=100)
    cp.add_argument('--method', type=str, default='sampling')
    cp.add_argument('--temperature', type=float, default=1.0)
    cp.add_argument('--top-k', type=int, default=40)
    cp.add_argument('--beam-width', type=int, default=3)

    # web
    wp = subparsers.add_parser('web')
    wp.add_argument('topic', type=str, nargs='?')
    wp.add_argument('--general', action='store_true')
    wp.add_argument('--num-topics', type=int, default=50)
    wp.add_argument('--vocab-size', type=int, default=2000)
    wp.add_argument('--d-model', type=int, default=64)
    wp.add_argument('--num-layers', type=int, default=2)
    wp.add_argument('--num-heads', type=int, default=4)
    wp.add_argument('--epochs', type=int, default=5)
    wp.add_argument('--save-interval', type=int, default=500)
    wp.add_argument('--save-model', type=str)

    # generate
    gp = subparsers.add_parser('generate')
    gp.add_argument('--prompt', type=str, required=True)
    gp.add_argument('--vocab-size', type=int, default=2000)
    gp.add_argument('--load-model', type=str)
    gp.add_argument('--max-length', type=int, default=100)
    gp.add_argument('--method', type=str, default='sampling')
    gp.add_argument('--temperature', type=float, default=1.0)
    gp.add_argument('--top-k', type=int, default=40)
    gp.add_argument('--beam-width', type=int, default=3)

    # eval
    ep = subparsers.add_parser('eval')
    ep.add_argument('--file', type=str, required=True)
    ep.add_argument('--vocab-size', type=int, default=2000)
    ep.add_argument('--load-model', type=str)
    ep.add_argument('--block-size', type=int, default=128)

    subparsers.add_parser('help')

    args = parser.parse_args()

    if args.command == 'help' or args.command is None:
        print("""
USAGE:
  python main.py train --file data.txt --epochs 50 --save-model mymodel
  python main.py chat --load-model models/mymodel
  python main.py web --general --num-topics 3 --epochs 5
  python main.py generate --prompt "Hello" --max-length 50
  python main.py eval --file test.txt
        """)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'chat':
        chat_command(args)
    elif args.command == 'web':
        web_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'eval':
        eval_command(args)


if __name__ == '__main__':
    main()
