#!/usr/bin/env python3
"""
Main entry point for the LLM.
Supports training, interactive chat, web learning, and text generation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse

from llm.llm_model import LLM, generate_text, beam_search, generate_text_advanced, save_model, load_model
from llm.tokenization import Tokenizer
from llm.enhanced_training import EnhancedTrainer, cross_entropy_loss, compute_perplexity
from llm.optimizer import AdamW
from llm.utils import set_seed, print_model_info, MetricsTracker, Timer


def create_model(vocab_size=5000, d_model=512, num_layers=8, num_heads=8, 
                 max_seq_length=128, dropout=0.1):
    """Create and initialize the LLM."""
    model = LLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    return model


def train_command(args):
    """Train the model."""
    print("Initializing training...")
    
    model = create_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    )
    
    print_model_info(model)
    
    texts = []
    
    if args.text:
        texts.append(args.text)
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    if args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith(('.txt', '.md')):
                filepath = os.path.join(args.folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    
    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog. " * 100]
    
    print(f"Loaded {len(texts)} text sources")
    
    trainer = EnhancedTrainer(
        model={'embedding': model.embedding,
               'positional_encoding': model.positional_encoding,
               'transformer_stack': model.transformer_stack,
               'output_linear': model.output_linear},
        learning_rate=args.learning_rate,
        optimizer='adamw',
        clip_norm=args.clip_norm,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    trainer.train_on_texts(
        texts, 
        epochs=args.epochs, 
        block_size=args.block_size,
        batch_size=args.batch_size
    )
    
    if args.save_model:
        from llm.llm_model import save_model
        
        save_path = args.save_model
        if not save_path.startswith('models/') and not save_path.startswith('/'):
            save_path = 'models/' + save_path
        
        os.makedirs('models', exist_ok=True)
        save_model(model, save_path)
        print(f"Model saved to {save_path}")


def chat_command(args):
    """Interactive chat mode."""
    print("Loading model...")
    
    model = create_model(vocab_size=args.vocab_size)
    
    if args.load_model:
        import os
        from llm.llm_model import load_model, load_checkpoint
        
        # Try full model first, then checkpoint
        try:
            model = load_model(args.load_model)
            print(f"Loaded model from {args.load_model}")
        except FileNotFoundError:
            try:
                model = load_checkpoint(args.load_model)
                print(f"Loaded checkpoint from {args.load_model}")
            except FileNotFoundError:
                print(f"No model found at {args.load_model}, using fresh model")
    
    tokenizer = Tokenizer(texts=["sample text for vocabulary " * 100], vocab_size=args.vocab_size)
    
    print(f"\n{'='*50}")
    print("Chat Mode - Type 'quit' to exit")
    print(f"{'='*50}\n")
    
    while True:
        try:
            prompt = input("You: ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            tokens = tokenizer.encode(prompt)
            
            generated = generate_text_advanced(
                model, tokens, 
                max_length=args.max_length,
                method=args.method,
                temperature=args.temperature,
                top_k=args.top_k,
                beam_width=args.beam_width
            )
            
            response = tokenizer.decode(generated)
            
            print(f"AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def web_command(args):
    """Learn from web content."""
    topics = []
    
    if args.general:
        print(f"Fetching {args.num_topics} diverse topics...")
        
        # Large list of 500+ diverse topics
        all_topics = [
            # Science (~80)
            "physics", "chemistry", "biology", "astronomy", "geology", "botany", "zoology",
            "quantum_mechanics", "thermodynamics", "electromagnetism", "optics", "acoustics",
            "biochemistry", "molecular_biology", "genetics", "evolution", "ecology", "microbiology",
            "neuroscience", "immunology", "pathology", "pharmacology", "toxicology",
            "particle_physics", "astrophysics", "cosmology", "relativity", "string_theory",
            "organic_chemistry", "inorganic_chemistry", "physical_chemistry", "analytical_chemistry",
            "cell_biology", "developmental_biology", "marine_biology", "paleontology", "virology",
            "entomology", "herpetology", "ornithology", "mammalogy", "mycology", "bacteriology",
            "geophysics", "oceanography", "meteorology", "climatology", "seismology", "volcanology",
            "hydrology", "glaciology", "cartography", "mineralogy", "crystallography", "petrology",
            "sedimentology", "stratigraphy", "paleoclimatology", "paleoecology", "anthropology",
            # Technology (~60)
            "artificial_intelligence", "machine_learning", "deep_learning", "neural_network",
            "robotics", "automation", "blockchain", "cryptocurrency", "quantum_computing",
            "software_engineering", "computer_programming", "web_development", "database",
            "computer_network", "cybersecurity", "cloud_computing", "virtual_reality",
            "augmented_reality", "3d_printing", "internet_of_things", "nanotechnology",
            "biotechnology", "genetic_engineering", "space_technology", "satellite",
            "telecommunication", "wireless_technology", "mobile_computing", "computer_hardware",
            "operating_system", "algorithm", "data_structure", "artificial_general_intelligence",
            "computer_vision", "natural_language_processing", "reinforcement_learning",
            "supervised_learning", "unsupervised_learning", "transfer_learning",
            "edge_computing", "fog_computing", "grid_computing", "distributed_computing",
            "parallel_computing", "high_performance_computing", "computer_graphics",
            "image_processing", "signal_processing", "information_technology",
            "software_as_a_service", "platform_as_a_service", "infrastructure_as_a_service",
            # History (~70)
            "ancient_history", "medieval_history", "modern_history", "contemporary_history",
            "world_war_i", "world_war_ii", "cold_war", "american_revolution", "french_revolution",
            "industrial_revolution", "renaissance", "enlightenment", "reformation",
            "crusades", "roman_empire", "greek_empire", "byzantine_empire", "ottoman_empire",
            "mongol_empire", "british_empire", "spanish_empire", "portuguese_empire",
            "napoleonic_wars", "american_civil_war", "civil_war", "vietnam_war", "korean_war",
            "world_war_i", "world_war_ii", "gulf_war", "war_in_afghanistan", "iraq_war",
            "september_11_attacks", "cold_war", "berlin_wall", "soviet_union", "communism",
            "fascism", "national_socialism", "colonialism", "imperialism", "slavery",
            "civil_rights_movement", "women_suffrage", "industrial_revolution",
            "information_revolution", "digital_revolution", "age_of_discovery",
            "age_of_enlightenment", "baroque", "rococo", "impressionism", "modernism",
            # Geography (~80)
            "africa", "asia", "europe", "north_america", "south_america", "antarctica", "australia",
            "alps", "himalayas", "andes", "rocky_mountains", "appalachians", "ural_mountains",
            "nile", "amazon", "mississippi", "yangtze", "ganges", "danube", "rhine", "thames",
            "pacific_ocean", "atlantic_ocean", "indian_ocean", "arctic_ocean", "southern_ocean",
            "sahara", "gobi", "amazon_rainforest", "siberian_wilderness", "great_barrier_reef",
            "mount_everest", "k2", "kilimanjaro", "denali", "mont_blanc", "matterhorn",
            "canada", "united_states", "mexico", "brazil", "argentina", "chile", "peru",
            "united_kingdom", "france", "germany", "italy", "spain", "portugal", "poland",
            "russia", "ukraine", "china", "japan", "india", "australia", "egypt", "south_africa",
            "nigeria", "kenya", "morocco", "indonesia", "thailand", "vietnam", "south_korea",
            # Culture/Art (~50)
            "art", "painting", "sculpture", "architecture", "photography", "dance", "theater",
            "cinema", "opera", "ballet", "modern_art", "contemporary_art", "classical_art",
            "renaissance_art", "baroque_art", "impressionism", "expressionism", "surrealism",
            "abstract_art", "pop_art", "minimalism", "conceptual_art", "digital_art",
            "street_art", "graffiti", "graphic_design", "illustration", "cartoon", "anime",
            "museum", "gallery", "conservation", "restoration", "art_history", "aesthetics",
            "visual_arts", "performing_arts", "fine_arts", "applied_arts", "decorative_arts",
            # Literature (~40)
            "literature", "poetry", "novel", "short_story", "drama", "essay", "mythology",
            "folklore", "epic", "tragedy", "comedy", "romance", "science_fiction", "fantasy",
            "horror", "mystery", "thriller", "detective", "adventure", "historical_fiction",
            "realism", "naturalism", "symbolism", "romanticism", "classic_literature",
            "contemporary_literature", "world_literature", "american_literature", "british_literature",
            "french_literature", "german_literature", "russian_literature", "japanese_literature",
            "chinese_literature", "indian_literature", "latin_literature", "greek_literature",
            # Sport (~40)
            "football", "soccer", "basketball", "tennis", "golf", "baseball", "hockey",
            "cricket", "rugby", "boxing", "wrestling", "swimming", "athletics", "gymnastics",
            "cycling", "motorsport", "formula_1", "nascar", "rally", "motogp", "football",
            "volleyball", "handball", "badminton", "table_tennis", "squash", "fencing",
            "archery", "shooting", "judo", "karate", "taekwondo", "boxing", "mma", "ufc",
            "olympics", "paralympics", "winter_olympics", "summer_olympics", "world_cup",
            # Philosophy (~30)
            "philosophy", "metaphysics", "epistemology", "ethics", "aesthetics", "logic",
            "existentialism", "rationalism", "empiricism", "idealism", "materialism",
            "pragmatism", "phenomenology", "structuralism", "postmodernism", "stoicism",
            "nihilism", "utilitarianism", "kantianism", "hegelianism", "marxism",
            "existentialism", "absurdism", "skepticism", "cynicism", "epicureanism",
            # Psychology (~30)
            "psychology", "cognitive_psychology", "developmental_psychology", "social_psychology",
            "clinical_psychology", "industrial_psychology", "educational_psychology",
            "neuropsychology", "abnormal_psychology", "behaviorism", "psychoanalysis",
            "humanistic_psychology", "positive_psychology", "cognitive_science",
            "perception", "memory", "attention", "learning", "motivation", "emotion",
            "personality", "intelligence", "creativity", "decision_making", "problem_solving",
            # Economy/Business (~30)
            "economics", "microeconomics", "macroeconomics", "econometrics", "political_economy",
            "capitalism", "socialism", "communalism", " Keyensian_economics", "monetarism",
            "supply_side_economics", "international_trade", "international_finance",
            "stock_market", "bond_market", "currency", "cryptocurrency", "banking",
            "investment", "entrepreneurship", "management", "marketing", "accounting",
            "corporate_finance", "personal_finance", "real_estate", "commodities",
            # Music (~30)
            "music", "classical_music", "jazz", "rock", "pop", "hip_hop", "electronic",
            "country", "blues", "reggae", "metal", "punk", "folk", "r_and_b", "soul",
            "funk", "disco", "dance", "techno", "house", "ambient", "experimental",
            "orchestra", "symphony", "chamber_music", "opera", "choral", "instrumental",
            "piano", "guitar", "violin", "drums", "bass", "saxophone", "trumpet",
            # Film/TV (~25)
            "film", "cinema", "movie", "documentary", "animation", "indie_film",
            "blockbuster", "action_movie", "comedy", "drama", "horror_movie", "thriller",
            "science_fiction_film", "fantasy_film", "romance_film", "war_film",
            "western_film", "noir", "silent_film", "avant_garde", "television", "tv_series",
            "sitcom", "drama_series", "comedy_series", "documentary_series", "reality_tv",
            # Mathematics (~25)
            "mathematics", "algebra", "geometry", "calculus", "statistics", "probability",
            "number_theory", "combinatorics", "graph_theory", "topology", "analysis",
            "linear_algebra", "differential_equations", "complex_analysis", "set_theory",
            "mathematical_logic", "category_theory", "mathematical_physics",
            # Medicine (~25)
            "medicine", "anatomy", "physiology", "pathology", "pharmacology", "surgery",
            "pediatrics", "cardiology", "oncology", "neurology", "psychiatry", "dermatology",
            "gastroenterology", "pulmonology", "endocrinology", "nephrology", "urology",
            "orthopedics", "ophthalmology", "otolaryngology", "radiology", "emergency_medicine",
        ]
        
        # Randomly select num_topics
        import random
        selected = random.sample(all_topics, min(args.num_topics, len(all_topics)))
        topics = selected
        
    elif args.topic:
        topics = [args.topic]
    
    print(f"Learning from {len(topics)} topics...")
    
    texts = []
    
    for topic in topics:
        print(f"Fetching: {topic}")
        
        try:
            import urllib.request
            
            wikipedia_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            
            try:
                req = urllib.request.Request(wikipedia_url, headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req, timeout=10)
                data = response.read().decode('utf-8')
                
                import json
                wiki_data = json.loads(data)
                
                if 'extract' in wiki_data and wiki_data['extract']:
                    texts.append(wiki_data['extract'])
                    print(f"  - Fetched {len(wiki_data['extract'])} chars")
                    
            except Exception as e:
                print(f"  - Error fetching {topic}: {e}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    if not texts:
        print("No content fetched. Using default text.")
        texts = ["This is a sample text for training. " * 100]
    
    model = create_model(vocab_size=args.vocab_size)
    
    print(f"\nTraining on {len(texts)} text sources...")
    
    from llm.simple_training import train_model
    train_fn = train_model
    
    save_interval = getattr(args, 'save_interval', 0) or 0
    save_path = getattr(args, 'save_model', None)
    
    train_fn(
        model={
            'embedding': model.embedding,
            'positional_encoding': model.positional_encoding,
            'transformer_stack': model.transformer_stack,
            'output_linear': model.output_linear
        },
        texts=texts,
        epochs=args.epochs,
        learning_rate=0.01,
        block_size=64,
        save_interval=save_interval,
        save_path=save_path
    )
    
    if args.save_model:
        from llm.llm_model import save_model
        save_path = 'models/' + args.save_model
        os.makedirs('models', exist_ok=True)
        save_model(model, save_path)
        print(f"Model saved to {save_path}")
    
    print("\nLearning complete! You can now chat with the model.")
    print("Run: python main.py chat")


def generate_command(args):
    """Generate text from prompt."""
    model = create_model(vocab_size=args.vocab_size)
    
    if args.load_model:
        import os
        from llm.llm_model import load_model, load_checkpoint
        
        try:
            model = load_model(args.load_model)
        except FileNotFoundError:
            try:
                model = load_checkpoint(args.load_model)
            except FileNotFoundError:
                print(f"No model found at {args.load_model}, using fresh model")
    
    tokenizer = Tokenizer(texts=["sample text for vocabulary " * 100], vocab_size=args.vocab_size)
    
    tokens = tokenizer.encode(args.prompt)
    
    generated = generate_text_advanced(
        model, tokens,
        max_length=args.max_length,
        method=args.method,
        temperature=args.temperature,
        top_k=args.top_k,
        beam_width=args.beam_width
    )
    
    result = tokenizer.decode(generated)
    print(result)


def eval_command(args):
    """Evaluate model on text."""
    model = create_model(vocab_size=args.vocab_size)
    
    if args.load_model:
        from llm.llm_model import load_model
        model = load_model(args.load_model)
    
    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = Tokenizer(vocab_size=args.vocab_size)
    tokens = tokenizer.encode(text)
    
    from llm.enhanced_training import cross_entropy_loss, compute_perplexity
    
    total_loss = 0
    num_batches = 0
    
    block_size = args.block_size
    
    for i in range(0, len(tokens) - block_size, block_size // 2):
        input_ids = np.array(tokens[i:i+block_size-1]).reshape(1, -1)
        target_ids = np.array(tokens[i+1:i+block_size]).reshape(1, -1)
        
        logits = model.forward(input_ids, training=False)
        loss = cross_entropy_loss(logits, target_ids)
        
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = compute_perplexity(avg_loss)
    
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")


def main():
    parser = argparse.ArgumentParser(description='LLM Training and Inference')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    train_parser = subparsers.add_parser('train', help='Train the model from text/file')
    train_parser.add_argument('--text', type=str, help='Training text (direct string)')
    train_parser.add_argument('--file', type=str, help='Path to training text file')
    train_parser.add_argument('--folder', type=str, help='Path to folder with .txt/.md files')
    train_parser.add_argument('--vocab-size', type=int, default=5000, help='Vocabulary size (default: 5000)')
    train_parser.add_argument('--d-model', type=int, default=512, help='Model dimension (default: 512)')
    train_parser.add_argument('--num-layers', type=int, default=8, help='Number of transformer layers (default: 8)')
    train_parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads (default: 8)')
    train_parser.add_argument('--max-seq-length', type=int, default=128, help='Maximum sequence length (default: 128)')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--block-size', type=int, default=128, help='Training block size (default: 128)')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    train_parser.add_argument('--clip-norm', type=float, default=1.0, help='Gradient clipping norm (default: 1.0)')
    train_parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    train_parser.add_argument('--warmup-steps', type=int, default=500, help='LR warmup steps (default: 500)')
    train_parser.add_argument('--gradient-accumulation', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    train_parser.add_argument('--checkpoint-dir', type=str, default='models', help='Checkpoint directory (default: models)')
    train_parser.add_argument('--log-interval', type=int, default=100, help='Logging interval (default: 100)')
    train_parser.add_argument('--save-interval', type=int, default=0, help='Auto-save interval in steps (0 = disabled, default: disabled)')
    train_parser.add_argument('--save-model', type=str, help='Model name to save (saved to models/<name>)')
    
    subparsers.add_parser('help', help='Show help for commands')
    
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--vocab-size', type=int, default=5000)
    chat_parser.add_argument('--load-model', type=str, help='Load model from file')
    chat_parser.add_argument('--max-length', type=int, default=100)
    chat_parser.add_argument('--method', type=str, default='sampling', 
                           choices=['greedy', 'sampling', 'top_k', 'beam'])
    chat_parser.add_argument('--temperature', type=float, default=1.0)
    chat_parser.add_argument('--top-k', type=int, default=40)
    chat_parser.add_argument('--beam-width', type=int, default=3)
    
    web_parser = subparsers.add_parser('web', help='Learn from Wikipedia (auto-save supported)')
    web_parser.add_argument('topic', type=str, nargs='?', help='Single topic to learn about')
    web_parser.add_argument('--general', action='store_true', 
                           help='Learn from 500+ diverse random topics (recommended)')
    web_parser.add_argument('--num-topics', type=int, default=500,
                           help='Number of random topics to fetch (default: 500)')
    web_parser.add_argument('--vocab-size', type=int, default=5000, help='Vocabulary size (default: 5000)')
    web_parser.add_argument('--epochs', type=int, default=5, help='Training epochs (default: 5)')
    web_parser.add_argument('--save-interval', type=int, default=0, 
                           help='Auto-save every N steps (0=disabled, recommended: 500)')
    web_parser.add_argument('--save-model', type=str, help='Model name to save (saved to models/)')
    
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--prompt', type=str, required=True)
    gen_parser.add_argument('--vocab-size', type=int, default=5000)
    gen_parser.add_argument('--load-model', type=str)
    gen_parser.add_argument('--max-length', type=int, default=100)
    gen_parser.add_argument('--method', type=str, default='sampling')
    gen_parser.add_argument('--temperature', type=float, default=1.0)
    gen_parser.add_argument('--top-k', type=int, default=40)
    gen_parser.add_argument('--beam-width', type=int, default=3)
    
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--file', type=str, required=True)
    eval_parser.add_argument('--vocab-size', type=int, default=5000)
    eval_parser.add_argument('--load-model', type=str)
    eval_parser.add_argument('--block-size', type=int, default=128)
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.command == 'help':
        print("""
LLM - Training and Inference Commands:

TRAIN:
  python main.py train --file <path> --epochs 10
  python main.py train --folder <path> --save-model mymodel
  python main.py train --text "your text here" --epochs 5

CHAT:
  python main.py chat --load-model <model>
  python main.py chat --load-model models/mymodel --max-length 50

WEB:
  python main.py web <topic>
  python main.py web --general --num-topics 100
  python main.py web --general --save-interval 500 --save-model mymodel

GENERATE:
  python main.py generate --prompt "Hello" --max-length 50
  python main.py generate --prompt "Hi" --method beam --beam-width 3

EVAL:
  python main.py eval --file <textfile> --block-size 128

OPTIONS:
  --vocab-size    : Vocabulary size (default: 5000)
  --d-model     : Model dimension (default: 512)
  --num-layers  : Transformer layers (default: 8)
  --learning-rate : Learning rate (default: 0.001)
  --save-interval : Auto-save every N steps (0=disabled)
  --save-model  : Save model to models/<name>
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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()