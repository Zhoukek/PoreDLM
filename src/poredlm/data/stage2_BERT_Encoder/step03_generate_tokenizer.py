import json
import argparse
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description="Generate tokenizer.json matching exact original format")
    parser.add_argument("--K", type=int, required=True, help="Number of <|bwav:i|> tokens (codebook size)")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Output path")
    args = parser.parse_args()

    K = args.K
    if K <= 0:
        raise ValueError("K must be a positive integer")

    # Build vocab in required order
    vocab = OrderedDict()

    # 0: <|unk|>
    vocab["<|unk|>"] = 0
    # 1: <|pad|>
    vocab["<|pad|>"] = 1
    # 2: <|bos|>
    vocab["<|bos|>"] = 2
    # 3: <|eos|>
    vocab["<|eos|>"] = 3
    
    # 4: <mask>
    vocab["<|mask|>"] = 4

    # 4–127: phonemes
    for i in range(124):
        vocab[f"<|ph_{i}|>"] = 5 + i

    # 128 onward: bwav tokens
    for i in range(K):
        vocab[f"<|bwav:{i}|>"] = 129 + i

    total_vocab_size = len(vocab)
    assert total_vocab_size == 129 + K, f"Vocab size mismatch: expected {129 + K}, got {total_vocab_size}"

    # added_tokens: ONLY bos, eos, pad — exactly as in your original file
    # Note: unk is NOT in added_tokens in your example!
    added_tokens = [
        {
            "id": 0,
            "content": "<|unk|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        },

        {
            "id": 1,
            "content": "<|pad|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        },
        {
            "id": 2,
            "content": "<|bos|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        },
        {
            "id": 3,
            "content": "<|eos|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        },
        {
            "id": 4,
            "content": "<|mask|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        }
    ]

    # Pre-tokenizer regex: must match all <|...|> patterns (including ph and bwav)
    pre_tokenizer = {
        "type": "Split",
        "pattern": {
            "Regex": r"<\|[^>]+\|>"
        },
        "behavior": "Isolated",
        "invert": False
    }

    # Assemble full config EXACTLY like your original structure
    tokenizer_config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {
            "type": "NFC"
        },
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "unk_token": "<|unk|>",
            "vocab": vocab
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated tokenizer.json with K={K}")
    print(f"   Vocab size: {total_vocab_size}")
    print(f"   Special token IDs: unk=0, pad=1, bos=2, eos=3, mask=4")
    print(f"   Phonemes: <|ph_0|> (ID=5) to <|ph_123|> (ID=128)")
    print(f"   Bwav tokens: <|bwav:0|> (ID=129) to <|bwav:{K-1}|> (ID={128 + K})")
    print(f"   Output: {args.output}")

if __name__ == "__main__":
    main()
