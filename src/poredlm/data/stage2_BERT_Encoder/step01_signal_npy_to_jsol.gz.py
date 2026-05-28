# -*- coding: utf-8 -*-
"""
Sequentially tokenize .npy files generated from fast5_to_chank.py using VQETokenizer.
Each chunk in a .npy file becomes one line in the corresponding .jsonl.gz file.
This version processes files sequentially, one after another.
"""

import os
import gzip
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from .vqe_tokenizer import VQETokenizer
import torch


def process_npy_file(npy_file_path,  output_path, tokenizer,max_batch_size):
    """
    Process a single .npy file: load chunks, tokenize them in batches,
    and write results to a .jsonl.gz file.

    Args:
        npy_file_path (str or Path): Path to the input .npy file.
        tokenizer (VQETokenizer): An instance of the VQETokenizer class.
        output_dir (str or Path): Directory to save the output .jsonl.gz file.
        max_batch_size (int): Batch size for tokenization during inference.
    """
    npy_file_path = Path(npy_file_path)


    try:
        # Load the .npy file
        # Assuming the structure from your previous script: list of 1D arrays
        chunks_list = np.load(npy_file_path, allow_pickle=True)
        if not isinstance(chunks_list, list) and not isinstance(chunks_list, np.ndarray):
             print(f"Warning: {npy_file_path} does not contain a list or array. Skipping.")
             return

        if isinstance(chunks_list, np.ndarray):
            # If it's a 2D array where rows are chunks, convert to list
            if chunks_list.ndim == 2:
                chunks_list = [chunks_list[i] for i in range(chunks_list.shape[0])]
            # If it's a 1D array, wrap it in a list
            elif chunks_list.ndim == 1:
                 print(f"Warning: {npy_file_path} seems to contain a single 1D array. Wrapping as list.")
                 chunks_list = [chunks_list]
            else:
                 print(f"Warning: {npy_file_path} has unexpected shape {chunks_list.shape}. Skipping.")
                 return

        if len(chunks_list) == 0:
            print(f"Info: {npy_file_path} contains no chunks. Writing empty .jsonl.gz file.")
            with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
                pass # Create an empty file
            return

        # Prepare results list
        results = []

        num_chunks = len(chunks_list)

        # Process chunks in batches
        for i in tqdm(range(0, num_chunks, max_batch_size), desc=f"Tokenizing {npy_file_path.name}", leave=False):
            batch_chunks = chunks_list[i:i+max_batch_size]

            # Ensure all chunks in the batch have the same length
            # The VQETokenizer likely expects fixed-size inputs based on its initialization
            # Assuming each chunk should be 40000 samples long based on your previous script
            # If lengths vary significantly or are different, you might need padding/truncation here
            # or ensure the downstream _tokenize_chunked_signal handles variable lengths correctly.
            # For simplicity, let's assume they are all the expected length.
            # If not, you'd need to filter or process them differently before batching.
            batch_signal_np = np.array(batch_chunks, dtype=np.float32) # Shape: (B, L_chunk) e.g., (32, 40000)

            # Prepare input tensor for the model
            x = torch.from_numpy(batch_signal_np).float().unsqueeze(1).to(tokenizer.device) # Shape: (B, 1, L_chunk)

            # Perform batched inference
            with torch.no_grad():
                # reconstructed_signals, level_tokens_tensor, loss, tokens_tensor = tokenizer.model(x) # tokens_tensor shape: (B, T_tokens) or (B, T_tokens, C)
                reconstructed_signals, tokens_tensor, loss, loss_breakdown, distill_loss = tokenizer.model(x) # tokens_tensor shape: (B, T_tokens) or (B, T_tokens, C)


            # Move tokens to CPU and convert to numpy
            tokens_np = tokens_tensor.cpu().numpy() # Shape: (B, T_tokens) or (B, T_tokens, C)

            # Ensure shape is (B, T_tokens) if it was (B, T_tokens, 1)
            if tokens_np.ndim == 3 and tokens_np.shape[-1] == 1:
                tokens_np = tokens_np.squeeze(-1) # Shape: (B, T_tokens)
            elif tokens_np.ndim != 2:
                 print(f"Warning: Unexpected token shape {tokens_np.shape} for batch starting at {i}. Skipping batch results.")
                 continue # Skip this batch if shape is wrong

            # Iterate through the batch results (each row corresponds to one input chunk)
            for j in range(tokens_np.shape[0]):
                # Get the token IDs for the j-th chunk in the current batch
                chunk_tokens = tokens_np[j] # Shape: (T_tokens,)

                # Format tokens into the required string format
                token_strings = [f"<|bwav:{int(token_id)}|>" for token_id in chunk_tokens]
                joined_string = "".join(token_strings)

                # Determine a unique ID for this chunk within the original read/file
                # Since we don't have the original read ID here, we'll use the filename and chunk index
                # Modify this ID generation logic if you have access to original read IDs stored elsewhere
                chunk_id = f"{npy_file_path.stem}_chunk_{i+j}" # i is the batch start index, j is the index within the batch
                # Append the result dictionary
                results.append({
                    "id": chunk_id,
                    "text": joined_string
                })
    except Exception as e:
        print(f"❌ Error processing {npy_file_path}: {e}")
        # Optionally, you could write partial results or log the error to a separate file
        return # Exit this function on error for this file

    # Write all results to the .jsonl.gz file
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
            for item in tqdm(results):
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Wrote {len(results)} lines to {output_path}")
    except Exception as e:
        print(f"❌ Error writing to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Tokenize a single .npy file using VQETokenizer.')
    # Changed argument name from --input-dir to --input-file
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Input .npy file to tokenize.')
    # Changed argument name from --output-dir to --output-file
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='Output .jsonl.gz file to save tokens.')
    parser.add_argument('--model-ckpt', type=str, required=True,
                        help='Path to the VQ tokenizer model checkpoint (.pth file).')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on (default: cuda). Use "cpu" if CUDA is unavailable.')
    parser.add_argument('--batch-size', type=int, default=32,help='Batch size for tokenization (default: 32).')
    parser.add_argument('--model-type', type=int, default=1,help='Excpected Model type')

    args = parser.parse_args()

    # Changed variable names for clarity
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    model_ckpt = args.model_ckpt
    device = args.device
    batch_size = args.batch_size
    model_type = args.model_type

    # Validate input *file*
    if not input_file.exists() or not input_file.is_file():
        print(f"Error: Input file does not exist or is not a file: {input_file}")
        return

    # Validate parent directory of output file exists, create output's parent directory if needed
    output_parent_dir = output_file.parent
    if not output_parent_dir.exists():
        print(f"Creating output directory: {output_parent_dir}")
        output_parent_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing file: {input_file.name}")
    print(f"Model checkpoint: {model_ckpt}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)

    # Initialize the tokenizer once
    tokenizer = VQETokenizer(model_ckpt=model_ckpt, device=device)

    # Process the single file
    # Note: process_npy_file likely needs to be adapted or replaced
    # to handle a single file path and a specific output file path,
    # rather than finding files recursively.
    try:
        process_npy_file(input_file, output_file, tokenizer, batch_size)
        print(f"\n✅ File processed successfully: {input_file} -> {output_file}")
    except Exception as e:
        print(f"\n❌ Error processing file {input_file}: {e}")
        return

if __name__ == "__main__":
    # Need to import torch here as well
    main()
