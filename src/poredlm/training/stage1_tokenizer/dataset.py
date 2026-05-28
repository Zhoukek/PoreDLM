import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class NanoporeSignalDataset(Dataset):
    def __init__(self, shards_dir, logic_chunk_size=None, logic_chunk_overlap_size=100, max_cache_size=256):
        """
        Args:
            shards_dir (str): Directory containing shards.npy and shards.json.
            logic_chunk_size (int, optional): Logic chunk size for training.
                                            If None, uses the original chunk_size.
            logic_chunk_overlap_size (int): Overlap size between consecutive logic chunks.
                                          Default is 0 (no overlap).
            max_cache_size (int): Max number of memmap files to keep open (per process).
        """
        meta_path = os.path.join(shards_dir, "shards.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.shard_info = meta["shards"]  # [{"path": "...", "num_samples": N}, ...]
        self.chunk_size = meta["chunk_size"]  # physical chunk size (e.g., 12000)
        self.dtype = np.dtype(meta["dtype"])
        self.shards_dir = shards_dir
        self.max_cache_size = max_cache_size

        # Set logic chunk size and overlap
        if logic_chunk_size is None:
            self.logic_chunk_size = self.chunk_size
        else:
            if logic_chunk_size <= 0:
                raise ValueError(f"logic_chunk_size ({logic_chunk_size}) must be positive")
            self.logic_chunk_size = logic_chunk_size
        
        if logic_chunk_overlap_size < 0:
            raise ValueError(f"logic_chunk_overlap_size ({logic_chunk_overlap_size}) must be non-negative")
        if logic_chunk_overlap_size >= self.logic_chunk_size:
            raise ValueError(f"logic_chunk_overlap_size ({logic_chunk_overlap_size}) must be less than logic_chunk_size ({self.logic_chunk_size})")
        
        self.logic_chunk_overlap_size = logic_chunk_overlap_size
        self.effective_stride = self.logic_chunk_size - self.logic_chunk_overlap_size

        # Pre-calculate logic chunks per physical chunk
        self.logic_chunks_per_physical = self._calculate_logic_chunks_in_physical()

        # Build global offsets based on logic chunks per physical chunk
        self.offsets = [0]
        for info in self.shard_info:
            num_physical_chunks = info["num_samples"]
            num_logic_chunks = num_physical_chunks * self.logic_chunks_per_physical
            self.offsets.append(self.offsets[-1] + num_logic_chunks)
        self.total_logic_samples = self.offsets[-1]

        # LRU cache for memmap objects: {shard_path: memmap_array}
        self._cache = OrderedDict()

    def _calculate_logic_chunks_in_physical(self):
        """Calculate how many logic chunks fit in one physical chunk (excluding incomplete ones)."""
        if self.logic_chunk_size > self.chunk_size:
            return 0  # No complete logic chunks can fit
        
        if self.effective_stride == 0:  # No stride (only happens when overlap equals chunk size, but we prevent this)
            # In case of no stride, just see how many full chunks fit without overlap
            return max(0, self.chunk_size // self.logic_chunk_size)
        
        # Calculate number of logic chunks that fit with stride
        # First chunk starts at 0, second at effective_stride, third at 2*effective_stride, etc.
        # Last chunk starts at (n-1)*effective_stride and ends at (n-1)*effective_stride + logic_chunk_size
        # So: (n-1)*effective_stride + logic_chunk_size <= chunk_size
        # (n-1)*effective_stride <= chunk_size - logic_chunk_size
        # n-1 <= (chunk_size - logic_chunk_size) / effective_stride
        # n <= (chunk_size - logic_chunk_size) / effective_stride + 1
        available_space = self.chunk_size - self.logic_chunk_size
        if available_space < 0:
            return 0
        num_chunks = int(available_space // self.effective_stride) + 1
        
        # Make sure the last chunk doesn't exceed the physical chunk boundary
        last_start_pos = (num_chunks - 1) * self.effective_stride
        if last_start_pos + self.logic_chunk_size > self.chunk_size:
            num_chunks -= 1
            
        return max(0, num_chunks)

    def _get_memmap(self, shard_path):
        """Get memmap array with LRU caching."""
        if shard_path in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        # Open new memmap
        memmap_arr = np.load(shard_path, mmap_mode='r')

        # Evict oldest if cache is full
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)  # Remove least recently used

        self._cache[shard_path] = memmap_arr
        return memmap_arr

    def _find_physical_shard_and_chunk(self, logic_idx):
        """Find which physical shard and which logic chunk within that physical chunk."""
        if logic_idx < 0 or logic_idx >= self.total_logic_samples:
            raise IndexError(f"Logic index {logic_idx} out of bounds [0, {self.total_logic_samples})")
        
        # Find physical shard
        physical_shard_id = 0
        for i in range(len(self.offsets) - 1):
            if self.offsets[i] <= logic_idx < self.offsets[i + 1]:
                physical_shard_id = i
                break
        else:
            raise IndexError(f"Logic index {logic_idx} not found in any shard range")

        # Find which logic chunk within the physical shard
        logic_chunks_in_prev_shards = self.offsets[physical_shard_id]
        local_logic_idx = logic_idx - logic_chunks_in_prev_shards  # 0-based within this physical shard

        # Convert to physical chunk index and logic sub-chunk index within that physical chunk
        physical_chunk_idx = local_logic_idx // self.logic_chunks_per_physical
        logic_subchunk_idx = local_logic_idx % self.logic_chunks_per_physical

        return physical_shard_id, physical_chunk_idx, logic_subchunk_idx

    def __len__(self):
        return self.total_logic_samples

    def __getitem__(self, idx, debug=False):
        if idx < 0 or idx >= self.total_logic_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_logic_samples})")

        # Find the physical location and sub-chunk position
        physical_shard_id, physical_chunk_idx, logic_subchunk_idx = self._find_physical_shard_and_chunk(idx)

        shard_filename = self.shard_info[physical_shard_id]["path"]
        shard_path = os.path.join(self.shards_dir, shard_filename)

        # Get the physical chunk
        data = self._get_memmap(shard_path)
        physical_chunk = data[physical_chunk_idx]  # shape: (chunk_size,)

        # Calculate start position for this logic chunk
        start_idx = logic_subchunk_idx * self.effective_stride
        end_idx = start_idx + self.logic_chunk_size

        # Ensure we don't go out of bounds (shouldn't happen due to calculation in _calculate_logic_chunks_in_physical)
        end_idx = min(end_idx, len(physical_chunk))
        start_idx = min(start_idx, len(physical_chunk) - self.logic_chunk_size)

        if debug:
            print(f"DEBUG - getitem({idx}):")
            print(f"  Physical shard ID: {physical_shard_id}")
            print(f"  Physical chunk index: {physical_chunk_idx}")
            print(f"  Logic subchunk index: {logic_subchunk_idx}")
            print(f"  Shard filename: {shard_filename}")
            print(f"  Physical chunk shape: {physical_chunk.shape}")
            print(f"  Effective stride: {self.effective_stride}")
            print(f"  Logic chunk size: {self.logic_chunk_size}")
            print(f"  Start index: {start_idx}")
            print(f"  End index: {end_idx}")
            print(f"  Physical chunk length: {len(physical_chunk)}")

        if end_idx > len(physical_chunk):
            # Pad if necessary (though this shouldn't happen with correct calculations)
            logic_chunk = np.zeros(self.logic_chunk_size, dtype=self.dtype)
            actual_data_len = len(physical_chunk) - start_idx
            logic_chunk[:actual_data_len] = physical_chunk[start_idx:]
            if debug:
                print(f"  PADDING APPLIED: Needed {self.logic_chunk_size}, got {actual_data_len}")
        else:
            logic_chunk = physical_chunk[start_idx:end_idx]  # shape: (logic_chunk_size,)
            if debug:
                print(f"  Successfully extracted chunk of shape: {logic_chunk.shape}")

        result = torch.from_numpy(logic_chunk.copy()).unsqueeze(0)  # shape: (1, logic_chunk_size)
        
        if debug:
            print(f"  Final result shape: {result.shape}")
            print(f"  Result stats - min: {result.min():.4f}, max: {result.max():.4f}, mean: {result.mean():.4f}")
        
        return result

    def get_original_chunk_size(self):
        """Return the original physical chunk size."""
        return self.chunk_size

    def get_logic_chunk_size(self):
        """Return the current logic chunk size."""
        return self.logic_chunk_size

    def get_logic_chunk_overlap_size(self):
        """Return the current logic chunk overlap size."""
        return self.logic_chunk_overlap_size

    def get_total_physical_samples(self):
        """Return total number of physical (original) samples."""
        return sum(info["num_samples"] for info in self.shard_info)

    def get_logic_chunks_per_physical_chunk(self):
        """Return number of logic chunks that fit in one physical chunk."""
        return self.logic_chunks_per_physical

def main():
    """Main function to test the NanoporeSignalDataset with different parameters."""
    shards_dir = "/mnt/nas_syy/default/poregpt/shared/dataset/dna/human_min0_max2_read96655/memap/train"
    
    # Test configurations: (logic_chunk_size, logic_chunk_overlap_size, test_indices)
    test_configs = [
        # Test 1: Original chunk size (no splitting)
        (12000, 0, [0, 1]),
        
        # Test 2: Smaller chunk size, no overlap
        (2000, 0, [0, 1, 5, 10]),
        
        # Test 3: Smaller chunk size with overlap
        (2000, 100, [0, 1, 5, 10]),
        
        # Test 4: Different sizes and overlaps
        (3000, 500, [0, 1, 2]),
        
        # Test 5: Small overlap
        (1000, 1, [0, 1, 10]),
        
        # Test 6: Large overlap relative to chunk size
        (500, 400, [0, 1, 5])
    ]
    
    for i, (chunk_size, overlap_size, test_indices) in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"TEST {i+1}: logic_chunk_size={chunk_size}, overlap={overlap_size}")
        print(f"{'='*80}")
        
        try:
            # Create dataset with specific parameters
            dataset = NanoporeSignalDataset(
                shards_dir=shards_dir,
                logic_chunk_size=chunk_size,
                logic_chunk_overlap_size=overlap_size
            )
            
            print(f"Original physical chunk size: {dataset.get_original_chunk_size()}")
            print(f"Logic chunk size: {dataset.get_logic_chunk_size()}")
            print(f"Logic chunk overlap: {dataset.get_logic_chunk_overlap_size()}")
            print(f"Effective stride: {dataset.get_logic_chunk_size() - dataset.get_logic_chunk_overlap_size()}")
            print(f"Logic chunks per physical chunk: {dataset.get_logic_chunks_per_physical_chunk()}")
            print(f"Total logic samples: {len(dataset)}")
            print(f"Total physical samples: {dataset.get_total_physical_samples()}")
            
            # Test specified indices
            for idx in test_indices:
                if idx < len(dataset):
                    print(f"\n--- Testing index {idx} ---")
                    try:
                        sample = dataset[idx]
                        print(f"Sample shape: {sample.shape}")
                        print(f"Sample stats - min: {sample.min():.4f}, max: {sample.max():.4f}, mean: {sample.mean():.4f}")
                        
                        # Also test with debug
                        print("\nDebug information:")
                        sample_debug = dataset.__getitem__(idx, debug=True)
                        
                    except Exception as e:
                        print(f"Error getting item {idx}: {e}")
                else:
                    print(f"Index {idx} is out of bounds (dataset length: {len(dataset)})")
                    
        except Exception as e:
            print(f"Error creating dataset with params ({chunk_size}, {overlap_size}): {e}")
    
    # Additional edge case test: very large overlap
    print(f"\n{'='*80}")
    print(f"EDGE CASE TEST: Attempting invalid parameters")
    print(f"{'='*80}")
    
    try:
        # This should fail
        invalid_dataset = NanoporeSignalDataset(
            shards_dir=shards_dir,
            logic_chunk_size=1000,
            logic_chunk_overlap_size=1000  # This should cause an error
        )
        print("ERROR: Should have failed with overlap >= chunk_size")
    except ValueError as e:
        print(f"✓ Correctly caught invalid parameter error: {e}")
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
