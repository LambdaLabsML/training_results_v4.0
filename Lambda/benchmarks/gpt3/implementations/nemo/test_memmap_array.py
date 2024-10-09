import numpy as np
import os

def create_synthetic_data(file_path, num_elements=1000):
    """Create a synthetic dataset and save it to a memory-mapped file."""
    # Generate synthetic data
    data = np.random.rand(num_elements).astype(np.float32)
    
    # # Write synthetic data to memory-mapped file
    memmap_array = np.memmap(file_path, dtype='float32', mode='w+', shape=(num_elements,))
    memmap_array[:] = data[:]
    memmap_array.flush()  # Ensure data is written to disk

def test_memmap_access(file_path):
    """Test accessing the memory-mapped file."""
    try:
        # Attempt to read from the memory-mapped file
        memmap_array = np.memmap(file_path, dtype='float32', mode='r')
        print(f"Successfully accessed {len(memmap_array)} elements.")
    except OSError as e:
        print(f"Encountered an error: {e}")

if __name__ == "__main__":
    # Define file path for synthetic data
    synthetic_file_path = '/home/ubuntu/ml-1cc/storage_tmp/synthetic_data.mmap'
    # synthetic_file_path = '/home/ubuntu/storage_tmp/synthetic_data.mmap'

    # Create synthetic data
    create_synthetic_data(synthetic_file_path)

    # Test accessing the synthetic data
    test_memmap_access(synthetic_file_path)

    print("completed successfully")