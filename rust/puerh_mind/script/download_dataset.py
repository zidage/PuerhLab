#!/usr/bin/env python3
"""
Download CIFAR-10 dataset (256 images) to testdata directory.
Images will be extracted and organized by class.
"""

import os
import sys
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def download_cifar10(testdata_dir):
    """Download and extract CIFAR-10 dataset."""
    testdata_path = Path(testdata_dir)
    testdata_path.mkdir(parents=True, exist_ok=True)
    
    # CIFAR-10 dataset URL
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_file = testdata_path / "cifar-10-python.tar.gz"
    extract_dir = testdata_path / "cifar-10-python"
    
    # Download
    if not tar_file.exists():
        print(f"Downloading CIFAR-10 dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, tar_file)
            print(f"Downloaded to {tar_file}")
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    # Extract
    import tarfile
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir}...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(testdata_path)
        print("Extraction complete")
    
    # Load and save images
    # CIFAR-10 extraction puts files directly in testdata/cifar-10-batches-py
    cifar_dir = testdata_path / "cifar-10-batches-py"
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    images_saved = 0
    target_count = 256
    images_per_batch = target_count // 2  # Use 2 batches
    
    # Process data batches
    for batch_idx in range(1, 3):  # Use batches 1 and 2
        batch_file = cifar_dir / f"data_batch_{batch_idx}"
        if not batch_file.exists():
            continue
            
        print(f"Processing {batch_file.name}...")
        
        try:
            with open(batch_file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            
            images = data[b'data']
            labels = data[b'labels']
            
            # Save 128 images from this batch
            for i in range(min(images_per_batch, len(images))):
                # Reshape and convert to PIL Image
                img_data = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_data)
                
                # Save image directly to testdata (no subdirectories)
                img_path = testdata_path / f"image_{images_saved:04d}.png"
                img.save(img_path)
                images_saved += 1
                
                if images_saved >= target_count:
                    break
            
            if images_saved >= target_count:
                break
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"\n✓ Downloaded and saved {images_saved} images to {testdata_path}")
    
    # Cleanup
    if tar_file.exists():
        tar_file.unlink()
    
    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    testdata_dir = repo_root / "testdata"
    
    print(f"Will download dataset to: {testdata_dir}")
    
    try:
        success = download_cifar10(testdata_dir)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
