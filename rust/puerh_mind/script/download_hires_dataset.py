#!/usr/bin/env python3
"""
Download high-resolution images from COCO dataset using FiftyOne.
"""

import sys
from pathlib import Path


def download_coco_subset(testdata_dir, count=256):
    """Download COCO dataset subset using FiftyOne"""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        
        testdata_path = Path(testdata_dir)
        testdata_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {count} high-resolution images from COCO...")
        
        # Load COCO dataset (this downloads it)
        # We only take a subset by limiting max samples
        dataset = foz.load_zoo_dataset(
            "coco-2014-val",
            max_samples=count,
            download_dir=str(testdata_path),
        )
        
        print(f"\n✓ Dataset loaded with {len(dataset)} samples")
        
        # Extract and save images
        images_saved = 0
        for sample in dataset:
            try:
                import shutil
                src = sample.filepath
                dst = testdata_path / f"image_{images_saved:04d}.jpg"
                shutil.copy(src, dst)
                images_saved += 1
                
                if images_saved % 50 == 0:
                    print(f"Saved {images_saved}/{count} images...")
                    
            except Exception as e:
                pass
        
        print(f"\n✓ Saved {images_saved} images to {testdata_path}")
        return images_saved >= count // 2
        
    except ImportError:
        print("FiftyOne not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fiftyone"])
        return download_coco_subset(testdata_dir, count)
    except Exception as e:
        print(f"COCO download error: {e}")
        return False


def download_flickr30k_subset(testdata_dir, count=256):
    """Download Flickr30K dataset subset"""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        
        testdata_path = Path(testdata_dir)
        testdata_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {count} images from Flickr30K...")
        
        dataset = foz.load_zoo_dataset(
            "flickr30k",
            max_samples=count,
            download_dir=str(testdata_path),
        )
        
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Copy images
        images_saved = 0
        for sample in dataset:
            try:
                import shutil
                src = sample.filepath
                dst = testdata_path / f"image_{images_saved:04d}.jpg"
                shutil.copy(src, dst)
                images_saved += 1
                
                if images_saved % 50 == 0:
                    print(f"Saved {images_saved}/{count} images...")
                    
            except Exception as e:
                pass
        
        print(f"✓ Saved {images_saved} images")
        return images_saved >= count // 2
        
    except Exception as e:
        print(f"Flickr30K download failed: {e}")
        return False


def download_openimages_subset(testdata_dir, count=256):
    """Download OpenImages dataset subset"""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        
        testdata_path = Path(testdata_dir)
        testdata_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {count} images from Open Images...")
        
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            max_samples=count,
            download_dir=str(testdata_path),
        )
        
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Copy images
        images_saved = 0
        for sample in dataset:
            try:
                import shutil
                src = sample.filepath
                dst = testdata_path / f"image_{images_saved:04d}.jpg"
                shutil.copy(src, dst)
                images_saved += 1
                
                if images_saved % 50 == 0:
                    print(f"Saved {images_saved}/{count} images...")
                    
            except Exception as e:
                pass
        
        print(f"✓ Saved {images_saved} images")
        return images_saved >= count // 2
        
    except Exception as e:
        print(f"Open Images download failed: {e}")
        return False


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    testdata_dir = repo_root / "testdata"
    
    print("Downloading high-resolution dataset...")
    print(f"Target: {testdata_dir}\n")
    
    try:
        # Try COCO first
        success = download_coco_subset(str(testdata_dir), count=256)
        
        if not success:
            # Try Flickr30K
            print("\nTrying Flickr30K...")
            success = download_flickr30k_subset(str(testdata_dir), count=256)
        
        if not success:
            # Try Open Images
            print("\nTrying Open Images...")
            success = download_openimages_subset(str(testdata_dir), count=256)
        
        # Count final images
        img_count = len(list(testdata_dir.glob("*.jpg"))) + len(list(testdata_dir.glob("*.png")))
        print(f"\nFinal image count: {img_count}")
        
        sys.exit(0 if img_count >= 200 else 1)
        
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
