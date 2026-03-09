"""
Data Splitter
Utilities for intelligent dataset splitting with stratified sampling and sequence checks.
"""
import random
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class DataSplitter:
    """Handles dataset splitting logic with support for stratification and sequence integrity"""
    
    @staticmethod
    def split_dataset(
        data_dir: str,
        output_dir: str,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        group_by_sequence: bool = False
    ):
        """
        Split dataset into train/val/test directories.
        
        Args:
            data_dir: Source directory containing images/ and labels/
            output_dir: Destination directory for train/val/test split
            ratios: Tuple of (train, val, test) ratios. Must sum to 1.
            seed: Random seed for reproducibility
            group_by_sequence: If True, attempts to keep files with similar prefixes together (simple sequence detection)
        """
        random.seed(seed)
        data_path = Path(data_dir)
        dest_path = Path(output_dir)
        
        # Validate input
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
            
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_images = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in valid_extensions]
        
        if not all_images:
            raise ValueError("No images found in data directory")
            
        # Create output structure
        for split in ['train', 'val', 'test']:
            (dest_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dest_path / split / 'labels').mkdir(parents=True, exist_ok=True)
            
        # Group logic
        if group_by_sequence:
            # Assume sequence format like seq01_001.jpg, seq01_002.jpg
            groups = defaultdict(list)
            for img in all_images:
                # Heuristic: split by underscore or hyphen, use first part as group ID
                prefix = img.split('_')[0] if '_' in img else img.split('-')[0]
                groups[prefix].append(img)
            
            group_keys = list(groups.keys())
            random.shuffle(group_keys)
            
            n_total = len(all_images)
            splits = {'train': [], 'val': [], 'test': []}
            counts = {'train': 0, 'val': 0, 'test': 0}
            
            # Distribute groups to satisfy ratios roughly
            for key in group_keys:
                imgs = groups[key]
                
                # Determine target split based on current counts
                current_total = sum(counts.values()) or 1
                current_ratios = {k: v/current_total for k,v in counts.items()}
                
                # Greedy assignment
                if counts['train'] / (n_total or 1) < ratios[0]:
                    target = 'train'
                elif counts['val'] / (n_total or 1) < ratios[1]:
                    target = 'val'
                else:
                    target = 'test'
                    
                splits[target].extend(imgs)
                counts[target] += len(imgs)
                
        else:
            # Simple random split
            random.shuffle(all_images)
            n = len(all_images)
            train_end = int(n * ratios[0])
            val_end = int(n * (ratios[0] + ratios[1]))
            
            splits = {
                'train': all_images[:train_end],
                'val': all_images[train_end:val_end],
                'test': all_images[val_end:]
            }
            
        # Move files
        print(f"Splitting data into {output_dir}...")
        for split, files in splits.items():
            print(f"Processing {split} set: {len(files)} images")
            for img_file in tqdm(files, desc=f"Copying {split}"):
                # Copy image
                src_img = images_dir / img_file
                dst_img = dest_path / split / 'images' / img_file
                shutil.copy2(src_img, dst_img)
                
                # Copy label if exists
                label_file = Path(img_file).stem + ".txt"
                src_label = labels_dir / label_file
                if src_label.exists():
                    dst_label = dest_path / split / 'labels' / label_file
                    shutil.copy2(src_label, dst_label)
                    
        print("Dataset split complete.")
