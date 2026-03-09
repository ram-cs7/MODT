"""
Convert Airbus Ship Detection dataset to YOLO format

Converts CSV segmentation masks to YOLO bounding boxes for training.
Input:  data/ships/airbus/train_ship_segmentations_v2.csv
Output: data/ships/airbus_yolo/ (YOLO format)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm


def rle_decode(mask_rle, shape=(768, 768)):
    """Decode Run-Length Encoding to binary mask"""
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        img[start:end] = 1
    
    return img.reshape(shape).T


def mask_to_bbox(mask):
    """Convert binary mask to YOLO bounding box"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    h, w = mask.shape
    x_center = ((cmin + cmax) / 2) / w
    y_center = ((rmin + rmax) / 2) / h
    width = (cmax - cmin + 1) / w
    height = (rmax - rmin + 1) / h
    
    return (x_center, y_center, width, height)


def main():
    print("=" * 80)
    print("AIRBUS SHIP DETECTION TO YOLO CONVERTER")
    print("=" * 80)
    
    airbus_dir = project_root / "data" / "ships" / "airbus"
    csv_file = airbus_dir / "train_ship_segmentations_v2.csv"
    images_dir = airbus_dir / "train_v2"
    output_dir = project_root / "data" / "ships" / "airbus_yolo"
    
    if not csv_file.exists():
        print(f"[ERROR] CSV file not found: {csv_file}")
        print("\nTIP: Please download Airbus Ship Detection dataset first:")
        print("   python scripts/download_datasets.py")
        print("   Select option: 6 (Airbus Ship Detection)")
        return
    
    print(f"\nInput CSV: {csv_file}")
    print(f"Input images: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    print("\nReading CSV annotations...")
    df = pd.read_csv(csv_file)
    print(f"Total entries: {len(df)}")
    
    grouped = df.groupby('ImageId')
    print(f"Unique images: {len(grouped)}")
    
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_images = len(grouped)
    train_count = int(total_images * 0.7)
    val_count = int(total_images * 0.2)
    
    print(f"\nDataset split:")
    print(f"  Train: {train_count} images (70%)")
    print(f"  Val: {val_count} images (20%)")
    print(f"  Test: {total_images - train_count - val_count} images (10%)")
    
    processed = 0
    skipped = 0
    
    for idx, (image_id, group) in enumerate(tqdm(grouped, desc="Converting")):
        if idx < train_count:
            split = 'train'
        elif idx < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
        
        img_path = images_dir / image_id
        if not img_path.exists():
            skipped += 1
            continue
        
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue
        
        output_img = output_dir / split / 'images' / image_id
        shutil.copy(img_path, output_img)
        
        bboxes = []
        for _, row in group.iterrows():
            if pd.isna(row['EncodedPixels']):
                continue
            
            mask = rle_decode(row['EncodedPixels'], shape=(img_h, img_w))
            bbox = mask_to_bbox(mask)
            if bbox is not None:
                bboxes.append(bbox)
        
        label_file = output_dir / split / 'labels' / image_id.replace('.jpg', '.txt')
        with open(label_file, 'w') as f:
            for bbox in bboxes:
                x_c, y_c, w, h = bbox
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        processed += 1
    
    yaml_content = f"""# Airbus Ship Detection (YOLO format)
path: {output_dir.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  0: ship

total_images: {processed}
"""
    
    yaml_file = output_dir / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"[OK] Processed: {processed} images")
    print(f"[ERROR] Skipped: {skipped} images")
    print(f"[OK] Dataset YAML: {yaml_file}")
    print("\nNext: python scripts/convert_flir_to_yolo.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
