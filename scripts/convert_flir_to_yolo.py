"""
Convert FLIR Thermal Dataset to YOLO format

Converts COCO JSON annotations to YOLO format for training.
Input:  data/thermal/flir/ (COCO JSON)
Output: data/thermal/flir_yolo/ (YOLO format)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from PIL import Image
import shutil
from tqdm import tqdm


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox to YOLO format (normalized)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return (x_center, y_center, width, height)


def main():
    print("=" * 80)
    print("FLIR THERMAL TO YOLO CONVERTER")
    print("=" * 80)
    
    flir_dir = project_root / "data" / "thermal" / "flir"
    output_dir = project_root / "data" / "thermal" / "flir_yolo"
    
    if not flir_dir.exists():
        print(f"[ERROR] FLIR directory not found: {flir_dir}")
        print("\nTIP: Please download FLIR Thermal dataset first:")
        print("   python scripts/download_datasets.py")  
        print("   Select option: 7 (FLIR Thermal ADAS)")
        return
    
    print(f"\nInput directory: {flir_dir}")
    print(f"Output directory: {output_dir}")
    
    annotation_files = list(flir_dir.glob("**/*.json"))
    
    if not annotation_files:
        print(f"\n[WARNING] No annotation JSON files found in {flir_dir}")
        print("\nTIP: Expected structure:")
        print("   data/thermal/flir/")
        print("   ├── train/")
        print("   │   ├── images/")
        print("   │   └── annotations.json")
        print("   └── val/...")
        return
    
    print(f"\nFound {len(annotation_files)} annotation file(s)")
    
    class_mapping = {
        'person': 6,
        'car': 4,
        'bicycle': 4,
        'dog': -1
    }
    
    print("\nClass mapping:")
    print("  FLIR 'person' -> MODT 'soldier' (class 6)")
    print("  FLIR 'car' -> MODT 'military_vehicle' (class 4)")
    print("  FLIR 'bicycle' -> MODT 'military_vehicle' (class 4)")
    print("  FLIR 'dog' -> SKIP")
    
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_annotations = 0
    skipped = 0
    
    for ann_file in annotation_files:
        if 'train' in str(ann_file):
            split = 'train'
        elif 'val' in str(ann_file):
            split = 'val'
        elif 'test' in str(ann_file):
            split = 'test'
        else:
            split = 'train'
        
        print(f"\nProcessing {split}: {ann_file.name}")
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        images = {img['id']: img for img in coco_data.get('images', [])}
        
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        for img_id, img_info in tqdm(images.items(), desc=f"Converting {split}"):
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            source_img = None
            for possible_path in [
                flir_dir / split / 'images' / img_filename,
                flir_dir / 'images' / split / img_filename,
                flir_dir / img_filename,
            ]:
                if possible_path.exists():
                    source_img = possible_path
                    break
            
            if source_img is None:
                skipped += 1
                continue
            
            dest_img = output_dir / split / 'images' / img_filename
            shutil.copy(source_img, dest_img)
            
            yolo_annotations = []
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    cat_id = ann['category_id']
                    cat_name = categories.get(cat_id, 'unknown').lower()
                    
                    modt_class = class_mapping.get(cat_name, -1)
                    if modt_class == -1:
                        continue
                    
                    coco_bbox = ann['bbox']
                    yolo_bbox = convert_coco_bbox_to_yolo(coco_bbox, img_width, img_height)
                    
                    yolo_annotations.append((modt_class, yolo_bbox))
                    total_annotations += 1
            
            label_file = dest_img.with_suffix('.txt').parent.parent / 'labels' / dest_img.with_suffix('.txt').name
            with open(label_file, 'w') as f:
                for cls_id, (x_c, y_c, w, h) in yolo_annotations:
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            total_images += 1
    
    yaml_content = f"""# FLIR Thermal Dataset (YOLO format)
path: {output_dir.as_posix()}
train: train/images
val: val/images
test: test/images

names:
  4: military_vehicle
  6: soldier

total_images: {total_images}
total_annotations: {total_annotations}
"""
    
    yaml_file = output_dir / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"[OK] Processed: {total_images} images")
    print(f"[OK] Annotations: {total_annotations}")
    print(f"[ERROR] Skipped: {skipped} images")
    print(f"[OK] Dataset YAML: {yaml_file}")
    print("\nNext: python scripts/merge_datasets.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
