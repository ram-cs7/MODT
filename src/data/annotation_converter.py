"""
Annotation Converter
Utilities for converting between dataset formats (COCO <-> YOLO)
"""
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

class AnnotationConverter:
    """Format conversion utilities for COCO and YOLO formats"""
    
    @staticmethod
    def coco_to_yolo(json_path: str, output_dir: str, use_segments: bool = False):
        """
        Convert COCO JSON to YOLO txt files.
        
        Args:
            json_path: Path to .json file (e.g., instances_train.json)
            output_dir: Directory to save .txt files
            use_segments: If True, exports polygon segments; else bounding boxes
        """
        json_path = Path(json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(json_path) as f:
            data = json.load(f)
            
        images = {img['id']: img for img in data['images']}
        annotations = data['annotations']
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Map COCO category IDs to 0-indexed YOLO IDs
        # Note: This simply enumerates them. For consistency, ensure sorted or mapped specific way.
        sorted_cat_ids = sorted(categories.keys())
        cat_map = {id: i for i, id in enumerate(sorted_cat_ids)}
        
        print(f"Converting {len(annotations)} annotations for {len(images)} images...")
        
        # Group by image_id
        img_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)
            
        # Write files
        for img_id, anns in tqdm(img_anns.items(), desc="Converting"):
            img_info = images.get(img_id)
            if not img_info:
                continue
                
            img_w = img_info['width']
            img_h = img_info['height']
            file_name = Path(img_info['file_name']).stem + ".txt"
            
            with open(output_dir / file_name, 'w') as f:
                for ann in anns:
                    cls_id = cat_map.get(ann['category_id'])
                    if cls_id is None:
                        continue
                        
                    if use_segments and 'segmentation' in ann:
                        # Handle segmentation (simplest polygon)
                        if isinstance(ann['segmentation'], list):
                             for seg in ann['segmentation']:
                                 # Normalize points
                                 norm_points = []
                                 for i in range(0, len(seg), 2):
                                     norm_points.append(seg[i] / img_w)
                                     norm_points.append(seg[i+1] / img_h)
                                 line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in norm_points)
                                 f.write(line + "\n")
                    else:
                        # Bounding Box (x, y, w, h) -> YOLO (x_c, y_c, w, h) normalized
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        
                        x_c = (x + w / 2) / img_w
                        y_c = (y + h / 2) / img_h
                        w_n = w / img_w
                        h_n = h / img_h
                        
                        # Clip to [0, 1]
                        x_c = max(0, min(1, x_c))
                        y_c = max(0, min(1, y_c))
                        w_n = max(0, min(1, w_n))
                        h_n = max(0, min(1, h_n))
                        
                        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                        
        print(f"Conversion complete. Saved to {output_dir}")
        print("Category Mapping:", {v: categories[k] for k, v in cat_map.items()})

    @staticmethod
    def yolo_to_coco(yolo_dir: str, image_dir: str, output_json: str, class_names: List[str] = None):
        """
        Convert YOLO txt files to a single COCO JSON.
        
        Args:
            yolo_dir: Directory containing .txt labels
            image_dir: Directory containing images (needed for dimensions)
            output_json: Output .json file path
            class_names: List of class names corresponding to IDs 0, 1, ...
        """
        import cv2
        
        yolo_path = Path(yolo_dir)
        img_path = Path(image_dir)
        
        dataset = {
            "info": {},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        if class_names:
            for i, name in enumerate(class_names):
                dataset['categories'].append({"id": i+1, "name": name, "supercategory": "object"})
        
        ann_id = 1
        img_files = list(img_path.glob("*.*"))
        
        print(f"Converting YOLO to COCO for {len(img_files)} images...")
        
        for i, img_file in enumerate(tqdm(img_files, desc="Converting")):
            # Image info
            img = cv2.imread(str(img_file))
            if img is None: continue
            h, w = img.shape[:2]
            
            image_info = {
                "id": i + 1,
                "file_name": img_file.name,
                "width": w,
                "height": h
            }
            dataset['images'].append(image_info)
            
            # Label info
            label_file = yolo_path / (img_file.stem + ".txt")
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    cls_id = int(parts[0])
                    x_c, y_c, wb, hb = parts[1:5]
                    
                    # De-normalize to pixel values (COCO: x_top_left, y_top_left, width, height)
                    width = wb * w
                    height = hb * h
                    x = (x_c * w) - (width / 2)
                    y = (y_c * h) - (height / 2)
                    
                    ann = {
                        "id": ann_id,
                        "image_id": i + 1,
                        "category_id": cls_id + 1, # COCO 1-indexed usually
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": [] 
                    }
                    dataset['annotations'].append(ann)
                    ann_id += 1
                    
        with open(output_json, 'w') as f:
            json.dump(dataset, f)
            
        print(f"Saved COCO JSON to {output_json}")
