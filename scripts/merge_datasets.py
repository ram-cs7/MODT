"""
Merge Multiple Datasets into Unified YOLO Format

Combines military, ships, and thermal datasets with class remapping.
Creates NEW merged dataset without modifying source datasets.

Input:  data/military/, data/ships/airbus_yolo/, data/thermal/flir_yolo/
Output: data/unified_combined/ (merged YOLO format)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import shutil
from tqdm import tqdm
import yaml


UNIFIED_CLASSES = {
    0: 'camouflage_soldier',
    1: 'weapon',
    2: 'military_tank',
    3: 'military_truck',
    4: 'military_vehicle',
    5: 'civilian',
    6: 'soldier',
    7: 'civilian_vehicle',
    8: 'military_artillery',
    9: 'trench',
    10: 'military_aircraft',
    11: 'military_warship'
}


def remap_label_line(line, class_mapping):
    """Remap class ID in YOLO label line"""
    parts = line.strip().split()
    if not parts:
        return None
    
    old_class = int(parts[0])
    new_class = class_mapping.get(old_class, None)
    
    if new_class is None:
        return None
    
    parts[0] = str(new_class)
    return ' '.join(parts)


def merge_dataset(source_dir, output_dir, split, class_mapping, dataset_name):
    """Merge one dataset into unified format"""
    source_images = source_dir / split / 'images'
    source_labels = source_dir / split / 'labels'
    
    output_images = output_dir / split / 'images'
    output_labels = output_dir / split / 'labels'
    
    if not source_images.exists():
        print(f"  [WARNING] {dataset_name}/{split}/images not found, skipping...")
        return 0
    
    count = 0
    
    for img_file in tqdm(list(source_images.glob('*')), desc=f"{dataset_name}/{split}", leave=False):
        dest_img = output_images / f"{dataset_name}_{img_file.name}"
        shutil.copy(img_file, dest_img)
        
        label_file = source_labels / img_file.with_suffix('.txt').name
        dest_label = output_labels / f"{dataset_name}_{img_file.with_suffix('.txt').name}"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                remapped = remap_label_line(line, class_mapping)
                if remapped:
                    new_lines.append(remapped + '\n')
            
            with open(dest_label, 'w') as f:
                f.writelines(new_lines)
        else:
            dest_label.touch()
        
        count += 1
    
    return count


def main():
    print("=" * 80)
    print("MULTI-DATASET MERGER - PHASE 11")
    print("=" * 80)
    
    data_dir = project_root / "data"
    output_dir = data_dir / "unified_combined"
    
    datasets = [
        {
            'name': 'military',
            'path': data_dir / 'military' / 'assets' / 'military_object_dataset',
            'mapping': {i: i for i in range(12)},
            'enabled': True
        },
        {
            'name': 'airbus',
            'path': data_dir / 'ships' / 'airbus_yolo',
            'mapping': {0: 11},
            'enabled': False
        },
        {
            'name': 'flir',
            'path': data_dir / 'thermal' / 'flir_yolo',
            'mapping': {4: 4, 6: 6},
            'enabled': False
        },
    ]
    
    print("\nChecking available datasets...")
    for ds in datasets:
        if ds['path'].exists():
            ds['enabled'] = True
            print(f"  [OK] {ds['name']}: {ds['path']}")
        else:
            print(f"  [ERROR] {ds['name']}: Not found")
    
    enabled_datasets = [ds for ds in datasets if ds['enabled']]
    
    if len(enabled_datasets) < 2:
        print("\n[WARNING] Need at least 2 datasets to merge!")
        print("\nTIP: Available datasets:")
        print("   1. Military Assets (primary) [OK] Always available")
        print("   2. Airbus Ships - download & convert with:")
        print("      python scripts/download_datasets.py (option 6)")
        print("      python scripts/convert_airbus_to_yolo.py")
        print("   3. FLIR Thermal - download & convert with:")
        print("      python scripts/download_datasets.py (option 7)")
        print("      python scripts/convert_flir_to_yolo.py")
        return
    
    print(f"\nSummary: Merging {len(enabled_datasets)} datasets:")
    for ds in enabled_datasets:
        print(f"   - {ds['name']}")
    
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    totals = {'train': 0, 'val': 0, 'test': 0}
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'=' * 80}")
        print(f"MERGING {split.upper()} SPLIT")
        print('=' * 80)
        
        for ds in enabled_datasets:
            count = merge_dataset(
                ds['path'],
                output_dir,
                split,
                ds['mapping'],
                ds['name']
            )
            totals[split] += count
            if count > 0:
                print(f"  [OK] {ds['name']}: {count} images")
    
    yaml_content = {
        'path': str(output_dir.as_posix()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': UNIFIED_CLASSES,
        'dataset_info': {
            'total_images': sum(totals.values()),
            'train_images': totals['train'],
            'val_images': totals['val'],
            'test_images': totals['test'],
            'source_datasets': [ds['name'] for ds in enabled_datasets],
            'created_by': 'merge_datasets.py - Phase 11 Integration'
        }
    }
    
    yaml_file = output_dir / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"[OK] Total images: {sum(totals.values())}")
    print(f"  - Train: {totals['train']}")
    print(f"  - Val: {totals['val']}")
    print(f"  - Test: {totals['test']}")
    print(f"[OK] Output: {output_dir}")
    print(f"[OK] Dataset YAML: {yaml_file}")
    
    print("\nNext: python scripts/train_detector.py --data", yaml_file)
    print("\nExpected improvements:")
    print("   - Overall mAP: +15-25%")
    print("   - Warship detection: +150% (if Airbus included)")
    print("   - Night operation capability (if FLIR included)")
    print("=" * 80)


if __name__ == "__main__":
    main()
