"""
Unified Dataset Management Script
Comprehensive solution for downloading and managing all datasets
"""

import os
import sys
import urllib.request
import ssl
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict


class UnifiedDatasetManager:
    """Manages all dataset downloads from multiple sources"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SSL context that doesn't verify certificates
        self.ssl_context = ssl._create_unverified_context()
    
    def download_file(self, url, output_path, desc="Downloading"):
        """Download file with progress and error handling"""
        print(f"\n{desc}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            print(f"✓ Downloaded: {Path(output_path).name}")
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def download_kaggle_dataset(self, dataset_id, output_dir, dataset_name):
        """Download dataset from Kaggle using API"""
        print(f"\n{'='*80}")
        print(f"Downloading: {dataset_name}")
        print(f"{'='*80}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if len(list(output_dir.glob('*'))) > 2:  # More than just hidden files
            print(f"✓ Dataset appears to exist at: {output_dir}")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                print("Skipping...")
                return True
        
        try:
            import subprocess
            
            print(f"\nDownloading {dataset_id} via Kaggle API...")
            cmd = [sys.executable, '-m', 'kaggle', 'datasets', 'download', '-d', dataset_id, '-p', str(output_dir), '--unzip']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                print(f"Output: {result.stdout}")
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "Unknown error"
                print(f"✗ Kaggle API download failed!")
                print(f"Error: {error_msg}")
                print(f"\n💡 TIP: Download manually from:")
                print(f"   https://www.kaggle.com/datasets/{dataset_id}")
                print(f"   Save to: {output_dir}")
                return False
            
            print(f"✓ {dataset_name} downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print(f"\n💡 MANUAL DOWNLOAD:")
            print(f"   1. Visit: https://www.kaggle.com/datasets/{dataset_id}")
            print(f"   2. Click 'Download'")
            print(f"   3. Extract to: {output_dir}")
            return False
    
    def download_military_datasets(self):
        """Download military datasets from Kaggle"""
        print("\n" + "="*80)
        print("MILITARY DATASETS (KAGGLE)")
        print("="*80)
        
        # Primary dataset
        dataset = {
            'id': 'rawsi18/military-assets-dataset-12-classes-yolo8-format',
            'name': 'Military Assets Dataset (12 Classes)',
            'output': str(self.data_dir / 'military' / 'assets')
        }
        
        print(f"\nDataset: {dataset['name']}")
        print(f"Size: 26,315 labeled images")
        print(f"Classes: 12 (tanks, aircraft, vehicles, soldiers, etc.)")
        print(f"Format: YOLOv8")
        
        response = input("\nDownload this dataset? (y/n): ")
        if response.lower() == 'y':
            return self.download_kaggle_dataset(
                dataset['id'],
                dataset['output'],
                dataset['name']
            )
        return False
    
    # =========================================================================
    # PRIORITY DATASETS (Multi-Dataset Integration - Phase 11)
    # =========================================================================
    
    def download_airbus_ships(self):
        """Download Airbus Ship Detection Challenge dataset"""
        print("\n" + "="*80)
        print("AIRBUS SHIP DETECTION CHALLENGE")
        print("="*80)
        
        dataset = {
            'id': 'airbus-ship-detection',
            'name': 'Airbus Ship Detection Challenge',
            'output': str(self.data_dir / 'ships' / 'airbus'),
            'type': 'competition'
        }
        
        print(f"\nDataset: {dataset['name']}")
        print(f"Size: ~28 GB (192K+ satellite images)")
        print(f"Purpose: Warship detection improvement")
        print(f"Format: PNG images + CSV segmentation masks")
        print(f"\n⚠️  This is a Kaggle COMPETITION dataset")
        print(f"    You must JOIN the competition first:")
        print(f"    https://www.kaggle.com/c/{dataset['id']}")
        
        response = input("\nHave you joined the competition? (y/n): ")
        if response.lower() != 'y':
            print("\n💡 Please join the competition first, then re-run this script")
            return False
        
        # Competition downloads use different command
        try:
            output_dir = Path(dataset['output'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nDownloading {dataset['name']}...")
            print("⏳ This is a large dataset (~28 GB), please be patient...")
            
            cmd = [sys.executable, '-m', 'kaggle', 'competitions', 'download', '-c', dataset['id'], '-p', str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"✗ Download failed!")
                print(f"\n💡 Manual download:")
                print(f"   1. Visit: https://www.kaggle.com/c/{dataset['id']}/data")
                print(f"   2. Click 'Download All'")
                print(f"   3. Extract to: {output_dir}")
                return False
            
            # Unzip files
            print("\nExtracting files...")
            for zip_file in output_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(output_dir)
                zip_file.unlink()  # Remove zip after extraction
            
            print(f"✓ {dataset['name']} downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def download_flir_thermal(self):
        """Download FLIR Thermal Images Dataset ADAS"""
        print("\n" + "="*80)
        print("FLIR THERMAL IMAGES DATASET (ADAS)")
        print("="*80)
        
        dataset = {
            'id': 'deepnewbie/flir-thermal-images-dataset',
            'name': 'FLIR Thermal ADAS',
            'output': str(self.data_dir / 'thermal' / 'flir')
        }
        
        print(f"\nDataset: {dataset['name']}")
        print(f"Size: ~3 GB (14,000+ thermal images)")
        print(f"Purpose: Night operation capability, IR detection")
        print(f"Classes: person, car, bike, dog")
        print(f"Format: COCO JSON annotations")
        
        response = input("\nDownload this dataset? (y/n): ")
        if response.lower() == 'y':
            return self.download_kaggle_dataset(
                dataset['id'],
                dataset['output'],
                dataset['name']
            )
        return False
    

    def download_priority_datasets(self):
        """Download 2 priority datasets for Phase 11 integration (simplified)"""
        print("\n" + "="*80)
        print("PRIORITY DATASETS - PHASE 11 INTEGRATION")
        print("="*80)
        
        print("\nThis will download 2 datasets:")
        print("1. Airbus Ship Detection (~28 GB) - Warship improvement")
        print("2. FLIR Thermal ADAS (~15 GB actual) - Night operation capability")
        print("\nTotal: ~43 GB download")
        print("Estimated time: 2-4 hours (depending on connection)")
        print("\n⚠️  Note: These are LARGE downloads. Manual download recommended.")
        
        response = input("\nProceed with downloads? (y/n): ")
        if response.lower() != 'y':
            print("\n💡 You can download datasets individually:")
            print("   - Option 6: Airbus Ships only")
            print("   - Option 7: FLIR Thermal only")
            print("\n📋 Or download manually from:")
            print("   Airbus: https://www.kaggle.com/c/airbus-ship-detection/data")
            print("   FLIR: https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset")
            return False
        
        results = []
        
        # Download each dataset
        print("\n" + "="*80)
        print("DOWNLOADING DATASET 1/2: AIRBUS SHIPS")
        print("="*80)
        results.append(('Airbus Ships', self.download_airbus_ships()))
        
        print("\n" + "="*80)
        print("DOWNLOADING DATASET 2/2: FLIR THERMAL")
        print("="*80)
        results.append(('FLIR Thermal', self.download_flir_thermal()))
        
        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        for name, success in results:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{name}: {status}")
        
        successful = sum(1 for _, s in results if s)
        print(f"\nCompleted: {successful}/2 datasets")
        
        if successful == 2:
            print("\n🎉 All priority datasets downloaded successfully!")
            print("\n📋 Next steps:")
            print("   1. python scripts/convert_airbus_to_yolo.py")
            print("   2. python scripts/convert_flir_to_yolo.py")
            print("   3. python scripts/merge_datasets.py")
            print("   4. python scripts/train_detector.py --data data/unified_combined/dataset.yaml")
        elif successful > 0:
            print(f"\n⚠️  {successful}/2 datasets downloaded. You can:")
            print("   - Retry failed downloads")
            print("   - Download manually from Kaggle website")
            print("   - Continue with available datasets")
        
        return successful > 0
    
    def download_sample_images(self):
        """Download 10 sample military images from Wikimedia Commons"""
        print("\n" + "="*80)
        print("SAMPLE IMAGES DOWNLOAD")
        print("="*80)
        
        images_dir = self.data_dir / "sample" / "test" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample images from Wikimedia Commons
        images = [
            ("https://upload.wikimedia.org/wikipedia/commons/5/54/Croatian_M-84_tank_during_the_2025_military_parade.jpg", "croatian_tank.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/4/4e/Military_tank_near_Stavanger_Airport.jpg", "stavanger_tank.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/2/23/Tank_in_Taiwan.jpg", "taiwan_tank.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/0/05/US_military_tank_on_Flamenco_Beach%2C_Culebra%2C_Puerto_Rico.jpg", "flamenco_tank.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/e/ef/Saab_JAS-39_Gripen_of_the_Czech_Air_Force_taking_off_from_AFB_%C4%8C%C3%A1slav.jpg", "jas39_gripen.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/1/13/Military_aircraft_at_Daniel_K._Inouye_International_Airport.jpg", "military_aircraft.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/b/b3/Flickr_-_The_U.S._Army_-_Apache_takeoff.jpg", "apache_helicopter.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/0/0e/Royal_Thai_Army%2C_Hino_300_military_truck..jpg", "military_truck1.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/6/60/Royal_Thai_Army%2C_Tata_LPTA_military_truck..jpg", "military_truck2.jpg"),
            ("https://upload.wikimedia.org/wikipedia/commons/4/4b/Military_truck_vehicle_deck_HMS_Albion.jpg", "military_truck3.jpg"),
        ]
        
        print(f"\nDownloading 10 sample images to: {images_dir}")
        
        downloaded = 0
        for url, filename in images:
            output_path = images_dir / filename
            
            if output_path.exists():
                print(f"✓ Exists: {filename}")
                downloaded += 1
                continue
            
            if self.download_file(url, output_path, f"Downloading {filename}"):
                downloaded += 1
        
        print(f"\n✓ Sample images ready: {downloaded}/10")
        return downloaded > 0
   
    def download_coco_val2017(self):
        """Download COCO val2017 dataset"""
        print("\n" + "="*80)
        print("COCO VAL2017 DATASET")
        print("="*80)
        
        coco_dir = self.data_dir / "coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = [
            ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations", "Annotations (241MB)"),
            ("http://images.cocodataset.org/zips/val2017.zip", "val2017", "Val2017 Images (1GB)")
        ]
        
        for url, subdir, desc in datasets:
            target_dir = coco_dir / subdir
            if target_dir.exists():
                print(f"✓ Already exists: {subdir}")
                continue
            
            zip_path = coco_dir / (subdir + ".zip")
            
            if self.download_file(url, zip_path, f"Downloading {desc}"):
                print(f"Extracting {subdir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(coco_dir)
                zip_path.unlink()
                print(f"✓ Extracted: {subdir}")
        
        return True


def print_dataset_catalog():
    """Print comprehensive dataset catalog"""
    print("\n" + "="*80)
    print("DATASET CATALOG")
    print("="*80)
    
    print("\n📦 AVAILABLE DATASETS:\n")
    
    print("1. Military Assets Dataset (Kaggle) - VERIFIED WORKING")
    print("   URL: https://kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format")
    print("   Size: 26,315 images")
    print("   Classes: 12 (tanks, trucks, aircraft, warships, soldiers, etc.)")
    print("   Format: YOLOv8")
    print("   Download: Manual (Kaggle website) or Kaggle API")
    print("   Path: data/military/assets/")
    
    print("\n2. COCO val2017 (Auto-download)")
    print("   URL: http://cocodataset.org")
    print("   Size: 5,000 images")
    print("   Classes: 80 general objects")
    print("   Format: COCO JSON")
    print("   Download: Automatic (this script)")
    print("   Path: data/coco/")
    
    print("\n3. Roboflow Universe Datasets (Public)")
    print("   Multiple military detection datasets available:")
    print("   - Military Vehicle Detection (9,683 images)")
    print("     URL: https://universe.roboflow.com/military-vehicle-detection")
    print("   - YOLO Military Object Detection (2,929 images, 11 classes)")
    print("     URL: https://universe.roboflow.com/uni/yolo-military-object-detection")
    print("   - Military Vehicles (1,994 images)")
    print("     URL: https://universe.roboflow.com/ojlehuha/military-vehicles_v1")
    print("   Format: YOLOv5/v7/v8/v9/v11")
    print("   Download: Free Roboflow account + web download")
    
    print("\n4. Sample Test Images (Auto-download)")
    print("   Source: Wikimedia Commons (Public Domain)")
    print("   Size: 10 images")
    print("   Types: Tanks, Aircraft, Military Vehicles")
    print("   Download: Automatic (this script)")
    print("   Path: data/sample/test/images/")


def main():
    print("="*80)
    print("UNIFIED DATASET MANAGEMENT SYSTEM")
    print("="*80)
    print("\nManages all datasets from Kaggle, COCO, Roboflow, and sample images")
    
    manager = UnifiedDatasetManager()
    
    # Print catalog
    print_dataset_catalog()
    
    print("\n" + "="*80)
    print("AUTO-DOWNLOAD OPTIONS")
    print("="*80)
    print("\n1. Kaggle Military Dataset (26K images) - RECOMMENDED")
    print("2. Sample Images (10 images) - Quick testing")
    print("3. COCO val2017 (~1.2GB) - General objects")
    print("4. All (Military + Sample + COCO)")
    print("5. Priority Datasets (Airbus + FLIR) - PHASE 11 ⭐")
    print("6. Airbus Ship Detection (~28 GB) - Individual")
    print("7. FLIR Thermal ADAS (~15 GB) - Individual")
    print("8. Skip downloads (show manual instructions)")
    
    choice = input("\nSelect option (1-8): ")
    
    if choice == "1":
        manager.download_military_datasets()
    elif choice == "2":
        manager.download_sample_images()
    elif choice == "3":
        manager.download_coco_val2017()
    elif choice == "4":
        manager.download_military_datasets()
        manager.download_sample_images()
        manager.download_coco_val2017()
    elif choice == "5":
        manager.download_priority_datasets()
    elif choice == "6":
        manager.download_airbus_ships()
    elif choice == "7":
        manager.download_flir_thermal()
    
    # Manual download instructions
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    
    print("\n### KAGGLE DATASETS (Recommended for Military Training)")
    print("\n1. Visit: https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format")
    print("2. Login to Kaggle account")
    print("3. Click 'Download' button")
    print("4. Save ZIP to: c:\\MODT\\data\\military\\assets\\")
    print("5. Unzip files in place")
    
    print("\n### ROBOFLOW DATASETS (Alternative)")
    print("\n1. Create free account at: https://roboflow.com")
    print("2. Visit: https://universe.roboflow.com/military-vehicle-detection")
    print("3. Click 'Download Dataset'")
    print("4. Select YOLOv8 format")
    print("5. Extract to: c:\\MODT\\data\\roboflow\\")
    
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    sample_exists = (Path("./data/sample/test/images").exists() and 
                     len(list(Path("./data/sample/test/images").glob('*.jpg'))) > 0)
    coco_exists = Path("./data/coco/val2017").exists()
    
    print(f"\n✓ Sample Images: {'YES' if sample_exists else 'NO'}")
    print(f"✓ COCO val2017: {'YES' if coco_exists else 'NO'}")
    print(f"✓ Military Assets: Manual download required")
    print(f"✓ Roboflow: Manual download required")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run demo with sample images: python scripts/demo.py")
    print("2. Test detection on samples: python scripts/deploy_edge.py --source data/sample/test/images/")
    print("3. After downloading military dataset, train: python scripts/train_detector.py")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
