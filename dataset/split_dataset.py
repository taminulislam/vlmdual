import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

class DatasetSplitter:
    def __init__(self, annotations_csv, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/val/test for mmsegmentation framework
        
        Args:
            annotations_csv: Path to augmented_annotations.csv
            output_dir: Root directory for split dataset
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
        """
        self.annotations_csv = annotations_csv
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        print("="*70)
        print("DATASET SPLITTING FOR MMSEGMENTATION")
        print("="*70)
        print(f"\nSplit ratios:")
        print(f"  Train:      {train_ratio*100:.1f}%")
        print(f"  Validation: {val_ratio*100:.1f}%")
        print(f"  Test:       {test_ratio*100:.1f}%")
        print()
    
    def create_directories(self):
        """Create directory structure for train/val/test"""
        print("Creating directory structure...")
        
        for split in ['train', 'val', 'test']:
            frames_dir = Path(self.output_dir) / split / 'frames'
            masks_dir = Path(self.output_dir) / split / 'masks'
            
            frames_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {split}/frames/")
            print(f"  {split}/masks/")
        print()
    
    def split_dataset(self):
        """Perform stratified split and copy files"""
        
        # Load annotations
        print(f"Loading annotations from: {self.annotations_csv}")
        df = pd.read_csv(self.annotations_csv)
        print(f"Total samples: {len(df)}")
        print()
        
        # Create stratification key (class_id + gas_type)
        df['stratify_key'] = df['class_id'].astype(str) + '_' + df['gas_type']
        
        print("Dataset composition:")
        print("-" * 50)
        for key in df['stratify_key'].unique():
            count = len(df[df['stratify_key'] == key])
            print(f"  {key}: {count} samples")
        print()
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=df['stratify_key'],
            random_state=42
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df['stratify_key'],
            random_state=42
        )
        
        print("Split results:")
        print("-" * 50)
        print(f"  Training:   {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        print()
        
        # Create directories
        self.create_directories()
        
        # Copy files for each split
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        split_annotations = {}
        
        for split_name, split_df in splits.items():
            print(f"Processing {split_name} set...")
            print("-" * 50)
            
            split_data = []
            success_count = 0
            failed_count = 0
            
            for idx, row in split_df.iterrows():
                try:
                    # Get source paths
                    src_frame = row['frame_path']
                    src_mask = row['mask_path']
                    
                    # Create destination paths
                    frame_filename = Path(src_frame).name
                    mask_filename = Path(src_mask).name
                    
                    dst_frame = Path(self.output_dir) / split_name / 'frames' / frame_filename
                    dst_mask = Path(self.output_dir) / split_name / 'masks' / mask_filename
                    
                    # Copy files
                    if os.path.exists(src_frame) and os.path.exists(src_mask):
                        shutil.copy2(src_frame, dst_frame)
                        shutil.copy2(src_mask, dst_mask)
                        
                        # Store annotation info
                        split_data.append({
                            'sample_id': row['sample_id'],
                            'original_sample_id': row['original_sample_id'],
                            'ph_value': row['ph_value'],
                            'class_id': row['class_id'],
                            'class_name': row['class_name'],
                            'gas_type': row['gas_type'],
                            'frame_path': str(dst_frame),
                            'mask_path': str(dst_mask),
                            'is_augmented': row['is_augmented'],
                            'augmentation_id': row['augmentation_id']
                        })
                        
                        success_count += 1
                    else:
                        if not os.path.exists(src_frame):
                            print(f"  Warning: Frame not found: {src_frame}")
                        if not os.path.exists(src_mask):
                            print(f"  Warning: Mask not found: {src_mask}")
                        failed_count += 1
                
                except Exception as e:
                    print(f"  Error processing {row['sample_id']}: {str(e)}")
                    failed_count += 1
            
            print(f"  Copied: {success_count} samples")
            if failed_count > 0:
                print(f"  Failed: {failed_count} samples")
            print()
            
            # Save split annotations
            split_df_clean = pd.DataFrame(split_data)
            split_csv = Path(self.output_dir) / f"{split_name}_annotations.csv"
            split_df_clean.to_csv(split_csv, index=False)
            print(f"  Saved annotations: {split_csv}")
            print()
            
            split_annotations[split_name] = split_data
        
        # Print statistics
        self.print_statistics(split_annotations)
        
        # Save split configuration
        self.save_split_config(split_annotations)
        
        return split_annotations
    
    def print_statistics(self, split_annotations):
        """Print detailed statistics for each split"""
        print("="*70)
        print("SPLIT STATISTICS")
        print("="*70)
        print()
        
        for split_name in ['train', 'val', 'test']:
            data = split_annotations[split_name]
            df = pd.DataFrame(data)
            
            print(f"{split_name.upper()} SET ({len(df)} samples)")
            print("-" * 50)
            
            # By class
            print("  By Class:")
            for class_name in sorted(df['class_name'].unique()):
                count = len(df[df['class_name'] == class_name])
                pct = count / len(df) * 100
                print(f"    {class_name:15s}: {count:4d} ({pct:5.1f}%)")
            
            # By gas type
            print("  By Gas Type:")
            for gas_type in sorted(df['gas_type'].unique()):
                count = len(df[df['gas_type'] == gas_type])
                pct = count / len(df) * 100
                print(f"    {gas_type.upper():15s}: {count:4d} ({pct:5.1f}%)")
            
            # Original vs augmented
            orig_count = len(df[df['is_augmented'] == False])
            aug_count = len(df[df['is_augmented'] == True])
            print("  Data Type:")
            print(f"    Original:       {orig_count:4d} ({orig_count/len(df)*100:5.1f}%)")
            print(f"    Augmented:      {aug_count:4d} ({aug_count/len(df)*100:5.1f}%)")
            
            print()
    
    def save_split_config(self, split_annotations):
        """Save split configuration to JSON"""
        config = {
            'dataset_info': {
                'name': 'Gas Emission Dataset',
                'num_classes': 3,
                'classes': ['background', 'tube', 'gas'],
                'class_mapping': {
                    0: 'Healthy',
                    1: 'Transitional',
                    2: 'Acidotic'
                }
            },
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'split_counts': {
                'train': len(split_annotations['train']),
                'val': len(split_annotations['val']),
                'test': len(split_annotations['test']),
                'total': sum(len(split_annotations[s]) for s in ['train', 'val', 'test'])
            },
            'directory_structure': {
                'train': {
                    'frames': str(Path(self.output_dir) / 'train' / 'frames'),
                    'masks': str(Path(self.output_dir) / 'train' / 'masks')
                },
                'val': {
                    'frames': str(Path(self.output_dir) / 'val' / 'frames'),
                    'masks': str(Path(self.output_dir) / 'val' / 'masks')
                },
                'test': {
                    'frames': str(Path(self.output_dir) / 'test' / 'frames'),
                    'masks': str(Path(self.output_dir) / 'test' / 'masks')
                }
            }
        }
        
        config_path = Path(self.output_dir) / 'split_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("="*70)
        print(f"Split configuration saved: {config_path}")
        print("="*70)
        print()


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    
    print()
    print("GAS EMISSION DATASET SPLITTER")
    print("For MMSegmentation Framework")
    print()
    
    # ========== CONFIGURATION ==========
    
    annotations_csv = "augmented_dataset/augmented_annotations.csv"
    output_dir = "mmseg_dataset"
    
    # Split ratios (must sum to 1.0)
    train_ratio = 0.7   # 70% for training
    val_ratio = 0.15    # 15% for validation
    test_ratio = 0.15   # 15% for testing
    
    # ===================================
    
    # Create splitter
    splitter = DatasetSplitter(
        annotations_csv=annotations_csv,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Perform split
    split_annotations = splitter.split_dataset()
    
    print("NEXT STEPS")
    print("="*70)
    print()
    print("1. Dataset split completed successfully!")
    print(f"2. Check output directory: {output_dir}/")
    print("3. Directory structure:")
    print(f"   {output_dir}/")
    print("     train/")
    print("       frames/")
    print("       masks/")
    print("     val/")
    print("       frames/")
    print("       masks/")
    print("     test/")
    print("       frames/")
    print("       masks/")
    print()
    print("4. Use this dataset with MMSegmentation")
    print("5. Update your config file with dataset paths")
    print()

