import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from tqdm import tqdm
import shutil

class GasDatasetAugmenter:
    def __init__(self, annotations_csv, output_dir, augmentations_per_image=20):
        """
        Augment gas emission dataset
        
        Args:
            annotations_csv: Path to annotations.csv
            output_dir: Where to save augmented data
            augmentations_per_image: How many augmented versions per original (default: 20)
        """
        self.annotations_csv = annotations_csv
        self.output_dir = output_dir
        self.augmentations_per_image = augmentations_per_image
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/frames", exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        print(f"Augmentations per image: {augmentations_per_image}")
        print()
        
        # Define augmentation pipeline
        self.transform = self.get_augmentation_pipeline()
    
    def get_augmentation_pipeline(self):
        """
        Define augmentation transformations
        
        These are specifically designed for gas plume images:
        - Preserve gas visibility
        - Simulate real-world variations
        - Keep masks aligned with frames
        """
        return A.Compose([
            # Geometric transformations (50% chance each)
            A.HorizontalFlip(p=0.5),
            
            A.Rotate(
                limit=15,  # Rotate up to ±15 degrees
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ),
            
            A.ShiftScaleRotate(
                shift_limit=0.1,    # Shift up to 10%
                scale_limit=0.15,   # Scale 85-115%
                rotate_limit=15,    # Rotate ±15 degrees
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            ),
            
            # Intensity transformations (important for gas visibility)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,   # ±20% brightness
                contrast_limit=0.2,     # ±20% contrast
                p=0.7
            ),
            
            A.RandomGamma(
                gamma_limit=(80, 120),  # Gamma correction
                p=0.5
            ),
            
            # Noise (simulates camera sensor noise)
            A.GaussNoise(
                var_limit=(10.0, 50.0),  # Gaussian noise
                p=0.3
            ),
            
            # Blur (simulates gas dispersion and camera focus)
            A.GaussianBlur(
                blur_limit=(3, 5),  # Slight blur
                p=0.3
            ),
            
            A.MotionBlur(
                blur_limit=5,  # Simulates camera/object movement
                p=0.2
            ),
            
            # Elastic transform (simulates gas flow variations)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3
            ),
            
            # Grid distortion (simulates turbulence)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3
            ),
            
        ], additional_targets={'mask': 'mask'})
    
    def augment_single_sample(self, image, mask):
        """Apply augmentation to a single image-mask pair"""
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def augment_dataset(self):
        """
        Main function to augment entire dataset
        """
        print("="*70)
        print("DATASET AUGMENTATION")
        print("="*70)
        print()
        
        # Load annotations
        print(f"Loading annotations from: {self.annotations_csv}")
        df = pd.read_csv(self.annotations_csv)
        
        print(f"Found {len(df)} original samples")
        print(f"Will generate {self.augmentations_per_image} augmentations per sample")
        print(f"Total samples after augmentation: {len(df) * (self.augmentations_per_image + 1):,}")
        print()
        
        # Store new annotations
        new_annotations = []
        
        # Statistics
        total_processed = 0
        total_failed = 0
        
        # Step 1: Copy original data
        print("Step 1/2: Copying original data...")
        print("-" * 70)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying originals"):
            try:
                # Load original frame and mask
                frame_path = row['frame_path']
                mask_path = row['mask_path']
                
                if not os.path.exists(frame_path):
                    print(f"  Warning: Frame not found: {frame_path}")
                    total_failed += 1
                    continue
                
                if not os.path.exists(mask_path):
                    print(f"  Warning: Mask not found: {mask_path}")
                    total_failed += 1
                    continue
                
                # Create new paths
                new_frame_path = f"{self.output_dir}/frames/{row['sample_id']}_orig.png"
                new_mask_path = f"{self.output_dir}/masks/{row['sample_id']}_orig.png"
                
                # Copy files
                shutil.copy(frame_path, new_frame_path)
                shutil.copy(mask_path, new_mask_path)
                
                # Add to annotations
                new_annotations.append({
                    'sample_id': f"{row['sample_id']}_orig",
                    'original_sample_id': row['sample_id'],
                    'ph_value': row['ph_value'],
                    'class_id': row['class_id'],
                    'class_name': row['class_name'],
                    'gas_type': row['gas_type'],
                    'frame_path': new_frame_path,
                    'mask_path': new_mask_path,
                    'is_augmented': False,
                    'augmentation_id': -1
                })
                
                total_processed += 1
                
            except Exception as e:
                print(f"  Error processing {row['sample_id']}: {str(e)}")
                total_failed += 1
        
        print(f"\nCopied {total_processed} original samples")
        if total_failed > 0:
            print(f"Warning: Failed to copy {total_failed} samples")
        print()
        
        # Step 2: Generate augmentations
        print("Step 2/2: Generating augmentations...")
        print("-" * 70)
        
        aug_processed = 0
        aug_failed = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            try:
                # Load original image and mask
                image = cv2.imread(row['frame_path'], cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"  Warning: Could not read image: {row['frame_path']}")
                    aug_failed += 1
                    continue
                
                if mask is None:
                    print(f"  Warning: Could not read mask: {row['mask_path']}")
                    aug_failed += 1
                    continue
                
                # Generate N augmented versions
                for aug_idx in range(self.augmentations_per_image):
                    try:
                        # Apply augmentation
                        aug_image, aug_mask = self.augment_single_sample(image, mask)
                        
                        # Create file paths
                        aug_sample_id = f"{row['sample_id']}_aug_{aug_idx:03d}"
                        aug_frame_path = f"{self.output_dir}/frames/{aug_sample_id}.png"
                        aug_mask_path = f"{self.output_dir}/masks/{aug_sample_id}.png"
                        
                        # Save augmented files
                        cv2.imwrite(aug_frame_path, aug_image)
                        cv2.imwrite(aug_mask_path, aug_mask)
                        
                        # Add to annotations
                        new_annotations.append({
                            'sample_id': aug_sample_id,
                            'original_sample_id': row['sample_id'],
                            'ph_value': row['ph_value'],
                            'class_id': row['class_id'],
                            'class_name': row['class_name'],
                            'gas_type': row['gas_type'],
                            'frame_path': aug_frame_path,
                            'mask_path': aug_mask_path,
                            'is_augmented': True,
                            'augmentation_id': aug_idx
                        })
                        
                        aug_processed += 1
                        
                    except Exception as e:
                        print(f"  Error augmenting {row['sample_id']} (aug {aug_idx}): {str(e)}")
                        aug_failed += 1
                
            except Exception as e:
                print(f"  Error processing {row['sample_id']}: {str(e)}")
                aug_failed += self.augmentations_per_image
        
        print(f"\nGenerated {aug_processed} augmented samples")
        if aug_failed > 0:
            print(f"Warning: Failed to generate {aug_failed} augmentations")
        print()
        
        # Save new annotations
        print("Saving augmented annotations...")
        new_df = pd.DataFrame(new_annotations)
        output_csv = f"{self.output_dir}/augmented_annotations.csv"
        new_df.to_csv(output_csv, index=False)
        
        print(f"Saved to: {output_csv}")
        print()
        
        # Print statistics
        self.print_statistics(df, new_df)
        
        return new_df
    
    def print_statistics(self, original_df, augmented_df):
        """Print detailed statistics about augmentation"""
        print("="*70)
        print("AUGMENTATION SUMMARY")
        print("="*70)
        print()
        
        print("Sample Counts:")
        print("-" * 50)
        print(f"  Original samples:   {len(original_df):6,}")
        print(f"  Augmented samples:  {len(augmented_df):6,}")
        print(f"  Multiplication:     {len(augmented_df) / len(original_df):6.1f}x")
        print()
        
        print("By Classification Class:")
        print("-" * 50)
        for class_name in ['Healthy', 'Transitional', 'Acidotic']:
            orig_count = len(original_df[original_df['class_name'] == class_name])
            aug_count = len(augmented_df[augmented_df['class_name'] == class_name])
            if orig_count > 0:
                print(f"  {class_name:15s}: {orig_count:4,} -> {aug_count:6,} ({aug_count/orig_count:.1f}x)")
        print()
        
        print("By Gas Type:")
        print("-" * 50)
        for gas_type in ['co2', 'ch4']:
            orig_count = len(original_df[original_df['gas_type'] == gas_type])
            aug_count = len(augmented_df[augmented_df['gas_type'] == gas_type])
            if orig_count > 0:
                print(f"  {gas_type.upper():15s}: {orig_count:4,} -> {aug_count:6,} ({aug_count/orig_count:.1f}x)")
        print()
        
        print("By pH Level:")
        print("-" * 50)
        for ph_value in sorted(original_df['ph_value'].unique()):
            orig_count = len(original_df[original_df['ph_value'] == ph_value])
            aug_count = len(augmented_df[augmented_df['ph_value'] == ph_value])
            class_name = original_df[original_df['ph_value'] == ph_value].iloc[0]['class_name']
            print(f"  pH {ph_value} ({class_name:12s}): {orig_count:4,} -> {aug_count:6,}")
        print()
        
        print("="*70)
        print("NEXT STEPS")
        print("="*70)
        print()
        print("1. Dataset augmented successfully!")
        print(f"2. Check output directory: {self.output_dir}/")
        print("3. Verify augmentation quality (run verification)")
        print("4. Split into train/val/test sets")
        print("5. Start training your model!")
        print()
    
    def verify_augmentation(self, num_samples=6):
        """
        Create visualization to verify augmentation quality
        """
        import matplotlib.pyplot as plt
        
        print("="*70)
        print("AUGMENTATION VERIFICATION")
        print("="*70)
        print()
        
        # Load augmented annotations
        aug_csv = f"{self.output_dir}/augmented_annotations.csv"
        if not os.path.exists(aug_csv):
            print(f"Error: Augmented annotations not found: {aug_csv}")
            return
        
        df = pd.read_csv(aug_csv)
        
        # Get random augmented samples (one from each class if possible)
        samples = []
        for class_name in ['Healthy', 'Transitional', 'Acidotic']:
            class_samples = df[(df['class_name'] == class_name) & (df['is_augmented'] == True)]
            if len(class_samples) > 0:
                samples.append(class_samples.sample(min(2, len(class_samples))))
        
        samples_df = pd.concat(samples).head(num_samples)
        
        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, row) in enumerate(samples_df.iterrows()):
            # Load frame and mask
            frame = cv2.imread(row['frame_path'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            
            if frame is None or mask is None:
                print(f"Warning: Could not load sample: {row['sample_id']}")
                continue
            
            # Create colored overlay
            overlay = np.zeros((*frame.shape, 3), dtype=np.uint8)
            overlay[mask == 0] = [0, 0, 0]        # Background - black
            overlay[mask == 1] = [128, 128, 128]  # Tube - gray
            overlay[mask == 2] = [255, 0, 0]      # Gas - red
            
            # Plot
            axes[i, 0].imshow(frame, cmap='gray')
            axes[i, 0].set_title(f"{row['sample_id']}\n{row['gas_type'].upper()}, pH {row['ph_value']} ({row['class_name']})")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=2)
            axes[i, 1].set_title('Segmentation Mask\n(0=bg, 1=tube, 2=gas)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay\n(Red=Gas, Gray=Tube)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{self.output_dir}/augmentation_verification.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Verification plot saved: {save_path}")
        
        # Try to show (may not work in all environments)
        try:
            plt.show()
        except:
            print("  (Could not display plot, but it's saved to file)")
        
        print()


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    
    print()
    print("GAS EMISSION DATASET AUGMENTATION")
    print()
    
    # ========== CONFIGURATION ==========
    
    annotations_csv = "annotations.csv"      # Your annotations file
    output_dir = "augmented_dataset"         # Where to save augmented data
    augmentations_per_image = 20             # How many augmentations per image
    
    # ===================================
    
    # Create augmenter
    augmenter = GasDatasetAugmenter(
        annotations_csv=annotations_csv,
        output_dir=output_dir,
        augmentations_per_image=augmentations_per_image
    )
    
    # Run augmentation
    print("Starting augmentation process...\n")
    augmented_df = augmenter.augment_dataset()
    
    # Verify augmentation quality
    print("\nGenerating verification plots...\n")
    augmenter.verify_augmentation(num_samples=6)
    
    print("\nALL DONE!")
    print(f"\nYour augmented dataset is ready in: {output_dir}/")
    print(f"Use file: {output_dir}/augmented_annotations.csv for training\n")