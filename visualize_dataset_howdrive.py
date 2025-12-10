import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from data.howdrive_dataset import HowDriveDataset
import argparse

# ==================
# CONFIGURATION
# ==================
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HowDir Dataset Structure")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Path to the HowDir dataset root (e.g., /kaggle/input/.../HowDir)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Directory to save plots (default: ./visualizations)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject split (default: 42)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots inline (only save to files)"
    )
    return parser.parse_args()


CLASS_NAMES = [f"c{i}" for i in range(10)]

# ==========================
# HELPER: Count images safely
# ==========================
def count_images_per_subject_class(root_dir, subjects, class_names):
    """Returns dict: {subject: {class: count}}"""
    counts = defaultdict(lambda: defaultdict(int))
    missing = []

    for subject in subjects:
        for cls in class_names:
            cls_path = os.path.join(root_dir, subject, cls)
            if not os.path.exists(cls_path):
                missing.append(cls_path)
                continue
            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[subject][cls] = len(imgs)
    if missing:
        print("⚠️  Warning: Missing directories:")
        for p in missing[:5]:  # show first 5
            print(f"   {p}")
        if len(missing) > 5:
            print(f"   ... and {len(missing) - 5} more.")
    return dict(counts)

# ==========================
# MAIN ANALYSIS
# ==========================
def main():
    args = parse_args()
    ROOT_DIR = args.root_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate root dir
    if not os.path.exists(ROOT_DIR):
        raise FileNotFoundError(f"Root directory not found: {ROOT_DIR}")
    
    SUBJECTS = sorted([
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ])
    
    if not SUBJECTS:
        raise ValueError(f"No subject directories found in {ROOT_DIR}")
    
    print(f"🔍 Found {len(SUBJECTS)} subjects in: {ROOT_DIR}")

    # Set seed for reproducible split
    np.random.seed(args.seed)
    shuffled_subjects = np.random.permutation(SUBJECTS).tolist()
    
    # Subject-wise split: 5 train, 2 val, 2 test (works for 9 subjects)
    if len(shuffled_subjects) != 9:
        print(f"⚠️  Expected 9 subjects, got {len(shuffled_subjects)}. Adjusting split proportions.")

    TRAIN_SUBJECTS = shuffled_subjects[:5]
    VAL_SUBJECTS = shuffled_subjects[5:7]
    TEST_SUBJECTS = shuffled_subjects[7:]

    SPLITS = {
        'Train': TRAIN_SUBJECTS,
        'Val': VAL_SUBJECTS,
        'Test': TEST_SUBJECTS
    }

    # Count data
    all_counts = count_images_per_subject_class(ROOT_DIR, SUBJECTS, CLASS_NAMES)

    global_class_counts = {cls: 0 for cls in CLASS_NAMES}
    subject_totals = {}
    for subject, class_dict in all_counts.items():
        total = sum(class_dict.values())
        subject_totals[subject] = total
        for cls, cnt in class_dict.items():
            global_class_counts[cls] += cnt

    # ---- Plot 1: Global Class Distribution ----
    plt.figure(figsize=(10, 5))
    bars = plt.bar(global_class_counts.keys(), global_class_counts.values(), color='steelblue')
    plt.title("Global Class Distribution (All Subjects)", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_global.png"))
    if not args.no_display:
        plt.show()
    plt.close()

    # ---- Plot 2: Images per Subject ----
    plt.figure(figsize=(12, 5))
    subjects_sorted = sorted(subject_totals.keys())
    totals_sorted = [subject_totals[s] for s in subjects_sorted]
    bars = plt.bar(subjects_sorted, totals_sorted, color='salmon')
    plt.title("Total Images per Subject", fontsize=14)
    plt.xlabel("Subject")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "images_per_subject.png"))
    if not args.no_display:
        plt.show()
    plt.close()

    # ---- Plot 3: Subject × Class Heatmap ----
    matrix = []
    for subject in sorted(SUBJECTS):
        row = [all_counts.get(subject, {}).get(cls, 0) for cls in CLASS_NAMES]
        matrix.append(row)
    matrix = np.array(matrix)

    plt.figure(figsize=(10, max(6, len(SUBJECTS) * 0.5)))
    sns.heatmap(matrix,
                xticklabels=CLASS_NAMES,
                yticklabels=sorted(SUBJECTS),
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Image Count'})
    plt.title("Images per Subject × Class", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "subject_class_heatmap.png"))
    if not args.no_display:
        plt.show()
    plt.close()

    # ---- Plot 4: Split-wise Class Distribution ----
    split_class_counts = {split: {cls: 0 for cls in CLASS_NAMES} for split in SPLITS}
    for split, subjects in SPLITS.items():
        for subject in subjects:
            for cls in CLASS_NAMES:
                split_class_counts[split][cls] += all_counts.get(subject, {}).get(cls, 0)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, [split_class_counts['Train'][c] for c in CLASS_NAMES], width, label='Train', color='#1f77b4')
    plt.bar(x,        [split_class_counts['Val'][c] for c in CLASS_NAMES],   width, label='Val',   color='#ff7f0e')
    plt.bar(x + width, [split_class_counts['Test'][c] for c in CLASS_NAMES], width, label='Test',  color='#2ca02c')
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.title("Class Distribution Across Subject-Wise Splits", fontsize=14)
    plt.xticks(x, CLASS_NAMES)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "split_class_distribution.png"))
    if not args.no_display:
        plt.show()
    plt.close()

    # ---- Print Summary ----
    print(f"\n✅ Analysis complete!")
    print(f"📊 Total images: {sum(subject_totals.values())}")
    print(f"👥 Subjects: {len(SUBJECTS)}")
    print(f"📚 Classes: {len(CLASS_NAMES)}")
    print("\nSplits (subject-wise):")
    for split, subs in SPLITS.items():
        total_imgs = sum(subject_totals[s] for s in subs)
        print(f"  {split:5s}: {len(subs):2d} subjects, {total_imgs:3d} images")
    print(f"\n📁 Plots saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()