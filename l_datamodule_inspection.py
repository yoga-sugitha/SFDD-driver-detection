import os
from data.howdrive_datamodule import HowDriveDataModule  # adjust import to your project structure
import torch

def inspect_dataloader(name, loader):
    print(f"\n--- Inspecting {name} dataloader ---")
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"❌ Error fetching batch: {e}")
        return

    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        imgs, labels = batch
    else:
        print("❌ Unexpected batch structure:", type(batch))
        return

    print(f"Batch image tensor shape: {imgs.shape}")
    print(f"Batch label shape:        {labels.shape}")
    print(f"Batch dtype:              {imgs.dtype}")
    print(f"Label dtype:              {labels.dtype}")

    # Quick assertions
    assert imgs.ndim == 4, "Images must be [B,C,H,W]"
    assert labels.ndim == 1, "Labels must be [B]"
    assert imgs.shape[2] == imgs.shape[3], "H and W must be equal (square images)"

    print("✔ Structure OK")

def main():
    data_dir = "/kaggle/input/howdir-safedriving-benchmarking-sample/HowDir"  # adjust path

    dm = HowDriveDataModule(
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        img_size=224,
        seed=42,
        task_type="multiclass"
    )

    print("\n===== Running setup() =====")
    dm.setup()

    # Print dataset sizes
    print("\n===== Dataset Sizes =====")
    print(f"Train samples: {len(dm._train_paths)}")
    print(f"Val samples:   {len(dm._val_paths)}")
    print(f"Test samples:  {len(dm._test_paths)}")

    # Build loaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Inspect 1 batch from each
    inspect_dataloader("train", train_loader)
    inspect_dataloader("val", val_loader)
    inspect_dataloader("test", test_loader)

    # Test get_test_sample()
    print("\n===== Testing get_test_sample() =====")
    try:
        img, label = dm.get_test_sample(0)
        print(f"Sample image shape: {img.shape}")
        print(f"Sample label:       {label}")
        print("✔ get_test_sample() OK")
    except Exception as e:
        print("❌ Error:", e)

    print("\nAll checks done.")

if __name__ == "__main__":
    main()
