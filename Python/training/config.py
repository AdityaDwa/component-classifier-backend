"""Training configuration and hyperparameters for YOLOv11s fine-tuning"""

TRAINING_CONFIG = {
    "model": "yolo11s.pt",          # Pre-trained YOLO model
    "epochs": 5,                    # Maximum training epochs
    "imgsz": 1024,                  # Image size (1024x1024)
    "batch": 4,                     # Batch size (adjust based on GPU)
    "patience": 50,                 # Early stopping patience
    "device": 0,                    # GPU index (0 = first GPU)
    "workers": 4,                   # Data loading workers
    "project": "checkpoints",       # Where to save runs
    "name": "train",                # Run name
}

SPLIT_CONFIG = {
    "train_ratio": 0.8,    # 80% training
    "val_ratio": 0.1,      # 10% validation
    "test_ratio": 0.1,     # 10% test
    "random_seed": 42,     # Reproducibility
}