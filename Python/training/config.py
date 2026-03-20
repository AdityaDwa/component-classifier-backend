"""Training configuration and hyperparameters for YOLOv11s fine-tuning"""

TRAINING_CONFIG = {
    "model": "yolo11s.pt",          # Pre-trained YOLO model
    "epochs": 5,                  # Maximum training epochs
    "imgsz": 1024,                  # Image size (1024x1024)
    "batch": 4,                    # Batch size (adjust based on GPU)
    "patience": 50,                 # Early stopping patience
    "device": 0,                    # GPU index (0 = first GPU)
    "workers": 4,                   # Data loading workers
    "project": "checkpoints",       # Where to save runs
    "name": "train",                # Run name
}
    
# Class weights for imbalanced classes
# Higher weight = model penalized more for missing that class
# CLASS_WEIGHTS = {
#     0: 1.0,   # heading
#     1: 0.5,   # link (very common, lower weight)
#     2: 1.0,   # image
#     3: 1.0,   # text
#     4: 1.0,   # list
#     5: 2.0,   # header
#     6: 2.0,   # footer
#     7: 1.0,   # table
#     8: 1.5,   # input
#     9: 1.0,   # button
#     10: 1.5,  # navigation
#     11: 10.0, # sidebar (only 65 samples, very rare)
#     12: 8.0,  # dialog (rare)
#     13: 0.8,  # container (very common, noisy)
# }

SPLIT_CONFIG = {
    "train_ratio": 0.8,    # 80% training
    "val_ratio": 0.1,      # 10% validation
    "test_ratio": 0.1,     # 10% test
    "random_seed": 42,     # Reproducibility
}