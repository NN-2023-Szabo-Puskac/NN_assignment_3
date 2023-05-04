WANDB_LOGGING = True
FREEZE_FEATURE_EXTRACTOR = True
CONFIG = {
    "project_name": "card_detector",
    "optimizer": {
        "lr": 0.0001,
        "num_epochs": 20,
    },
    "dataset": {
        "img_dir": "",
        "annotations_file_train": "data/train_labels_22k.csv",
        "annotations_file_val": "data/val_labels_22k.csv",
        "annotations_file_test": "data/test_labels_22k.csv",
        "annotations_file_real": "data/real/real_labels.csv",
        "img_w": 640,
        "img_h": 640,
        "num_anchors_per_cell": 3,
        "limit": 1000,
        "num_max_boxes": 5,
    },
    "dataloader": {
        "batch_size": 64,
    },
    "training": {
        "SOURCE_PATH": None,
        "DEST_PATH": "models/present_run_local_weight_5_10x10",
        "ANCHOR_BOXES": [
            [198.27963804, 206.74086672],
            [129.59395666, 161.90171490],
            [161.65437828, 232.34624509],
        ],
        "LOCALIZATION_WEIGHT": 5,
    },
}
