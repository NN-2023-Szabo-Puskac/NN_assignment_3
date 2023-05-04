WANDB_LOGGING = True
FREEZE_FEATURE_EXTRACTOR = True
CONFIG = {
    "project_name": "card_detector_22k",
    "optimizer": {
        "lr": 0.0001,
        "num_epochs": 20,
    },
    "dataset": {
        "img_dir": "",
        "annotations_file_train": "data/train_labels_22k.csv",
        "annotations_file_val": "data/val_labels_22k.csv",
        "annotations_file_test": "data/test_labels_22k.csv",
        "img_w": 640,
        "img_h": 640,
        "num_anchors_per_cell": 3,
        "limit": None,
        "num_max_boxes": 5,
    },
    "dataloader": {
        "batch_size": 64,
    },
}
