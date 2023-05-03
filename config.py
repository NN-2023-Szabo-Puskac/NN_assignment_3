WANDB_LOGGING = False
FREEZE_FEATURE_EXTRACTOR = True
CONFIG = {
    "project_name": "card detector",
    "optimizer": {
        "lr": 0.0001,
        "num_epochs": 100,
    },
    "dataset": {
        "img_dir": "",
        "annotations_file_train": "data/train_labels.csv",
        "annotations_file_val": "data/val_labels.csv",
        "img_w": 640,
        "img_h": 640,
        "num_anchors_per_cell": 3,
        "limit": None,
    },
    "dataloader": {
        "batch_size": 32,
    },

}
