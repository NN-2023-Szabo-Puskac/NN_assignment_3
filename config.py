WANDB_LOGGING = False
FREEZE_FEATURE_EXTRACTOR = False
CONFIG = {
    "project_name": "card detector",
    "optimizer": {
        "lr": 0.0001,
    },
    "dataset": {
        "img_dir": "data/images/",
        "annotations_file": "data/labels.csv",
        "img_w": 640,
        "img_h": 640,
        "num_anchors_per_cell": 3,
    },
    "dataloader": {
        "batch_size": 32,
    }
}