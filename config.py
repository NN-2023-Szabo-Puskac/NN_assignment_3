WANDB_LOGGING = False
FREEZE_FEATURE_EXTRACTOR = False
CONFIG = {
    "project_name": "card detector",
    "optimizer": {
        "lr": 0.0001,
        "num_epochs": 100,
    },
    "dataset": {
        "img_dir": "",
        "annotations_file": "data/labels_new.csv",
        "img_w": 640,
        "img_h": 640,
        "num_anchors_per_cell": 3,
        "limit": 13,
    },
    "dataloader": {
        "batch_size": 32,
    },

}
