import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.ops import box_iou, box_convert
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms
import math


class MTGCardsDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        anchor_boxes,
        feature_map_dims,
        img_dims,
        num_anchors_per_cell,
        num_max_boxes,
        transform=None,
        target_transform=None,
        limit=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.limit = len(self.img_labels) if limit is None else limit

        self.img_dir = img_dir

        self.anchor_boxes = anchor_boxes
        self.feature_map_w = feature_map_dims[0]
        self.feature_map_h = feature_map_dims[1]
        self.img_w = img_dims[0]
        self.img_h = img_dims[1]
        self.scale_w = self.img_w / self.feature_map_w
        self.scale_h = self.img_h / self.feature_map_h
        self.num_anchors_per_cell = num_anchors_per_cell
        self.num_max_boxes = num_max_boxes

        self.transform = transform
        self.target_transform = target_transform

        self.target_labels = self.generate_target_labels()

    def generate_target_labels(self):
        target_labels = []
        for idx in range(self.limit):
            label = np.array(
                self.img_labels.iloc[idx, 1:].values, dtype=float
            )
            target_labels.append(self.generate_feature_label(label=label))
        return target_labels
    
    def get_ground_truth(self, label):
        real_cx = label[-4]
        real_cy = label[-3]
        real_w = label[-2]
        real_h = label[-1]

        label_cell_x = int(
            real_cx / self.scale_w
        )  
        label_cell_y = int(real_cy / self.scale_h)
        label_cx = (
            real_cx / self.scale_w
        ) - label_cell_x  
        label_cy = (real_cy / self.scale_h) - label_cell_y
        label_w = real_w / self.scale_w
        label_h = real_h / self.scale_h

        return {
            "cell_x": label_cell_x, # the cell in whice the box is centered
            "cell_y": label_cell_y,
            "cx": label_cx, # offset from the 0,0 of the cell
            "cy": label_cy,
            "w": label_w,   # width of the box in cells
            "h": label_h,   # height of the box in cells
        }

    def generate_feature_label(self, label):
        # we are only doing this for the first box
        box_label = label[:4]
        gt_cell = self.get_ground_truth(label=box_label)
        gt_xyxy = box_convert(torch.Tensor(box_label), in_fmt="cxcywh", out_fmt="xyxy")

        max_iou = 0
        target_anchor_idx = 0
        for anchor_idx in range(len(self.anchor_boxes)):
            anchor_cxcywh = torch.Tensor([box_label[0], box_label[1], self.anchor_boxes[anchor_idx][0], self.anchor_boxes[anchor_idx][1]])
            anchor_xyxy = box_convert(anchor_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
            iou = box_iou(gt_xyxy.unsqueeze(0), anchor_xyxy.unsqueeze(0))
            if iou > max_iou:
                max_iou = iou
                target_anchor_idx = anchor_idx

        target = torch.zeros((self.feature_map_w, self.feature_map_h, self.num_anchors_per_cell, 5))  # create an empty ground truth
        cell_x_min = int(gt_cell["cell_x"] + gt_cell["cx"] - (gt_cell["w"] / 2))
        cell_y_min = int(gt_cell["cell_y"] + gt_cell["cy"] - (gt_cell["h"] / 2))
        cell_x_max = int(gt_cell["cell_x"] + gt_cell["cx"] + (gt_cell["w"] / 2))
        cell_y_max = int(gt_cell["cell_y"] + gt_cell["cy"] + (gt_cell["h"] / 2))

        tw = math.log(box_label[-2] / self.anchor_boxes[target_anchor_idx][0])
        th = math.log(box_label[-1] / self.anchor_boxes[target_anchor_idx][1])
        for x in range(cell_x_min, cell_x_max + 1):
            for y in range(cell_y_min, cell_y_max):
                offset_x = gt_cell["cell_x"] + gt_cell["cx"] - x
                offset_y = gt_cell["cell_y"] + gt_cell["cy"] - y

                target[x, y,target_anchor_idx, 0] = 1.0
                target[x, y,target_anchor_idx, 1:] = torch.Tensor([offset_x, offset_y, tw, th])

        return target


    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        # label = np.array(self.img_labels.iloc[idx, [1,2,3,4]].values, dtype=float)
        label = self.target_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_transform_pipe(img_w, img_h):
    transform_pipe = transforms.Compose(
        [
            transforms.Resize([img_w, img_h]),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform_pipe


if __name__ == "__main__":
    anchor_boxes = torch.Tensor([[198.27963804, 206.74086672],
       [129.59395666, 161.90171490],
       [161.65437828, 232.34624509]
    ])
    train_dataset = MTGCardsDataset(
        annotations_file="data/val_labels.csv",
        img_dir="",
        anchor_boxes=anchor_boxes,
        feature_map_dims=(20,20),
        img_dims= (640, 640),
        num_anchors_per_cell=3,
        num_max_boxes=1,
        transform=get_transform_pipe(640,640),
        limit=None
    )
    target = train_dataset.generate_feature_label([300, 300, 200, 50])
    print(target[torch.where(target[:,:,:,0] == 1)])
    print(torch.where(target[:,:,:,0] == 1))


#class MTGCardsDataset(Dataset):
#    def __init__(
#        self,
#        annotations_file,
#        img_dir,
#        anchor_boxes,
#        feature_map_dims,
#        img_dims,
#        num_anchors_per_cell,
#        num_max_boxes,
#        transform=None,
#        target_transform=None,
#        limit=None,
#    ):
#        self.img_labels = pd.read_csv(annotations_file)
#        self.limit = len(self.img_labels) if limit is None else limit
#
#        self.img_dir = img_dir
#
#        self.anchor_boxes = anchor_boxes
#        self.feature_map_w = feature_map_dims[0]
#        self.feature_map_h = feature_map_dims[1]
#        self.img_w = img_dims[0]
#        self.img_h = img_dims[1]
#        self.scale_w = self.img_w / self.feature_map_w
#        self.scale_h = self.img_h / self.feature_map_h
#        self.num_anchors_per_cell = num_anchors_per_cell
#        self.num_max_boxes = num_max_boxes
#
#        self.transform = transform
#        self.target_transform = target_transform
#
#        self.target_labels = self.generate_target_labels()
#
#    def generate_target_labels(self):
#        target_labels = []
#        for idx in range(self.limit):
#            label = np.array(
#                self.img_labels.iloc[idx, 1:].values, dtype=float
#            )
#            target_labels.append(self.generate_feature_label(label=label))
#        return target_labels
#
#    def get_ground_truth(self, label):
#        real_cx = label[-4]
#        real_cy = label[-3]
#        real_w = label[-2]
#        real_h = label[-1]
#
#        label_cell_x = int(
#            real_cx / self.scale_w
#        )  # the cell in whice the box is centered
#        label_cell_y = int(real_cy / self.scale_h)
#        label_cx = (
#            real_cx / self.scale_w
#        ) - label_cell_x  # offset from the 0,0 of the cell
#        label_cy = (real_cy / self.scale_h) - label_cell_y
#        label_w = math.log(real_w / self.scale_w)  # width and height of the box in cells
#        label_h = math.log(real_h / self.scale_h)
#        return torch.Tensor(
#            [label_cell_x, label_cell_y, label_cx, label_cy, label_w, label_h]
#        )
#    
#
#    def get_ground_truth_real_coords(
#        self, label
#    ):  # convets the box from (cx, cy, w, h) to (x1, y1, x2, y2) - TODO: maybe we can do this using torchvision.ops.box_convert
#        real_cx = label[-4]
#        real_cy = label[-3]
#        real_w = label[-2]
#        real_h = label[-1]
#        real_x1 = real_cx - (real_w / 2)
#        real_y1 = real_cy - (real_h / 2)
#        real_x2 = real_cx + (real_w / 2)
#        real_y2 = real_cy + (real_h / 2)
#        return torch.Tensor([[real_x1, real_y1, real_x2, real_y2]])
#
#    def get_anchor(self, anchor, pos_x, pos_y):
#        anchor_cx = pos_x * self.scale_w
#        anchor_cy = pos_y * self.scale_h
#        anchor_x1 = anchor_cx - (anchor[0] / 2)
#        anchor_y1 = anchor_cy - (anchor[1] / 2)
#        anchor_x2 = anchor_cx + (anchor[0] / 2)
#        anchor_y2 = anchor_cy + (anchor[1] / 2)
#        return torch.Tensor([[anchor_x1, anchor_y1, anchor_x2, anchor_y2]])
#
#
#    def generate_feature_label(self, label):
#        
#        box_labels = []
#        for i in range(0, (self.num_max_boxes * 4), 4):
#            if not np.all(label[i:i+4] == 0):
#                box_labels.append(label[i:i+4])
#
#        gt_objects = []
#        gt_real_coords = []
#        for box_label in box_labels:
#            gt_object = self.get_ground_truth(label=box_label)
#            gt_real_coord = self.get_ground_truth_real_coords(label=box_label)
#            gt_objects.append(gt_object)
#            gt_real_coords.append(gt_real_coord)
#
#        target = torch.zeros(
#            (self.feature_map_w, self.feature_map_h, self.num_anchors_per_cell, 5)
#        )  # create an empty ground truth
#
#        target_objects = []
#        for gt_obj_idx in range(len(gt_objects)):
#
#            cell_x_idx = int(gt_objects[gt_obj_idx][0])
#            cell_y_idx = int(gt_objects[gt_obj_idx][1])
#            x_offset = gt_objects[gt_obj_idx][2]
#            y_offset = gt_objects[gt_obj_idx][3]
#            #w = gt_objects[gt_obj_idx][4]
#            #h = gt_objects[gt_obj_idx][5]
#
#            max_iou = 0
#            target_anchor_idx = 0
#            for anchor_idx in range(len(self.anchor_boxes)):
#                anchor = self.get_anchor(
#                    self.anchor_boxes[anchor_idx],
#                    (cell_x_idx + x_offset),
#                    (cell_y_idx + y_offset),
#                ) 
#                iou = box_iou(gt_real_coords[gt_obj_idx], anchor)
#                if iou > max_iou:
#                    max_iou = iou
#                    target_anchor_idx = anchor_idx
#
#            tx = x_offset
#            ty = y_offset
#            tw = math.log(box_labels[gt_obj_idx][-2] / self.anchor_boxes[target_anchor_idx][0])
#            th = math.log(box_labels[gt_obj_idx][-1] / self.anchor_boxes[target_anchor_idx][1])
#            target_objects.append([cell_x_idx, cell_y_idx, target_anchor_idx, tx, ty, tw, th])
#
#        for target_object in target_objects:
#            #      cell_x_index      cell_y_index      target_anchor_idx
#            target[target_object[0], target_object[1], target_object[2], 0] = 1.0
#            target[target_object[0], target_object[1], target_object[2], 1:5] = torch.Tensor([
#                target_object[3],   # tx
#                target_object[4],   # ty
#                target_object[5],   # tw
#                target_object[6],   # th
#            ])
#        #print(f"RETURNED TARGET: {[target[target_object[0], target_object[1], target_object[2], :] for target_object in target_objects]}")
#        return target
#    
#    def get_ground_truthV2(self, label):
#        real_cx = label[-4]
#        real_cy = label[-3]
#        real_w = label[-2]
#        real_h = label[-1]
#
#        label_cell_x = int(
#            real_cx / self.scale_w
#        )  
#        label_cell_y = int(real_cy / self.scale_h)
#        label_cx = (
#            real_cx / self.scale_w
#        ) - label_cell_x  
#        label_cy = (real_cy / self.scale_h) - label_cell_y
#        label_w = real_w / self.scale_w
#        label_h = real_h / self.scale_h
#
#        return {
#            "cell_x": label_cell_x, # the cell in whice the box is centered
#            "cell_y": label_cell_y,
#            "cx": label_cx, # offset from the 0,0 of the cell
#            "cy": label_cy,
#            "w": label_w,   # width of the box in cells
#            "h": label_h,   # height of the box in cells
#        }
#
#
#    def generate_feature_labelV2(self, label):
#        # we are only doing this for the first box
#        box_label = label[:4]
#        gt_cell = self.get_ground_truthV2(label=box_label)
#        gt_xyxy = box_convert(torch.Tensor(box_label), in_fmt="cxcywh", out_fmt="xyxy")
#
#        max_iou = 0
#        target_anchor_idx = 0
#        for anchor_idx in range(len(self.anchor_boxes)):
#            anchor_cxcywh = torch.Tensor([box_label[0], box_label[1], self.anchor_boxes[anchor_idx][0], self.anchor_boxes[anchor_idx][1]])
#            anchor_xyxy = box_convert(anchor_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
#            iou = box_iou(gt_xyxy.unsqueeze(0), anchor_xyxy.unsqueeze(0))
#            if iou > max_iou:
#                max_iou = iou
#                target_anchor_idx = anchor_idx
#
#        target = torch.zeros((self.feature_map_w, self.feature_map_h, self.num_anchors_per_cell, 5))  # create an empty ground truth
#        cell_x_min = int(gt_cell["cell_x"] + gt_cell["cx"] - (gt_cell["w"] / 2))
#        cell_y_min = int(gt_cell["cell_y"] + gt_cell["cy"] - (gt_cell["h"] / 2))
#        cell_x_max = int(gt_cell["cell_x"] + gt_cell["cx"] + (gt_cell["w"] / 2))
#        cell_y_max = int(gt_cell["cell_y"] + gt_cell["cy"] + (gt_cell["h"] / 2))
#
#
#        tw = math.log(box_label[-2] / self.anchor_boxes[target_anchor_idx][0])
#        th = math.log(box_label[-1] / self.anchor_boxes[target_anchor_idx][1])
#        for x in range(cell_x_min, cell_x_max + 1):
#            for y in range(cell_y_min, cell_y_max):
#                offset_x = gt_cell["cell_x"] + gt_cell["cx"] - x
#                offset_y = gt_cell["cell_y"] + gt_cell["cy"] - y
#
#                target[x, y,target_anchor_idx, 0] = 1.0
#                target[x, y,target_anchor_idx, 1:] = torch.Tensor([offset_x, offset_y, tw, th])
#
#
#
#        return target
#
#
#    def __len__(self):
#        return self.limit
#
#    def __getitem__(self, idx):
#        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#        image = read_image(img_path, mode=ImageReadMode.RGB)
#        # label = np.array(self.img_labels.iloc[idx, [1,2,3,4]].values, dtype=float)
#        label = self.target_labels[idx]
#        if self.transform:
#            image = self.transform(image)
#        if self.target_transform:
#            label = self.target_transform(label)
#        return image, label