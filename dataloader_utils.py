import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.ops import box_iou
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms
import math


class MTGCardsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, anchor_boxes, feature_map_dims, img_dims, num_anchors_per_cell, num_max_boxes, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
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
        for idx in range(len(self.img_labels)):
            label = np.array(self.img_labels.iloc[idx, [1,2,3,4]].values, dtype=float)
            target_labels.append(self.generate_feature_label(label=label))
        return target_labels


    def get_ground_truth(self, label):
        real_cx = label[-4]
        real_cy = label[-3]
        real_w = label[-2]
        real_h = label[-1]

        label_cell_x = int(real_cx / self.scale_w)
        label_cell_y = int(real_cy / self.scale_h)
        label_cx = (real_cx / self.scale_w) - label_cell_x  # offset from the 0,0 of the cell
        label_cy = (real_cy / self.scale_h) - label_cell_y
        #print(f"label_cx: {label_cx}, label_cy:{label_cy}, {(real_cx / self.scale_w)}, {real_cy / self.scale_h}, {int(real_cx / self.scale_w)}, {int(real_cy / self.scale_h)} ")
        label_w = real_w / self.scale_w  # width of the box in cells
        label_h = real_h / self.scale_h  # height of the box in cells
        return torch.Tensor([label_cell_x, label_cell_y, label_cx, label_cy, label_w, label_h])
    
    def get_ground_truth_real_coords(self, label):
        real_cx = label[-4]
        real_cy = label[-3]
        real_w = label[-2]
        real_h = label[-1]

        real_x1 = real_cx - (real_w / 2)
        real_y1 = real_cy - (real_h / 2)
        real_x2 = real_cx + (real_w / 2)
        real_y2 = real_cy + (real_h / 2)
        return torch.Tensor([real_x1, real_y1, real_x2, real_y2])
    
    def get_anchor(self, anchor, pos_x, pos_y):
        anchor_cx = pos_x * self.scale_w
        anchor_cy = pos_y * self.scale_h
        anchor_x1 = anchor_cx - (anchor[0] / 2)
        anchor_y1 = anchor_cy - (anchor[1] / 2)
        anchor_x2 = anchor_cx + (anchor[0] / 2)
        anchor_y2 = anchor_cy + (anchor[1] / 2)
        return torch.Tensor([anchor_x1, anchor_y1, anchor_x2, anchor_y2])

    def generate_feature_label(self, label): 
        gt_object = self.get_ground_truth(label=label)
        gt_real_coords = self.get_ground_truth_real_coords(label=label)
        target = torch.zeros((self.feature_map_w, self.feature_map_h, self.num_anchors_per_cell, 5))    # create an empty ground truth

        #iou_matrix = torch.zeros((self.feature_map_w, self.feature_map_h, self.num_anchors_per_cell))
        #max_iou = 0
        #max_iou_coords = [0, 0, 0]
        #for i in range(self.feature_map_w):
        #    for j in range(self.feature_map_h):
        #        for k in range(self.num_anchors_per_cell):
        #            anchor = self.get_anchor(self.anchor_boxes[k], (i + gt_object[0]), (j + gt_object[1])) # box shifted by the x, y center offset of ground truth 
        #            iou = box_iou(gt_real_coords.unsqueeze(0), anchor.unsqueeze(0))
        #            if iou > max_iou:
        #                max_iou = iou
        #                max_iou_coords = (i, j, k)  # TODO: this needs a rework if we want to add more boxes per image
        #            #iou_matrix[i, j, k] = iou

        #print(f"GROUND TRUTH OBJ: {gt_object}")
        #print(f"real coords: {gt_real_coords}")
        cell_x_idx = int(gt_object[0])
        cell_y_idx = int(gt_object[1])
        
        max_iou = 0
        anchor_idx = 0
        for i in range(len(self.anchor_boxes)):
            anchor = self.get_anchor(self.anchor_boxes[i], (gt_object[0] + gt_object[2]), (gt_object[1] + gt_object[3])) # box shifted by the x, y center offset of ground truth 
            iou = box_iou(gt_real_coords.unsqueeze(0), anchor.unsqueeze(0))
            if iou > max_iou:
                max_iou = iou
                anchor_idx = i

        target[cell_x_idx, cell_y_idx, anchor_idx, 0] = 1.0 
        tx = gt_object[2]
        ty = gt_object[3]
        tw = label[-2] / self.anchor_boxes[anchor_idx][0]
        th = label[-1] / self.anchor_boxes[anchor_idx][1]
        target[cell_x_idx, cell_y_idx, anchor_idx, 1:5] = torch.Tensor([tx, ty, tw, th])

        #print(f"RETURNED TARGET: {target[cell_x_idx, cell_y_idx, anchor_idx, :]}")
        #print(f"gt_vals: {target[max_iou_coords[0], max_iou_coords[1], max_iou_coords[2], 1:5]}")
        #print(f"gt vals: cell_x:{max_iou_coords[0]}, cell_y:{max_iou_coords[1]}, k:{max_iou_coords[2]}, center_x_offset:{tx}, center_y_offset:{ty}, tw:{tw}, th:{th}")
        return target

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        #label = np.array(self.img_labels.iloc[idx, [1,2,3,4]].values, dtype=float)
        label = self.target_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class ImageTransformer:
    def __init__(self):
        pass

    def get_transform_pipe(self, img_w, img_h):
        transform_pipe = transforms.Compose([
            transforms.Resize([img_w,img_h]),
            #transforms.ToPILImage(),
            #ransforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        return transform_pipe
    
    def get_test_transform_pipe(self, img_w, img_h):
        transform_pipe = transforms.Compose([
            transforms.Resize([img_w,img_h]),
            transforms.ToTensor()
        ])
        return transform_pipe