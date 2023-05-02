import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from torchvision.ops import nms, box_convert
import torchvision.transforms.functional as fn
import torchmetrics
import wandb

from torch.utils.data import DataLoader

from config import FREEZE_FEATURE_EXTRACTOR, WANDB_LOGGING


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors_per_cell: int):
        super().__init__()
        
        out_channels = num_anchors_per_cell * 5
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)
    

class CardDetector(nn.Module):
    def __init__(self, img_dims, anchor_boxes: torch.Tensor, num_anchors_per_cell: int, num_max_boxes: int = 1):
        super(CardDetector, self).__init__()

        self.img_w = img_dims[0]
        self.img_h = img_dims[1]
        self.anchor_boxes = anchor_boxes
        self.num_anchors_per_cell = num_anchors_per_cell
        self.num_max_boxes = num_max_boxes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        if FREEZE_FEATURE_EXTRACTOR:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.detection_head = DetectionHead(in_channels=512, num_anchors_per_cell=self.num_anchors_per_cell)

        dummy_image = torch.randn(1, 3, self.img_w, self.img_h)
        features_shape = self.feature_extractor(dummy_image).shape
        self.features_w = features_shape[-2]
        self.features_h = features_shape[-1]
        self.scale_w = self.img_w / self.features_w
        self.scale_h = self.img_h / self.features_h

    def forward(self, input):
        # Get feature map
        features = self.feature_extractor(input)

        # Get detection vectors for each feature
        detection = self.detection_head(features)
        detection = detection.permute(0,2,3,1).contiguous()
        detection = detection.view(detection.shape[0], detection.shape[1], detection.shape[2], self.num_anchors_per_cell, 5)

        # Apply sigmoid to the first 3 elements of the (p, cx, cy, w, h) tensor
        #detection[:, :, :, :, :3] = torch.sigmoid(detection[:, :, :, :, :3])  

        # Square the last two numbers (scales of width and height)
        #detection[:, :, :, :, 3:] = torch.square(detection[:, :, :, :, 3:])
        
        return detection

    def predict(self, input, ground_truth=None):
        self.eval()

        if (len(input.shape) == 3): # If we get a single image with shape (C x W x H) we need to add a dimension at the beginning so that the forward function can process it (only works on batched input)
            input = input.unsqueeze(0) 

        detection = self.forward(input)

        if ground_truth != None:
            detection = ground_truth

        anchor_box_scales = self.create_anchor_box_scales(detection_shape=detection.shape)   #.to(self.device)
        anchor_box_offsets = self.create_anchor_box_offsets(detection_shape=detection.shape, scale_w=self.scale_w, scale_h=self.scale_h) #.to(self.device)

        detection[:,:,:,:,3:5] = torch.exp(detection[:,:,:,:,3:5])
        detection[:,:,:,:,3:5] = torch.mul(detection[:,:,:,:,3:5], anchor_box_scales[:,:,:,:,3:5]) # multiply the w, h coords of detection with predifined anchor box w, h
        detection[:,:,:,:,1] = torch.mul(detection[:,:,:,:,1], self.scale_w)   # scale the x offset from cell orgin
        detection[:,:,:,:,2] = torch.mul(detection[:,:,:,:,2], self.scale_h)   # scale the y offset from cell origin
        detection[:,:,:,:,1:3] = torch.add(detection[:,:,:,:,1:3], anchor_box_offsets[:,:,:,:,1:3])   # add offset from image origin

        # Apply sigmoid to the objectness scores
        detection[:,:,:,:,0] = torch.sigmoid(detection[:,:,:,:,0])

        wh_offsets = detection[:, :, :, :, 3:5].clone()
        wh_offsets = torch.mul(wh_offsets, 0.5)

        cx_cy = detection[:, :, :, :, 1:3].clone()

        pred_boxes = detection.clone()
        pred_boxes[:, :, :, :, 1:3] = torch.add(cx_cy, -1 * wh_offsets)
        pred_boxes[:, :, :, :, 3:5] = torch.add(cx_cy, wh_offsets)
        
        pred_boxes = pred_boxes.view(-1, pred_boxes.shape[1] * pred_boxes.shape[2] * self.num_anchors_per_cell, 5)


        final_boxes = torch.Tensor(input.shape[0], self.num_max_boxes, 4)
        for idx, image in enumerate(pred_boxes):
            boxes =  image[:, 1:]  # select the coordinate values
            objectness_scores = image[:, :1].squeeze(dim=1) # select the objectness score values, the squeeze to get rid of the extra dimension

            indices_to_keep = nms(boxes=boxes, scores=objectness_scores, iou_threshold=0.5)

            final_boxes[idx, :, :] = boxes[indices_to_keep[:self.num_max_boxes]]

        return final_boxes
        
    def create_anchor_box_offsets(self, detection_shape, scale_w, scale_h):
        addition_tensor = torch.zeros(detection_shape[0], detection_shape[1], detection_shape[2], detection_shape[3], detection_shape[4])
        for i in range(detection_shape[1]):
            for j in range(detection_shape[2]):
                addition_tensor[:, i, j, :, 1] = i * scale_w
                addition_tensor[:, i, j, :, 2] = j * scale_h      
        return addition_tensor

    def create_anchor_box_scales(self, detection_shape):
        tensor = torch.zeros(detection_shape[0], detection_shape[1], detection_shape[2], detection_shape[3], detection_shape[4])

        for k in range(detection_shape[3]): # num of anchors
            tensor[:, :, :, k, 3] = self.anchor_boxes[k][0]
            tensor[:, :, :, k, 4] = self.anchor_boxes[k][1]
        return tensor
    

from tqdm.auto import tqdm  # We use tqdm to display a simple progress bar, allowing us to observe the learning progression.
from torchmetrics.detection import mean_ap

def fit(
  model: nn.Module,
  num_epochs: int,
  optimizer: torch.optim.Optimizer,
  train_dataloader: DataLoader,
  val_dataloader: DataLoader,
  device: str,
  print_rate: int = 100
  ):
    # TODO: figure out accuacy
    #accuracy = torchmetrics.Accuracy(task='multiclass', average="weighted").to(model.device)
    accuracy = None
    model = model.to(device=device)
    box_loss = nn.MSELoss()
    obj_loss = nn.BCEWithLogitsLoss()
    
    # Iterate through epochs with tqdm
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n")
        train_loss = 0
        model.train()  # Set model to train
        
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            pred_boxes = outputs[..., 1:]
            pred_obj = outputs[..., 0]

            true_boxes = y[..., 1:]
            true_obj = y[..., 0]
            
             # localization loss
            box_loss_value = box_loss(pred_boxes, true_boxes)
            
            # objectness loss
            obj_loss_value = obj_loss(pred_obj, true_obj)

            # total loss
            loss = box_loss_value + obj_loss_value
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            if batch % print_rate == 0: 
                print(f"Looked at {batch} Batches\t---\t{batch * len(X)}/{len(train_dataloader.dataset)} Samples")
            elif batch == len(train_dataloader) - 1:
                print(f"Looked at {batch} Batches\t---\t{len(train_dataloader.dataset)}/{len(train_dataloader.dataset)} Samples")
        
        # Divide the train_loss by the number of batches to get the average train_loss
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        # Setup the Val Loss and Accuracy to accumulate over the batches in the val dataset
        val_loss = 0
        val_acc = 0
        ## Set model to evaluation mode and use torch.inference_mode to remove unnecessary training operations 
        model.eval()
        with torch.inference_mode():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                # localization loss
                box_loss_value = box_loss(pred_boxes, true_boxes)
                # objectness loss
                obj_loss_value = obj_loss(pred_obj, true_obj)
                # total loss
                loss = box_loss_value + obj_loss_value
                val_loss += loss.item()

                #TODO: calculate accuracy

        ## Get the average Val Loss and Accuracy
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_acc / len(val_dataloader)

        print(f"Train loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val Accuracy: {avg_val_acc}")
        if WANDB_LOGGING:
            wandb.log({"Train Loss": avg_train_loss,"Val Loss": avg_val_loss, "Val Accuracy": avg_val_acc})