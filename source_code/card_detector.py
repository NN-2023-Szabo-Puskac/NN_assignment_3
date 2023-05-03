import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

from torchvision.ops import nms, box_convert
import torchvision.transforms.functional as fn
import torchmetrics
import wandb

from torch.utils.data import DataLoader

from source_code.config import FREEZE_FEATURE_EXTRACTOR, WANDB_LOGGING

FEATURES_IN_ANCHOR = 5


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors_per_cell: int):
        super().__init__()

        out_channels = num_anchors_per_cell * FEATURES_IN_ANCHOR
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class DetectionHeadV2(nn.Module):
    def __init__(self, in_channels: int, num_anchors_per_cell: int):
        super().__init__()

        out_channels = num_anchors_per_cell * FEATURES_IN_ANCHOR
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.LeakyReLU(negative_slope=0.4)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class CardDetector(nn.Module):
    def __init__(
        self,
        img_dims,
        anchor_boxes: torch.Tensor,
        num_anchors_per_cell: int,
        num_max_boxes: int = 1,
    ):
        super(CardDetector, self).__init__()

        self.img_w = img_dims[0]
        self.img_h = img_dims[1]
        self.anchor_boxes = anchor_boxes
        self.num_anchors_per_cell = num_anchors_per_cell
        self.num_max_boxes = num_max_boxes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-2]
        )
        if FREEZE_FEATURE_EXTRACTOR:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.detection_head = DetectionHeadV2(
            in_channels=512, num_anchors_per_cell=self.num_anchors_per_cell
        )

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
        detection = detection.permute(0, 2, 3, 1).contiguous()
        detection = detection.view(
            detection.shape[0],
            detection.shape[1],
            detection.shape[2],
            self.num_anchors_per_cell,
            FEATURES_IN_ANCHOR,
        )
        # detection[:,:,:,:,0] = torch.sigmoid(detection[:,:,:,:,0])

        return detection

    def predict(
        self, input, keep_box_score_treshhold=0.51, num_max_boxes=5, ground_truth=None
    ):
        self.eval()

        if (
            len(input.shape) == 3
        ):  # If we get a single image with shape (C x W x H) we need to add a dimension at the beginning so that the forward function can process it (only works on batched input)
            input = input.unsqueeze(0)

        if ground_truth != None:
            detection = ground_truth
        else:
            detection = self.forward(input)

        detection = self.scale_prediction_to_input_shape(detection)

        # Apply sigmoid to the objectness scores
        detection[:, :, :, :, 0] = torch.sigmoid(detection[:, :, :, :, 0])

        pred_boxes = self.boxconvert_predicted_boxes(detection)

        final_boxes = torch.Tensor(input.shape[0], num_max_boxes, 4)
        for idx, image in enumerate(pred_boxes):
            boxes = image[:, 1:]  # select the coordinate values
            objectness_scores = image[:, :1].squeeze(
                dim=1
            )  # select the objectness score values, the squeeze to get rid of the extra dimension

            indices_to_keep = nms(
                boxes=boxes, scores=objectness_scores, iou_threshold=0.5
            )
            actual_len = num_max_boxes
            if len(indices_to_keep) < num_max_boxes:
                actual_len = len(indices_to_keep)

            kept_boxes = torch.zeros(num_max_boxes, 4)
            kept_boxes[:actual_len, :] = boxes[indices_to_keep[:actual_len]]
            kept_objectness_scores = torch.zeros(num_max_boxes)
            kept_objectness_scores[:actual_len] = objectness_scores[
                indices_to_keep[:actual_len]
            ]
            print(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            for kept_box_idx in range(len(kept_boxes)):
                if kept_objectness_scores[kept_box_idx] < keep_box_score_treshhold:
                    # print(f"score: {kept_objectness_scores[kept_box_idx]}")
                    kept_boxes[kept_box_idx, :] = torch.Tensor([0.0, 0.0, 0.0, 0.0])

            print(f"kept_objectness_scores: {kept_objectness_scores}")
            print(f"kept_boxes: {kept_boxes}")
            final_boxes[idx, :, :] = kept_boxes

        return final_boxes

    def scale_prediction_to_input_shape(self, detection):
        anchor_box_scales = self.create_anchor_box_scales(
            detection_shape=detection.shape
        ).to(self.device)
        anchor_box_offsets = self.create_anchor_box_offsets(
            detection_shape=detection.shape, scale_w=self.scale_w, scale_h=self.scale_h
        ).to(self.device)

        detection[:, :, :, :, 3:FEATURES_IN_ANCHOR] = torch.exp(
            detection[:, :, :, :, 3:FEATURES_IN_ANCHOR]
        )
        detection[:, :, :, :, 3:FEATURES_IN_ANCHOR] = torch.mul(
            detection[:, :, :, :, 3:FEATURES_IN_ANCHOR],
            anchor_box_scales[:, :, :, :, 3:FEATURES_IN_ANCHOR],
        )  # multiply the w, h coords of detection with predifined anchor box w, h
        detection[:, :, :, :, 1] = torch.mul(
            detection[:, :, :, :, 1], self.scale_w
        )  # scale the x offset from cell origin
        detection[:, :, :, :, 2] = torch.mul(
            detection[:, :, :, :, 2], self.scale_h
        )  # scale the y offset from cell origin
        detection[:, :, :, :, 1:3] = torch.add(
            detection[:, :, :, :, 1:3], anchor_box_offsets[:, :, :, :, 1:3]
        )  # add offset from image origin
        return detection

    def generate_predicted_boxes(self, detection):
        wh_offsets = detection[:, :, :, :, 3:FEATURES_IN_ANCHOR].clone()
        wh_offsets = torch.mul(wh_offsets, 0.5)

        cx_cy = detection[:, :, :, :, 1:3].clone()

        pred_boxes = detection.clone()
        pred_boxes[:, :, :, :, 1:3] = torch.add(cx_cy, -1 * wh_offsets)
        pred_boxes[:, :, :, :, 3:FEATURES_IN_ANCHOR] = torch.add(cx_cy, wh_offsets)

        return pred_boxes.view(
            -1,
            pred_boxes.shape[1] * pred_boxes.shape[2] * self.num_anchors_per_cell,
            FEATURES_IN_ANCHOR,
        )

    def boxconvert_predicted_boxes(self, detection):
        pred_boxes = detection.view(
            -1,
            detection.shape[1] * detection.shape[2] * self.num_anchors_per_cell,
            FEATURES_IN_ANCHOR,
        )
        for image in pred_boxes:
            image[:, 1:] = box_convert(image[:, 1:], in_fmt="cxcywh", out_fmt="xyxy")
        return pred_boxes

    def create_anchor_box_offsets(self, detection_shape, scale_w, scale_h):
        addition_tensor = torch.zeros(
            detection_shape[0],
            detection_shape[1],
            detection_shape[2],
            detection_shape[3],
            detection_shape[4],
        )
        for i in range(detection_shape[1]):
            for j in range(detection_shape[2]):
                addition_tensor[:, i, j, :, 1] = i * scale_w
                addition_tensor[:, i, j, :, 2] = j * scale_h
        return addition_tensor

    def create_anchor_box_scales(self, detection_shape):
        tensor = torch.zeros(
            detection_shape[0],
            detection_shape[1],
            detection_shape[2],
            detection_shape[3],
            detection_shape[4],
        )

        for k in range(detection_shape[3]):  # num of anchors
            tensor[:, :, :, k, 3] = self.anchor_boxes[k][0]
            tensor[:, :, :, k, 4] = self.anchor_boxes[k][1]
        return tensor


from tqdm.auto import (
    tqdm,
)  # We use tqdm to display a simple progress bar, allowing us to observe the learning progression.
from torchmetrics.detection import mean_ap


def fit(
    model: nn.Module,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    print_rate: int = 100,
):
    # TODO: figure out accuacy
    # accuracy = torchmetrics.Accuracy(task='multiclass', average="weighted").to(model.device)
    accuracy = None
    model = model.to(device=device)
    box_loss = nn.MSELoss()
    obj_loss = nn.BCEWithLogitsLoss()

    # Iterate through epochs with tqdm
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch: {epoch}")
        train_total_loss = 0
        train_objectness_loss = 0
        train_localization_loss = 0
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
            train_total_loss += loss.item()
            train_objectness_loss += obj_loss_value.item()
            train_localization_loss += box_loss_value.item()

            loss.backward()
            optimizer.step()
            if batch % print_rate == 0:
                print(
                    f"Looked at {batch} Batches\t---\t{batch * len(X)}/{len(train_dataloader.dataset)} Samples"
                )
            elif batch == len(train_dataloader) - 1:
                print(
                    f"Looked at {batch} Batches\t---\t{len(train_dataloader.dataset)}/{len(train_dataloader.dataset)} Samples"
                )

        # Divide the train_loss by the number of batches to get the average train_loss
        avg_train_total_loss = train_total_loss / len(train_dataloader)
        avg_train_objectness_loss = train_objectness_loss / len(train_dataloader)
        avg_train_localization_loss = train_localization_loss / len(train_dataloader)

        # Validation
        # Setup the Val Loss and Accuracy to accumulate over the batches in the val dataset
        val_total_loss = 0
        val_objectness_loss = 0
        val_localization_loss = 0
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
                val_total_loss += loss.item()
                val_objectness_loss += obj_loss_value.item()
                val_localization_loss += box_loss_value.item()
                # TODO: calculate accuracy

        ## Get the average Val Loss and Accuracy
        avg_val_total_loss = val_total_loss / len(val_dataloader)
        avg_val_objectness_loss = val_objectness_loss / len(val_dataloader)
        avg_val_localization_loss = val_localization_loss / len(val_dataloader)
        avg_val_acc = val_acc / len(val_dataloader)

        print(
            f"Train Total Loss: {avg_train_total_loss}\nTrain Objectness Loss: {avg_train_objectness_loss}\nTrain Localization Loss: {avg_train_localization_loss}\nVal Total Loss: {avg_val_total_loss}\nVal Objectness Loss: {avg_val_objectness_loss}\nVal Localization Loss: {avg_val_localization_loss}\nVal Accuracy: {avg_val_acc}"
        )
        if WANDB_LOGGING:
            wandb.log(
                {
                    "Train Total Loss": avg_train_total_loss,
                    "Train Objectness Loss": avg_train_objectness_loss,
                    "Train Localization Loss": avg_train_localization_loss,
                    "Val Total Loss": avg_val_total_loss,
                    "Val Objectness Loss": avg_val_objectness_loss,
                    "Val Localization Loss": avg_val_localization_loss,
                    "Val Accuracy": avg_val_acc,
                }
            )


if __name__ == "__main__":
    anchor_boxes = torch.Tensor(
        [
            [198.27963804, 206.74086672],
            [129.59395666, 161.90171490],
            [161.65437828, 232.34624509],
        ]
    )
    detector = CardDetector(
        img_dims=(640, 640), anchor_boxes=anchor_boxes, num_anchors_per_cell=3
    )
    dhv2 = DetectionHeadV2(512, num_anchors_per_cell=3)

    dummy_img = torch.randn(1, 3, 640, 640)
    features = detector.feature_extractor(dummy_img)

    detection1 = detector.detection_head(features)
    detection2 = dhv2(features)
    print(detection1.shape)
    print(detection2.shape)
