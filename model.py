"""
python :
pytorch :
torchinfo :

"""

import torch
from torch import nn
from torchinfo import summary


# Basic Conv Block 정의
class BasicBlock(nn.Module):
    def __int__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            # TODO (Conv, BatchNorm, LeakyReLU)
        )

    def forward(self, x):
        return self.conv(x)


# ResidualBlock 정의
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.residual = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        return self.residual(x) + x


# DarkNet53 정의
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO : define darknet53 (conv, res, res, res, res, res)

    def forward(self, x):
        # TODO
        feature_map_01 = None
        feature_map_02 = None
        feature_map_03 = None
        return feature_map_01, feature_map_02, feature_map_03


class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # TODO : define route conv & output conv


    def forward(self, x):
        route = self.route_conv(x)
        output = self.output_conv(route)
        return route, output


class DetectionLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.pred = nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, 1)

    def forward(self, x):
        output = self.pred(x)
        # TODO
        return output


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        return self.upsample(x)


class Yolov3(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.num_classes = num_classes

        self.darknet = Darknet53()

        self.yolo_block_01 = YoloBlock(1024, 512)
        self.detectlayer_01 = DetectionLayer(512, num_classes)
        self.upsample_01 = Upsampling(512, 256)

        self.yolo_block_02 = YoloBlock(512 + 256, 256)
        self.detectlayer_02 = DetectionLayer(256, num_classes)
        self.upsample_02 = Upsampling(256, 128)

        self.yolo_block_03 = YoloBlock(256 + 128, 128)
        self.detectlayer_03 = DetectionLayer(128, num_classes)

    def forward(self, x):
        self.feature_map_01, self.feature_map_02, self.feature_map_03 = self.darknet53

        x, output_01 = self.yolo_block_01(self.feature_map_03)
        output_01 = self.detectlayer_01(output_01)
        x = self.upsample_01(x)

        x, output_02 = self.yolo_block_02(torch.cat([x, self.feature_map_02], dim=1))
        output_02 = self.detectlayer_02(output_02)
        x = self.upsample_02(x)

        x, output_03 = self.yolo_block_03(torch.cat([x, self.feature_map_01], dim=1))
        output_03 = self.detectlayer_03(output_03)

        return output_01, output_02, output_03


# Anchors
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

GRID_SIZE = [13, 26, 52]

# Define Util & Loss function
def iou(box1, box2):

    # TODO
    iou_score = None

    return iou_score


def convert_cells_to_bboxes():
    # TODO
    converted_bboxes = None
    return converted_bboxes.tolist()


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()


    def forward(self, pred, target, anchors):


        # TODO
        box_loss = 0
        class_loss = 0
        object_loss = 0
        no_object_loss = 0

        return(
            box_loss
            + object_loss
            + no_object_loss
            + class_loss
        )

