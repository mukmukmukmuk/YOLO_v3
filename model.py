"""
python : 3.8.X
pytorch : 2.1.X
torchinfo : 1.8.X
version 동작하는지 추가 확인필요
"""

import torch
import matplotlib.pyplot as plt
from torch import nn
from torchinfo import summary


# Basic Conv Block 정의
class BasicBlock(nn.Module):
    def __int__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            # TODO : (Conv, BatchNorm, LeakyReLU) 스펙 보고 구현
        )

    def forward(self, x):
        return self.conv(x)


# ResidualBlock 정의
class ResidualBlock(nn.Module):
    def __init__(self, channels): # residual block은 input channel 수와 output channel 수가 동일하다.
        super().__init__()

        self.residual = nn.Sequential(
            # TODO : Spec 보고 구현
        )

    def forward(self, x):
        return self.residual(x) + x


# DarkNet53 정의
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO : define darknet53 (위에서 정의한 Conv block과 Res block 활용)

    def forward(self, x):
        # TODO : Darknet53에서 output으로 나오는 세가지 feature map 생산
        feature_map_01 = None
        feature_map_02 = None
        feature_map_03 = None
        return feature_map_01, feature_map_02, feature_map_03


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            # TODO : YOLO Network Architecture에서 Upsampling에 사용
        )

    def forward(self, x):
        return self.upsample(x)


class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # TODO : define route conv & output conv


    def forward(self, x):
        route = self.route_conv(x)
        output = self.output_conv(route)
        return route, output # route의 경우 다음 yolo block으로 전달되고 output의 경우 DetectionLayer로 전달


class DetectionLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # TODO : YOLO Network에서 output 된 결과를 이용하여 prediction

    def forward(self, x):
        output = self.pred(x)
        # TODO : output에 추가적인 처리 필요
        return output



class Yolov3(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        self.num_classes = num_classes

        self.darknet = Darknet53()

        self.yolo_block_01 = YoloBlock(0 + 0, 0)
        self.detectlayer_01 = DetectionLayer(0, num_classes)
        self.upsample_01 = Upsampling(0, 0)

        self.yolo_block_02 = YoloBlock(0 + 0, 0)
        self.detectlayer_02 = DetectionLayer(0, num_classes)
        self.upsample_02 = Upsampling(0, 0)

        self.yolo_block_03 = YoloBlock(0 + 0, 0)
        self.detectlayer_03 = DetectionLayer(0, num_classes)

    def forward(self, x):
        self.feature_map_01, self.feature_map_02, self.feature_map_03 = self.darknet

        x, output_01 = self.yolo_block_01(self.feature_map_03)
        output_01 = self.detectlayer_01(output_01)
        x = self.upsample_01(x)

        x, output_02 = self.yolo_block_02(torch.cat([x, self.feature_map_02], dim=1))
        output_02 = self.detectlayer_02(output_02)
        x = self.upsample_02(x)

        x, output_03 = self.yolo_block_03(torch.cat([x, self.feature_map_01], dim=1))
        output_03 = self.detectlayer_03(output_03)

        return output_01, output_02, output_03


# 모델 확인 코드
x = torch.randn((1, 3, 640, 640)) # RGB format의 640 x 640 랜덤 이미지
model = Yolov3(num_classes = 3)
out = model(x)
print(out[0].shape) # torch.Size([1, 3, 13, 13, 8]) / B, RGB, cell size, cell size, (c, x, y, w, h) + classes_prob
print(out[1].shape) # torch.Size([1, 3, 26, 26, 8])
print(out[2].shape) # torch.Size([1, 3, 52, 52, 8])

# torch summary
summary(model, input_size = (2, 3, 640, 640), device = "cpu")

# Anchors
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

GRID_SIZE = [13, 26, 52]

# Define Util & Loss function
# 참고 자료 : https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
def iou(box1, box2, is_pred = True):

    # TODO
    iou_score = None

    return iou_score



def nms(bboxes, iou_threshold, threshold):
    # TODO
    bboxes_nms = None
    return bboxes_nms


def convert_cells_to_bboxes():
    # TODO
    converted_bboxes = None
    return converted_bboxes.tolist()


def plot_image(image, boxes):

    plt.show()


def save_checkpoint(model, optimizer, filename = "dr_bee_checkpoint.ptr.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)



# Function to load checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




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

