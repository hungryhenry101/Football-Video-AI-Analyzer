import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50
import soccerpitch

MEAN_PATH = '../../models/pitch_seg_npy/mean.npy'
STD_PATH = '../../models/pitch_seg_npy/std.npy'
MODEL_PATH = '../../models/soccer_pitch_segmentation.pth'

class SegmentationNetwork:
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height

        self.mean = np.load(MEAN_PATH)
        self.std = np.load(STD_PATH)
        model = nn.DataParallel(deeplabv3_resnet50(weights=None,weights_backbone=None, num_classes=29))

        self.init_weight(model, nn.init.kaiming_normal_,
                         nn.BatchNorm2d, 1e-3, 0.1,
                         mode='fan_in')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("MPS (Apple Silicon) detected")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict["model"])
        model.eval()
        self.model = model.to(self.device)

        print(f"Device: {self.device}")
        print(f"Input resolution: {width}x{height}")


    def init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                    **kwargs):
        for name, m in feature.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def analyse_img(self, img): # img: BGR image
        img = cv2.resize(img, (self.width, self.height))
        img = np.asarray(img, np.float32) / 255. # Normalize
        img = (img - self.mean) / self.std # Standardize
        img = img.transpose((2, 0, 1)) # transpose to Channel, Height, Width
        img = torch.from_numpy(img).to(self.device, dtype=torch.float32).unsqueeze(0) # Add batch to shape

        with torch.no_grad():
            result = self.model(img)
        output = result['out'][0].cpu().numpy() # Classes, Height, Width
        output = np.asarray(np.argmax(output, axis=0), dtype=np.uint8) # 没有很理解 Classes 到底是什么

        return output
