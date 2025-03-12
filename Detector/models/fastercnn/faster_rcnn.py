import torch
import torch.nn as nn
import torchvision

from models.fastercnn.roi_head import ROIHead
from models.fastercnn.rpn import RegionProposalNetwork
from models.fastercnn.utils import *

# Link in youtube video implementation: https://www.youtube.com/watch?v=YA7BqiUTCwM
# other implementation in github here: https://github.com/DaHeller/Object-detection-using-Faster-RCNN/blob/master/src/training.ipynb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FasterRCNN(nn.Module):
    def __init__(self, model_config, num_classes):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]
        self.rpn = RegionProposalNetwork(model_config['backbone_out_channels'],
                                         scales=model_config['scales'],
                                         aspect_ratios=model_config['aspect_ratios'],
                                         model_config=model_config)
        self.roi_head = ROIHead(model_config, num_classes,
                                in_channels=model_config['backbone_out_channels'])
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device

        # Normalize
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]
        #############

        # Resize to 1000x600 such that lowest size dimension is scaled upto 600
        # but larger dimension is not more than 1000
        # So compute scale factor for both and scale is minimum of these two
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size,
                          float(self.max_size) / max_size)
        scale_factor = scale.item()

        # Resize image based on scale computed
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None and bboxes.size(1) > 0:
            # Resize boxes by
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        #else: 
        #    bboxes = torch.empty((1, 0, 4), dtype=torch.float32, device=bboxes.device)  # (1, 0, 4)

        return image, bboxes

    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        # Define the input shape for Faster-RCNN
        if self.training:
            # Normalize and resize boxes
            image, bboxes = self.normalize_resize_image_and_boxes(
                image, target['bboxes'])
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)

        # Call backbone
        feat = self.backbone(image)

        # Call RPN and get proposals
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        if not self.training:
            # Transform boxes to original image dimensions called only during inference
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'],
                                                                     image.shape[-2:],
                                                                     old_shape)
        return rpn_output, frcnn_output
