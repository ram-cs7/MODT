"""
Inference Engine
Optimized inference utilities for detection, including Non-Maximum Suppression (NMS)
"""
import torch
import numpy as np
import torchvision
import time
from typing import List, Tuple, Optional

class InferenceEngine:
    """Wrapper for batch inference and post-processing"""
    
    @staticmethod
    def non_max_suppression(
        prediction: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300,
    ) -> List[torch.Tensor]:
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        
        Args:
            prediction: [batch, num_anchors, 4 + 1 + num_classes] (cx, cy, w, h, obj_conf, cls_probs...)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold
            classes: Filter by class
            agnostic: Class-agnostic NMS
            multi_label: Allow multiple labels per box
            max_det: Maximum detections per image
            
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        
        # Checks
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model output is (preds, proto)
            prediction = prediction[0]  # Select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported
            prediction = prediction.cpu()
            
        bs = prediction.shape[0]  # batch size
        if prediction.shape[1] == 0:
            return [torch.zeros((0, 6), device=prediction.device)] * bs
            
        # prediction is [bs, anchors, 5 + nc]
        nc = prediction.shape[2] - 5  # number of classes
        
        # Candidate filtering based on object confidence > conf_thres
        # Index 4 is objectness confidence
        xc = prediction[..., 4] > conf_thres  # candidates

        #Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        
        start = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 4:mi] > conf_thres).any(1))] if multi_label else x[xc[xi]]
            
            # Simplified YOLO output handling (assumes xywh + scores)
            # Depending on model architecture, this tensor shape varies.
            # Assuming typical [N, 85]
            
            # Filter by confidence
            # x = x[x[:, 4] > conf_thres]
            
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            
            output[xi] = x[i]
            
            if (time.time() - start) > time_limit:
                print(f"WARNING: NMS time limit {time_limit}s exceeded")
                break
                
        return output

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
