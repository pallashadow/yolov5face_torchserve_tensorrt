# 系统import
from ts.torch_handler.base_handler import BaseHandler
import time
import json
import torch
import torch.nn
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True # distilbert batchsize 完全不变，所以benchmark打开比较快

import numpy as np
import torchvision
import pickle



# handler

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[:, 5:15] ,j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        #if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def decode(x, stride=[8,16,32], nc=1, 
            anchors=([4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]) # yolov5s
            ): 
    device = x[0].device
    no = nc + 5 + 10  # number of outputs per anchor
    nl = 3
    grid = [torch.zeros(1)] * nl  # init grid
    a = torch.tensor(anchors).float().view(nl, -1, 2)
    anchors = a.to(device)
    anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2).to(device) # shape(nl,1,na,1,1,2)
    z = []
    for i in range(nl):
        bs, _, ny, nx, _ = x[i].shape
        #if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        grid = make_grid(nx, ny).to(device)

        y = torch.full_like(x[i], 0)
        class_range = list(range(5)) + list(range(15,15+nc))
        y[..., class_range] = x[i][..., class_range].sigmoid()
        y[..., 5:15] = x[i][..., 5:15]

        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid.to(x[i].device)) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        y[..., 5:7]   = y[..., 5:7] *   anchor_grid[i] + grid * stride[i] # landmark x1 y1
        y[..., 7:9]   = y[..., 7:9] *   anchor_grid[i] + grid * stride[i]# landmark x2 y2
        y[..., 9:11]  = y[..., 9:11] *  anchor_grid[i] + grid * stride[i]# landmark x3 y3
        y[..., 11:13] = y[..., 11:13] * anchor_grid[i] + grid * stride[i]# landmark x4 y4
        y[..., 13:15] = y[..., 13:15] * anchor_grid[i] + grid * stride[i]# landmark x5 y5

        z.append(y.view(bs, -1, no))
    return torch.cat(z, 1)

def preprocess_server(imgList, device, fp16=0):
    img = np.stack(imgList, axis=0)
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    if fp16:
        img = img.half()
    else:
        img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def postprocess_server(x):
    x = decode(x)
    conf_thres = 0.3
    iou_thres = 0.5
    preds = non_max_suppression_face(x, conf_thres, iou_thres)
    preds = [x.cpu().numpy() for x in preds]
    return preds

class Yolov5FaceHandler(BaseHandler):
    """
    This handler takes a list of raw text
    and returns the tags of each text. 
    Ref. https://github.com/pytorch/serve/blob/master/docs/custom_service.md
    """
    def __init__(self, fp16=0):
        super().__init__()
        self.fp16=fp16
        
    def preprocess(self, data):
        """不得不在GPU服务器执行的依赖CPU的前处理，一般包括数据解压和图片预处理，这部分torchserve会自动CPU多进程"""
        imageList = []
        for dict1 in data:
            img1p = dict1.get("data") or dict1.get("body")
            img1 = pickle.loads(img1p)
            imageList.append(img1)
        data = preprocess_server(imageList, self.device, self.fp16)
        return data

    def inference(self, x):
        """gpu inference part, 这部分torchserve会自动调度batch size使qps最大化"""
        with torch.no_grad():
            x = self.model(x)
        return x
        
    def postprocess(self, preds):
        """不得不在GPU服务器执行的依赖CPU的后处理，一般包括nms和数据压缩，这部分torchserve会自动CPU多进程"""
        res = postprocess_server(preds)
        torch.cuda.synchronize()
        res = [pickle.dumps(x) for x in res]
        return res