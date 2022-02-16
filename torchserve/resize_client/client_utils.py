import numpy as np
import torch
import cv2

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def preprocess_client(orgimg, img_size=320, stride_max=32):
    assert((img_size%stride_max)==0)
    h0, w0 = orgimg.shape[:2]
    padh, padw = 0, 0
    
    if max(h0, w0)>img_size:
        if h0>w0:
            s = img_size / h0
            h1, w1 = img_size, int(w0*s//2*2)
            padw = (img_size - w1)//2
        else:
            s = img_size / w0
            h1, w1 = int(h0*s//2*2), img_size
            padh = (img_size - h1)//2
        img = cv2.resize(orgimg, (w1, h1), interpolation=cv2.INTER_LINEAR)
        if h0>w0:
            pad = np.ones([h1, padw, 3], np.uint8)*128
            img = np.hstack([pad, img, pad])
        else:
            pad = np.ones([padh, w1, 3], np.uint8)*128
            img = np.vstack([pad, img, pad])
    assert(img.shape[0]==img_size and img.shape[1]==img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    padsize = np.array([padw, padh])
    return img, padsize

    
def postprocess_client(dets, padsize, img_size=320):
    if len(dets)==0:
        return [],[],[],[]
    imgsz1 = np.array([img_size, img_size]) - padsize * 2
    xyxy = dets[:,:4] - np.tile(padsize, 2)[None]
    conf = dets[:,4]
    landmarks = dets[:, 5:15] - np.tile(padsize, 5)[None]
    
    xyxy_ratio = xyxy / np.tile(imgsz1, 2)[None]
    landmarks_ratio = landmarks / np.tile(imgsz1, 5)[None]
    xywh_ratio = xyxy2xywh(xyxy_ratio)
    return xywh_ratio, conf, landmarks_ratio

