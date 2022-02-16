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

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    #clip_coords(coords, img0_shape)
    return coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    return coords



def preprocess_client(orgimg, img_size, model_stride_max):
    imgsz0 = orgimg.shape[:2]  # orig hw
    h0, w0 = imgsz0
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(orgimg, (int(w0 * r), int(h0 * r)), interpolation=interp)
    
    img1w = check_img_size(img_size, s=model_stride_max)  # check img_size
    img = letterbox(img0, new_shape=img1w)[0]
    imgsz1 = img.shape[:2]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    return img, imgsz1, imgsz0

    
def postprocess_client(dets, imgsz0, imgsz1):
    if len(dets)==0:
        return [],[],[],[]
    xyxy = dets[:,:4]
    conf = dets[:,4]
    landmarks = dets[:, 5:15]
    xyxy_origin = scale_coords(imgsz1, xyxy, imgsz0).round() # 推算神经网络padding像素，还原到原图坐标，比较复杂
    landmarks_origin = scale_coords_landmarks(imgsz1, landmarks, imgsz0).round()
    
    gn = np.array(imgsz0)[[1, 0, 1, 0]]
    gn_lks = np.array(imgsz0)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
    xyxy_ratio = xyxy_origin / gn
    landmarks_ratio = landmarks_origin / gn_lks
    xywh_ratio = xyxy2xywh(xyxy_ratio)
    return xywh_ratio, conf, landmarks_ratio

