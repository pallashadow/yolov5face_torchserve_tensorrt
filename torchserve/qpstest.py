# -*- coding: UTF-8 -*-
import sys
sys.path.append("./")
import argparse
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json

from models.experimental import attempt_load
from torchserve.client import TorchServe_Local_Simulator, TorchServeClientBase


def show_results(img, xywh, conf, landmarks):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s-face.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/head.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--fp16', type=int, default=0, help='fp16 inference')
    parser.add_argument('--vis', type=int, default=0, help='visualization')
    parser.add_argument('--mode', type=int, default=1, choices=[1,2,3], help='test mode')
    
    opt = parser.parse_args()
    print(opt)
    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if opt.mode ==1:
        print("testing pytorch local inference")
        #model = attempt_load(opt.weights, map_location=device, fp16=opt.fp16)
        model = torch.load(opt.weights, map_location=device)['model']
        client = TorchServe_Local_Simulator(model, device, fp16=opt.fp16)
    elif opt.mode==2:
        print("testing tensorrt local inference")
        opt.weights = opt.weights.split(".")[0] + ".torch2trt"
        from torch2trt import TRTModule
        torch2trt_path = "weights/yolov5s-face.torch2trt" #torch2trt模型路径
        model = TRTModule()
        model.load_state_dict(torch.load(torch2trt_path))
        client = TorchServe_Local_Simulator(model, device, fp16=0)
    else: # opt.mode==3
        print("testing torchserve inference")
        client = TorchServeClientBase(url="http://127.0.0.1:8080/predictions/", deployment_name='fd1', grpcFlag=1)
    
    b_img = open(opt.image, "rb").read()
    if 1:
        orgimg = cv2.imdecode(np.frombuffer(b_img, np.uint8), cv2.IMREAD_COLOR) # BGR
        assert orgimg is not None, 'Image Not Found ' + opt.image
    result = client.batch_inference([b_img])[0] # omit cold start
    
    if not opt.vis:
        for batchsize in [1,1,1,1,4,16,32,64,128]:
            #b_img = bytes('', encoding = 'utf-8')
            result = client.batch_inference([b_img]*batchsize)[0]
    else:
        for face in result:
            xywh_ratio, conf, landmarks_ratio = face["xywh_ratio"], face["conf"], face["landmarks_ratio"]
            show_results(orgimg, xywh_ratio, conf, landmarks_ratio)
        print('det size:', img_size)
        print('orgimg.shape: ', orgimg.shape)
        
        cv2.imwrite('result.jpg', orgimg)
        cv2.imshow("orgimg", orgimg)
        cv2.waitKey()
    
