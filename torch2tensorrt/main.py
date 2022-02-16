import os
import sys
import cv2
import copy
import torch
import time
import json
root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from detect_face import show_results
cur_path=os.path.abspath(os.path.dirname(__file__))

from torchserve.handler import preprocess_client, preprocess_server, postprocess_server, postprocess_client


if __name__ == '__main__':
    # ============参数================
    #img_path=cur_path+"/sample.jpg" #测试图片路径
    #img_path = "./data/images/zidane.jpg"
    img_path = "./data/images/test.jpg"
    #onnx_model_path = cur_path+"/yolov5s-face.onnx" #ONNX模型路径
    fp16_mode=True  #True则FP16推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============图像预处理================
    #img,orgimg=img_process(img_path) #[1,3,640,640]
    orgimg = cv2.imread(img_path)
    img_size = 320
    
    # ============TensorRT推理================
    # 初始化TensorRT引擎
    traditional = 0
    if traditional: # traditional
        from torch2tensorrt.yolo_trt_model import YoloTrtModel
        trt_engine_path = "weights/yolov5s-face.trt" #TRT模型路径
        model_trt=YoloTrtModel(trt_engine_path, img_size=img_size) 
    else:
        from torch2trt import TRTModule
        x = torch.ones(1, 3, img_size, img_size).to(device)
        torch2trt_path = "weights/yolov5s-face.torch2trt" #torch2trt模型路径
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(torch2trt_path))
        pred = model_trt(x) # cold start

    # 耗时统计 = tensorrt推理 + torch后处理
    for i in range(10):
        t1 = time.time()
        img, padsize = preprocess_client(orgimg, img_size=img_size)
        img = preprocess_server([img], torch.device("cpu"), fp16=0)
        t2 = time.time()
        if traditional:
            img = img.cpu().numpy()
            pred = model_trt(img)
            pred = [torch.from_numpy(x).to("cpu") for x in pred]
        else:
            with torch.no_grad():
                pred = model_trt(img.to(device))
                torch.cuda.synchronize()
        t3 = time.time()
        pred = postprocess_server(pred)
        result = postprocess_client((pred[0], padsize))
        t4 = time.time()
        print(t2-t1, t3-t2, t4-t3)
   
    # ============可视化================
    if 0:
        result = json.loads(result)
        for face in result:
            xywh_ratio, conf, landmarks_ratio = face["xywh_ratio"], face["conf"], face["landmarks_ratio"]
            show_results(orgimg, xywh_ratio, conf, landmarks_ratio, 1)
        print('det size:', img_size)
        print('orgimg.shape: ', orgimg.shape)
        
        #cv2.imwrite('result.jpg', orgimg)
        cv2.imshow("orgimg", orgimg)
        cv2.waitKey()


