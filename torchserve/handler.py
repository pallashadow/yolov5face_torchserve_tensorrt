import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from ts.torch_handler.base_handler import BaseHandler
#from ts.utils.util import load_label_mapping
import time
import json
import torch
import torch.nn
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True
torch.set_num_threads(1)

import numpy as np
import torchvision
import cv2
import logging
logger = logging.getLogger(__name__)






def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def preprocess_client(b_img, img_size=320, stride_max=32):
    #t1 = time.time()
    orgimg = None
    try:
        if isinstance(b_img, (bytes, bytearray)):
            orgimg = cv2.imdecode(np.frombuffer(b_img, np.uint8), cv2.IMREAD_COLOR)
            # 40ms on 2000*1000
        else: # elif isinstance(np.ndarray)
            orgimg = b_img
        h0, w0 = orgimg.shape[:2]
    except Exception as e:
        print(e)
        logger.error("failed to load image")
        orgimg = np.zeros([img_size, img_size, 3], np.uint8)
        h0, w0 = img_size, img_size
    assert((img_size%stride_max)==0)
    
    padh, padw = 0, 0

    #t2 = time.time()
    if max(h0, w0)>img_size:
        if h0>w0:
            s = img_size / h0
            h1, w1 = img_size, int(w0*s//2*2)
        else:
            s = img_size / w0
            h1, w1 = int(h0*s//2*2), img_size
        img = cv2.resize(orgimg, (w1, h1), interpolation=cv2.INTER_LINEAR)
    else:
        h1, w1 = h0, w0
        img = orgimg.copy()
    #t3 = time.time()
    padw = (img_size - w1)//2
    padh = (img_size - h1)//2
    pad = np.ones([h1, padw, 3], np.uint8)*128
    img = np.hstack([pad, img, pad])
    pad = np.ones([padh, img_size, 3], np.uint8)*128
    img = np.vstack([pad, img, pad])

    assert(img.shape[0]==img_size and img.shape[1]==img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    #img = img.transpose(2, 0, 1).copy()  # RGB to 3x416x416
    padsize = np.array([padw, padh])
    #t4 = time.time()
    #print("prec, read:{:.4f}, resize:{:.4f}, pad:{:.4f}".format(t2-t1, t3-t2, t4-t3)) #
    return img, padsize


def preprocess_server(imgList, device, fp16=0):
    if isinstance(imgList[0], np.ndarray):
        img = np.stack(imgList, axis=0)
        imgT = torch.from_numpy(img).to(device)
        if fp16:
            imgT = imgT.half()
        else:
            imgT = imgT.float()
        if imgT.ndimension() == 3:
            imgT = imgT.unsqueeze(0)
        imgT /= 255.0  # 0 - 255 to 0.0 - 1.0
    else:
        imgT = torch.stack(imgList, dim=0).to(device)
    return imgT


def postprocess_client(pack, img_size=320):
    #t1 = time.time()
    dets, padsize = pack
    #dets = nms(dets).numpy()
    N = len(dets)
    if N==0:
        return json.dumps([])
    imgsz1 = np.array([img_size, img_size]) - padsize * 2
    xyxy = dets[:,:4] - np.tile(padsize, 2)[None]
    conf = dets[:,4]
    landmarks = dets[:, 5:15] - np.tile(padsize, 5)[None]

    xyxy_ratio = xyxy / np.tile(imgsz1, 2)[None]
    landmarks_ratio = (landmarks / np.tile(imgsz1, 5)[None]).round(3)
    xywh_ratio = xyxy2xywh(xyxy_ratio).round(3)
    result = [{"xywh_ratio": xywh_ratio[i].tolist(), "conf": conf[i].tolist(), "landmarks_ratio":landmarks_ratio[i].tolist()} for i in range(N)]

    #t2 = time.time()
    result = json.dumps(result) # slow if many faces
    #t3 = time.time()
    #print(t2-t1, t3-t2)
    return result



def postprocess_server(x):
    x = decode(x)
    #x = x.cpu()
    print(x.shape)
    x = non_max_suppression_face(x)
    x = [xx.cpu().numpy() for xx in x]
    print(x[0].shape)
    return x


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

def nms(x, conf_thres=0.3, iou_thres=0.5):
    x[:,4]*=x[:,15]
    i = x[:,4] > conf_thres
    x = x[i] # confidence
    if x.shape[0]==0:
        return x
    boxes = xywh2xyxy(x[:, :4])
    scores = x[:,4]
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    x = torch.cat((boxes, scores[:,None], x[:, 5:15], torch.ones_like(scores[:,None])), 1)
    return x[i]

def non_max_suppression_face(prediction, conf_thres=0.3, iou_thres=0.5):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    #prediction = prediction.cpu()
    output = []
    for x in prediction:  # image index, image inference
        x = nms(x)
        output.append(x)
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

        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        y[..., 5:15] = y[..., 5:15] * anchor_grid[i].tile(5) + (grid * stride[i]).tile(5)
        z.append(y.view(bs, -1, no))
    return torch.cat(z, 1)

class Yolov5FaceHandler(BaseHandler):
    """
    Ref. https://github.com/pytorch/serve/blob/master/docs/custom_service.md
    """
    def __init__(self, fp16=1):
        super().__init__()
        self.fp16=fp16
        import multiprocessing
        import os
        self.pool = multiprocessing.Pool(os.cpu_count())

    def initialize(self, context):
        serialized_file = context.manifest["model"]["serializedFile"]
        if serialized_file.split(".")[-1] == "torch2trt":
            self._load_torchscript_model = self._load_torch2trt_model # overwrite load model function
        super().initialize(context)

    def _load_torch2trt_model(self, torch2trt_path):
        logger.info("Loading torch2trt model")
        from torch2trt import TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(torch2trt_path))
        self.fp16=0
        self.pool = None
        return model_trt

    def preprocess(self, data):
        """不得不在GPU服务器执行的依赖CPU的前处理，一般包括数据解压和图片预处理"""
        taskList = [dict1.get("data") or dict1.get("body") for dict1 in data]
        if self.pool is None:
            packList = [preprocess_client(pack) for pack in taskList]
        else:
            packList = self.pool.map(preprocess_client, taskList)
        imageList, padsizeList = zip(*packList)
        #imageList = [pickle.loads(task) for task in taskList]
        data = preprocess_server(imageList, self.device, self.fp16)
        return data, padsizeList

    def inference(self, x):
        """gpu inference part, 这部分torchserve会自动调度batch size使qps最大化"""
        with torch.no_grad():
            x = self.model(x)
        return x

    def postprocess(self, preds, padsizeList):
        """不得不在GPU服务器执行的依赖CPU的后处理，一般包括nms和数据压缩"""
        t1 = time.time()
        res = postprocess_server(preds)
        torch.cuda.synchronize()
        t2 = time.time()
        taskList = [(dets, padsizeList[i]) for i, dets in enumerate(res)]
        if self.pool is None:
            res = [postprocess_client(task) for task in taskList]
        else:
            res = self.pool.map(postprocess_client, taskList)
        t3 = time.time()
        print("posts:{:.4f}, postc:{:.4f}".format(t2-t1, t3-t2))
        return res

    def handle(self, data, context):
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics
        
        data_preprocess, padsizeList = self.preprocess(data)

        if not self._is_explain():
            output = self.inference(data_preprocess)
            output = self.postprocess(output, padsizeList)
        else:
            output = self.explain_handle(data_preprocess, data)
    
        

        stop_time = time.time()
        metrics.add_time('HandlerTime', round((stop_time - start_time) * 1000, 2), None, 'ms')
        return output
    
    
'''
import io
from PIL import Image
from torchvision import transforms

def preprocess_client1(b_img, img_size=320, stride_max=32): 
    """
    使用PIL和transforms.Compose，但初始化速度太慢，废弃
    """
    orgimg = Image.open(io.BytesIO(b_img))
    w0, h0 = orgimg.size

    if max(h0, w0)>img_size:
        if h0>w0:
            s = img_size / h0
            h1, w1 = img_size, int(w0*s//2*2)
        else:
            s = img_size / w0
            h1, w1 = int(h0*s//2*2), img_size
    else:
        h1, w1 = h0, w0

    image_processing = transforms.Compose([
        transforms.Resize((h1,w1)),
        transforms.ToTensor(),
    ]) # slow to init
    imgT = image_processing(orgimg)

    padw = (img_size - w1)//2
    padh = (img_size - h1)//2
    pad = torch.ones([3, h1, padw])*0.5
    imgT = torch.cat([pad, imgT, pad], dim=2)
    pad = torch.ones([3, padh, img_size])*0.5
    imgT = torch.cat([pad, imgT, pad], dim=1)
    assert(imgT.shape[1]==img_size and imgT.shape[2]==img_size)
    padsize = np.array([padw, padh])
    return imgT, padsize

'''