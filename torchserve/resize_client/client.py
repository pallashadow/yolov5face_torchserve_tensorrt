import pickle
import requests
import time
import torch
from concurrent.futures import ThreadPoolExecutor

import grpc
from torchserve.grpc import inference_pb2, inference_pb2_grpc
from torchserve.client_utils import preprocess_client, postprocess_client


class Client_Base(object):
	def batch_inference(self, imgList0):
		N = len(imgList0)
		t1 = time.time()
		with ThreadPoolExecutor(max_workers=100) as executor:
			res = list(executor.map(self.post, imgList0))
		t2 = time.time()
		serverQPS = N/(t2-t1)
		print("batchsize:{}, time:{:.3f}, serverQPS:{}".format(N, t2-t1, serverQPS))
		return res

class TorchServeClientBase(Client_Base):
	"""
	torchserve 接口
	"""
	def __init__(self, img_size, stride_max=32, url="http://127.0.0.1:8080/predictions/", deployment_name='fd1', grpcFlag=1):
		self.url = url + deployment_name
		self.grpcFlag = grpcFlag
		self.model_name = deployment_name
		self.grpc_url = self.url.split("/")[2].replace("8080", "7070")
		self.img_size = img_size
		self.stride_max = stride_max

	def post(self, orgimg, user_params=None):
		img, padsize = preprocess_client(orgimg, self.img_size, self.stride_max)
		b_imgs = pickle.dumps(img)
		if self.grpcFlag:
			channel = grpc.insecure_channel(self.grpc_url)
			stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
			respond = stub.Predictions(inference_pb2.PredictionsRequest(model_name=self.model_name, input={"data":b_imgs}))
			result = pickle.loads(respond.prediction)
		else:
			respond = requests.put(url=self.url, headers={}, data=b_imgs)
			result = pickle.loads(respond.content)
		
		result = postprocess_client(result, padsize, self.img_size)
		return result


class TorchServe_Local_Simulator(object):
    def __init__(self, model, img_size, device, fp16=0):
        from torchserve.handler import Yolov5FaceHandler
        self.handler = Yolov5FaceHandler(fp16)
        self.handler.model = model
        self.handler.device = device
        self.stride_max = 32
        self.img_size = img_size
        
    def batch_inference(self, imgList0):
        N = len(imgList0)
        t1 = time.time()
        packs = [preprocess_client(orgimg, self.img_size, self.stride_max) for orgimg in imgList0] # client端多进程，大图resize缩小
        #imgList1, imgsz1List, imgsz0List = zip(*packs)
        imgList1, padsizeList = zip(*packs)
        t2 = time.time()
        imgT = self.handler.preprocess([{"data":pickle.dumps(img1)} for img1 in imgList1]) # 图片padding到特定尺寸，batching
        t3 = time.time()
        preds = self.handler.inference(imgT) # 神经网络
        torch.cuda.synchronize()
        t4 = time.time()
        preds = self.handler.postprocess(preds) # 解码和NMS
        torch.cuda.synchronize()
        t5 = time.time()
        preds = [pickle.loads(pred) for pred in preds]
        #packList = [postprocess_client(dets, imgsz0List[i], imgsz1List[i]) for i, dets in enumerate(preds)] # 还原到原图坐标
        packList = [postprocess_client(dets, padsizeList[i], self.img_size) for i, dets in enumerate(preds)] # 还原到原图坐标
        t6 = time.time()
        serverQPS = N/(t5-t2)
        print("batchsize:{}, pre:{:.3f}, server:{:.3f}, post:{:.3f}, serverQPS:{}".format(N, t2-t1,t5-t2,t6-t5, serverQPS))
        print("Gpre:{:.3f}, model:{:.3f}, Gpost:{:.3f}".format(t3-t2,t4-t3,t5-t4))
        return packList