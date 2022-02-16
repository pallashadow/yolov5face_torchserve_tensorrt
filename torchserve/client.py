import pickle
import requests
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import json

import grpc
from torchserve.grpc import inference_pb2, inference_pb2_grpc


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
	def __init__(self, url="http://127.0.0.1:8080/predictions/", deployment_name='fd1', grpcFlag=1):
		self.url = url + deployment_name
		self.grpcFlag = grpcFlag
		self.model_name = deployment_name
		self.grpc_url = self.url.split("/")[2].replace("8080", "7070")

	def post(self, b_imgs, user_params=None):
		#b_imgs = pickle.dumps(orgimg)
		if self.grpcFlag:
			channel = grpc.insecure_channel(self.grpc_url)
			stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
			respond = stub.Predictions(inference_pb2.PredictionsRequest(model_name=self.model_name, input={"data":b_imgs}))
			result = json.loads(respond.prediction)
		else:
			respond = requests.put(url=self.url, headers={}, data=b_imgs)
			result = json.loads(respond.content)
		return result


class TorchServe_Local_Simulator(object):
	def __init__(self, model, device, fp16=0):
		from torchserve.handler import Yolov5FaceHandler
		self.handler = Yolov5FaceHandler(fp16)
		self.handler.model = model
		self.handler.device = device
		
		
	def batch_inference(self, imgList0):
		N = len(imgList0)
		t1 = time.time()
		imgT, padsizeList = self.handler.preprocess([{"data":img1} for img1 in imgList0])
		t2 = time.time()
		preds = self.handler.inference(imgT) # 神经网络
		torch.cuda.synchronize()
		t3 = time.time()
		preds = self.handler.postprocess(preds, padsizeList) # 解码和NMS
		torch.cuda.synchronize()
		t4 = time.time()
		packList = [json.loads(pred) for pred in preds]
		serverQPS = N/(t4-t1)
		print("batchsize:{}, serverQPS:{}".format(N, serverQPS))
		print("Gpre:{:.3f}, model:{:.3f}, Gpost:{:.3f}".format(t2-t1,t3-t2,t4-t3))
		return packList