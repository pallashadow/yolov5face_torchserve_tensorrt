import sys
sys.path.append("./")
import cv2
import torch
import io

from torchserve.qpstest import show_results
from torchserve.client import TorchServe_Local_Simulator, TorchServeClientBase


if __name__ == "__main__":
	file1 = "data/images/zidane.jpg"
	if 0: # 初始化本地模拟API
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = torch.load("weights/yolov5s-face.pt", map_location=device)['model']
		client = TorchServe_Local_Simulator(model, device, fp16=0)
	else: # 初始化torchserve API
		client = TorchServeClientBase(url="http://127.0.0.1:8080/predictions/", deployment_name='fd1', grpcFlag=0)
	
	# 请求服务器
	#xywh_ratio, conf, landmarks_ratio = client.batch_inference([img])[0]

	b_img = open(file1, "rb").read()
	result = client.post(b_img)

	# 显示结果
	img = cv2.imread(file1) # 读取图片
	for face in result:
		xywh_ratio, conf, landmarks_ratio = face["xywh_ratio"], face["conf"], face["landmarks_ratio"]
		show_results(img, xywh_ratio, conf, landmarks_ratio)
	cv2.imshow("orgimg", img)
	cv2.waitKey()
