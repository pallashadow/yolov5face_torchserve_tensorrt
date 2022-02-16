import sys, os
sys.path.append("./")
import torch
import argparse
from models.experimental import attempt_load

def pth2pt(model_pth, ptPath=None, image_size=320, fp16=0):
    """标准pytorch模型转换成torchscript模型"""
    # ignore TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. 
    # https://stackoverflow.com/questions/66746307/torch-jit-trace-tracerwarning-converting-a-tensor-to-a-python-boolean-might-c
    imgT = (torch.zeros([1,3,image_size,image_size], dtype=torch.float32)).to(device)
    if fp16:
        imgT = imgT.half()
        model_pth.half()
    #print(model_pth.state_dict().keys())
    model = torch.jit.trace(model_pth, imgT)
    model.save(ptPath)

def pt2mar(ptPath, modelFile=None, marName="jit_fd1"):
    """所有相关资源打包为torchserve需要的.mar文件"""
    marDir = "./torchserve/model_store/"
    if not os.path.isdir(marDir):
        os.mkdir(marDir)
    handlerFile = "./torchserve/handler.py"
    extraFiles = []

    tma_path = "torch-model-archiver "  
    # 确保当前环境安装了torch-model-archiver 
    command = tma_path \
            + " --model-name " + marName \
            + " --version 1.0 " \
            + " --serialized-file " + ptPath \
            + " --export-path " + marDir \
            + " --handler " + handlerFile 
    if len(extraFiles)>0:
        command += " --extra-files " + ",".join(extraFiles) 
    if modelFile is not None:
        command += "--model-file " + modelFile
    command += " -f " 
    print("Run command:\n", command)
    os.system(command)
    print("Info:Exported [%s] to  [%s]\n"%(marName, marDir))
    #os.remove(ptPath)#删除中间生成的pt模型文件
    
def pth2mar(model_pth, pthPath):
    torch.save(model_pth.state_dict(), pthPath)
    pt2mar(pthPath)
    
def pth2trt(model_pth, torch2trtPath, fp16, image_size=320):
    from torch2trt import torch2trt
    import tensorrt as trt
    print("torch2trt, may take 1 minute...")
    x = torch.ones(1, 3, image_size, image_size).to(device)
    model_pth.float()
    model_trt = torch2trt(model_pth, [x], fp16_mode=fp16, 
                            log_level=trt.Logger.INFO, 
                            max_workspace_size=(1 << 32),)
    #能被torch2trt.TRTModule导入的pytorch模型
    pred = model_trt(x)
    torch.save(model_trt.state_dict(), torch2trtPath) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', type=int, default=1, help='fp16 inference')
    parser.add_argument('--trt', type=int, default=1, help='pack with torch2trt model, otherwise with torchscript model')
    parser.add_argument('--trt_rebuild', type=int, default=1, help='rebuild torch2trt model')
    args = parser.parse_args()
    
    pthPath0 = "weights/yolov5s-face.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pth = attempt_load(pthPath0, map_location=device, fp16=args.fp16)
    ptPath = "weights/yolov5s-jit.pt"
    if args.trt: # torch2trt
        torch2trtPath = "./weights/yolov5s-face.torch2trt"
        if args.trt_rebuild:
            pth2trt(model_pth, torch2trtPath, args.fp16)
        pt2mar(torch2trtPath, marName="trt_fd1")
    else: # jit
        pth2pt(model_pth, ptPath, fp16=args.fp16)
        pt2mar(ptPath, marName="jit_fd1")
    