from PIL import Image
import cv2
import numpy as np
from anti_spoofing.check import load_model, check_liveness

# 初始化模型（只需调用一次）
load_model(device_id=0)

# 读取人脸图片
img = Image.open("face3.jpg").convert("RGB")
img_np = np.array(img)  # RGB
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 转为 BGR

# 活体检测
if check_liveness(img_bgr):
    print("活体")
else:
    print("非活体（照片/视频攻击）")
