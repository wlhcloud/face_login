import os
import numpy as np
from PIL import Image
import cv2
from anti_spoofing.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.generate_patches import CropImage
from anti_spoofing.utility import parse_model_name

model = None
model_path = "anti_spoofing/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"


def load_model(device_id=0):
    global model
    model = AntiSpoofPredict(device_id)


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    # height, width, channel = image.shape
    # if width / height != 3 / 4:
    #     print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
    #     return False
    # else:
    return True


def check_liveness(image):
    """
    单张图片活体检测
    返回: True/False
    """
    global model
    # 获取人脸 bbox（假设单人脸）
    image_bbox = model.get_bbox(image)
    if image_bbox is None:
        return False

    # 裁剪人脸
    from anti_spoofing.generate_patches import CropImage

    image_cropper = CropImage()
    h_input, w_input, model_type, scale = parse_model_name(model_path.split("/")[-1])
    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    img_cropped = image_cropper.crop(**param)

    # 预测
    pred = model.predict(img_cropped, model_path)
    label = np.argmax(pred[0])  # 0=Fake, 1=Real
    score = pred[0][label]
    print(f"Label: {label}, Score: {score}")
    if label == 1:
        print(f"Image is Real Face. Score: {score:.2f}.")
    else:
        print(f"Image is Fake Face. Score: {score:.2f}.")
    # 输出 True = 活体
    return label == 1
