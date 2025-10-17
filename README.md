基于 **FastAPI + face_recognition +MiniFASNet** 的人脸识别登录系统，支持单模型 CPU 版静态活体检测。

**<font style="color:rgba(255, 255, 255, 1);">Silent-Face-Anti-Spoofing</font>**<font style="color:rgba(255, 255, 255, 1);">：活体检测</font>

[GitHub - minivision-ai/Silent-Face-Anti-Spoofing: 静默活体检测（Silent-Face-Anti-Spoofing）](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing?tab=readme-ov-file)

**face_recognition**：人脸识别

---

<h2 id="rHi32">📁 项目结构</h2>
face_recognition/  
│  
├─ main.py # FastAPI 主程序  
├─ anti_spoofing/ # 活体检测模块  
│ ├─ check.py # 单张图片活体检测函数

│ ├─ resources/anti_spoof_models/ # 训练好的模型文件  
│ └─ ...  
├─ models/ # 数据库和模型存放目录  
│ └─ user_encodings.db # SQLite 用户人脸编码  
│ requirements.txt # 项目依赖  
└─ README.md # 项目说明文档

---

<h2 id="XMwy3">⚙️ 环境依赖</h2>
+ Python 3.12（CPU）
+ Conda 虚拟环境建议
+ 主要 Python 库：

```plain
face-recognition==1.3.0
face_recognition_models==0.3.0
fastapi==0.119.0
numpy==2.3.4
python-multipart==0.0.20
uvicorn==0.37.0
```

注意：如果 CPU 版 PyTorch，需要安装 `torch` 对应 CPU 版本。

---

<h2 id="Kg0y4">🛠️ 数据库配置</h2>
+ SQLite 数据库路径：`./models/user_encodings.db`
+ 数据表 `users`：

| 字段名 | 类型  | 说明  |
| --- | --- | --- |
| username | TEXT | 用户名（主键） |
| encoding | BLOB | 人脸编码（face_recognition 输出） |
| created_at | TIMESTAMP | 用户注册时间 |

---

<h2 id="I9sSV">📝 接口说明</h2>
<h3 id="MLehs">1. 注册用户</h3>
```plain
POST /register
```

**参数**：

- `username` (str)
- `file` (上传人脸图片)

**返回**：

```plain
{
  "success": true,
  "username": "张三",
  "message": "注册成功"
}
```

---

<h3 id="pYsUi">2. 用户登录</h3>
```plain
POST /login
```

**参数**：

- `username` (str)
- `file` (上传人脸图片)

**返回**：

```plain
{
  "success": true,
  "username": "张三",
  "distance": 0.32,
  "similarity": 0.68,
  "message": "登录成功"
}
```

如果人脸不匹配或活体检测失败，会返回 `success: false`。

---

<h3 id="p15q8">3. 获取用户列表（调试用）</h3>
```plain
GET /users
```

**返回**：

```plain
{
  "success": true,
  "users": [
    {"username": "张三", "created_at": "2025-10-16 12:00:00"}
  ]
}
```

---

<h2 id="b1n0W">👁️ 活体检测</h2>
使用 **MiniFASNet 单模型 CPU 版**，通过 `anti_spoofing.check.check_liveness` 实现：

```python
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
```

- 返回值：`True` = 活体, `False` = 非活体
- 内部判断使用 `np.argmax(pred) == 1` 来确定活体

---

<h2 id="eanB3">⚡ 使用说明</h2>
1. 创建 Conda 环境：

```plain
conda create -n face_login python=3.12
conda activate face_login
pip install -r requirements.txt
```

1. 下载并放置活体检测模型：
2. 启动 FastAPI 服务：

```plain
python main.py
```

1. 使用 Postman 或前端进行注册和登录测试。

---

<h2 id="cOO42">🔒 防止照片登录</h2>
+ 活体检测使用 MiniFASNetV2 模型
+ 单人脸裁剪 + 模型预测概率
+ CPU 版也可运行
+ 对多模型可采用平均分数，但单模型时直接用 `argmax` 即可

---

<h2 id="CXLo0">⚖️ 商用说明</h2>
+ `face_recognition` 可用于商业用途（MIT License）
+ `Silent_Face_Anti_Spoofing` / MiniFASNetV2 需自行确认授权

---

<h2 id="kC8ky">📝 备注</h2>
+ CPU 测试性能有限，推荐使用小批量或单人脸图片
+ 多人脸或过大图片可能影响精度和速度