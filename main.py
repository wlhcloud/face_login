import cv2
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import sqlite3
from PIL import Image
import io
import os

from anti_spoofing.check import check_liveness
from fastapi.middleware.cors import CORSMiddleware
import logging

device = "cpu"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Login Demo", description="基于人脸识别的登录系统", version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库配置
DB_PATH = "./models/user_encodings.db"
os.makedirs("./models", exist_ok=True)

# 加载活体检测模型
load_anti_spoofing_model = True
if load_anti_spoofing_model:
    from anti_spoofing.check import load_model

    device_id = 0  # 使用第一个GPU，若无GPU则使用CPU
    load_model(device_id=device_id)
    device = "cuda:{}".format(device_id) if device_id >= 0 else "cpu"
    logger.info(f"活体检测模型加载完成，使用设备: {device}")


# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    conn.close()


init_db()


def image_to_encoding(file_bytes):
    """
    将图片转换为人脸编码
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        # 转换为RGB模式（处理PNG透明通道等问题）
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)

        # 检测人脸位置
        face_locations = face_recognition.face_locations(img)

        if len(face_locations) == 0:
            return None, "未检测到人脸"

        if len(face_locations) > 1:
            return None, "检测到多张人脸，请上传只包含一张人脸的图片"

        # 提取人脸编码
        encodings = face_recognition.face_encodings(
            img, known_face_locations=face_locations
        )

        if len(encodings) == 0:
            return None, "无法提取人脸特征"

        return encodings[0], None

    except Exception as e:
        logger.error(f"图片处理错误: {e}")
        return None, f"图片处理错误: {str(e)}"


@app.post("/register")
async def register(username: str = Form(...), file: UploadFile = None):
    """
    用户注册接口
    """
    if file is None:
        raise HTTPException(status_code=400, detail="请上传人脸图片")

    if not username.strip():
        raise HTTPException(status_code=400, detail="用户名不能为空")

    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        file_bytes = await file.read()

        # 检查图片大小（限制为5MB）
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片大小不能超过5MB")

        # 提取人脸编码
        encoding, error = image_to_encoding(file_bytes)
        if error:
            raise HTTPException(status_code=400, detail=error)

        # 活体检测
        img_array = np.array(Image.open(io.BytesIO(file_bytes)))
        img_array = np.array(Image.open(io.BytesIO(file_bytes)))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # 转为 BGR
        if not check_liveness(img_bgr):
            raise HTTPException(
                status_code=400, detail="活体检测失败，请确保是真实人脸"
            )

        # 保存到数据库
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users(username, encoding) VALUES(?,?)",
                (username.strip(), encoding.tobytes()),
            )
            conn.commit()
            logger.info(f"用户 {username} 注册成功")
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="用户名已存在")
        finally:
            conn.close()

        return {"success": True, "username": username, "message": "注册成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册错误: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.post("/login")
async def login(username: str = Form(...), file: UploadFile = None):
    """
    用户登录接口
    """
    if file is None:
        raise HTTPException(status_code=400, detail="请上传人脸图片")

    if not username.strip():
        raise HTTPException(status_code=400, detail="用户名不能为空")

    try:
        file_bytes = await file.read()

        # 检查图片大小
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片大小不能超过5MB")

        # 提取人脸编码
        encoding, error = image_to_encoding(file_bytes)
        if error:
            raise HTTPException(status_code=400, detail=error)

        # 活体检测
        img_array = np.array(Image.open(io.BytesIO(file_bytes)))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # 转为 BGR
        if not check_liveness(img_bgr):
            raise HTTPException(
                status_code=400, detail="活体检测失败，请确保是真实人脸"
            )

        # 从数据库查询用户
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT username, encoding FROM users WHERE username = ?",
            (username.strip(),),
        )
        user = c.fetchone()
        conn.close()

        if not user:
            return {"success": False, "message": "用户不存在"}

        # 比较人脸特征
        known_encoding = np.frombuffer(user[1], dtype=np.float64)
        distance = np.linalg.norm(known_encoding - encoding)

        # 可调整的相似度阈值
        threshold = 0.4

        if distance < threshold:
            logger.info(f"用户 {username} 登录成功，距离: {distance:.4f}")
            return {
                "success": True,
                "username": user[0],
                "distance": float(distance),
                "similarity": float(1 - distance),  # 添加相似度分数
                "message": "登录成功",
            }
        else:
            logger.info(f"用户 {username} 登录失败，距离: {distance:.4f}")
            return {
                "success": False,
                "message": "人脸不匹配",
                "distance": float(distance),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录错误: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@app.get("/users")
async def get_users():
    """
    获取所有用户列表（仅用于调试）
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, created_at FROM users ORDER BY created_at")
    users = c.fetchall()
    conn.close()

    return {
        "success": True,
        "users": [{"username": user[0], "created_at": user[1]} for user in users],
    }


@app.get("/")
async def root():
    """
    根路径
    """
    return {
        "message": "人脸识别登录系统 API",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register - 用户注册",
            "login": "POST /login - 用户登录",
            "users": "GET /users - 获取用户列表",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # 启动 FastAPI
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
