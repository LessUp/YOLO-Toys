"""
增强版 FastAPI 后端 - 支持多模型系统
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os
import asyncio
import json
from typing import Any, Dict, List, Optional
from app.model_manager import (
    model_manager,
    get_available_models,
    get_model_info,
    MODEL_REGISTRY,
    ModelCategory,
)
from app import __version__ as VERSION
import uvicorn

ALLOWED = os.getenv("ALLOW_ORIGINS", "*").strip()
ALLOW_ORIGINS = ["*"] if ALLOWED == "*" else [o.strip() for o in ALLOWED.split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
DEFAULT_MODEL = os.getenv("MODEL_NAME", "yolov8s.pt")

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

app = FastAPI(
    title="YOLO-Toys Vision API",
    description="多模型实时物体检测、分割、姿态估计和多模态分析 API",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def _get_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


@app.on_event("startup")
async def _warmup_model():
    """启动时预热默认模型"""
    if os.getenv("SKIP_WARMUP"):
        return
    try:
        model_manager.load_model(DEFAULT_MODEL)
        # 预热推理
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_manager.infer(DEFAULT_MODEL, dummy, conf=0.3, iou=0.45)
    except Exception as e:
        print(f"Warmup failed: {e}")


@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "ok",
        "version": VERSION,
        "device": model_manager.device,
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models")
async def list_models():
    """获取所有可用模型，按类别分组"""
    return {
        "default": DEFAULT_MODEL,
        "categories": get_available_models(),
    }


@app.get("/models/{model_id:path}")
async def model_info(model_id: str):
    """获取指定模型的详细信息"""
    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model_id,
        **info,
    }


@app.get("/labels")
async def labels(model: Optional[str] = Query(default=None)):
    """获取模型的标签列表"""
    model_id = model or DEFAULT_MODEL
    try:
        m = model_manager.load_model(model_id)
        # YOLO 模型
        names = getattr(getattr(m, "model", None), "names", None)
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, (list, tuple)):
            ordered = list(names)
        else:
            ordered = []
        return {"model": model_id, "labels": ordered}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(default=None, description="置信度阈值"),
    iou: Optional[float] = Query(default=None, description="IoU 阈值"),
    device: Optional[str] = Query(default=None, description="设备 (cpu/cuda/mps)"),
    max_det: Optional[int] = Query(default=None, description="最大检测数"),
    model: Optional[str] = Query(default=None, description="模型 ID"),
    imgsz: Optional[int] = Query(default=None, description="推理尺寸"),
    half: Optional[bool] = Query(default=None, description="FP16 半精度"),
    text_queries: Optional[str] = Query(default=None, description="文本查询（用于开放词汇检测）"),
    question: Optional[str] = Query(default=None, description="问题（用于 VQA）"),
):
    """
    统一推理端点
    
    支持：
    - YOLO 检测/分割/姿态估计
    - DETR 检测
    - OWL-ViT 开放词汇检测
    - BLIP 图像描述
    - BLIP VQA 视觉问答
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    
    model_id = model or DEFAULT_MODEL
    
    # 解析文本查询
    queries = None
    if text_queries:
        queries = [q.strip() for q in text_queries.split(",") if q.strip()]
    
    async with semaphore:
        try:
            result = model_manager.infer(
                model_id=model_id,
                image=img,
                conf=conf if conf is not None else 0.25,
                iou=iou if iou is not None else 0.45,
                max_det=max_det if max_det is not None else 300,
                device=device,
                imgsz=imgsz,
                half=half if half is not None else False,
                text_queries=queries,
                question=question,
            )
            result["model"] = model_id
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/caption")
async def caption(
    file: UploadFile = File(...),
    model: Optional[str] = Query(default="Salesforce/blip-image-captioning-base"),
):
    """图像描述生成"""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    
    async with semaphore:
        try:
            result = model_manager.infer(
                model_id=model,
                image=img,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/vqa")
async def vqa(
    file: UploadFile = File(...),
    question: str = Query(..., description="要问的问题"),
    model: Optional[str] = Query(default="Salesforce/blip-vqa-base"),
):
    """视觉问答"""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")
    
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    
    async with semaphore:
        try:
            result = model_manager.infer(
                model_id=model,
                image=img,
                question=question,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_infer(websocket: WebSocket):
    """WebSocket 实时推理"""
    params = websocket.query_params
    conf = _get_optional_float(params.get("conf")) or 0.25
    iou = _get_optional_float(params.get("iou")) or 0.45
    max_det = _get_optional_int(params.get("max_det")) or 300
    device = params.get("device") or None
    model_id = params.get("model") or DEFAULT_MODEL
    imgsz = _get_optional_int(params.get("imgsz"))
    half = _get_optional_bool(params.get("half")) or False
    text_queries_str = params.get("text_queries") or ""
    question = params.get("question") or None
    
    # 解析文本查询
    text_queries = None
    if text_queries_str:
        text_queries = [q.strip() for q in text_queries_str.split(",") if q.strip()]
    
    await websocket.accept()
    await websocket.send_text(json.dumps({
        "type": "ready",
        "message": "connected",
        "model": model_id,
        "device": model_manager.device,
    }, ensure_ascii=False))
    
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            data = message.get("bytes")
            if data is None:
                # 处理文本消息（配置更新）
                text = message.get("text")
                if text:
                    try:
                        config = json.loads(text)
                        if config.get("type") == "config":
                            model_id = config.get("model", model_id)
                            conf = config.get("conf", conf)
                            iou = config.get("iou", iou)
                            max_det = config.get("max_det", max_det)
                            device = config.get("device", device)
                            imgsz = config.get("imgsz", imgsz)
                            half = config.get("half", half)
                            if config.get("text_queries"):
                                text_queries = config["text_queries"]
                            if config.get("question"):
                                question = config["question"]
                            await websocket.send_text(json.dumps({
                                "type": "config_updated",
                                "model": model_id,
                            }, ensure_ascii=False))
                    except json.JSONDecodeError:
                        pass
                continue
            
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "detail": "file too large"
                }, ensure_ascii=False))
                continue
            
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "detail": "failed to decode image"
                }, ensure_ascii=False))
                continue
            
            try:
                async with semaphore:
                    result = await loop.run_in_executor(
                        None,
                        lambda: model_manager.infer(
                            model_id=model_id,
                            image=img,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            device=device,
                            imgsz=imgsz,
                            half=half,
                            text_queries=text_queries,
                            question=question,
                        ),
                    )
                result["model"] = model_id
            except Exception as exc:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "detail": str(exc)
                }, ensure_ascii=False))
                continue
            
            await websocket.send_text(json.dumps({
                "type": "result",
                "data": result
            }, ensure_ascii=False))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# 挂载静态文件
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
