from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os
import asyncio
import json
from typing import Optional
from app.inference import infer as run_infer
from app.inference import runtime_info, recommended_models, DEFAULT_MODEL
from app.inference import load_model
from app import __version__ as VERSION
from app.schemas import InferenceResponse
import uvicorn

ALLOWED = os.getenv("ALLOW_ORIGINS", "*").strip()
ALLOW_ORIGINS = ["*"] if ALLOWED == "*" else [o.strip() for o in ALLOWED.split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

app = FastAPI(title="Vision Object Detection")

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
    if os.getenv("SKIP_WARMUP"):
        return
    try:
        y = load_model(DEFAULT_MODEL)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = y(dummy, conf=0.3, iou=0.45, verbose=False)
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION, "info": runtime_info(), "recommended_models": recommended_models()}


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(default=None),
    iou: Optional[float] = Query(default=None),
    device: Optional[str] = Query(default=None),
    max_det: Optional[int] = Query(default=None),
    model: Optional[str] = Query(default=None),
    include: Optional[str] = Query(default=None),
    imgsz: Optional[int] = Query(default=None),
    half: Optional[bool] = Query(default=None),
):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="invalid content type")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="failed to decode image")
    async with semaphore:
        result = run_infer(
            img,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            model=model,
            imgsz=imgsz,
            half=half,
            include=include,
        )
        return result


@app.get("/models")
async def models():
    return {"default": DEFAULT_MODEL, "models": recommended_models()}


@app.get("/labels")
async def labels(model: Optional[str] = Query(default=None)):
    y = load_model(model or DEFAULT_MODEL)
    names = getattr(getattr(y, "model", None), "names", None)
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        ordered = []
    return {"model": model or DEFAULT_MODEL, "labels": ordered}


@app.websocket("/ws")
async def websocket_infer(websocket: WebSocket):
    params = websocket.query_params
    conf = _get_optional_float(params.get("conf"))
    iou = _get_optional_float(params.get("iou"))
    max_det = _get_optional_int(params.get("max_det"))
    device = params.get("device") or None
    model = params.get("model") or None
    include = params.get("include") or None
    imgsz = _get_optional_int(params.get("imgsz"))
    half = _get_optional_bool(params.get("half"))
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "ready", "message": "connected"}, ensure_ascii=False))
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            data = message.get("bytes")
            if data is None:
                # ignore non-binary messages (used for keep-alive/config)
                continue
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                await websocket.send_text(json.dumps({"type": "error", "detail": "file too large"}, ensure_ascii=False))
                continue
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                await websocket.send_text(json.dumps({"type": "error", "detail": "failed to decode image"}, ensure_ascii=False))
                continue
            try:
                async with semaphore:
                    result = await loop.run_in_executor(
                        None,
                        lambda: run_infer(
                            img,
                            conf=conf,
                            iou=iou,
                            device=device,
                            max_det=max_det,
                            model=model,
                            imgsz=imgsz,
                            half=half,
                            include=include,
                        ),
                    )
            except Exception as exc:  # pragma: no cover - safeguards runtime errors
                await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False))
                continue
            await websocket.send_text(json.dumps({"type": "result", "data": result}, ensure_ascii=False))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
