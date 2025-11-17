from functools import lru_cache
from typing import Any, Dict, List, Optional
import os
import time
import numpy as np
from ultralytics import YOLO
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


DEFAULT_MODEL = os.getenv("MODEL_NAME", "yolov8s.pt")
CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DET = int(os.getenv("MAX_DET", "300"))
DEVICE_ENV = (os.getenv("DEVICE", "").strip() or None)


def select_device() -> str:
    if DEVICE_ENV:
        return DEVICE_ENV
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda:0"
        if torch is not None and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=4)
def load_model(name: str) -> YOLO:
    return YOLO(name)


def get_task_from_result(model: YOLO, r: Any) -> str:
    t = getattr(getattr(model, "model", None), "task", None)
    if isinstance(t, str):
        if t in ("detect", "segment", "pose"):
            return t
    if getattr(r, "masks", None) is not None:
        return "segment"
    if getattr(r, "keypoints", None) is not None:
        return "pose"
    return "detect"


def infer(
    image: np.ndarray,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Optional[str] = None,
    max_det: Optional[int] = None,
    model: Optional[str] = None,
    imgsz: Optional[int] = None,
    half: Optional[bool] = None,
    include: Optional[str] = None,
) -> Dict[str, Any]:
    model_name = (model or DEFAULT_MODEL)
    yolo = load_model(model_name)
    device = device or select_device()
    c = float(conf) if conf is not None else CONF
    j = float(iou) if iou is not None else IOU
    m = int(max_det) if max_det is not None else MAX_DET
    t0 = time.time()
    inc_masks = True
    inc_kps = True
    if include is not None:
        parts = {p.strip().lower() for p in str(include).split(',') if p.strip()}
        inc_masks = 'masks' in parts
        inc_kps = 'keypoints' in parts
    use_half = False
    try:
        if half and device and device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            use_half = True
    except Exception:
        use_half = False
    kwargs: Dict[str, Any] = {"conf": c, "iou": j, "max_det": m, "verbose": False, "device": device}
    if imgsz:
        kwargs["imgsz"] = int(imgsz)
    if use_half:
        kwargs["half"] = True
    results = yolo(image, **kwargs)
    elapsed = (time.time() - t0) * 1000.0
    r = results[0]
    task = get_task_from_result(yolo, r)
    boxes = r.boxes
    dets: List[Dict[str, Any]] = []
    if boxes is not None:
        xyxy = boxes.xyxy
        scores = boxes.conf
        classes = boxes.cls
        if xyxy is not None and xyxy.shape[0] > 0:
            xyxy = xyxy.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            classes = classes.detach().cpu().numpy().astype(int)
            names = r.names
            for i in range(xyxy.shape[0]):
                item: Dict[str, Any] = {
                    "bbox": [float(v) for v in xyxy[i].tolist()],
                    "score": float(scores[i]),
                    "label": names.get(classes[i], str(classes[i])) if isinstance(names, dict) else str(classes[i]),
                }
                if inc_masks and task == "segment" and getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None:
                    try:
                        polys = r.masks.xy[i]
                        if polys is not None:
                            # each polygon is Nx2; flatten each to [x,y,...]
                            item["polygons"] = [[float(v) for v in p.reshape(-1)] for p in polys]
                    except Exception:
                        pass
                if inc_kps and task == "pose" and getattr(r, "keypoints", None) is not None and getattr(r.keypoints, "xy", None) is not None:
                    try:
                        kps = r.keypoints.xy[i]  # shape: Kx2
                        if kps is not None:
                            item["keypoints"] = [[float(x), float(y)] for x, y in kps.tolist()]
                    except Exception:
                        pass
                dets.append(item)
    h, w = image.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "detections": dets,
        "inference_time": float(elapsed),
        "task": task,
        "model": model_name,
        "params": {
            "conf": c,
            "iou": j,
            "max_det": m,
            "device": device,
            "imgsz": int(imgsz) if imgsz else None,
            "half": use_half,
            "model": model_name,
        },
    }


def runtime_info() -> Dict[str, Any]:
    return {
        "model": DEFAULT_MODEL,
        "device": select_device(),
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
    }


def recommended_models() -> List[str]:
    return [
        "yolov8s.pt",
        "yolov8s-seg.pt",
        "yolov8s-pose.pt",
    ]
