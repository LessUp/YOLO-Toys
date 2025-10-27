from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel


class Detection(BaseModel):
    bbox: List[float]
    score: float
    label: str
    polygons: Optional[List[List[float]]] = None
    keypoints: Optional[List[List[float]]] = None


class InferenceResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]
    inference_time: float
    task: Literal["detect", "segment", "pose"]
    model: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
