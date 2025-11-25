"""
多模型管理器 - 支持 YOLO、HuggingFace Transformers 和多模态模型
"""
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import numpy as np
from PIL import Image
import io

try:
    import torch
except ImportError:
    torch = None

# YOLO 系列
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# HuggingFace Transformers
try:
    from transformers import (
        DetrForObjectDetection,
        DetrImageProcessor,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        pipeline,
        BlipProcessor,
        BlipForConditionalGeneration,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    DetrForObjectDetection = None
    DetrImageProcessor = None


class ModelCategory:
    """模型类别定义"""
    YOLO_DETECT = "yolo_detect"
    YOLO_SEGMENT = "yolo_segment"
    YOLO_POSE = "yolo_pose"
    HF_DETR = "hf_detr"
    HF_OWLVIT = "hf_owlvit"
    HF_GROUNDING_DINO = "hf_grounding_dino"
    MULTIMODAL_CAPTION = "multimodal_caption"
    MULTIMODAL_VQA = "multimodal_vqa"


# 模型注册表
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # YOLO 系列 - 检测
    "yolov8n.pt": {
        "category": ModelCategory.YOLO_DETECT,
        "name": "YOLOv8 Nano",
        "description": "超轻量检测模型，适合实时场景，速度最快",
        "speed": "极快",
        "accuracy": "中等",
    },
    "yolov8s.pt": {
        "category": ModelCategory.YOLO_DETECT,
        "name": "YOLOv8 Small",
        "description": "轻量检测模型，平衡速度与精度",
        "speed": "快",
        "accuracy": "较好",
    },
    "yolov8m.pt": {
        "category": ModelCategory.YOLO_DETECT,
        "name": "YOLOv8 Medium",
        "description": "中等规模检测模型，精度更高",
        "speed": "中等",
        "accuracy": "高",
    },
    "yolov8l.pt": {
        "category": ModelCategory.YOLO_DETECT,
        "name": "YOLOv8 Large",
        "description": "大规模检测模型，高精度",
        "speed": "较慢",
        "accuracy": "很高",
    },
    "yolov8x.pt": {
        "category": ModelCategory.YOLO_DETECT,
        "name": "YOLOv8 XLarge",
        "description": "超大规模检测模型，最高精度",
        "speed": "慢",
        "accuracy": "最高",
    },
    # YOLO 系列 - 分割
    "yolov8n-seg.pt": {
        "category": ModelCategory.YOLO_SEGMENT,
        "name": "YOLOv8 Nano Seg",
        "description": "超轻量实例分割模型",
        "speed": "极快",
        "accuracy": "中等",
    },
    "yolov8s-seg.pt": {
        "category": ModelCategory.YOLO_SEGMENT,
        "name": "YOLOv8 Small Seg",
        "description": "轻量实例分割模型",
        "speed": "快",
        "accuracy": "较好",
    },
    "yolov8m-seg.pt": {
        "category": ModelCategory.YOLO_SEGMENT,
        "name": "YOLOv8 Medium Seg",
        "description": "中等规模实例分割模型",
        "speed": "中等",
        "accuracy": "高",
    },
    # YOLO 系列 - 姿态估计
    "yolov8n-pose.pt": {
        "category": ModelCategory.YOLO_POSE,
        "name": "YOLOv8 Nano Pose",
        "description": "超轻量姿态估计模型",
        "speed": "极快",
        "accuracy": "中等",
    },
    "yolov8s-pose.pt": {
        "category": ModelCategory.YOLO_POSE,
        "name": "YOLOv8 Small Pose",
        "description": "轻量姿态估计模型，检测人体关键点",
        "speed": "快",
        "accuracy": "较好",
    },
    "yolov8m-pose.pt": {
        "category": ModelCategory.YOLO_POSE,
        "name": "YOLOv8 Medium Pose",
        "description": "中等规模姿态估计模型",
        "speed": "中等",
        "accuracy": "高",
    },
    # HuggingFace DETR
    "facebook/detr-resnet-50": {
        "category": ModelCategory.HF_DETR,
        "name": "DETR ResNet-50",
        "description": "Facebook 的 DEtection TRansformer，端到端目标检测",
        "speed": "中等",
        "accuracy": "高",
    },
    "facebook/detr-resnet-101": {
        "category": ModelCategory.HF_DETR,
        "name": "DETR ResNet-101",
        "description": "DETR 大规模版本，更高精度",
        "speed": "较慢",
        "accuracy": "很高",
    },
    # OWL-ViT (开放词汇检测)
    "google/owlvit-base-patch32": {
        "category": ModelCategory.HF_OWLVIT,
        "name": "OWL-ViT Base",
        "description": "开放词汇检测，可检测任意文本描述的物体",
        "speed": "中等",
        "accuracy": "高",
    },
    # Grounding DINO
    "IDEA-Research/grounding-dino-tiny": {
        "category": ModelCategory.HF_GROUNDING_DINO,
        "name": "Grounding DINO Tiny",
        "description": "先进的开放集检测模型，支持文本提示",
        "speed": "中等",
        "accuracy": "很高",
    },
    # 多模态 - 图像描述
    "Salesforce/blip-image-captioning-base": {
        "category": ModelCategory.MULTIMODAL_CAPTION,
        "name": "BLIP Caption Base",
        "description": "图像描述生成模型，自动生成图像内容描述",
        "speed": "中等",
        "accuracy": "高",
    },
    "Salesforce/blip-image-captioning-large": {
        "category": ModelCategory.MULTIMODAL_CAPTION,
        "name": "BLIP Caption Large",
        "description": "大规模图像描述模型，更丰富的描述",
        "speed": "较慢",
        "accuracy": "很高",
    },
    # 多模态 - VQA
    "Salesforce/blip-vqa-base": {
        "category": ModelCategory.MULTIMODAL_VQA,
        "name": "BLIP VQA Base",
        "description": "视觉问答模型，可回答关于图像的问题",
        "speed": "中等",
        "accuracy": "高",
    },
}


def get_device() -> str:
    """自动选择最佳设备"""
    device_env = os.getenv("DEVICE", "").strip()
    if device_env:
        return device_env
    if torch is not None:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
    return "cpu"


class ModelManager:
    """统一模型管理器"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self._device = get_device()
    
    @property
    def device(self) -> str:
        return self._device
    
    def _load_yolo(self, model_id: str) -> Any:
        """加载 YOLO 模型"""
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        return YOLO(model_id)
    
    def _load_detr(self, model_id: str) -> Tuple[Any, Any]:
        """加载 DETR 模型"""
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed")
        processor = DetrImageProcessor.from_pretrained(model_id)
        model = DetrForObjectDetection.from_pretrained(model_id)
        if self._device.startswith("cuda") and torch is not None:
            model = model.to("cuda")
        return model, processor
    
    def _load_owlvit(self, model_id: str) -> Tuple[Any, Any]:
        """加载 OWL-ViT 模型"""
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if self._device.startswith("cuda") and torch is not None:
            model = model.to("cuda")
        return model, processor
    
    def _load_blip_caption(self, model_id: str) -> Tuple[Any, Any]:
        """加载 BLIP 图像描述模型"""
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed")
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        if self._device.startswith("cuda") and torch is not None:
            model = model.to("cuda")
        return model, processor
    
    def _load_blip_vqa(self, model_id: str) -> Tuple[Any, Any]:
        """加载 BLIP VQA 模型"""
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed")
        from transformers import BlipForQuestionAnswering
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForQuestionAnswering.from_pretrained(model_id)
        if self._device.startswith("cuda") and torch is not None:
            model = model.to("cuda")
        return model, processor
    
    def load_model(self, model_id: str) -> Any:
        """加载模型（带缓存）"""
        if model_id in self._models:
            return self._models[model_id]
        
        info = MODEL_REGISTRY.get(model_id, {})
        category = info.get("category", "")
        
        # YOLO 系列
        if category in (ModelCategory.YOLO_DETECT, ModelCategory.YOLO_SEGMENT, ModelCategory.YOLO_POSE):
            model = self._load_yolo(model_id)
            self._models[model_id] = model
            return model
        
        # 如果是 .pt 文件但不在注册表中，尝试作为 YOLO 加载
        if model_id.endswith(".pt"):
            model = self._load_yolo(model_id)
            self._models[model_id] = model
            return model
        
        # DETR
        if category == ModelCategory.HF_DETR:
            model, processor = self._load_detr(model_id)
            self._models[model_id] = model
            self._processors[model_id] = processor
            return model
        
        # OWL-ViT
        if category == ModelCategory.HF_OWLVIT:
            model, processor = self._load_owlvit(model_id)
            self._models[model_id] = model
            self._processors[model_id] = processor
            return model
        
        # BLIP Caption
        if category == ModelCategory.MULTIMODAL_CAPTION:
            model, processor = self._load_blip_caption(model_id)
            self._models[model_id] = model
            self._processors[model_id] = processor
            return model
        
        # BLIP VQA
        if category == ModelCategory.MULTIMODAL_VQA:
            model, processor = self._load_blip_vqa(model_id)
            self._models[model_id] = model
            self._processors[model_id] = processor
            return model
        
        # 尝试作为 HuggingFace 模型 ID 加载
        if "/" in model_id:
            try:
                model, processor = self._load_detr(model_id)
                self._models[model_id] = model
                self._processors[model_id] = processor
                return model
            except Exception:
                pass
        
        raise ValueError(f"Unknown model: {model_id}")
    
    def get_processor(self, model_id: str) -> Optional[Any]:
        """获取模型对应的处理器"""
        return self._processors.get(model_id)
    
    def infer_yolo(
        self,
        model: Any,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        half: bool = False,
    ) -> Dict[str, Any]:
        """YOLO 推理"""
        t0 = time.time()
        device = device or self._device
        
        kwargs: Dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "verbose": False,
            "device": device,
        }
        if imgsz:
            kwargs["imgsz"] = int(imgsz)
        if half and device.startswith("cuda"):
            kwargs["half"] = True
        
        results = model(image, **kwargs)
        elapsed = (time.time() - t0) * 1000.0
        
        r = results[0]
        
        # 确定任务类型
        task = "detect"
        model_task = getattr(getattr(model, "model", None), "task", None)
        if model_task in ("detect", "segment", "pose"):
            task = model_task
        elif getattr(r, "masks", None) is not None:
            task = "segment"
        elif getattr(r, "keypoints", None) is not None:
            task = "pose"
        
        # 解析结果
        boxes = r.boxes
        dets: List[Dict[str, Any]] = []
        
        if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            names = r.names
            
            for i in range(xyxy.shape[0]):
                item: Dict[str, Any] = {
                    "bbox": [float(v) for v in xyxy[i].tolist()],
                    "score": float(scores[i]),
                    "label": names.get(classes[i], str(classes[i])) if isinstance(names, dict) else str(classes[i]),
                }
                
                # 分割掩膜
                if task == "segment" and getattr(r, "masks", None) is not None:
                    try:
                        masks_xy = getattr(r.masks, "xy", None)
                        if masks_xy is not None and i < len(masks_xy):
                            polys = masks_xy[i]
                            if polys is not None:
                                item["polygons"] = [[float(v) for v in p.reshape(-1)] for p in [polys]]
                    except Exception:
                        pass
                
                # 关键点
                if task == "pose" and getattr(r, "keypoints", None) is not None:
                    try:
                        kps_xy = getattr(r.keypoints, "xy", None)
                        if kps_xy is not None and i < len(kps_xy):
                            kps = kps_xy[i]
                            if kps is not None:
                                item["keypoints"] = [[float(x), float(y)] for x, y in kps.tolist()]
                    except Exception:
                        pass
                
                dets.append(item)
        
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "detections": dets,
            "inference_time": elapsed,
            "task": task,
        }
    
    def infer_detr(
        self,
        model_id: str,
        image: np.ndarray,
        conf: float = 0.5,
    ) -> Dict[str, Any]:
        """DETR 推理"""
        t0 = time.time()
        
        model = self._models.get(model_id)
        processor = self._processors.get(model_id)
        
        if model is None or processor is None:
            raise RuntimeError(f"Model {model_id} not loaded")
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
        
        # 预处理
        inputs = processor(images=pil_image, return_tensors="pt")
        if self._device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([pil_image.size[::-1]])
        if self._device.startswith("cuda"):
            target_sizes = target_sizes.to("cuda")
        
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]
        
        elapsed = (time.time() - t0) * 1000.0
        
        dets = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            dets.append({
                "bbox": [float(v) for v in box],
                "score": float(score),
                "label": model.config.id2label.get(int(label), str(label)),
            })
        
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "detections": dets,
            "inference_time": elapsed,
            "task": "detect",
        }
    
    def infer_owlvit(
        self,
        model_id: str,
        image: np.ndarray,
        text_queries: List[str],
        conf: float = 0.1,
    ) -> Dict[str, Any]:
        """OWL-ViT 开放词汇检测推理"""
        t0 = time.time()
        
        model = self._models.get(model_id)
        processor = self._processors.get(model_id)
        
        if model is None or processor is None:
            raise RuntimeError(f"Model {model_id} not loaded")
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image[:, :, ::-1])
        
        # 预处理
        inputs = processor(text=text_queries, images=pil_image, return_tensors="pt")
        if self._device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]
        
        elapsed = (time.time() - t0) * 1000.0
        
        dets = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            dets.append({
                "bbox": [float(v) for v in box],
                "score": float(score),
                "label": text_queries[int(label)] if int(label) < len(text_queries) else str(label),
            })
        
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "detections": dets,
            "inference_time": elapsed,
            "task": "detect",
            "text_queries": text_queries,
        }
    
    def infer_caption(
        self,
        model_id: str,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """图像描述生成"""
        t0 = time.time()
        
        model = self._models.get(model_id)
        processor = self._processors.get(model_id)
        
        if model is None or processor is None:
            raise RuntimeError(f"Model {model_id} not loaded")
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image[:, :, ::-1])
        
        # 预处理
        inputs = processor(pil_image, return_tensors="pt")
        if self._device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 生成描述
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        elapsed = (time.time() - t0) * 1000.0
        
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "caption": caption,
            "inference_time": elapsed,
            "task": "caption",
        }
    
    def infer_vqa(
        self,
        model_id: str,
        image: np.ndarray,
        question: str,
    ) -> Dict[str, Any]:
        """视觉问答"""
        t0 = time.time()
        
        model = self._models.get(model_id)
        processor = self._processors.get(model_id)
        
        if model is None or processor is None:
            raise RuntimeError(f"Model {model_id} not loaded")
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image[:, :, ::-1])
        
        # 预处理
        inputs = processor(pil_image, question, return_tensors="pt")
        if self._device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 生成答案
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        
        answer = processor.decode(out[0], skip_special_tokens=True)
        elapsed = (time.time() - t0) * 1000.0
        
        h, w = image.shape[:2]
        return {
            "width": w,
            "height": h,
            "question": question,
            "answer": answer,
            "inference_time": elapsed,
            "task": "vqa",
        }
    
    def infer(
        self,
        model_id: str,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        half: bool = False,
        text_queries: Optional[List[str]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """统一推理接口"""
        # 确保模型已加载
        model = self.load_model(model_id)
        
        info = MODEL_REGISTRY.get(model_id, {})
        category = info.get("category", "")
        
        # YOLO 系列
        if category in (ModelCategory.YOLO_DETECT, ModelCategory.YOLO_SEGMENT, ModelCategory.YOLO_POSE):
            return self.infer_yolo(model, image, conf, iou, max_det, device, imgsz, half)
        
        # 如果是 .pt 文件，作为 YOLO 处理
        if model_id.endswith(".pt"):
            return self.infer_yolo(model, image, conf, iou, max_det, device, imgsz, half)
        
        # DETR
        if category == ModelCategory.HF_DETR:
            return self.infer_detr(model_id, image, conf)
        
        # OWL-ViT
        if category == ModelCategory.HF_OWLVIT:
            queries = text_queries or ["object"]
            return self.infer_owlvit(model_id, image, queries, conf)
        
        # BLIP Caption
        if category == ModelCategory.MULTIMODAL_CAPTION:
            return self.infer_caption(model_id, image)
        
        # BLIP VQA
        if category == ModelCategory.MULTIMODAL_VQA:
            q = question or "What is in this image?"
            return self.infer_vqa(model_id, image, q)
        
        raise ValueError(f"Unknown model category for {model_id}")


# 全局模型管理器实例
model_manager = ModelManager()


def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
    """获取可用模型列表，按类别分组"""
    categories = {
        "yolo_detect": {"name": "YOLO 检测", "models": []},
        "yolo_segment": {"name": "YOLO 分割", "models": []},
        "yolo_pose": {"name": "YOLO 姿态", "models": []},
        "hf_detr": {"name": "DETR 检测", "models": []},
        "hf_owlvit": {"name": "开放词汇检测", "models": []},
        "hf_grounding_dino": {"name": "Grounding DINO", "models": []},
        "multimodal_caption": {"name": "图像描述", "models": []},
        "multimodal_vqa": {"name": "视觉问答", "models": []},
    }
    
    for model_id, info in MODEL_REGISTRY.items():
        cat = info.get("category", "")
        if cat in categories:
            categories[cat]["models"].append({
                "id": model_id,
                "name": info.get("name", model_id),
                "description": info.get("description", ""),
                "speed": info.get("speed", ""),
                "accuracy": info.get("accuracy", ""),
            })
    
    # 只返回有模型的类别
    return {k: v for k, v in categories.items() if v["models"]}


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """获取模型信息"""
    return MODEL_REGISTRY.get(model_id)
