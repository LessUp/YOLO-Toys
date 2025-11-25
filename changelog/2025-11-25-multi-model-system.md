# 多模型系统增强 - 2025-11-25

## 概述
全面增强项目，支持多种 AI 模型的动态切换，包括 YOLO 系列、HuggingFace Transformers 模型和多模态模型。同时深度优化前端界面的设计和交互体验。

## 新功能

### 1. 多模型支持系统
- **YOLO 检测系列**: YOLOv8n/s/m/l/x 目标检测模型
- **YOLO 分割系列**: YOLOv8n/s/m-seg 实例分割模型  
- **YOLO 姿态系列**: YOLOv8n/s/m-pose 人体姿态估计模型
- **HuggingFace DETR**: facebook/detr-resnet-50/101 端到端目标检测
- **OWL-ViT**: 开放词汇检测，支持任意文本描述检测
- **多模态图像描述**: BLIP 图像自动描述生成
- **视觉问答 (VQA)**: BLIP VQA 回答关于图像的问题

### 2. 新增后端文件
- `app/model_manager.py`: 统一模型管理器
  - 支持多种模型类别和动态切换
  - 模型缓存和自动设备选择
  - 统一推理接口

### 3. API 增强
- `GET /models`: 返回所有可用模型，按类别分组
- `GET /models/{model_id}`: 获取指定模型详细信息
- `POST /infer`: 增加 `text_queries` 和 `question` 参数
- `POST /caption`: 新增图像描述生成端点
- `POST /vqa`: 新增视觉问答端点
- WebSocket 支持运行时配置更新

### 4. 前端 UI 全面重新设计
- 现代化深色/浅色主题设计
- 模型类别标签页快速切换
- 顶部快速模型选择下拉框
- 实时性能叠加层显示
- 图像描述结果展示区域

### 5. 鼠标悬浮提示 (Tooltips)
为所有设置项添加了详细的功能说明：
- **置信度阈值**: 过滤低置信度检测结果
- **IoU 阈值**: 非极大值抑制的重叠阈值
- **最大检测数**: 每帧最多物体数量
- **推理尺寸**: 图像缩放尺寸，影响速度和精度
- **发送帧率**: 每秒发送帧数
- **FP16 半精度**: 加速推理（仅 NVIDIA GPU）
- **WebSocket**: 低延迟实时连接
- 等等...

## 文件变更

### 新增文件
- `app/model_manager.py` - 多模型管理器
- `changelog/2025-11-25-multi-model-system.md` - 本更新日志

### 修改文件
- `app/main.py` - 使用新的模型管理器，增加 API 端点
- `frontend/index.html` - 全新 UI 设计
- `frontend/style.css` - 现代化样式
- `frontend/app.js` - 支持多模型切换和新功能
- `requirements.txt` - 添加 transformers 和 accelerate 依赖

### 备份文件
- `app/main_old.py` - 原 main.py 备份
- `frontend/index_old.html` - 原 HTML 备份
- `frontend/style_old.css` - 原 CSS 备份
- `frontend/app_old.js` - 原 JS 备份

## 依赖更新
```
transformers>=4.35.0  # HuggingFace Transformers
accelerate>=0.24.0    # 推理加速
```

## 使用说明

### 切换模型
1. 在设置面板中选择模型类别标签页
2. 点击模型卡片选择具体模型
3. 或使用顶部快速选择下拉框

### 多模态功能
1. 选择"多模态"类别
2. 图像描述：选择 BLIP Caption 模型，自动生成描述
3. 视觉问答：选择 BLIP VQA 模型，在设置中输入问题

### 开放词汇检测
1. 选择 OWL-ViT 模型
2. 在多模态设置中输入要检测的物体（逗号分隔）
3. 例如: "person, red car, dog"

## 注意事项
- HuggingFace 模型首次使用需要下载，请确保网络连接
- 多模态模型需要更多内存，建议使用 GPU
- 原有的 YOLO 功能完全兼容
