# 2025-11-23 GPU 与精度相关修改

- 更新 `docker-compose.yml` 中 `MODEL_NAME` 默认值为 `yolov8s.pt`，提升默认识别精度。
- 更新 `README.md`：
  - 说明默认模型为 `YOLOv8s`，并同步本地运行 / Docker 示例。
  - 新增多硬件 GPU 使用说明（NVIDIA / AMD ROCm / Apple M 系列，本机与 Docker 场景）。
