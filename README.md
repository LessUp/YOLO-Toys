# 实时视频流物体识别（FastAPI + YOLOv8 + 原生前端）

## 文档导航
- [详细教学文档（docs/README.md）](docs/README.md)

### 使用 Docker Compose
```
docker compose up --build -d
# 停止
docker compose down --remove-orphans
```

### 使用 Makefile
```
make install       # 安装运行依赖
make dev           # 安装开发依赖并安装 pre-commit 钩子
make lint          # 代码规范检查
make test          # 运行测试
make run           # 本地开发启动 uvicorn --reload
make docker-build  # 构建镜像 vision-det:latest
make docker-run    # 运行镜像（读取 .env）
make compose-up    # docker compose up --build -d
make compose-down  # docker compose down
```


## 功能
- **前端**：
  - 摄像头采集、Canvas 叠加渲染。
  - 模型选择（检测/分割/姿态），可自定义模型名或权重路径。
  - 可配置服务地址、发送帧率、上传宽度、JPEG 质量。
  - 动态调参：置信度、IoU、最大检测数、设备（auto/cpu/mps/cuda）。
  - 显示开关：框、标签、掩膜、关键点。设置自动持久化到浏览器本地。
- **后端**：
  - FastAPI 提供 `/infer`、`/health`、`/models` 接口。
  - Ultralytics YOLOv8：检测（.pt）、分割（-seg.pt）、姿态（-pose.pt）。
  - 启动预热（可跳过）、自动选择设备（CUDA/MPS/CPU），支持按请求覆盖推理参数。

## 目录结构
```
vision/
├─ app/
│  ├─ main.py           # FastAPI 入口，挂载静态前端
│  ├─ inference.py      # YOLOv8 推理逻辑
│  └─ schemas.py        # Pydantic 返回结构
├─ frontend/
│  ├─ index.html
│  ├─ style.css
│  └─ app.js
├─ requirements.txt
└─ README.md
```

## 本地运行
1) 创建虚拟环境并安装依赖（首次会自动下载 `yolov8s.pt` 权重，需联网）：
```
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt
```

2) 启动服务：
```
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3) 打开浏览器访问：
```
http://localhost:8000/
```
点击“开始”按钮授予摄像头权限即可看到识别结果。

## API 参考
- `POST /infer`
  - 查询参数（可选）：
    - `conf` 浮点，置信度阈值，默认 0.3
    - `iou` 浮点，NMS IoU 阈值，默认 0.45
    - `max_det` 整数，最大检测数，默认 300
    - `device` 字符串，可选 `cpu`/`mps`/`cuda:0`；默认自动
    - `model` 字符串，模型名或路径（如 `yolov8s.pt`、`yolov8s-seg.pt`）
    - `include` 字符串，逗号分隔，支持 `masks`、`keypoints`，用于控制返回体中是否包含掩膜/关键点（默认都包含）
    - `imgsz` 整数，推理尺寸（如 640），可加速/影响精度
    - `half` 布尔，是否使用 FP16（仅 CUDA 可用）
  - 表单数据：`file` 单张图像（JPEG/PNG 等）
  - 返回：
    - `width`、`height`：后端解析的输入图像尺寸
    - `task`：`detect` | `segment` | `pose`
    - `detections`：数组
      - `bbox`: [x1,y1,x2,y2]
      - `score`: 置信度
      - `label`: 类别名
      - `polygons`?: 针对分割，列表，每个为 `[x,y,...]`
      - `keypoints`?: 针对姿态，列表，每个为 `[x,y]`
    - `inference_time`：毫秒

- `GET /health`
  - 返回运行时信息（默认模型、设备、阈值等）与推荐模型列表

- `GET /models`
  - 返回 `default` 与 `models`（推荐模型名）

- `GET /labels`
  - 查询参数：`model`（可选），返回该模型的类别标签列表

- `WEBSOCKET /ws`
  - 支持与 `/infer` 相同的参数（通过查询字符串或前端控件传入）
  - 连接后发送 JPEG 二进制帧，服务器会按请求参数推理并返回 `{type:"result", data}` JSON 消息
  - 返回数据结构与 `/infer` 相同；`type:"error"` 时包含 `detail`
  - 前端在“WebSocket”开关开启时自动使用 WS，否则回退 HTTP

## 说明与建议
- 默认使用 `YOLOv8n`（轻量，速度更快）。可在 `app/inference.py` 中将 `"yolov8n.pt"` 替换为 `yolov8s.pt` 等模型。
- 若是 Apple Silicon（M1/M2/M3），安装的 `torch` 会自动使用 CPU/MPS（Metal）后端（若可用）。
- 可通过前端“发送帧率”下拉选择调节与后端交互频率，平衡带宽与延迟。
- 若想跨端访问（手机访问），确保设备与电脑在同一局域网，使用电脑的局域网 IP 访问 `http://<LAN_IP>:8000/`。

## 常见问题
- 首次推理较慢：模型和权重下载+首次加载编译，后续会缓存。
- 浏览器权限：请允许访问摄像头；如被拦截，检查浏览器地址栏的权限设置。
- 依赖安装失败：尝试升级 pip 或使用国内镜像源；也可使用 Conda 创建环境后再安装。

## Docker 运行
1) 构建镜像：
```
docker build -t vision-det .
```

2) 运行容器（CPU 示例）：
```
docker run --rm -it -p 8000:8000 \
  -e MODEL_NAME=yolov8s.pt \
  -e CONF_THRESHOLD=0.3 \
  -e IOU_THRESHOLD=0.45 \
  -e DEVICE=cpu \
  vision-det
```

GPU（NVIDIA）示例：
```
docker run --rm -it -p 8000:8000 \
  --gpus all \
  -e MODEL_NAME=yolov8s.pt \
  -e CONF_THRESHOLD=0.3 \
  -e IOU_THRESHOLD=0.45 \
  -e DEVICE=cuda:0 \
  vision-det
```

3) 访问前端：
```
http://localhost:8000/
```

## 配置
- 后端环境变量（在 shell、.env 或 Docker -e 中设置）：
  - `MODEL_NAME`：默认 `yolov8s.pt`（精度更高，推荐在有 GPU 时使用）
  - `CONF_THRESHOLD`：默认 `0.3`
  - `IOU_THRESHOLD`：默认 `0.45`
  - `MAX_DET`：默认 `300`
  - `DEVICE`：可选 `cuda:0`、`mps`、`cpu`，默认自动选择（CUDA/MPS/CPU）
  - `SKIP_WARMUP`：任意非空值则跳过启动预热
  - `ALLOW_ORIGINS`：CORS 允许的来源，默认 `*`，可用逗号分隔多个域
  - `MAX_UPLOAD_MB`：上传图片大小上限（MB），默认 `10`
  - `MAX_CONCURRENCY`：后端并发推理限制，默认 `4`
  - WebSocket 同样受上述参数限制
- GPU 使用要点：
  - 本机 NVIDIA / AMD（ROCm）：确保安装 GPU 版 `torch`，不设置 `DEVICE` 时会自动选择；也可显式设置为 `cuda:0`。
  - 本机 Apple M1/M2/M3：安装支持 MPS 的 `torch`，不设置 `DEVICE` 时自动选择；也可设置为 `mps`。
  - Docker + NVIDIA：运行容器时添加 `--gpus all`，并通过 `-e DEVICE=cuda:0` 或留空让后端自动选择。
  - Docker + AMD（ROCm）：使用 ROCm 官方 PyTorch 镜像并映射 ROCm 设备，运行时同样通过 `DEVICE=cuda:0` 或自动选择。
  - Docker on Apple Silicon：当前仅支持 CPU 推理，容器内不会使用到 M 系列 GPU。
- 前端参数（页面顶部控件）：
  - 服务地址：默认 `window.location.origin`，跨设备访问可填 `http://<LAN_IP>:8000`
  - 发送帧率：3/5/10 fps
  - 发送宽度：320/480/640 px（上传会按此宽度下采样，降低带宽与延迟）
  - 模型：从下拉选择或填写自定义模型（需后端可访问）
  - 开关：框/标签/掩膜/关键点

## 开发与贡献
- 安装开发依赖与钩子：
```
.venv/bin/pip install -r requirements-dev.txt
pre-commit install
```
- 代码检查与测试：
```
pre-commit run --all-files
pytest -q
```
- 代码风格：Black + Ruff + isort（见 `pyproject.toml`）
- 贡献指南：见 `CONTRIBUTING.md`，行为准则见 `CODE_OF_CONDUCT.md`，安全披露见 `SECURITY.md`
 - PR 模板：`.github/PULL_REQUEST_TEMPLATE.md`
 - Issue 模板：`.github/ISSUE_TEMPLATE/bug_report.md`、`.github/ISSUE_TEMPLATE/feature_request.md`

## Release 准备（开源到 GitHub）
- 初始化仓库并首个提交：
```
git init
git add .
git commit -m "feat: initial release (FastAPI + YOLOv8 detect/seg/pose, web UI, Docker, CI)"
```
- 推送到 GitHub：
```
git branch -M main
git remote add origin git@github.com:<your-org>/<your-repo>.git
git push -u origin main
```
- 启用 GitHub Actions：默认 `CI` 工作流会在 PR/Push 上运行 Lint 和测试

