const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const fpsSelect = document.getElementById('fps');
const statsEl = document.getElementById('stats');
const serverInput = document.getElementById('server');
const sendSizeSelect = document.getElementById('sendSize');
const confInput = document.getElementById('conf');
const iouInput = document.getElementById('iou');
const maxDetInput = document.getElementById('maxDet');
const deviceSelect = document.getElementById('device');
const qualitySelect = document.getElementById('quality');
const modelSelect = document.getElementById('model');
const customModelInput = document.getElementById('customModel');
const showBoxesCb = document.getElementById('showBoxes');
const showLabelsCb = document.getElementById('showLabels');
const showMasksCb = document.getElementById('showMasks');
const showKeypointsCb = document.getElementById('showKeypoints');
const imgszInput = document.getElementById('imgsz');
const halfCb = document.getElementById('half');
const maskAlphaInput = document.getElementById('maskAlpha');
const showSkeletonCb = document.getElementById('showSkeleton');
const useWsCb = document.getElementById('useWs');
const permissionStatus = document.getElementById('permissionStatus');

let running = false;
let busy = false;
let detections = [];
let sendInterval = 200;
let baseUrl = window.location.origin;
let sendWidth = parseInt(sendSizeSelect?.value || '640', 10);
let lastInferSize = { width: canvas.width || 1, height: canvas.height || 1 };
let lastTask = 'detect';
let ws = null;
let wsReady = false;

const SETTINGS_KEY = 'vision_settings_v1';
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'); } catch { return {}; }
}
function saveSettings(s) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

const NOTICE_PRIORITY = { info: 1, warning: 2, error: 3 };
function updatePermissionMessage(message, level = 'info', force = false) {
  if (!permissionStatus) return;
  const currentLevel = permissionStatus.dataset.level || '';
  if (!force && currentLevel && NOTICE_PRIORITY[currentLevel] > NOTICE_PRIORITY[level]) {
    return;
  }
  permissionStatus.hidden = false;
  permissionStatus.textContent = message;
  permissionStatus.dataset.level = level;
  permissionStatus.classList.remove('info', 'warning', 'error');
  permissionStatus.classList.add(level);
}
function clearPermissionMessage() {
  if (!permissionStatus) return;
  permissionStatus.hidden = true;
  permissionStatus.textContent = '';
  delete permissionStatus.dataset.level;
  permissionStatus.classList.remove('info', 'warning', 'error');
}

function isTrustedLocalhost() {
  return ['localhost', '127.0.0.1'].includes(window.location.hostname);
}

function preflightCameraCheck() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updatePermissionMessage('当前浏览器不支持摄像头 API（navigator.mediaDevices.getUserMedia），请使用最新版 Chrome/Edge/Firefox。', 'error', true);
    return false;
  }
  if (!window.isSecureContext && !isTrustedLocalhost()) {
    updatePermissionMessage('浏览器要求 HTTPS 或 localhost/127.0.0.1 才能访问摄像头。请改用安全地址，或在浏览器设置中将当前地址标记为安全。', 'warning', true);
  } else if (permissionStatus && (permissionStatus.hidden || permissionStatus.dataset.level === 'info')) {
    updatePermissionMessage('点击“开始”后浏览器会请求摄像头权限，请选择“允许”。', 'info');
  }
  return true;
}

async function monitorCameraPermission() {
  if (!navigator.permissions?.query) {
    return;
  }
  try {
    const status = await navigator.permissions.query({ name: 'camera' });
    applyPermissionState(status.state);
    status.onchange = () => applyPermissionState(status.state);
  } catch (e) {
    // 某些浏览器（如 Safari）不支持 camera 权限查询，忽略错误
  }
}

function applyPermissionState(state) {
  if (state === 'granted') {
    updatePermissionMessage('摄像头权限已授予，可直接点击“开始”进入推理。', 'info');
  } else if (state === 'prompt') {
    updatePermissionMessage('点击“开始”后浏览器会弹出权限提示，请选择“允许”以继续。', 'info');
  } else if (state === 'denied') {
    updatePermissionMessage('摄像头权限被拒绝，请在浏览器地址栏右侧的权限设置中允许访问后刷新页面。', 'error', true);
  }
}

function handleCameraError(err) {
  if (!err) return;
  const errorName = err.name || '';
  if (errorName === 'NotAllowedError' || errorName === 'SecurityError') {
    updatePermissionMessage('摄像头权限被浏览器拒绝，请在浏览器权限设置中允许访问后重新点击“开始”。', 'error', true);
  } else if (errorName === 'NotFoundError' || errorName === 'OverconstrainedError') {
    updatePermissionMessage('未检测到可用的摄像头设备，请确认外设连接正常后重试。', 'error', true);
  } else {
    updatePermissionMessage('摄像头初始化失败：' + (err.message || '未知错误'), 'error', true);
  }
}
function applySettings() {
  const s = loadSettings();
  if (s.server) serverInput.value = s.server;
  if (s.fps) fpsSelect.value = String(s.fps);
  if (s.sendWidth) sendSizeSelect.value = String(s.sendWidth);
  if (s.conf) confInput.value = String(s.conf);
  if (s.iou) iouInput.value = String(s.iou);
  if (s.maxDet) maxDetInput.value = String(s.maxDet);
  if (s.device) deviceSelect.value = String(s.device);
  if (s.quality) qualitySelect.value = String(s.quality);
  if (s.model) modelSelect.value = String(s.model);
  if (s.customModel) customModelInput.value = String(s.customModel);
  if (typeof s.showBoxes === 'boolean') showBoxesCb.checked = s.showBoxes;
  if (typeof s.showLabels === 'boolean') showLabelsCb.checked = s.showLabels;
  if (typeof s.showMasks === 'boolean') showMasksCb.checked = s.showMasks;
  if (typeof s.showKeypoints === 'boolean') showKeypointsCb.checked = s.showKeypoints;
  if (s.imgsz) imgszInput.value = String(s.imgsz);
  if (typeof s.half === 'boolean') halfCb.checked = s.half;
  if (s.maskAlpha) maskAlphaInput.value = String(s.maskAlpha);
  if (typeof s.showSkeleton === 'boolean') showSkeletonCb.checked = s.showSkeleton;
  if (typeof s.useWs === 'boolean') useWsCb.checked = s.useWs;
}
function updateSetting(k, v) {
  const s = loadSettings();
  s[k] = v; saveSettings(s);
}

if (serverInput && !serverInput.value) serverInput.value = baseUrl;
sendSizeSelect?.addEventListener('change', () => {
  sendWidth = parseInt(sendSizeSelect.value, 10);
  updateSetting('sendWidth', sendWidth);
});
serverInput?.addEventListener('change', () => updateSetting('server', serverInput.value));
fpsSelect?.addEventListener('change', () => updateSetting('fps', parseInt(fpsSelect.value, 10)));
confInput?.addEventListener('change', () => updateSetting('conf', parseFloat(confInput.value)));
iouInput?.addEventListener('change', () => updateSetting('iou', parseFloat(iouInput.value)));
maxDetInput?.addEventListener('change', () => updateSetting('maxDet', parseInt(maxDetInput.value, 10)));
deviceSelect?.addEventListener('change', () => updateSetting('device', deviceSelect.value));
qualitySelect?.addEventListener('change', () => updateSetting('quality', parseFloat(qualitySelect.value)));
modelSelect?.addEventListener('change', () => updateSetting('model', modelSelect.value));
customModelInput?.addEventListener('change', () => updateSetting('customModel', customModelInput.value));
showBoxesCb?.addEventListener('change', () => updateSetting('showBoxes', showBoxesCb.checked));
showLabelsCb?.addEventListener('change', () => updateSetting('showLabels', showLabelsCb.checked));
showMasksCb?.addEventListener('change', () => updateSetting('showMasks', showMasksCb.checked));
showKeypointsCb?.addEventListener('change', () => updateSetting('showKeypoints', showKeypointsCb.checked));
imgszInput?.addEventListener('change', () => updateSetting('imgsz', parseInt(imgszInput.value || '')));
halfCb?.addEventListener('change', () => updateSetting('half', !!halfCb.checked));
maskAlphaInput?.addEventListener('input', () => updateSetting('maskAlpha', parseFloat(maskAlphaInput.value)));
showSkeletonCb?.addEventListener('change', () => updateSetting('showSkeleton', showSkeletonCb.checked));
useWsCb?.addEventListener('change', () => updateSetting('useWs', !!useWsCb.checked));

function updateFps() {
  const fps = parseInt(fpsSelect.value, 10);
  sendInterval = Math.max(1000 / fps, 50);
}
fpsSelect.addEventListener('change', updateFps);
updateFps();

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function draw() {
  if (!running) return;
  if (video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawDetections();
  }
  requestAnimationFrame(draw);
}

function drawDetections() {
  ctx.lineWidth = 2;
  ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto';
  const sx = canvas.width / (lastInferSize.width || canvas.width || 1);
  const sy = canvas.height / (lastInferSize.height || canvas.height || 1);
  const cocoPairs = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [0, 1], [0, 2], [1, 3], [2, 4]
  ];
  for (const d of detections) {
    let [x1, y1, x2, y2] = d.bbox;
    x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
    const label = `${d.label} ${(d.score * 100).toFixed(1)}%`;
    const h = Math.abs([...String(d.label)].reduce((a, c) => ((a * 31 + c.charCodeAt(0)) | 0), 0)) % 360;
    const stroke = `hsla(${h},85%,55%,0.95)`;
    const a = Math.max(0.05, Math.min(0.9, parseFloat(maskAlphaInput?.value || '0.2')));
    const fill = `hsla(${h},85%,55%,${a})`;
    if (showMasksCb?.checked && Array.isArray(d.polygons)) {
      ctx.fillStyle = fill;
      for (const poly of d.polygons) {
        if (!Array.isArray(poly) || poly.length < 6) continue;
        ctx.beginPath();
        for (let i = 0; i < poly.length; i += 2) {
          const px = poly[i] * sx;
          const py = poly[i + 1] * sy;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
      }
    }
    if (showBoxesCb?.checked) {
      ctx.strokeStyle = stroke;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    if (showLabelsCb?.checked) {
      const tw = ctx.measureText(label).width;
      const th = 16;
      ctx.fillStyle = `hsla(${h},85%,55%,0.9)`;
      ctx.fillRect(x1, Math.max(0, y1 - th), tw + 8, th);
      ctx.fillStyle = '#0b0e11';
      ctx.fillText(label, x1 + 4, Math.max(12, y1 - 4));
    }
    if (showKeypointsCb?.checked && Array.isArray(d.keypoints)) {
      ctx.fillStyle = `hsla(${h},85%,55%,0.95)`;
      for (const [kx, ky] of d.keypoints) {
        const px = kx * sx, py = ky * sy;
        ctx.beginPath(); ctx.arc(px, py, 2.2, 0, Math.PI * 2); ctx.fill();
      }
      if (showSkeletonCb?.checked) {
        ctx.strokeStyle = `hsla(${h},85%,55%,0.9)`;
        for (const [aIdx, bIdx] of cocoPairs) {
          const pa = d.keypoints[aIdx];
          const pb = d.keypoints[bIdx];
          if (!pa || !pb) continue;
          const ax = pa[0] * sx, ay = pa[1] * sy;
          const bx = pb[0] * sx, by = pb[1] * sy;
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        }
      }
    }
  }
}

async function sendFrame() {
  if (!running || busy) return;
  busy = true;
  const t0 = performance.now();
  const vw = video.videoWidth || canvas.width;
  const vh = video.videoHeight || canvas.height;
  const scale = sendWidth / Math.max(1, vw);
  const sw = Math.max(1, Math.round(vw * scale));
  const sh = Math.max(1, Math.round(vh * scale));
  const sc = document.createElement('canvas');
  sc.width = sw; sc.height = sh;
  const scx = sc.getContext('2d');
  scx.drawImage(video, 0, 0, sw, sh);
  sc.toBlob(async (blob) => {
    const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
    const cv = parseFloat(confInput?.value || '');
    const iv = parseFloat(iouInput?.value || '');
    const mv = parseInt(maxDetInput?.value || '');
    const dv = (deviceSelect?.value || 'auto');
    const model = (customModelInput?.value?.trim()) || (modelSelect?.value || '');
    const include = [showMasksCb?.checked ? 'masks' : '', showKeypointsCb?.checked ? 'keypoints' : ''].filter(Boolean).join(',');
    const imgszVal = parseInt(imgszInput?.value || '');
    const halfVal = !!halfCb?.checked;
    try {
      if (useWsCb?.checked && ws && wsReady) {
        ws.send(blob);
      } else {
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const url = new URL(base + '/infer');
        if (!Number.isNaN(cv)) url.searchParams.set('conf', String(cv));
        if (!Number.isNaN(iv)) url.searchParams.set('iou', String(iv));
        if (!Number.isNaN(mv)) url.searchParams.set('max_det', String(mv));
        if (dv && dv !== 'auto') url.searchParams.set('device', dv);
        if (model) url.searchParams.set('model', model);
        url.searchParams.set('include', include);
        if (!Number.isNaN(imgszVal)) url.searchParams.set('imgsz', String(imgszVal));
        if (halfVal) url.searchParams.set('half', '1');
        const res = await fetch(url.toString(), { method: 'POST', body: fd });
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        handleResult(data, sw, sh, t0, dv, halfVal, imgszVal);
      }
    } catch (e) {
      statsEl.textContent = '请求失败';
    } finally {
      busy = false;
      if (running) {
        setTimeout(sendFrame, sendInterval);
      }
    }
  }, 'image/jpeg', Math.max(0.5, Math.min(0.95, parseFloat(qualitySelect?.value || '0.8'))));
}

function handleResult(data, sw, sh, t0, deviceValue, halfVal, imgszVal) {
  detections = data.detections || [];
  if (data.width && data.height) {
    lastInferSize = { width: data.width, height: data.height };
  }
  if (data.task) lastTask = data.task;
  const t1 = performance.now();
  const dvShow = deviceValue || 'auto';
  const halfShow = halfVal ? 'fp16' : 'fp32';
  const imgszShow = !Number.isNaN(imgszVal) ? imgszVal : '-';
  statsEl.textContent = `图像 ${canvas.width}x${canvas.height} | 上传 ${sw}x${sh} | 设备 ${dvShow} | 任务 ${lastTask} | 精度 ${halfShow} | imgsz ${imgszShow} | 间隔 ${Math.round(sendInterval)}ms | 后端 ${data.inference_time?.toFixed?.(1) || '-'}ms | 往返 ${(t1 - t0).toFixed(1)}ms | 检测 ${detections.length}`;
}

function initWebSocket() {
  if (!useWsCb?.checked) return;
  const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
  const url = new URL((base.replace('http', 'ws')) + '/ws');
  const cv = parseFloat(confInput?.value || '');
  const iv = parseFloat(iouInput?.value || '');
  const mv = parseInt(maxDetInput?.value || '');
  const dv = (deviceSelect?.value || 'auto');
  const model = (customModelInput?.value?.trim()) || (modelSelect?.value || '');
  const include = [showMasksCb?.checked ? 'masks' : '', showKeypointsCb?.checked ? 'keypoints' : ''].filter(Boolean).join(',');
  const imgszVal = parseInt(imgszInput?.value || '');
  const halfVal = !!halfCb?.checked;
  if (!Number.isNaN(cv)) url.searchParams.set('conf', String(cv));
  if (!Number.isNaN(iv)) url.searchParams.set('iou', String(iv));
  if (!Number.isNaN(mv)) url.searchParams.set('max_det', String(mv));
  if (dv && dv !== 'auto') url.searchParams.set('device', dv);
  if (model) url.searchParams.set('model', model);
  url.searchParams.set('include', include);
  if (!Number.isNaN(imgszVal)) url.searchParams.set('imgsz', String(imgszVal));
  if (halfVal) url.searchParams.set('half', '1');
  ws = new WebSocket(url.toString());
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    wsReady = true;
  };
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'result' && payload.data) {
        const sw = video.videoWidth || canvas.width;
        const sh = video.videoHeight || canvas.height;
        const t0 = performance.now();
        handleResult(payload.data, sw, sh, t0, dv, halfVal, imgszVal);
      } else if (payload.type === 'error') {
        statsEl.textContent = payload.detail || '推理错误';
      }
    } catch (e) {
      console.error('ws message parse error', e);
    }
  };
  ws.onclose = () => {
    wsReady = false;
    ws = null;
  };
  ws.onerror = () => {
    wsReady = false;
  };
}

function closeWebSocket() {
  if (ws) {
    try {
      ws.close();
    } catch (e) {
      // ignore
    }
    ws = null;
  }
  wsReady = false;
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  try {
    await initModels();
    applySettings();
    await setupCamera();
    closeWebSocket();
    initWebSocket();
    running = true;
    stopBtn.disabled = false;
    draw();
    sendFrame();
  } catch (e) {
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', () => {
  running = false;
  closeWebSocket();
  const tracks = video.srcObject ? video.srcObject.getTracks() : [];
  tracks.forEach(t => t.stop());
  video.srcObject = null;
  stopBtn.disabled = true;
  startBtn.disabled = false;
});

async function initModels() {
  try {
    const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
    const res = await fetch(base + '/models');
    if (!res.ok) throw new Error('models');
    const data = await res.json();
    const arr = data.models || [];
    modelSelect.innerHTML = '';
    for (const m of arr) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m; modelSelect.appendChild(opt);
    }
    const s = loadSettings();
    if (s.model && arr.includes(s.model)) modelSelect.value = s.model;
  } catch {
    // fallback defaults
    const arr = ['yolov8n.pt','yolov8n-seg.pt','yolov8n-pose.pt'];
    modelSelect.innerHTML = '';
    for (const m of arr) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m; modelSelect.appendChild(opt);
    }
  }
}

// Apply saved settings at load time
applySettings();
