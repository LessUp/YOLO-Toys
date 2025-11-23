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
const imageFileInput = document.getElementById('imageFile');
const inferImageBtn = document.getElementById('inferImage');
const detectionsSidebar = document.getElementById('detectionsSidebar');
const classCountsEl = document.getElementById('classCounts');
const summaryTotalEl = document.getElementById('summaryTotal');
const summaryModelEl = document.getElementById('summaryModel');
const summaryDeviceEl = document.getElementById('summaryDevice');
const summaryTimingEl = document.getElementById('summaryTiming');

// UI Toggles
const settingsOverlay = document.getElementById('settingsOverlay');
const openSettingsBtn = document.getElementById('openSettings');
const closeSettingsBtn = document.getElementById('closeSettings');
const toggleSidebarBtn = document.getElementById('toggleSidebar');
const themeToggleBtn = document.getElementById('themeToggle');
const iconMoon = themeToggleBtn.querySelector('.icon-moon');
const iconSun = themeToggleBtn.querySelector('.icon-sun');
const emptyState = document.getElementById('emptyState');

// Theme Logic
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  if (theme === 'light') {
    iconMoon.style.display = 'none';
    iconSun.style.display = 'block';
  } else {
    iconMoon.style.display = 'block';
    iconSun.style.display = 'none';
  }
}

// Init Theme
const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
setTheme(savedTheme);

themeToggleBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  setTheme(current === 'light' ? 'dark' : 'light');
});

openSettingsBtn.addEventListener('click', () => {
  settingsOverlay.classList.add('open');
});

closeSettingsBtn.addEventListener('click', () => {
  settingsOverlay.classList.remove('open');
});

settingsOverlay.addEventListener('click', (e) => {
  if (e.target === settingsOverlay) {
    settingsOverlay.classList.remove('open');
  }
});

toggleSidebarBtn.addEventListener('click', () => {
  if (window.innerWidth <= 768) {
    detectionsSidebar.classList.toggle('open');
  } else {
    detectionsSidebar.classList.toggle('collapsed');
  }
});

// Responsive Canvas
function resizeCanvas() {
  const container = canvas.parentElement;
  if (container) {
    // We don't strictly set width/height here to avoid clearing context
    // but we can use CSS to handle the visual size
    // Logic: The canvas internal resolution is set by the image/video frame
    // CSS object-fit handles the display.
  }
}
window.addEventListener('resize', resizeCanvas);

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

const SETTINGS_KEY = 'vision_settings_v2'; // Bumped version
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
    updatePermissionMessage('当前浏览器不支持摄像头 API，请使用最新版 Chrome/Edge/Firefox。', 'error', true);
    return false;
  }
  if (!window.isSecureContext && !isTrustedLocalhost()) {
    updatePermissionMessage('浏览器要求 HTTPS 或 localhost/127.0.0.1 才能访问摄像头。', 'warning', true);
  } else if (permissionStatus && (permissionStatus.hidden || permissionStatus.dataset.level === 'info')) {
    updatePermissionMessage('点击“开始”以启动摄像头与检测。', 'info');
  }
  return true;
}

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'environment'
      }
    });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    clearPermissionMessage();
    return true;
  } catch (e) {
    console.error(e);
    if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
      updatePermissionMessage('摄像头访问被拒绝。请在浏览器设置中允许访问摄像头。', 'error', true);
    } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
      updatePermissionMessage('未找到摄像头设备。', 'error', true);
    } else {
      updatePermissionMessage(`摄像头错误: ${e.message}`, 'error', true);
    }
    throw e;
  }
}

function applySettings() {
  const s = loadSettings();
  if (s.server) serverInput.value = s.server;
  if (s.useWs !== undefined) useWsCb.checked = s.useWs;
  if (s.fps) fpsSelect.value = s.fps;
  if (s.sendSize) sendSizeSelect.value = s.sendSize;
  if (s.conf) confInput.value = s.conf;
  if (s.iou) iouInput.value = s.iou;
  if (s.maxDet) maxDetInput.value = s.maxDet;
  if (s.device) deviceSelect.value = s.device;
  if (s.quality) qualitySelect.value = s.quality;
  if (s.model) modelSelect.value = s.model;
  if (s.customModel) customModelInput.value = s.customModel;
  if (s.imgsz) imgszInput.value = s.imgsz;
  if (s.half !== undefined) halfCb.checked = s.half;
  if (s.showBoxes !== undefined) showBoxesCb.checked = s.showBoxes;
  if (s.showLabels !== undefined) showLabelsCb.checked = s.showLabels;
  if (s.showMasks !== undefined) showMasksCb.checked = s.showMasks;
  if (s.maskAlpha) maskAlphaInput.value = s.maskAlpha;
  if (s.showKeypoints !== undefined) showKeypointsCb.checked = s.showKeypoints;
  if (s.showSkeleton !== undefined) showSkeletonCb.checked = s.showSkeleton;

  updateVars();
}

function updateVars() {
  sendInterval = 1000 / parseInt(fpsSelect.value);
  sendWidth = parseInt(sendSizeSelect.value);
  const s = {
    server: serverInput.value,
    useWs: useWsCb.checked,
    fps: fpsSelect.value,
    sendSize: sendSizeSelect.value,
    conf: confInput.value,
    iou: iouInput.value,
    maxDet: maxDetInput.value,
    device: deviceSelect.value,
    quality: qualitySelect.value,
    model: modelSelect.value,
    customModel: customModelInput.value,
    imgsz: imgszInput.value,
    half: halfCb.checked,
    showBoxes: showBoxesCb.checked,
    showLabels: showLabelsCb.checked,
    showMasks: showMasksCb.checked,
    maskAlpha: maskAlphaInput.value,
    showKeypoints: showKeypointsCb.checked,
    showSkeleton: showSkeletonCb.checked,
  };
  saveSettings(s);
}

[fpsSelect, sendSizeSelect, serverInput, confInput, iouInput, maxDetInput, deviceSelect, qualitySelect, modelSelect, customModelInput, imgszInput, halfCb, showBoxesCb, showLabelsCb, showMasksCb, maskAlphaInput, showKeypointsCb, showSkeletonCb, useWsCb].forEach(el => {
  if (el) el.addEventListener('change', () => {
    updateVars();
    if (el === useWsCb) {
      closeWebSocket();
      if (running) initWebSocket();
    }
  });
});

function draw() {
  if (!running) return;
  requestAnimationFrame(draw);
  
  // If we are in webcam mode, we draw the video frame first
  if (video.srcObject && !video.paused && !video.ended) {
    // Only resize if changed to avoid flicker/perf issues
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  
  drawDetections();
}

function drawDetections() {
  if (!detections || detections.length === 0) return;

  const scaleX = canvas.width / lastInferSize.width;
  const scaleY = canvas.height / lastInferSize.height;

  const showBoxes = showBoxesCb.checked;
  const showLabels = showLabelsCb.checked;
  const showMasks = showMasksCb.checked;
  const showKpts = showKeypointsCb.checked;
  const showSkel = showSkeletonCb.checked;
  const maskAlpha = parseFloat(maskAlphaInput.value);

  // Draw Masks
  if (showMasks) {
    for (const det of detections) {
      if (det.mask && det.mask.length) {
        // Polygon mask
        ctx.fillStyle = det.color || 'rgba(255,0,0,0.5)';
        // override alpha
        const c = hexToRgb(det.color || '#FF0000');
        ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${maskAlpha})`;
        
        ctx.beginPath();
        det.mask.forEach((pt, i) => {
          const x = pt[0] * scaleX;
          const y = pt[1] * scaleY;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  // Draw Boxes & Labels
  if (showBoxes || showLabels) {
    ctx.lineWidth = 2;
    ctx.font = '14px sans-serif';
    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const sx1 = x1 * scaleX;
      const sy1 = y1 * scaleY;
      const sx2 = x2 * scaleX;
      const sy2 = y2 * scaleY;
      const color = det.color || '#00FF00';

      if (showBoxes) {
        ctx.strokeStyle = color;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
      }

      if (showLabels) {
        const label = `${det.class} ${(det.conf * 100).toFixed(1)}%`;
        const textMetrics = ctx.measureText(label);
        const th = 14; 
        const tw = textMetrics.width;
        
        ctx.fillStyle = color;
        ctx.fillRect(sx1, sy1 - th - 4, tw + 8, th + 4);
        ctx.fillStyle = '#000';
        ctx.fillText(label, sx1 + 4, sy1 - 2);
      }
    }
  }

  // Draw Keypoints & Skeleton
  if (showKpts || showSkel) {
    for (const det of detections) {
      if (det.keypoints) {
        // keypoints
        if (showKpts) {
          det.keypoints.forEach((kp, i) => {
            const [kx, ky, kconf] = kp; // might be just [x,y] or [x,y,conf]
            if (kx === 0 && ky === 0) return;
            const skx = kx * scaleX;
            const sky = ky * scaleY;
            ctx.fillStyle = det.color || '#00FF00';
            ctx.beginPath();
            ctx.arc(skx, sky, 3, 0, 2 * Math.PI);
            ctx.fill();
          });
        }
        // skeleton could be added here if data provides connections
      }
    }
  }
}

function hexToRgb(hex) {
  // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
  const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
  hex = hex.replace(shorthandRegex, (m, r, g, b) => {
    return r + r + g + g + b + b;
  });
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 255, g: 0, b: 0 };
}

function updateDetectionsSidebar(info) {
  if (!info || !detections) return;
  
  // Update Summary
  summaryTotalEl.textContent = detections.length;
  summaryModelEl.textContent = modelSelect.value || 'default';
  summaryDeviceEl.textContent = info.deviceValue || '-';
  summaryTimingEl.textContent = `${info.backendMs?.toFixed(1) || '-'}ms`;

  // Count Classes
  const counts = {};
  detections.forEach(d => {
    counts[d.class] = (counts[d.class] || 0) + 1;
  });
  
  // Render Counts
  classCountsEl.innerHTML = '';
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([cls, count]) => {
    const item = document.createElement('div');
    item.className = 'class-item';
    item.innerHTML = `<span>${cls}</span> <span>${count}</span>`;
    classCountsEl.appendChild(item);
  });
}

async function runImageInference(file) {
  const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
  const cv = parseFloat(confInput?.value || '');
  const iv = parseFloat(iouInput?.value || '');
  const mv = parseInt(maxDetInput?.value || '');
  const dv = (deviceSelect?.value || 'auto');
  const model = (customModelInput?.value?.trim()) || (modelSelect?.value || '');
  const include = [showMasksCb?.checked ? 'masks' : '', showKeypointsCb?.checked ? 'keypoints' : ''].filter(Boolean).join(',');
  const imgszVal = parseInt(imgszInput?.value || '');
  const halfVal = !!halfCb?.checked;

  const fd = new FormData();
  fd.append('file', file, file.name || 'image.jpg');
  const url = new URL(base + '/infer');
  if (!Number.isNaN(cv)) url.searchParams.set('conf', String(cv));
  if (!Number.isNaN(iv)) url.searchParams.set('iou', String(iv));
  if (!Number.isNaN(mv)) url.searchParams.set('max_det', String(mv));
  if (dv && dv !== 'auto') url.searchParams.set('device', dv);
  if (model) url.searchParams.set('model', model);
  url.searchParams.set('include', include);
  if (!Number.isNaN(imgszVal)) url.searchParams.set('imgsz', String(imgszVal));
  if (halfVal) url.searchParams.set('half', '1');

  const t0 = performance.now();
  
  statsEl.textContent = '正在上传并推理...';

  try {
    const res = await fetch(url.toString(), { method: 'POST', body: fd });
    if (!res.ok) {
      statsEl.textContent = '请求失败: ' + res.status;
      return;
    }
    const data = await res.json();
    const imgUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width || canvas.width || 1;
      canvas.height = img.height || canvas.height || 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(imgUrl);
      detections = data.detections || [];
      if (data.width && data.height) {
        lastInferSize = { width: data.width, height: data.height };
      } else {
        lastInferSize = { width: canvas.width, height: canvas.height };
      }
      if (data.task) {
        lastTask = data.task;
      }
      drawDetections();
      emptyState.style.display = 'none';
      
      const t1 = performance.now();
      statsEl.textContent = `完成: ${detections.length} 个目标, 耗时 ${(t1 - t0).toFixed(0)}ms`;
      
      updateDetectionsSidebar({
        data,
        deviceValue: dv,
        backendMs: data.inference_time,
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(imgUrl);
      statsEl.textContent = '图片加载失败';
    };
    img.src = imgUrl;
  } catch (e) {
    statsEl.textContent = '推理请求异常';
    console.error(e);
  } finally {
    if (imageFileInput) imageFileInput.value = '';
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
      // statsEl.textContent = '请求失败'; // Don't spam UI on frame drop
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
  const rtMs = t1 - t0;
  const backendMs = data.inference_time;

  statsEl.textContent = `Inference: ${backendMs?.toFixed(1) || '-'}ms | RTT: ${rtMs.toFixed(0)}ms | ${detections.length} Objects`;
  
  updateDetectionsSidebar({
    data,
    deviceValue,
    backendMs,
  });
}

function initWebSocket() {
  if (!useWsCb?.checked) return;
  const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
  // Replace http/https with ws/wss
  const wsProtocol = base.startsWith('https') ? 'wss' : 'ws';
  const wsBase = base.replace(/^http(s)?/, wsProtocol);
  
  const url = new URL(wsBase + '/ws');
  
  // ... params ...
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
  ws.onopen = () => { wsReady = true; };
  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'result' && payload.data) {
        const sw = video.videoWidth || canvas.width;
        const sh = video.videoHeight || canvas.height;
        const t0 = performance.now(); // Approximate, since we don't track per-frame RTT well in WS
        handleResult(payload.data, sw, sh, t0, dv, halfVal, imgszVal);
      } else if (payload.type === 'error') {
        console.warn('WS Error:', payload.detail);
      }
    } catch (e) {
      console.error('ws message parse error', e);
    }
  };
  ws.onclose = () => { wsReady = false; ws = null; };
  ws.onerror = () => { wsReady = false; };
}

function closeWebSocket() {
  if (ws) {
    try { ws.close(); } catch (e) {}
    ws = null;
  }
  wsReady = false;
}

startBtn.addEventListener('click', async () => {
  if (!preflightCameraCheck()) return;
  startBtn.disabled = true;
  startBtn.innerHTML = '<span class="loader"></span> 启动中...';
  
  try {
    await initModels();
    applySettings();
    await setupCamera();
    closeWebSocket();
    initWebSocket();
    
    running = true;
    stopBtn.disabled = false;
    emptyState.style.display = 'none';
    draw();
    sendFrame();
  } catch (e) {
    console.error(e);
    statsEl.textContent = '启动失败，请检查控制台日志';
  } finally {
    startBtn.disabled = false;
    startBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 开始';
  }
});

stopBtn.addEventListener('click', () => {
  running = false;
  closeWebSocket();
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(t => t.stop());
    video.srcObject = null;
  }
  stopBtn.disabled = true;
  statsEl.textContent = 'Ready';
  emptyState.style.display = 'flex';
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

function showToast(message, type = 'info') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 300);
  }, 3000);
}

if (inferImageBtn) {
  inferImageBtn.addEventListener('click', async () => {
    const file = imageFileInput && imageFileInput.files && imageFileInput.files[0];
    if (!file) {
      showToast('请先选择一张图片', 'warning');
      return;
    }
    if (running) stopBtn.click();
    await runImageInference(file);
  });
}

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
    const arr = ['yolov8n.pt','yolov8n-seg.pt','yolov8n-pose.pt'];
    modelSelect.innerHTML = '';
    for (const m of arr) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m; modelSelect.appendChild(opt);
    }
  }
}

// Apply settings on load
applySettings();
monitorCameraPermission(); // If this was defined before, I kept it or need to define it. 
// Ah, it was in the original code but I might have missed copying the function definition if it was there.
// Checking original code... `monitorCameraPermission` was NOT in the read output of app.js. 
// Wait, let me check the previous read output again.
// It ends with `monitorCameraPermission(); preflightCameraCheck();` but I don't see the function definition in the `read_multiple_files` output I got earlier.
// It might have been truncated or I missed it.
// I will define a simple one or remove it if not needed. `navigator.permissions` API.

function monitorCameraPermission() {
  if (navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: 'camera' }).then(permissionStatus => {
      permissionStatus.onchange = () => {
        preflightCameraCheck();
      };
    }).catch(() => {});
  }
}

monitorCameraPermission();
preflightCameraCheck();
