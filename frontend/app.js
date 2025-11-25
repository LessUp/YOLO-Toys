/**
 * YOLO-Toys 增强版前端 - 多模型实时视觉识别
 */
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const statsEl = document.getElementById('stats');
const emptyState = document.getElementById('emptyState');
const overlayInfo = document.getElementById('overlayInfo');
const captionResult = document.getElementById('captionResult');
const captionText = document.getElementById('captionText');
const serverInput = document.getElementById('server');
const useWsCb = document.getElementById('useWs');
const quickModelSelect = document.getElementById('quickModelSelect');
const modelTabs = document.getElementById('modelTabs');
const modelList = document.getElementById('modelList');
const customModelInput = document.getElementById('customModel');
const deviceSelect = document.getElementById('device');
const halfCb = document.getElementById('half');
const confInput = document.getElementById('conf');
const iouInput = document.getElementById('iou');
const maxDetInput = document.getElementById('maxDet');
const imgszInput = document.getElementById('imgsz');
const fpsSelect = document.getElementById('fps');
const sendSizeSelect = document.getElementById('sendSize');
const qualitySelect = document.getElementById('quality');
const showBoxesCb = document.getElementById('showBoxes');
const showLabelsCb = document.getElementById('showLabels');
const showMasksCb = document.getElementById('showMasks');
const showKeypointsCb = document.getElementById('showKeypoints');
const showSkeletonCb = document.getElementById('showSkeleton');
const showOverlayCb = document.getElementById('showOverlay');
const maskAlphaInput = document.getElementById('maskAlpha');
const textQueriesInput = document.getElementById('textQueries');
const vqaQuestionInput = document.getElementById('vqaQuestion');
const detectionsSidebar = document.getElementById('detectionsSidebar');
const classCountsEl = document.getElementById('classCounts');
const summaryTotalEl = document.getElementById('summaryTotal');
const summaryModelEl = document.getElementById('summaryModel');
const summaryDeviceEl = document.getElementById('summaryDevice');
const summaryTimingEl = document.getElementById('summaryTiming');
const overlayInferTime = document.getElementById('overlayInferTime');
const overlayObjects = document.getElementById('overlayObjects');
const permissionStatus = document.getElementById('permissionStatus');
const imageFileInput = document.getElementById('imageFile');
const inferImageBtn = document.getElementById('inferImage');
const settingsOverlay = document.getElementById('settingsOverlay');
const openSettingsBtn = document.getElementById('openSettings');
const closeSettingsBtn = document.getElementById('closeSettings');
const toggleSidebarBtn = document.getElementById('toggleSidebar');
const themeToggleBtn = document.getElementById('themeToggle');

let running = false, busy = false, detections = [], sendInterval = 200;
let baseUrl = window.location.origin, sendWidth = 640;
let lastInferSize = { width: 1, height: 1 }, lastTask = 'detect';
let ws = null, wsReady = false, currentModel = 'yolov8s.pt', currentCategory = 'yolo_detect';
let modelCategories = {};

const toastContainer = document.getElementById('toastContainer');

const SETTINGS_KEY = 'yolo_toys_v3';
const SKELETON = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const COLORS = ['#ef4444','#f97316','#f59e0b','#84cc16','#22c55e','#06b6d4','#3b82f6','#6366f1','#a855f7','#ec4899'];

// Toast 通知系统
function showToast(message, type = 'info', duration = 3000) {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, duration);
}

// Theme
function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('theme', t);
  const moon = themeToggleBtn?.querySelector('.icon-moon');
  const sun = themeToggleBtn?.querySelector('.icon-sun');
  if (moon) moon.style.display = t === 'light' ? 'none' : 'block';
  if (sun) sun.style.display = t === 'light' ? 'block' : 'none';
}
setTheme(localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme:light)').matches ? 'light' : 'dark'));
themeToggleBtn?.addEventListener('click', () => setTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light'));

// UI
openSettingsBtn?.addEventListener('click', () => settingsOverlay?.classList.add('open'));
closeSettingsBtn?.addEventListener('click', () => settingsOverlay?.classList.remove('open'));
settingsOverlay?.addEventListener('click', e => e.target === settingsOverlay && settingsOverlay.classList.remove('open'));
toggleSidebarBtn?.addEventListener('click', () => detectionsSidebar?.classList.toggle(innerWidth <= 768 ? 'open' : 'collapsed'));

// Settings
function loadSettings() { try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'); } catch { return {}; } }
function saveSettings(s) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); }

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
  if (s.imgsz) imgszInput.value = s.imgsz;
  if (s.half !== undefined) halfCb.checked = s.half;
  if (s.showBoxes !== undefined) showBoxesCb.checked = s.showBoxes;
  if (s.showLabels !== undefined) showLabelsCb.checked = s.showLabels;
  if (s.showMasks !== undefined) showMasksCb.checked = s.showMasks;
  if (s.showKeypoints !== undefined) showKeypointsCb.checked = s.showKeypoints;
  if (s.showSkeleton !== undefined) showSkeletonCb.checked = s.showSkeleton;
  if (s.showOverlay !== undefined && showOverlayCb) showOverlayCb.checked = s.showOverlay;
  if (s.maskAlpha) maskAlphaInput.value = s.maskAlpha;
  if (s.model) currentModel = s.model;
  updateVars();
}

function updateVars() {
  sendInterval = 1000 / parseInt(fpsSelect?.value || '5');
  sendWidth = parseInt(sendSizeSelect?.value || '640');
  saveSettings({
    server: serverInput?.value, useWs: useWsCb?.checked, fps: fpsSelect?.value, sendSize: sendSizeSelect?.value,
    conf: confInput?.value, iou: iouInput?.value, maxDet: maxDetInput?.value, device: deviceSelect?.value,
    quality: qualitySelect?.value, imgsz: imgszInput?.value, half: halfCb?.checked,
    showBoxes: showBoxesCb?.checked, showLabels: showLabelsCb?.checked, showMasks: showMasksCb?.checked,
    showKeypoints: showKeypointsCb?.checked, showSkeleton: showSkeletonCb?.checked,
    showOverlay: showOverlayCb?.checked, maskAlpha: maskAlphaInput?.value, model: currentModel
  });
}

[fpsSelect, sendSizeSelect, serverInput, confInput, iouInput, maxDetInput, deviceSelect, qualitySelect,
 imgszInput, halfCb, showBoxesCb, showLabelsCb, showMasksCb, showKeypointsCb, showSkeletonCb, showOverlayCb,
 maskAlphaInput, useWsCb].forEach(el => el?.addEventListener('change', updateVars));

// Models
async function loadModels() {
  const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
  try {
    const res = await fetch(`${base}/models`);
    const data = await res.json();
    modelCategories = data.categories || {};
  } catch {
    modelCategories = {
      yolo_detect: { name: 'YOLO 检测', models: [{ id: 'yolov8n.pt', name: 'YOLOv8 Nano' }, { id: 'yolov8s.pt', name: 'YOLOv8 Small' }] },
      yolo_segment: { name: '分割', models: [{ id: 'yolov8s-seg.pt', name: 'YOLOv8 Seg' }] },
      yolo_pose: { name: '姿态', models: [{ id: 'yolov8s-pose.pt', name: 'YOLOv8 Pose' }] }
    };
  }
  renderModels();
}

function renderModels() {
  if (!modelList || !quickModelSelect) return;
  quickModelSelect.innerHTML = '';
  modelList.innerHTML = '';
  
  // 判断是否应该显示该类别的模型
  const shouldShowCategory = (cat) => {
    if (cat === currentCategory) return true;
    // YOLO 类别相互显示
    if (currentCategory.startsWith('yolo') && cat.startsWith('yolo') && currentCategory === cat) return true;
    // 多模态标签页显示所有多模态类别
    if (currentCategory === 'multimodal' && cat.startsWith('multimodal')) return true;
    // 精确匹配
    return cat === currentCategory;
  };
  
  Object.entries(modelCategories).forEach(([cat, data]) => {
    const grp = document.createElement('optgroup');
    grp.label = data.name;
    data.models?.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.name;
      if (m.id === currentModel) opt.selected = true;
      grp.appendChild(opt);
      
      if (shouldShowCategory(cat)) {
        const item = document.createElement('div');
        item.className = `model-item${m.id === currentModel ? ' selected' : ''}`;
        item.dataset.model = m.id;
        item.innerHTML = `<div class="model-item-name">${m.name}</div><div class="model-item-desc">${m.description || ''}</div>${m.speed ? `<div class="model-item-meta"><span>速度: ${m.speed}</span><span>精度: ${m.accuracy || '-'}</span></div>` : ''}`;
        item.onclick = () => { currentModel = m.id; renderModels(); updateVars(); showToast(`已选择 ${m.name}`, 'info'); };
        modelList.appendChild(item);
      }
    });
    quickModelSelect.appendChild(grp);
  });
  
  if (!modelList.children.length) {
    modelList.innerHTML = '<div class="empty-hint">此类别暂无可用模型</div>';
  }
}

modelTabs?.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    currentCategory = btn.dataset.cat;
    modelTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderModels();
  });
});

quickModelSelect?.addEventListener('change', () => { currentModel = quickModelSelect.value; renderModels(); updateVars(); });

// Camera
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' } });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    showToast('摄像头启动成功', 'success');
  } catch (e) {
    if (e.name === 'NotAllowedError') {
      showToast('摄像头权限被拒绝', 'error');
      if (permissionStatus) { permissionStatus.hidden = false; permissionStatus.textContent = '请在浏览器设置中允许摄像头访问'; permissionStatus.className = 'notice error'; }
    } else if (e.name === 'NotFoundError') {
      showToast('未找到摄像头设备', 'error');
    } else {
      showToast(`摄像头错误: ${e.message}`, 'error');
    }
    throw e;
  }
}

// Drawing
function getColor(label) { let h = 0; for (let i = 0; i < label.length; i++) h = label.charCodeAt(i) + ((h << 5) - h); return COLORS[Math.abs(h) % COLORS.length]; }
function hexToRgb(hex) { const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex); return r ? { r: parseInt(r[1], 16), g: parseInt(r[2], 16), b: parseInt(r[3], 16) } : { r: 255, g: 0, b: 0 }; }

function draw() {
  if (!running) return;
  requestAnimationFrame(draw);
  if (video.srcObject && !video.paused) {
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
  }
  drawDetections();
}

function drawDetections() {
  if (!detections?.length) return;
  const sx = canvas.width / lastInferSize.width, sy = canvas.height / lastInferSize.height;
  const alpha = parseFloat(maskAlphaInput?.value || '0.3');

  if (showMasksCb?.checked) detections.forEach(d => {
    if (!d.polygons?.length) return;
    const rgb = hexToRgb(getColor(d.label));
    ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
    d.polygons.forEach(p => { if (!p?.length) return; ctx.beginPath(); for (let i = 0; i < p.length; i += 2) i === 0 ? ctx.moveTo(p[i] * sx, p[i+1] * sy) : ctx.lineTo(p[i] * sx, p[i+1] * sy); ctx.closePath(); ctx.fill(); });
  });

  ctx.lineWidth = 2; ctx.font = 'bold 12px sans-serif';
  detections.forEach(d => {
    const [x1, y1, x2, y2] = d.bbox.map((v, i) => v * (i % 2 ? sy : sx));
    const c = getColor(d.label);
    if (showBoxesCb?.checked) { ctx.strokeStyle = c; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); }
    if (showLabelsCb?.checked) { const l = `${d.label} ${(d.score * 100).toFixed(0)}%`; ctx.fillStyle = c; ctx.fillRect(x1, y1 - 16, ctx.measureText(l).width + 6, 16); ctx.fillStyle = '#000'; ctx.fillText(l, x1 + 3, y1 - 4); }
    if (d.keypoints?.length && (showKeypointsCb?.checked || showSkeletonCb?.checked)) {
      const kps = d.keypoints.map(k => [k[0] * sx, k[1] * sy]);
      if (showSkeletonCb?.checked) { ctx.strokeStyle = c; SKELETON.forEach(([i, j]) => { if (kps[i] && kps[j] && kps[i][0] > 0 && kps[j][0] > 0) { ctx.beginPath(); ctx.moveTo(...kps[i]); ctx.lineTo(...kps[j]); ctx.stroke(); } }); }
      if (showKeypointsCb?.checked) kps.forEach(([x, y]) => { if (x > 0 && y > 0) { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); } });
    }
  });
}

// Inference
function getParams() {
  const base = (serverInput?.value?.trim() || baseUrl).replace(/\/$/, '');
  return { base, conf: parseFloat(confInput?.value || '0.25'), iou: parseFloat(iouInput?.value || '0.45'),
    maxDet: parseInt(maxDetInput?.value || '300'), device: deviceSelect?.value || 'auto',
    model: customModelInput?.value?.trim() || currentModel, imgsz: parseInt(imgszInput?.value || '640'), half: halfCb?.checked };
}

async function sendFrame() {
  if (!running || busy) return;
  busy = true;
  const t0 = performance.now();
  const scale = sendWidth / video.videoWidth;
  const sw = Math.round(video.videoWidth * scale), sh = Math.round(video.videoHeight * scale);
  const sc = document.createElement('canvas'); sc.width = sw; sc.height = sh;
  sc.getContext('2d').drawImage(video, 0, 0, sw, sh);

  sc.toBlob(async blob => {
    const p = getParams();
    try {
      if (useWsCb?.checked && ws && wsReady) { ws.send(blob); }
      else {
        const fd = new FormData(); fd.append('file', blob, 'f.jpg');
        const u = new URL(`${p.base}/infer`);
        u.searchParams.set('conf', p.conf); u.searchParams.set('iou', p.iou); u.searchParams.set('max_det', p.maxDet);
        if (p.device !== 'auto') u.searchParams.set('device', p.device);
        u.searchParams.set('model', p.model); u.searchParams.set('imgsz', p.imgsz);
        if (p.half) u.searchParams.set('half', '1');
        const res = await fetch(u, { method: 'POST', body: fd });
        if (res.ok) handleResult(await res.json(), t0, p.device);
      }
    } catch (e) { 
      console.error(e);
      if (!e.message?.includes('fetch')) showToast('推理请求失败', 'error');
    }
    busy = false;
    if (running) setTimeout(sendFrame, sendInterval);
  }, 'image/jpeg', parseFloat(qualitySelect?.value || '0.8'));
}

function handleResult(data, t0, device) {
  detections = data.detections || [];
  if (data.width && data.height) lastInferSize = { width: data.width, height: data.height };
  const rt = performance.now() - t0, bt = data.inference_time;
  statsEl.textContent = `推理: ${bt?.toFixed(0) || '-'}ms | RTT: ${rt.toFixed(0)}ms | ${detections.length} 目标`;
  if (showOverlayCb?.checked && overlayInfo) { overlayInfo.hidden = false; overlayInferTime.textContent = `${bt?.toFixed(0)}ms`; overlayObjects.textContent = detections.length; }
  else if (overlayInfo) overlayInfo.hidden = true;
  if (data.caption) { captionResult.hidden = false; captionText.textContent = data.caption; } else captionResult.hidden = true;
  updateSidebar(data, device, bt);
}

function updateSidebar(data, device, bt) {
  summaryTotalEl.textContent = detections.length;
  summaryModelEl.textContent = data.model || currentModel;
  summaryDeviceEl.textContent = device || '-';
  summaryTimingEl.textContent = `${bt?.toFixed(0) || '-'}ms`;
  const counts = {}; detections.forEach(d => counts[d.label] = (counts[d.label] || 0) + 1);
  classCountsEl.innerHTML = Object.keys(counts).length
    ? Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([c, n]) => `<div class="class-item"><span style="color:${getColor(c)}">${c}</span><span class="count">${n}</span></div>`).join('')
    : '<div class="empty-hint">暂无检测</div>';
}

// WebSocket
function initWS() {
  if (!useWsCb?.checked) return;
  const p = getParams(), proto = p.base.startsWith('https') ? 'wss' : 'ws';
  const u = new URL(`${p.base.replace(/^http(s)?/, proto)}/ws`);
  u.searchParams.set('conf', p.conf); u.searchParams.set('iou', p.iou); u.searchParams.set('max_det', p.maxDet);
  if (p.device !== 'auto') u.searchParams.set('device', p.device);
  u.searchParams.set('model', p.model); u.searchParams.set('imgsz', p.imgsz);
  ws = new WebSocket(u); ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsReady = true; showToast('WebSocket 已连接', 'success'); };
  ws.onmessage = e => { 
    try { 
      const d = JSON.parse(e.data); 
      if (d.type === 'result') handleResult(d.data, performance.now(), p.device);
      else if (d.type === 'error') showToast(d.detail || '推理错误', 'error');
    } catch {} 
  };
  ws.onclose = () => { wsReady = false; ws = null; if (running) showToast('WebSocket 连接断开', 'error'); };
  ws.onerror = () => { wsReady = false; showToast('WebSocket 连接失败', 'error'); };
}
function closeWS() { try { ws?.close(); } catch {} ws = null; wsReady = false; }

// Image Inference
async function runImageInference(file) {
  const p = getParams(), fd = new FormData(); fd.append('file', file);
  const u = new URL(`${p.base}/infer`);
  u.searchParams.set('conf', p.conf); u.searchParams.set('iou', p.iou); u.searchParams.set('model', p.model);
  statsEl.textContent = '推理中...';
  try {
    const res = await fetch(u, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const img = new Image();
    img.onload = () => { 
      canvas.width = img.width; canvas.height = img.height; ctx.drawImage(img, 0, 0); 
      detections = data.detections || []; 
      lastInferSize = { width: data.width || img.width, height: data.height || img.height }; 
      drawDetections(); emptyState.style.display = 'none'; 
      statsEl.textContent = `完成: ${detections.length} 目标 | ${data.inference_time?.toFixed(0) || '-'}ms`; 
      updateSidebar(data, p.device, data.inference_time);
      showToast(`检测到 ${detections.length} 个目标`, 'success');
    };
    img.src = URL.createObjectURL(file);
  } catch (e) { 
    statsEl.textContent = '推理失败'; 
    showToast(`推理失败: ${e.message}`, 'error');
  }
}

// Event Handlers
startBtn?.addEventListener('click', async () => {
  startBtn.disabled = true; startBtn.innerHTML = '<span class="loader"></span> 启动中...';
  try {
    await loadModels(); applySettings(); await setupCamera();
    closeWS(); initWS(); running = true; stopBtn.disabled = false; emptyState.style.display = 'none';
    draw(); sendFrame();
  } catch (e) { 
    console.error(e); 
    statsEl.textContent = '启动失败'; 
    showToast(`启动失败: ${e.message || '未知错误'}`, 'error');
  }
  startBtn.disabled = false; startBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> 开始摄像头';
});

stopBtn?.addEventListener('click', () => {
  running = false; closeWS();
  video.srcObject?.getTracks().forEach(t => t.stop()); video.srcObject = null;
  stopBtn.disabled = true; statsEl.textContent = '就绪'; emptyState.style.display = 'flex';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

inferImageBtn?.addEventListener('click', () => { const f = imageFileInput?.files?.[0]; if (f) { if (running) stopBtn.click(); runImageInference(f); } });
imageFileInput?.addEventListener('change', () => { if (imageFileInput.files?.[0]) { if (running) stopBtn.click(); runImageInference(imageFileInput.files[0]); } });

// Init
applySettings();
loadModels();
