/** Bbox labeling client — canvas overlays and keyboard-first editing. */

const state = {
  frames: [],
  index: 0,
  faces: [],
  plates: [],
  drag: null,
  selected: null, // { kind, idx }
  imgNatural: { w: 1, h: 1 },
  display: { x: 0, y: 0, w: 1, h: 1, scale: 1 },
};

const el = {
  meta: document.getElementById('meta'),
  indicator: document.getElementById('frame-indicator'),
  img: document.getElementById('frame-img'),
  canvas: document.getElementById('overlay'),
  stage: document.getElementById('stage'),
};

function currentClass() {
  const r = document.querySelector('input[name="cls"]:checked');
  return r ? r.value : 'face';
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function syncDisplayMetrics() {
  const img = el.img;
  const canvas = el.canvas;
  const rect = el.stage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  const nw = img.naturalWidth || 1;
  const nh = img.naturalHeight || 1;
  state.imgNatural = { w: nw, h: nh };
  const scale = Math.min(canvas.width / nw, canvas.height / nh);
  const dw = nw * scale;
  const dh = nh * scale;
  state.display = {
    x: (canvas.width - dw) / 2,
    y: (canvas.height - dh) / 2,
    w: dw,
    h: dh,
    scale,
  };
}

function pixelToNorm(x, y) {
  const d = state.display;
  const nx = (x - d.x) / d.w;
  const ny = (y - d.y) / d.h;
  return { nx: clamp(nx, 0, 1), ny: clamp(ny, 0, 1) };
}

function normBoxToCanvas(box) {
  const d = state.display;
  const [x, y, w, h] = box;
  const nw = state.imgNatural.w;
  const nh = state.imgNatural.h;
  const px = d.x + (x / nw) * d.w;
  const py = d.y + (y / nh) * d.h;
  const pw = (w / nw) * d.w;
  const ph = (h / nh) * d.h;
  return { px, py, pw, ph };
}

function canvasRectToPixelBox(x0, y0, x1, y1) {
  const d = state.display;
  const nw = state.imgNatural.w;
  const nh = state.imgNatural.h;
  const left = clamp(Math.min(x0, x1) - d.x, 0, d.w);
  const top = clamp(Math.min(y0, y1) - d.y, 0, d.h);
  const right = clamp(Math.max(x0, x1) - d.x, 0, d.w);
  const bottom = clamp(Math.max(y0, y1) - d.y, 0, d.h);
  const x = Math.round((left / d.w) * nw);
  const y = Math.round((top / d.h) * nh);
  const w = Math.round(((right - left) / d.w) * nw);
  const h = Math.round(((bottom - top) / d.h) * nh);
  if (w < 2 || h < 2) return null;
  return [x, y, w, h];
}

function draw() {
  const ctx = el.canvas.getContext('2d');
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
  const drawList = [
    ...state.faces.map((b, i) => ({ kind: 'face', idx: i, box: b, color: '#3dd68c' })),
    ...state.plates.map((b, i) => ({ kind: 'license_plate', idx: i, box: b, color: '#6eb5ff' })),
  ];
  for (const item of drawList) {
    const { px, py, pw, ph } = normBoxToCanvas(item.box);
    const sel = state.selected && state.selected.kind === item.kind && state.selected.idx === item.idx;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = sel ? 3 : 2;
    ctx.strokeRect(px, py, pw, ph);
  }
  if (state.drag) {
    const { x0, y0, x1, y1 } = state.drag;
    ctx.strokeStyle = '#f0b429';
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(
      Math.min(x0, x1),
      Math.min(y0, y1),
      Math.abs(x1 - x0),
      Math.abs(y1 - y0)
    );
    ctx.setLineDash([]);
  }
}

function hitTest(mx, my) {
  const lists = [
    ['face', state.faces],
    ['license_plate', state.plates],
  ];
  for (const [kind, arr] of lists) {
    for (let i = arr.length - 1; i >= 0; i--) {
      const { px, py, pw, ph } = normBoxToCanvas(arr[i]);
      if (mx >= px && mx <= px + pw && my >= py && my <= py + ph) {
        return { kind, idx: i };
      }
    }
  }
  return null;
}

async function loadFrame(index) {
  if (!state.frames.length) return;
  state.index = clamp(index, 0, state.frames.length - 1);
  const name = state.frames[state.index];
  el.indicator.textContent = `${state.index + 1} / ${state.frames.length} — ${name}`;
  el.img.src = `/api/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
  await new Promise((resolve, reject) => {
    el.img.onload = () => resolve();
    el.img.onerror = reject;
  });
  const row = await api(`/api/labels/${encodeURIComponent(name)}`);
  state.faces = (row.faces || []).map((b) => b.slice(0, 4));
  state.plates = (row.license_plates || []).map((b) => b.slice(0, 4));
  state.selected = null;
  syncDisplayMetrics();
  draw();
}

async function saveFrame() {
  const name = state.frames[state.index];
  await api('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frame: name,
      faces: state.faces,
      license_plates: state.plates,
      note: 'human_reviewed',
    }),
  });
  el.meta.textContent = `Saved ${name} at ${new Date().toLocaleTimeString()}`;
}

function setupCanvas() {
  el.canvas.addEventListener('mousedown', (ev) => {
    const rect = el.canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = hitTest(mx, my);
    if (hit) {
      state.selected = hit;
      draw();
      return;
    }
    state.selected = null;
    state.drag = { x0: mx, y0: my, x1: mx, y1: my };
    draw();
  });
  window.addEventListener('mousemove', (ev) => {
    if (!state.drag) return;
    const rect = el.canvas.getBoundingClientRect();
    state.drag.x1 = ev.clientX - rect.left;
    state.drag.y1 = ev.clientY - rect.top;
    draw();
  });
  window.addEventListener('mouseup', () => {
    if (!state.drag) return;
    const { x0, y0, x1, y1 } = state.drag;
    state.drag = null;
    const box = canvasRectToPixelBox(x0, y0, x1, y1);
    if (box) {
      const cls = currentClass();
      if (cls === 'face') state.faces.push(box);
      else state.plates.push(box);
    }
    draw();
  });
}

function deleteSelected() {
  if (!state.selected) return;
  const { kind, idx } = state.selected;
  if (kind === 'face') state.faces.splice(idx, 1);
  else state.plates.splice(idx, 1);
  state.selected = null;
  draw();
}

async function init() {
  const meta = await api('/api/meta');
  el.meta.textContent = `${meta.batch_id} — ${meta.title} (${meta.frame_count} frames)`;
  const frames = await api('/api/frames');
  state.frames = frames.frames || [];
  setupCanvas();
  window.addEventListener('resize', () => {
    syncDisplayMetrics();
    draw();
  });
  document.getElementById('btn-prev').onclick = () => loadFrame(state.index - 1);
  document.getElementById('btn-next').onclick = () => loadFrame(state.index + 1);
  document.getElementById('btn-save').onclick = () => saveFrame();
  document.getElementById('btn-delete').onclick = deleteSelected;
  document.getElementById('btn-clear').onclick = () => {
    state.faces = [];
    state.plates = [];
    state.selected = null;
    draw();
  };
  window.addEventListener('keydown', (ev) => {
    if (ev.target && ev.target.tagName === 'INPUT') return;
    if (ev.key === 'n' || ev.key === 'N') loadFrame(state.index + 1);
    if (ev.key === 'p' || ev.key === 'P') loadFrame(state.index - 1);
    if (ev.key === 's' || ev.key === 'S') saveFrame();
    if (ev.key === '1') document.querySelector('input[value="face"]').checked = true;
    if (ev.key === '2') document.querySelector('input[value="license_plate"]').checked = true;
    if (ev.key === 'Delete' || ev.key === 'Backspace') deleteSelected();
  });
  await loadFrame(0);
}

init().catch((err) => {
  el.meta.textContent = `Studio error: ${err.message}`;
  console.error(err);
});
