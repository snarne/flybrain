import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { createFlyView } from './fly.js';

// ------------------------------------------------------------------ data
const [posBuf, metaBuf, meta, meshBuf, meshIdx, presets] = await Promise.all([
  fetch('data/neurons.bin').then(r => r.arrayBuffer()),
  fetch('data/meta.bin').then(r => r.arrayBuffer()),
  fetch('data/meta.json').then(r => r.json()),
  fetch('data/meshes.bin').then(r => r.arrayBuffer()),
  fetch('data/meshes.json').then(r => r.json()),
  fetch('api/presets').then(r => r.json()),
]);
const N = meta.n;
const positions = new Float32Array(posBuf);
const M = {};
meta.order.forEach((k, i) => { M[k] = new Uint16Array(metaBuf, i * N * 2, N); });
const REG = meta.regions;
const NR = REG.length;

// ------------------------------------------------------------------ scene
const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x07090d);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.setScissorTest(true);
const viewBrain = document.getElementById('view-brain');
const viewFly = document.getElementById('view-fly');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
camera.position.set(520, -330, 1550);
const controls = new OrbitControls(camera, viewBrain);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, -430, 0);   // whole CNS: brain on top, nerve cord below

// ---- stage layout: the fly (left) and the brain (right) share the space between the panels
let viewMode = 'split';
try { viewMode = localStorage.getItem('flybrain-view') || 'split'; } catch (e) { /* default */ }
const rects = { brain: null, fly: null };
function stage() {
  const L = document.getElementById('left'), R = document.getElementById('right');
  const x0 = L.getBoundingClientRect().right + 10, x1 = R.getBoundingClientRect().left - 10;
  return { x: x0, y: 56, w: Math.max(200, x1 - x0), h: innerHeight - 56 - 10 };
}
function layout() {
  const s = stage();
  if (viewMode === 'brain') { rects.brain = { ...s }; rects.fly = null; }
  else if (viewMode === 'fly') { rects.fly = { ...s }; rects.brain = null; }
  else if (s.w / s.h > 1.25) {
    const w = Math.floor(s.w / 2) - 4;
    rects.fly = { x: s.x, y: s.y, w, h: s.h };
    rects.brain = { x: s.x + w + 8, y: s.y, w, h: s.h };
  } else {
    const h = Math.floor(s.h / 2) - 4;
    rects.fly = { x: s.x, y: s.y, w: s.w, h };
    rects.brain = { x: s.x, y: s.y + h + 8, w: s.w, h };
  }
  for (const [el, r] of [[viewBrain, rects.brain], [viewFly, rects.fly]]) {
    el.hidden = !r;
    if (r) Object.assign(el.style, { left: r.x + 'px', top: r.y + 'px', width: r.w + 'px', height: r.h + 'px' });
  }
  if (rects.brain) {
    camera.aspect = rects.brain.w / rects.brain.h;
    camera.updateProjectionMatrix();
    pointsMat.uniforms.uScale.value = rects.brain.h * renderer.getPixelRatio() / 2;
  }
  if (rects.fly && flyView) {
    const { w, h } = rects.fly, cam = flyView.camera, a = w / h;
    cam.aspect = a;
    // keep the whole fly in frame in tall/narrow views, and centre it in the space above the experiment bar
    cam.fov = a < 1.5 ? 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(13)) * 1.5 / a) * 180 / Math.PI : 26;
    const bar = viewFly.querySelector('.experiments').offsetHeight || 0;
    cam.setViewOffset(w, h, 0, (bar - 10) / 2, w, h);
    cam.updateProjectionMatrix();
  }
  // the "why" card sits over the brain whenever the brain is on screen, where its pathway lights up
  const card = document.getElementById('decision'), home = rects.brain ? viewBrain : viewFly;
  if (card.parentElement !== home) home.append(card);
  document.querySelectorAll('.viewmode [data-view]').forEach(b => b.classList.toggle('on', b.dataset.view === viewMode));
}
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  layout();
}
document.querySelectorAll('.viewmode [data-view]').forEach(b => b.onclick = () => {
  viewMode = b.dataset.view;
  try { localStorage.setItem('flybrain-view', viewMode); } catch (e) { /* ignore */ }
  layout();
});
new ResizeObserver(() => layout()).observe(document.getElementById('left'));

// neurons as a point cloud: dim region colour at rest, hot glow when spiking
const activity = new Float32Array(N);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const actAttr = new THREE.BufferAttribute(activity, 1).setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('act', actAttr);
geo.setAttribute('rgn', new THREE.BufferAttribute(new Float32Array(M.region), 1));
const highlight = new Float32Array(N);
const hlAttr = new THREE.BufferAttribute(highlight, 1).setUsage(THREE.DynamicDrawUsage);
geo.setAttribute('hl', hlAttr);
geo.computeBoundingSphere();

const regionColors = REG.map(r => new THREE.Color(r.color));
const regionVisible = new Array(NR).fill(1);

const pointsMat = new THREE.ShaderMaterial({
  uniforms: {
    uColors: { value: regionColors },
    uVisible: { value: regionVisible },
    uScale: { value: 400 },
    uBase: { value: 0.32 },   // resting brightness (slider in Brain regions)
  },
  vertexShader: /* glsl */`
    attribute float act;
    attribute float rgn;
    attribute float hl;
    uniform vec3 uColors[${NR}];
    uniform float uVisible[${NR}];
    uniform float uScale;
    uniform float uBase;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      int r = int(rgn + 0.5);
      float vis = uVisible[r];
      float a = clamp(act, 0.0, 1.5);
      vec3 base = uColors[r];
      vec3 hot = mix(vec3(1.0, 0.72, 0.30), vec3(1.0, 0.97, 0.88), clamp(a - 0.4, 0.0, 1.0));
      vColor = mix(base, hot, clamp(a * 1.4, 0.0, 1.0));
      vAlpha = vis * (uBase + a * 1.1);
      if (hl > 0.0) {                       // part of the pathway being explained
        vColor = mix(vec3(0.35, 0.95, 1.0), vec3(1.0), clamp(a, 0.0, 1.0));
        vAlpha = max(vAlpha, 0.55 * hl + 0.4);
      }
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      float worldSize = 3.5 + 8.0 * uBase + a * 12.0 + hl * 8.0;
      gl_PointSize = vis * max(1.6, worldSize * uScale / -mv.z);
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      float d = dot(c, c);
      if (d > 0.25) discard;
      float soft = smoothstep(0.25, 0.0, d);
      gl_FragColor = vec4(vColor * soft, vAlpha * soft);
    }`,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});
const points = new THREE.Points(geo, pointsMat);
scene.add(points);

// neuropil surfaces
const meshV = new Float32Array(meshBuf, 0, meshIdx.nv * 3);
const meshF = new Uint32Array(meshBuf, meshIdx.nv * 12, meshIdx.nf * 3);
const shells = new THREE.Group();
const regionShellMats = REG.map(r => new THREE.MeshBasicMaterial({
  color: r.color, transparent: true, opacity: 0.05, depthWrite: false, side: THREE.DoubleSide,
}));
let outline = null;
for (const m of meshIdx.meshes) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(meshV.slice(m.v[0] * 3, (m.v[0] + m.v[1]) * 3), 3));
  const f = meshF.slice(m.f[0] * 3, (m.f[0] + m.f[1]) * 3);
  g.setIndex(new THREE.BufferAttribute(f, 1));
  if (m.name === 'BRAIN') {
    const edges = new THREE.EdgesGeometry(g, 25);
    outline = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x8a93a3, transparent: true, opacity: 0.12, depthWrite: false }));
    scene.add(outline);
  } else {
    const mesh = new THREE.Mesh(g, regionShellMats[m.region]);
    mesh.userData = { name: m.name, region: m.region };
    shells.add(mesh);
  }
}
scene.add(shells);

// selection marker + partner lines
const selMarker = new THREE.Mesh(new THREE.SphereGeometry(4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
selMarker.visible = false;
scene.add(selMarker);
const linkMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 });
const links = new THREE.LineSegments(new THREE.BufferGeometry(), linkMat);
scene.add(links);

// ------------------------------------------------------------------ websocket
let ws, connected = false;
const conn = document.getElementById('conn');
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { connected = true; conn.textContent = 'live'; conn.className = 'conn on'; dispatchEvent(new Event('flybrain:open')); };
  ws.onclose = () => { connected = false; conn.textContent = 'reconnecting…'; conn.className = 'conn off'; setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      if (m.type === 'status') onStatus(m);
      else if (m.type === 'body') onBody(m);
      else if (m.type === 'decision') onDecision(m);
      else if (m.type === 'skill' || m.type === 'skills') { onSkill(m); dispatchEvent(new CustomEvent('flybrain:msg', { detail: m })); }
      else dispatchEvent(new CustomEvent('flybrain:msg', { detail: m }));
    } else onSpikes(ev.data);
  };
}
function send(cmd) { if (connected) ws.send(JSON.stringify(cmd)); }
connect();

function onSpikes(buf) {
  const dv = new DataView(buf);
  const t = dv.getFloat64(0, true);
  const n = dv.getUint32(8, true);
  const idx = new Uint32Array(buf, 12, n);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    activity[i] = Math.min(activity[i] + 0.55, 1.5);
  }
  document.getElementById('t-sim').textContent = (t / 1000).toFixed(1) + ' s';
}

// ------------------------------------------------------------------ UI: senses
let pct = 100;
const presetByKey = Object.fromEntries(presets.stimuli.map(s => [s.key, s]));
const presetRate = key => Math.max(1, Math.round(presetByKey[key].rate * pct / 100));
const manualRate = () => Math.round(150 * pct / 100);
const activeStim = new Map();
const stimEl = document.getElementById('stimuli');
const groups = {};
for (const s of presets.stimuli) (groups[s.group] ||= []).push(s);
for (const [g, list] of Object.entries(groups)) {
  const label = document.createElement('div');
  label.className = 'group-label';
  label.textContent = g;
  stimEl.append(label);
  const chips = document.createElement('div');
  chips.className = 'chips';
  for (const s of list) {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = s.label;
    b.title = `${s.about}\n${s.n} neurons · ${s.rate} Hz at 100%`;
    b.dataset.key = s.key;
    b.onclick = () => {
      const on = !activeStim.has(s.key);
      if (on) activeStim.set(s.key, presetRate(s.key)); else activeStim.delete(s.key);
      send({ cmd: 'preset', key: s.key, rate: on ? presetRate(s.key) : 0 });
      b.classList.toggle('on', on);
    };
    chips.append(b);
  }
  stimEl.append(chips);
}
const rateEl = document.getElementById('rate');
rateEl.oninput = () => {
  pct = +rateEl.value;
  document.getElementById('rate-out').textContent = pct + '%';
};
rateEl.onchange = () => {
  for (const k of activeStim.keys()) { activeStim.set(k, presetRate(k)); send({ cmd: 'preset', key: k, rate: presetRate(k) }); }
};

document.querySelectorAll('[data-cmd]').forEach(b => b.onclick = () => {
  send({ cmd: b.dataset.cmd });
  if (b.dataset.cmd === 'clear') {
    activeStim.clear();
    document.querySelectorAll('.chip.on').forEach(c => c.classList.remove('on'));
  }
  if (b.dataset.cmd === 'reset') activity.fill(0);
});
const pauseBtn = document.getElementById('pause');
let paused = false;
pauseBtn.onclick = () => { paused = !paused; send({ cmd: 'pause', on: paused }); pauseBtn.textContent = paused ? 'Resume' : 'Pause'; };
document.getElementById('speed').onchange = e => send({ cmd: 'speed', value: +e.target.value });
const modeHints = {
  stable: 'The Shiu et al. (2024) model on the BANC wiring, plus spike-frequency adaptation for the neurons that otherwise lock into endless self-excitation after odours.',
  paper: 'The plain Shiu et al. (2024) model on the BANC wiring, no adaptation. Strong odours can trigger runaway firing.',
};
const modeEl = document.getElementById('mode');
const modeHint = document.getElementById('mode-hint');
modeHint.textContent = modeHints.stable;
modeEl.onchange = () => { send({ cmd: 'mode', value: modeEl.value }); modeHint.textContent = modeHints[modeEl.value]; activity.fill(0); };

// search by type
const searchEl = document.getElementById('search');
const resultsEl = document.getElementById('search-results');
let searchTimer;
searchEl.oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = searchEl.value.trim();
    if (!q) { resultsEl.innerHTML = ''; return; }
    const res = await fetch('api/search?q=' + encodeURIComponent(q)).then(r => r.json());
    resultsEl.innerHTML = '';
    for (const r of res) {
      const row = document.createElement('div');
      row.className = 'result';
      if (r.neuron) {
        row.innerHTML = `<span class="name">${esc(r.neuron.type)}</span><span class="n">neuron</span>`;
        const b = mkBtn('Inspect', () => select(r.neuron.idx));
        row.append(b);
      } else {
        row.innerHTML = `<span class="name">${esc(r.type)}</span><span class="n">${r.n}</span>`;
        let on = false;
        const b = mkBtn('Stimulate', () => {
          on = !on;
          send({ cmd: 'type', type: r.type, rate: on ? manualRate() : 0 });
          b.classList.toggle('primary', on);
          b.textContent = on ? 'On' : 'Stimulate';
        });
        const f = mkBtn('Find', async () => {
          const d = await fetch('api/type/' + encodeURIComponent(r.type)).then(x => x.json());
          if (d.neurons.length) select(d.neurons[0]);
        });
        row.append(f, b);
      }
      resultsEl.append(row);
    }
    if (!res.length) resultsEl.innerHTML = '<div class="hint">No matching cell types</div>';
  }, 200);
};

// ------------------------------------------------------------------ UI: right panel
const regEl = document.getElementById('regions');
const regRows = REG.map((r, i) => {
  const d = document.createElement('div');
  d.className = 'meter region';
  d.innerHTML = `<span class="label"><span class="dot" style="background:${r.color}"></span>${esc(r.label)}</span><span class="val"></span><span class="about">${esc(r.about)} · ${meta.counts[r.key].toLocaleString()} neurons</span><div class="bar"><i style="background:${r.color}"></i></div>`;
  d.onclick = () => {
    regionVisible[i] = regionVisible[i] ? 0 : 1;
    d.classList.toggle('hidden', !regionVisible[i]);
    shells.children.forEach(m => { if (m.userData.region === i) m.visible = !!regionVisible[i]; });
  };
  regEl.append(d);
  return d;
});
document.getElementById('show-shells').onchange = e => { shells.visible = e.target.checked; };
if (!meshIdx.meshes.length) document.querySelector('.row.toggles').hidden = true;
const brightEl = document.getElementById('brightness');
try { const v = +localStorage.getItem('flybrain-bright'); if (v) brightEl.value = v; } catch (e) { /* default */ }
const setBright = () => { pointsMat.uniforms.uBase.value = +brightEl.value / 100; try { localStorage.setItem('flybrain-bright', brightEl.value); } catch (e) { /* ignore */ } };
brightEl.oninput = setBright;
setBright();   // no neuropil meshes for BANC yet
document.getElementById('show-outline').onchange = e => { if (outline) outline.visible = e.target.checked; };

const regionLevel = new Float32Array(NR);
const topEl = document.getElementById('top');
let topHover = false;
topEl.onmouseenter = () => { topHover = true; };
topEl.onmouseleave = () => { topHover = false; };
const topEmpty = document.createElement('div');
topEmpty.className = 'empty';
topEmpty.textContent = 'Nothing yet. Switch on a sense.';
topEl.append(topEmpty);
const topRows = Array.from({ length: 10 }, () => {
  const d = document.createElement('div');
  d.className = 'item';
  d.hidden = true;
  d.innerHTML = '<span class="t"></span><span class="r"></span>';
  d.onclick = () => select(+d.dataset.idx);
  topEl.append(d);
  return d;
});
function onStatus(s) {
  const achieved = Math.min(s.speed, s.sim_ratio);
  document.getElementById('t-ratio').textContent = achieved >= s.speed * 0.95 ? `${s.speed}× real time` : `${achieved.toFixed(2)}× real time (CPU-limited)`;
  document.getElementById('t-firing').textContent = `${s.firing.toLocaleString()} neurons firing`;
  document.getElementById('runaway').hidden = !s.runaway;
  if (modeEl.value !== s.mode) { modeEl.value = s.mode; modeHint.textContent = modeHints[s.mode]; }
  s.regions.forEach((r, i) => {
    const lvl = Math.min(1, Math.log10(1 + r.hz_per_neuron * 10) / 2);
    regionLevel[i] = lvl;
    regRows[i].querySelector('.val').textContent = r.hz_total >= 1000 ? (r.hz_total / 1000).toFixed(1) + 'k spikes/s' : r.hz_total.toFixed(0) + ' spikes/s';
    regRows[i].querySelector('i').style.width = (lvl * 100).toFixed(1) + '%';
  });
  if (!topHover) {
    topEmpty.hidden = s.top.length > 0;
    topRows.forEach((row, k) => {
      const t = s.top[k];
      row.hidden = !t;
      if (!t) return;
      row.dataset.idx = t.idx;
      row.firstChild.textContent = t.type;
      row.lastChild.textContent = `${t.region} · ${t.hz.toFixed(0)} Hz`;
    });
  }
  // sync chips with server state (e.g. commands sent by another client or the API)
  document.querySelectorAll('.chip').forEach(c => {
    const on = c.dataset.key in s.inputs.presets;
    c.classList.toggle('on', on);
    if (on) activeStim.set(c.dataset.key, s.inputs.presets[c.dataset.key]); else activeStim.delete(c.dataset.key);
  });
}

// ------------------------------------------------------------------ picking + inspector
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 3;
const mouse = new THREE.Vector2();
function pick(ev) {
  const r = rects.brain;
  if (!r) return -1;
  mouse.set(((ev.clientX - r.x) / r.w) * 2 - 1, -((ev.clientY - r.y) / r.h) * 2 + 1);
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(points);
  for (const h of hits) if (regionVisible[M.region[h.index]]) return h.index;
  return -1;
}
const hoverEl = document.getElementById('hover');
let hoverTimer = 0;
viewBrain.addEventListener('pointermove', ev => {
  const now = performance.now();
  if (now - hoverTimer < 60 || ev.buttons) { if (ev.buttons) hoverEl.hidden = true; return; }
  hoverTimer = now;
  const i = pick(ev);
  if (i < 0) { hoverEl.hidden = true; return; }
  hoverEl.hidden = false;
  hoverEl.style.left = ev.clientX + 'px';
  hoverEl.style.top = ev.clientY + 'px';
  hoverEl.innerHTML = `${esc(meta.type[M.type[i]] || '(unnamed)')} <span class="m">· ${esc(REG[M.region[i]].label)}</span>`;
});
let downAt = null;
viewBrain.addEventListener('pointerdown', ev => { downAt = [ev.clientX, ev.clientY]; });
viewBrain.addEventListener('pointerup', ev => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
  const i = pick(ev);
  if (i >= 0) select(i);
});

const insp = document.getElementById('inspector');
let selected = -1;
async function select(i) {
  selected = i;
  const d = await fetch('api/neuron/' + i).then(r => r.json());
  if (selected !== i) return;
  selMarker.position.fromArray(positions, i * 3);
  selMarker.visible = true;
  drawLinks(i, d);
  insp.hidden = false;
  insp.innerHTML = `
    <button class="close" aria-label="Close" title="Back to controls">×</button>
    <h3>${esc(d.type)}</h3>
    <div class="sub">${esc(d.class || d.super_class)} · ${esc(d.region)} · ${esc(d.side)}</div>
    <dl>
      <dt>Firing</dt><dd id="insp-hz">${d.hz} Hz</dd>
      <dt>Neuropil</dt><dd>${esc(d.neuropil)}</dd>
      <dt>Transmitter</dt><dd>${esc(d.transmitter)} (${d.sign})</dd>
      <dt>Outputs</dt><dd>${d.n_out.toLocaleString()} neurons · ${d.syn_out.toLocaleString()} synapses</dd>
      <dt>Inputs</dt><dd>${d.n_in.toLocaleString()} neurons · ${d.syn_in.toLocaleString()} synapses</dd>
      <dt>BANC ID</dt><dd><a href="${d.codex_url}" target="_blank" rel="noopener">${d.root_id}</a></dd>
    </dl>
    <div class="actions">
      <button class="btn ${d.stimulated ? 'primary' : ''}" id="insp-stim">${d.stimulated ? 'Stimulating' : 'Stimulate'}</button>
      <button class="btn danger ${d.blocked ? 'on' : ''}" id="insp-block">${d.blocked ? 'Silenced' : 'Silence'}</button>
    </div>
    <h4>Strongest outputs <span style="color:#ffb35c">●</span></h4><div id="insp-out"></div>
    <h4>Strongest inputs <span style="color:#6fb6ff">●</span></h4><div id="insp-in"></div>`;
  insp.querySelector('.close').onclick = () => { insp.hidden = true; selMarker.visible = false; links.visible = false; selected = -1; };
  let stim = d.stimulated, blk = d.blocked;
  const sb = insp.querySelector('#insp-stim'), bb = insp.querySelector('#insp-block');
  sb.onclick = () => { stim = !stim; send({ cmd: 'neuron', idx: i, rate: stim ? manualRate() : 0 }); sb.classList.toggle('primary', stim); sb.textContent = stim ? 'Stimulating' : 'Stimulate'; };
  bb.onclick = () => { blk = !blk; send({ cmd: 'block', idx: i, on: blk }); bb.classList.toggle('on', blk); bb.textContent = blk ? 'Silenced' : 'Silence'; };
  const fill = (el, list) => list.forEach(p => {
    const r = document.createElement('div');
    r.className = 'partner';
    r.innerHTML = `<span class="t">${esc(p.type)}</span><span class="r">${esc(p.region)} · ${p.synapses > 0 ? '+' : ''}${p.synapses}</span>`;
    r.onclick = () => select(p.idx);
    el.append(r);
  });
  fill(insp.querySelector('#insp-out'), d.outputs);
  fill(insp.querySelector('#insp-in'), d.inputs);
}
function drawLinks(i, d) {
  const pts = [], cols = [];
  const a = positions.subarray(i * 3, i * 3 + 3);
  const add = (list, c) => list.forEach(p => {
    pts.push(...a, ...positions.subarray(p.idx * 3, p.idx * 3 + 3));
    cols.push(...c, ...c);
  });
  add(d.outputs, [1.0, 0.7, 0.36]);
  add(d.inputs, [0.43, 0.71, 1.0]);
  links.geometry.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  links.geometry = g;
  links.visible = true;
}
setInterval(async () => {
  if (selected < 0 || insp.hidden) return;
  const el = document.getElementById('insp-hz');
  if (!el) return;
  const d = await fetch('api/neuron/' + selected).then(r => r.json()).catch(() => null);
  if (d && el.isConnected) el.textContent = d.hz + ' Hz';
}, 1000);

// ------------------------------------------------------------------ helpers
function mkBtn(label, fn) { const b = document.createElement('button'); b.className = 'btn'; b.textContent = label; b.onclick = fn; return b; }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

// ------------------------------------------------------------------ the fly: brain -> muscles
let flyView = null;
createFlyView(viewFly).then(v => { flyView = v; layout(); }).catch(e => console.error('fly view', e));
const flyMode = document.getElementById('fly-mode');
const flySenses = document.getElementById('fly-senses');
const MODE_TEXT = {
  rest: 'Resting', legs: 'Moving its legs', groom: 'Grooming its antennae', feed: 'Feeding: proboscis extended',
  jump: 'Escape jump!', fly: 'Flying', skill: 'Skill',
};
const pathEl = document.getElementById('pathways');
let pathRows = [];
fetch('api/body/pathways').then(r => r.json()).then(p => {
  pathRows = p.rows.map(row => {
    const d = document.createElement('div');
    d.className = 'meter path ' + row.kind;
    d.title = `${row.from} → ${row.to}\nclick to find these ${row.neurons.length} neurons in the nervous system`;
    d.innerHTML = `<span class="label"><span class="via ${row.kind === 'dn' ? 'cord' : 'brain'}"></span>${esc(row.from)}</span><span class="val">–</span><span class="about">${row.kind === 'dn' ? '' : '→ '}${esc(row.to)}</span><div class="bar"><i></i></div>`;
    d.onclick = () => highlightNeurons(row.neurons, 6);
    pathEl.append(d);
    return { row, el: d, bar: d.querySelector('i'), val: d.querySelector('.val') };
  });
});
function onBody(b) {
  flyView?.onBody(b);
  const sk = b.skill;
  flyMode.textContent = sk ? `${sk.kind === 'learn' ? 'Practising' : 'Doing'}: ${sk.name.replace(/_/g, ' ')}${sk.attempt ? ` (try ${sk.attempt})` : ''}` : (MODE_TEXT[b.mode] || b.mode);
  flyMode.className = 'mode-badge ' + (sk ? 'skill' : b.mode);
  const s = b.senses || {};
  const bits = [];
  if (s.loom_left || s.loom_right) bits.push(`👁 looming ${s.loom_left ? 'left' : 'right'} eye · LPLC2 ${Math.max(s.loom_left || 0, s.loom_right || 0)} Hz`);
  if (s.sugar) bits.push(`sugar taste · ${s.sugar} Hz`);
  if (s.bitter) bits.push(`bitter taste · ${s.bitter} Hz`);
  if (s.touch) bits.push(`antenna touch · ${s.touch} Hz`);
  for (const a of b.active || []) bits.push(`activating ${a.toUpperCase()}`);
  flySenses.textContent = bits.join('   ·   ');
  for (const p of pathRows) {
    if (p.row.kind === 'dn') {
      const hz = (b.rates || {})[p.row.key] || 0;
      p.val.textContent = hz.toFixed(0) + ' Hz';
      p.bar.style.width = Math.min(100, hz / 40 * 100).toFixed(1) + '%';
      p.el.classList.toggle('active', hz > 5);
    } else {
      const a = (b.parts || {})[p.row.key] || 0;
      p.val.textContent = Math.round(a * 100) + '%';
      p.bar.style.width = Math.min(100, a * 100).toFixed(1) + '%';
      p.el.classList.toggle('active', a > 0.15);
    }
  }
}
// keep the fly camera's orbit controls from capturing clicks meant for the overlays
for (const el of viewFly.querySelectorAll('.experiments, .decision, .fly-hud'))
  for (const t of ['pointerdown', 'wheel']) el.addEventListener(t, e => e.stopPropagation());
document.querySelectorAll('[data-exp]').forEach(btn => btn.onclick = () => send({ cmd: 'body', action: btn.dataset.exp, side: btn.dataset.side }));
let demoTimers = [];
const demoBtn = document.getElementById('demo');
function stopDemo() {
  demoTimers.forEach(clearTimeout);
  demoTimers = [];
  demoBtn.textContent = '▶ Demo';
  demoBtn.classList.remove('running');
}
demoBtn.onclick = () => {
  if (demoTimers.length) {                       // running: stop it and calm the fly down
    stopDemo();
    send({ cmd: 'body', action: 'stop' });
    send({ cmd: 'skill', action: 'stop' });
    return;
  }
  const steps = [[0, 'reset'], [600, 'sugar'], [6000, 'loom', 'left'], [9500, 'bitter'], [14500, 'loom', 'right']];
  demoTimers = steps.map(([t, action, side]) => setTimeout(() => send({ cmd: 'body', action, side }), t));
  const learned = [...document.querySelectorAll('#skill-list .skill-item')].length;
  if (learned) demoTimers.push(setTimeout(() => document.querySelector('#skill-list .skill-item .btn')?.click(), 18000));
  demoTimers.push(setTimeout(stopDemo, learned ? 21000 : 18000));
  demoBtn.textContent = '■ Stop demo';
  demoBtn.classList.add('running');
};

// ------------------------------------------------------------------ skills: learned movements
const skillList = document.getElementById('skill-list');
const skillLive = document.getElementById('skill-live');
function renderSkills(list) {
  skillList.innerHTML = list.length ? '' : '<div class="empty">None yet. Ask Fly in chat to learn a movement, e.g. “learn to tap your front left leg, then type hi”.</div>';
  for (const s of list) {
    const d = document.createElement('div');
    d.className = 'skill-item';
    d.innerHTML = `<div class="grow"><div class="t">${esc(s.name.replace(/_/g, ' '))}</div><div class="m">${esc(s.description)} · score ${(s.score ?? 0).toFixed(2)} · via ${esc(s.levels.join(', '))}${s.uses ? ` · used ${s.uses}×` : ''}</div></div>
      <button class="btn" title="Do it">▶</button><button class="link" title="Forget this skill">×</button>`;
    const [play, forget] = d.querySelectorAll('button');
    play.onclick = () => send({ cmd: 'skill', action: 'play', name: s.name });
    forget.onclick = () => { if (confirm(`Forget “${s.name}”?`)) send({ cmd: 'skill', action: 'forget', name: s.name }); };
    skillList.append(d);
  }
}
fetch('api/skills').then(r => r.json()).then(r => renderSkills(r.skills));
const LEVEL_SHORT = { descending: 'descending', premotor: 'premotor', motor: 'motor neuron' };
function onSkill(m) {
  if (m.type === 'skills') return renderSkills(m.skills);
  if (m.event === 'search') {
    skillLive.hidden = false;
    skillLive.innerHTML = `<div class="t"><span class="spinner"></span> Learning “${esc(m.name.replace(/_/g, ' '))}”: searching the wiring for neurons to drive…</div>`;
  } else if (m.event === 'run') {
    highlightNeurons(m.neurons, Math.max(2, m.duration + 1));
    if (m.kind === 'learn') {
      skillLive.hidden = false;
      const rows = m.drivers.map(d => `<div class="drv"><span class="lvl ${d.level}">${LEVEL_SHORT[d.level]}</span> <b>${esc(d.channel)}</b> ← ${esc(d.types.slice(0, 3).join(', '))}${d.gain !== 1 ? ` <span class="m">×${d.gain}</span>` : ''}</div>`).join('');
      skillLive.innerHTML = `<div class="t"><span class="spinner"></span> Practising “${esc(m.name.replace(/_/g, ' '))}”, try ${m.attempt}</div>${rows}<div class="scores" id="skill-scores">${skillLive.dataset.scores || ''}</div>`;
    }
  } else if (m.event === 'attempt') {
    skillLive.dataset.scores = (skillLive.dataset.scores || '') + `<span>try ${m.attempt}: ${m.score.toFixed(2)}</span>`;
    const sc = document.getElementById('skill-scores');
    if (sc) sc.innerHTML = skillLive.dataset.scores;
  } else if (m.event === 'learned') {
    skillLive.innerHTML = `<div class="t">Learned “${esc(m.name.replace(/_/g, ' '))}” in ${m.seconds} s · best score ${m.score.toFixed(2)}</div><div class="scores">${skillLive.dataset.scores || ''}</div>`;
    skillLive.dataset.scores = '';
    fetch('api/skills').then(r => r.json()).then(r => renderSkills(r.skills));
    setTimeout(() => { skillLive.hidden = true; }, 15000);
  } else if (m.event === 'error') {
    skillLive.hidden = false;
    skillLive.innerHTML = `<div class="t bad">${esc(m.error)}</div>`;
  }
}

// "why did it do that?": the traced pathway, as a card and lit up in the brain
const decisionEl = document.getElementById('decision');
let hlSet = [], hlFade = 0, decisionTimer = 0;
const chainLines = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x7fe8ff, transparent: true, opacity: 0.7, depthWrite: false, blending: THREE.AdditiveBlending }));
chainLines.visible = false;
scene.add(chainLines);
function highlightNeurons(list, seconds = 8, groups = null) {
  for (const i of hlSet) highlight[i] = 0;
  hlSet = list.slice();
  for (const i of hlSet) highlight[i] = 1;
  hlAttr.needsUpdate = true;
  hlFade = seconds;
  const pts = [];
  if (groups) for (let g = 0; g + 1 < groups.length; g++) {
    const a = groups[g], b = groups[g + 1];
    for (let k = 0; k < Math.min(24, a.length * b.length); k++) {
      const i = a[k % a.length], j = b[Math.floor(k / a.length) % b.length];
      pts.push(...positions.subarray(i * 3, i * 3 + 3), ...positions.subarray(j * 3, j * 3 + 3));
    }
  }
  chainLines.geometry.dispose();
  chainLines.geometry = new THREE.BufferGeometry();
  chainLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  chainLines.visible = pts.length > 0;
}
function onDecision(d) {
  if (!d.chain || !d.chain.length) return;
  const direct = d.chain.length === 1;
  const steps = d.chain.map((g, k) => `
    <div class="step${g.sensory ? ' sensory' : ''}" title="${esc(g.region)}${g.share ? ` · supplies ${Math.round(g.share * 100)}% of the next cell's excitation` : ''}">
      <div class="t">${esc(g.type)}${g.n > 1 ? ` <span class="n">×${g.n}</span>` : ''}<span class="hz">${g.hz} Hz</span></div>
      <div class="m">${esc(g.region)}</div>
    </div>${k < d.chain.length - 1 ? '<div class="arrow">↓</div>' : ''}`).join('');
  decisionEl.innerHTML = `
    <div class="head"><span class="title">Why: ${esc(d.title)}</span><button class="x" aria-label="Close">×</button></div>
    <div class="because">${esc(d.why)}${direct ? ' (you activated these neurons directly)' : ''}</div>
    <div class="chain">${steps}<div class="arrow">↓</div><div class="step body">${esc(d.title)}</div></div>
    ${d.inhibitor ? `<div class="inhib">Pushing against it: <b>${esc(d.inhibitor.type)}</b> (inhibition ${Math.round(d.inhibitor.vs_excitation * 100)}% as strong as the excitation)</div>` : ''}
    <div class="foot">Traced backwards through the connectome: at each step, the cell type supplying the most excitation right now (firing rate × synapses). Lit up in cyan in the brain.</div>`;
  decisionEl.hidden = false;
  decisionEl.querySelector('.x').onclick = () => { decisionEl.hidden = true; };
  const groups = d.chain.map(g => g.neurons);
  highlightNeurons(groups.flat(), 9, groups);
  clearTimeout(decisionTimer);
  decisionTimer = setTimeout(() => { decisionEl.hidden = true; }, 12000);
}

// ------------------------------------------------------------------ render loop
addEventListener('resize', resize);
resize();
let last = performance.now();
function tick(now) {
  const dt = Math.min(now - last, 100);
  last = now;
  const decay = Math.exp(-dt / 140);
  for (let i = 0; i < N; i++) {
    const a = activity[i];
    if (a > 0.002) activity[i] = a * decay; else if (a !== 0) activity[i] = 0;
  }
  actAttr.needsUpdate = true;
  for (let r = 0; r < NR; r++) regionShellMats[r].opacity = 0.03 + regionLevel[r] * 0.08;
  const pulse = 1 + 0.25 * Math.sin(now / 180);
  selMarker.scale.setScalar(pulse);
  controls.update();
  if (hlFade > 0) {
    hlFade = Math.max(0, hlFade - dt / 1000);
    const v = Math.min(1, hlFade / 2);
    for (const i of hlSet) highlight[i] = v;
    hlAttr.needsUpdate = true;
    chainLines.material.opacity = 0.7 * v;
    if (hlFade === 0) { for (const i of hlSet) highlight[i] = 0; hlSet = []; chainLines.visible = false; }
  }
  renderer.setScissorTest(true);
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setScissor(0, 0, innerWidth, innerHeight);
  renderer.clear();
  const draw = (r, sc, cam) => {
    if (!r) return;
    const y = innerHeight - r.y - r.h;
    renderer.setViewport(r.x, y, r.w, r.h);
    renderer.setScissor(r.x, y, r.w, r.h);
    renderer.render(sc, cam);
  };
  if (flyView && rects.fly) { flyView.update(dt / 1000); draw(rects.fly, flyView.scene, flyView.camera); }
  draw(rects.brain, scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
window.flybrain = { send, select, camera, controls, highlightNeurons };
