// The activity view: a spike raster and firing-rate traces for neuron groups from the senses through
// the brain's learning and command neurons to the motor neurons, like an electrophysiology rig's
// display. Groups come from the server (api/activity_groups); more cell types can be added.

const WINDOW_MS = 4000;          // raster time span
const RATE_MS = 10000;           // rate plot time span
const BIN_MS = 100;              // rate bin
const STAGE_COL = { sense: '#6fb6ff', brain: '#c792ea', command: '#ffb35c', motor: '#7ee787', custom: '#f0f0f0' };

export function createActivityView(root, { N, onPickType }) {
  root.innerHTML = `
    <div class="act-head">
      <b>Activity</b><span class="m">spike raster (a sample of each group's neurons) and group firing rates</span>
      <form class="act-add"><input placeholder="add a cell type, e.g. LC4, MBON01, DNg62" autocomplete="off"><button class="btn">Add</button></form>
    </div>
    <div class="act-body">
      <div class="act-labels"></div>
      <canvas class="act-raster"></canvas>
    </div>
    <div class="act-rates-head"><span class="m">mean firing rate per group (Hz), last 10 s · click a label to hide its trace</span></div>
    <canvas class="act-rates"></canvas>`;
  const raster = root.querySelector('.act-raster'), rates = root.querySelector('.act-rates');
  const labelsEl = root.querySelector('.act-labels');
  let groups = [];
  let rowOf = new Int32Array(N).fill(-1);     // neuron -> raster row
  let memberOf = [];                          // neuron -> list of group indices (built as a Map for sparse use)
  let rows = 0;
  let events = [];                            // [t_ms, row]
  let counts = [];                            // per group: spikes in the current bin
  let history = [];                           // per group: [t_ms, Hz]
  let binStart = null, lastT = 0;
  const hidden = new Set();

  function rebuild() {
    rowOf = new Int32Array(N).fill(-1);
    memberOf = new Map();
    rows = 0;
    labelsEl.innerHTML = '';
    groups.forEach((g, gi) => {
      g.row0 = rows;
      g.slots = 12;                                    // equal bands: a pair of command neurons is as readable as 4,500 Kenyon cells
      const pad = Math.floor((g.slots - g.sample.length) / 2);
      g.sample.forEach((n, k) => { rowOf[n] = g.row0 + pad + k; });
      rows += g.slots;
      for (const n of g.neurons) {
        const l = memberOf.get(n);
        if (l) l.push(gi); else memberOf.set(n, [gi]);
      }
      const d = document.createElement('div');
      d.className = 'act-label';
      d.style.setProperty('--c', STAGE_COL[g.stage] || '#ddd');
      d.innerHTML = `<i></i><span>${g.label}</span><em>${g.neurons.length.toLocaleString()}</em>`;
      d.title = `${g.label}: ${g.neurons.length} neurons (${g.sample.length} shown in the raster)\nclick: hide/show its rate trace · double-click: find them in the brain`;
      d.onclick = () => { hidden.has(g.key) ? hidden.delete(g.key) : hidden.add(g.key); d.classList.toggle('off', hidden.has(g.key)); };
      d.ondblclick = () => onPickType && onPickType(g.neurons);
      labelsEl.append(d);
      g.el = d;
    });
    counts = groups.map(() => 0);
    history = groups.map(() => []);
    events = [];
    layoutLabels();
  }

  function layoutLabels() {
    const h = raster.clientHeight;
    groups.forEach(g => {
      const y0 = g.row0 / Math.max(rows, 1) * h, y1 = (g.row0 + g.slots) / Math.max(rows, 1) * h;
      Object.assign(g.el.style, { top: y0 + 'px', height: Math.max(12, y1 - y0) + 'px' });
    });
  }

  fetch('api/activity_groups').then(r => r.json()).then(gs => { groups = gs; rebuild(); });

  root.querySelector('.act-add').onsubmit = async ev => {
    ev.preventDefault();
    const inp = root.querySelector('.act-add input');
    const name = inp.value.trim();
    if (!name) return;
    const d = await fetch('api/type/' + encodeURIComponent(name)).then(r => r.json()).catch(() => null);
    if (!d || !d.neurons.length) { inp.value = ''; inp.placeholder = `no cell type "${name}"`; return; }
    const pick = d.neurons.slice().sort(() => Math.random() - 0.5).slice(0, 12).sort((a, b) => a - b);
    groups.push({ key: 'custom_' + name, label: name, stage: 'custom', neurons: d.neurons, sample: pick });
    inp.value = '';
    rebuild();
  };

  // called with each frame of spikes from the server
  function onSpikes(t, idx) {
    if (!groups.length) return;
    if (t < lastT) { events = []; history = groups.map(() => []); binStart = null; }   // the brain was reset
    lastT = t;
    if (binStart === null) binStart = t;
    for (let k = 0; k < idx.length; k++) {
      const n = idx[k];
      const r = rowOf[n];
      if (r >= 0) events.push([t, r]);
      const gl = memberOf.get(n);
      if (gl) for (const gi of gl) counts[gi]++;
    }
    if (t - binStart >= BIN_MS) {
      const dt = (t - binStart) / 1000;
      groups.forEach((g, gi) => {
        history[gi].push([t, counts[gi] / g.neurons.length / dt]);
        while (history[gi].length && history[gi][0][0] < t - RATE_MS) history[gi].shift();
        counts[gi] = 0;
      });
      binStart = t;
    }
    let cut = 0;
    while (cut < events.length && events[cut][0] < t - WINDOW_MS) cut++;
    if (cut) events.splice(0, cut);
  }

  function fit(c) {
    const r = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
    const g = c.getContext('2d');
    g.setTransform(r, 0, 0, r, 0, 0);
    return [g, w, h];
  }

  function draw() {
    if (root.hidden || !root.clientWidth) return;
    // raster
    let [g, w, h] = fit(raster);
    g.clearRect(0, 0, w, h);
    const rh = h / Math.max(rows, 1);
    groups.forEach((gr, gi) => {
      if (gi % 2) { g.fillStyle = 'rgba(255,255,255,0.025)'; g.fillRect(0, gr.row0 * rh, w, gr.slots * rh); }
    });
    const t1 = lastT, t0 = t1 - WINDOW_MS;
    const colOfRow = new Array(rows);
    groups.forEach(gr => { for (let k = 0; k < gr.slots; k++) colOfRow[gr.row0 + k] = STAGE_COL[gr.stage] || '#ddd'; });
    for (const [t, r] of events) {
      g.fillStyle = colOfRow[r];
      g.fillRect((t - t0) / WINDOW_MS * w, r * rh + rh * 0.15, 1.5, Math.max(1, rh * 0.7));
    }
    g.fillStyle = 'rgba(255,255,255,0.35)';
    g.font = '10px system-ui';
    for (let s = Math.ceil(t0 / 1000); s * 1000 <= t1; s++) {
      const x = (s * 1000 - t0) / WINDOW_MS * w;
      g.fillRect(x, 0, 1, h);
      g.fillText(s + ' s', x + 3, h - 3);
    }
    layoutLabels();
    // rates
    [g, w, h] = fit(rates);
    g.clearRect(0, 0, w, h);
    let top = 5;
    groups.forEach((gr, gi) => { if (!hidden.has(gr.key)) for (const [, v] of history[gi]) top = Math.max(top, v); });
    top = Math.pow(10, Math.ceil(Math.log10(top) * 4) / 4);
    g.fillStyle = 'rgba(255,255,255,0.4)';
    g.fillText(`${top.toFixed(top < 10 ? 1 : 0)} Hz`, 4, 11);
    const r0 = t1 - RATE_MS;
    const used = [];
    groups.forEach((gr, gi) => {
      if (hidden.has(gr.key) || history[gi].length < 2) return;
      g.strokeStyle = STAGE_COL[gr.stage] || '#ddd';
      g.globalAlpha = 0.85;
      g.lineWidth = 1.3;
      g.beginPath();
      history[gi].forEach(([t, v], k) => {
        const x = (t - r0) / RATE_MS * w, y = h - 2 - (v / top) * (h - 16);
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
      const [tl, vl] = history[gi][history[gi].length - 1];
      const ly = h - 4 - (vl / top) * (h - 16);
      if (vl > top * 0.08 && used.every(u => Math.abs(u - ly) > 11)) {
        used.push(ly);
        g.fillStyle = g.strokeStyle;
        g.fillText(gr.label.split(' (')[0], Math.min(w - 120, (tl - r0) / RATE_MS * w - 118), ly);
      }
    });
    g.globalAlpha = 1;
  }

  return { onSpikes, draw };
}
