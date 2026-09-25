// Chat with the agent (open LLM + Laya System 1), model setup, and the body-link feed.
const $ = id => document.getElementById(id);
const send = cmd => window.flybrain?.send(cmd);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtBytes = b => b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.round(b / 1e3) + ' kB';
const fmtN = n => (n ?? 0).toLocaleString();

// ------------------------------------------------------------------ tabs
const left = $('left');
function showTab(name) {
  document.querySelectorAll('.tabs [data-tab]').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  $('tab-chat').hidden = name !== 'chat';
  $('tab-senses').hidden = name !== 'senses';
  left.classList.toggle('wide', name === 'chat');
  document.body.classList.toggle('chat-open', name === 'chat');
  try { localStorage.setItem('flybrain-tab', name); } catch (e) { /* storage unavailable */ }
}
document.querySelectorAll('.tabs [data-tab]').forEach(b => b.onclick = () => showTab(b.dataset.tab));
let startTab = 'chat';
try { startTab = localStorage.getItem('flybrain-tab') || 'chat'; } catch (e) { /* default */ }
showTab(startTab);

// ------------------------------------------------------------------ markdown (small, safe: escapes first)
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src) {
  const parts = String(src || '').split('```');
  let html = '';
  parts.forEach((part, k) => {
    if (k % 2 === 1) {
      const nl = part.indexOf('\n');
      const lang = nl > 0 ? part.slice(0, nl).trim() : '';
      const code = nl >= 0 ? part.slice(nl + 1) : part;
      html += `<div class="code"><div class="code-head"><span>${esc(lang || 'code')}</span><button class="copy" type="button">Copy</button></div><pre>${esc(code.replace(/\n$/, ''))}</pre></div>`;
      return;
    }
    const blocks = esc(part).split(/\n{2,}/);
    for (const b of blocks) {
      const t = b.trim();
      if (!t) continue;
      const lines = t.split('\n');
      if (lines.every(l => /^\s*[-*] /.test(l))) {
        html += '<ul>' + lines.map(l => `<li>${inline(l.replace(/^\s*[-*] /, ''))}</li>`).join('') + '</ul>';
      } else if (lines.every(l => /^\s*\d+[.)] /.test(l))) {
        html += '<ol>' + lines.map(l => `<li>${inline(l.replace(/^\s*\d+[.)] /, ''))}</li>`).join('') + '</ol>';
      } else if (/^#{1,4} /.test(t) && lines.length === 1) {
        html += `<h4>${inline(t.replace(/^#+ /, ''))}</h4>`;
      } else {
        html += `<p>${lines.map(inline).join('<br>')}</p>`;
      }
    }
  });
  return html;
}
$('chat-log').addEventListener('click', e => {
  const b = e.target.closest('.copy');
  if (!b) return;
  navigator.clipboard?.writeText(b.closest('.code').querySelector('pre').textContent);
  b.textContent = 'Copied';
  setTimeout(() => { b.textContent = 'Copy'; }, 1200);
});

// ------------------------------------------------------------------ chat log
const log = $('chat-log');
const streams = new Map();
let stickToBottom = true;
log.addEventListener('scroll', () => { stickToBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60; });
function scroll() { if (stickToBottom) log.scrollTop = log.scrollHeight; }
function add(html, cls) {
  const d = document.createElement('div');
  d.className = cls;
  d.innerHTML = html;
  log.append(d);
  scroll();
  return d;
}
function emptyState() {
  if (log.children.length) return;
  add(`<p><strong>Fly</strong> is an open language model running on this computer, with a fast System 1 (Laya) in front of it and the fly brain as its body.</p>
       <p>Try: <em>“Design a URL shortener that scales”</em> · <em>“Write a Python script that finds primes and run it”</em> · <em>“Taste some sugar and tell me what your brain did”</em></p>`, 'hello');
}

const INTENT = { chat: 'Chat', question: 'Question', code: 'Code', design: 'Design', brain: 'Brain' };
function renderItem(it) {
  switch (it.kind) {
    case 'user':
      log.querySelector('.hello')?.remove();
      add(esc(it.text).replace(/\n/g, '<br>'), 'msg user');
      break;
    case 'system1': {
      const conf = it.intent_conf != null ? ` ${Math.round(it.intent_conf * 100)}%` : '';
      const depth = ['instant', 'some thought', 'deep'][Math.min(2, Math.round(it.depth))];
      const probs = Object.entries(it.intent_probs || {}).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(', ');
      add(`<span class="s1-tag ${it.source}">${it.source === 'laya' ? 'System 1 · Laya' : 'System 1 · rules'}</span>
           <span>${INTENT[it.intent] || it.intent}${conf}</span>
           <span>${depth} → thinking ${it.thinking ? 'on' : 'off'}</span>
           <span>tone: ${esc(it.tone)}</span>
           <span class="ms">${it.ms} ms</span>`, 's1').title = probs ? `Intent probabilities: ${probs}` : 'Keyword rules (Laya not loaded)';
      break;
    }
    case 'assistant': {
      let s = streams.get(it.id);
      if (!s) s = newStream(it.id, false);
      s.reasoning = it.reasoning || '';
      s.content = it.text || '';
      finishStream(s, it.stats);
      break;
    }
    case 'tool': {
      const pending = log.querySelector('.tool.pending');
      pending?.remove();
      const arg = it.name === 'run_command' ? it.args?.command : (it.args?.path || it.args?.sense || it.args?.query || it.args?.note || '');
      const extra = it.exit_code !== undefined ? ` · exit ${it.exit_code ?? 'timeout'} · ${it.seconds}s` : '';
      const d = add(`<details><summary><span class="dot ${it.ok ? 'ok' : 'bad'}"></span><strong>${esc(it.name)}</strong> <code>${esc(String(arg ?? '').slice(0, 80))}</code><span class="ms">${extra}</span></summary><pre>${esc(it.result)}</pre></details>`, 'tool');
      if (it.name === 'write_file' && it.args?.content) {
        d.querySelector('pre').textContent = it.args.content;
      }
      break;
    }
    case 'error':
      add(esc(it.text), 'sys bad');
      break;
    case 'kv':
      kvNotice(it);
      break;
    case 'notice':
      add(esc(it.text), 'sys notice');
      break;
  }
}
function kvNotice(m) {
  const txt = m.event === 'saved'
    ? `KV cache saved to disk: ${fmtN(m.tokens)} tokens (${fmtBytes(m.bytes)})`
    : `KV cache restored from disk: ${fmtN(m.tokens)} tokens in ${m.ms} ms, so the model didn't have to re-read this conversation`;
  const last = log.lastElementChild;
  if (m.event === 'saved' && last?.classList.contains('kv') && last.dataset.event === 'saved') last.innerHTML = esc(txt);
  else add(esc(txt), 'sys kv').dataset.event = m.event;
}
function newStream(id, live) {
  const el = add(`<details class="think" hidden><summary>Thinking</summary><div class="think-body"></div></details><div class="body"></div><div class="stats"></div>`, 'msg bot' + (live ? ' live' : ''));
  const s = { id, el, reasoning: '', content: '', dirty: false };
  streams.set(id, s);
  return s;
}
function paint(s) {
  const t = s.el.querySelector('.think');
  if (s.reasoning) {
    t.hidden = false;
    s.el.querySelector('.think-body').textContent = s.reasoning;
  }
  s.el.querySelector('.body').innerHTML = md(s.content);
  scroll();
}
function finishStream(s, stats) {
  s.el.classList.remove('live');
  s.el.querySelector('.think').open = false;
  if (!s.reasoning) s.el.querySelector('.think').hidden = true;
  paint(s);
  if (!s.content && !s.reasoning) { s.el.remove(); return; }
  if (stats && stats.cache_n != null) {
    const reused = stats.cache_n, fresh = stats.prompt_n;
    s.el.querySelector('.stats').innerHTML =
      `<span title="Tokens whose keys and values were already in the KV cache, so the model skipped them">KV cache reused ${fmtN(reused)}</span> · processed ${fmtN(fresh)} new · ${stats.tok_s} tok/s · ${stats.seconds}s` +
      (stats.context_size ? ` · <span title="Tokens in the model's context window">context ${fmtN(stats.context_used)} / ${fmtN(stats.context_size)}</span>` : '');
  }
}
let raf = 0;
function schedulePaint() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    for (const s of streams.values()) if (s.dirty) { s.dirty = false; paint(s); }
  });
}

// ------------------------------------------------------------------ incoming events
let busy = false;
let rt = null;
addEventListener('flybrain:msg', ev => {
  const m = ev.detail;
  if (m.type === 'agent_status') return onAgentStatus(m);
  if (m.type === 'bridge') return onBridge(m);
  if (m.type === 'conversation') return loadConversation(m);
  if (m.type === 'kv') return kvNotice(m);
  if (m.type !== 'agent') return;
  switch (m.kind) {
    case 'stream_start': {
      const s = newStream(m.id, true);
      if (m.thinking) { const t = s.el.querySelector('.think'); t.hidden = false; t.open = true; }
      break;
    }
    case 'delta': {
      const s = streams.get(m.id);
      if (!s) break;
      if (m.field === 'reasoning') s.reasoning += m.text; else {
        if (!s.content) s.el.querySelector('.think').open = false;
        s.content += m.text;
      }
      s.dirty = true;
      schedulePaint();
      break;
    }
    case 'stream_abort':
      streams.get(m.id)?.el.remove();
      streams.delete(m.id);
      break;
    case 'tool_start':
      add(`<span class="spinner"></span> Running <strong>${esc(m.name)}</strong>…`, 'tool pending');
      break;
    case 'approval': {
      const d = add(`<div>Fly wants to run a command in its workspace:</div><pre>${esc(m.args?.command ?? JSON.stringify(m.args))}</pre>
        <div class="approval-actions"><button class="btn primary" data-a="allow">Run it</button><button class="btn" data-a="deny">Don't run</button><button class="btn" data-a="always">Always allow</button></div>`, 'approval');
      d.querySelectorAll('button').forEach(b => b.onclick = () => {
        const a = b.dataset.a;
        send({ cmd: 'approve', call_id: m.call_id, allow: a !== 'deny', always: a === 'always' });
        if (a === 'always') $('auto-approve').checked = true;
        d.querySelectorAll('button').forEach(x => { x.disabled = true; });
        b.classList.add('chosen');
      });
      break;
    }
    case 'idle':
      setBusy(false);
      refreshHistory();
      break;
    default:
      renderItem(m);
  }
});
addEventListener('flybrain:open', () => { refreshConversation(); refreshHistory(); });

// ------------------------------------------------------------------ model card
const card = $('model-card');
let chooserOpen = false;
function onAgentStatus(s) {
  rt = s.runtime;
  setBusy(s.busy);
  $('auto-approve').checked = s.auto_approve;
  $('bridge-on').checked = s.bridge;
  if (s.conversation) $('chat-title').textContent = s.conversation.title;
  renderCard(s);
}
function hwLine(r) {
  const g = r.hardware.gpus?.[0];
  const gpu = g ? `${esc(g.name)} (${(g.total_mb / 1024).toFixed(0)} GB, ${esc(r.backend)})`
    : r.state === 'installing' ? 'checking the GPU…'
    : r.engine ? 'no GPU found, will use the CPU' : (r.hardware.os === 'darwin' ? 'Apple GPU (Metal)' : 'GPU not checked yet');
  return `${gpu} · ${r.hardware.ram_gb} GB RAM`;
}
function renderCard(s) {
  const r = s.runtime;
  const s1 = s.system1;
  const s1dot = { ready: 'ok', loading: 'wait', waiting: '' }[s1.state] ?? 'warn';
  const s1line = `<div class="s1line"><span class="dot ${s1dot}"></span>System 1: ${esc(s1.detail)}</div>`;
  if (r.state === 'ready' && !chooserOpen) {
    card.innerHTML = `<div class="ready"><span class="dot ok"></span><span class="grow">${esc(r.detail)}</span><button class="link" id="change-model">Change</button></div>${s1line}`;
    $('change-model').onclick = () => { chooserOpen = true; renderCard(s); };
    return;
  }
  if (r.state === 'external' && !chooserOpen) {
    card.innerHTML = `<div class="ready"><span class="dot ok"></span><span class="grow">${esc(r.model_label)}</span><button class="link" id="change-model">Change</button></div>${s1line}`;
    $('change-model').onclick = () => { chooserOpen = true; renderCard(s); };
    return;
  }
  if (['installing', 'downloading', 'starting'].includes(r.state)) {
    const p = r.progress;
    const pct = p && p.total ? (100 * p.done / p.total) : null;
    card.innerHTML = `<div class="setup"><div class="grow">${esc(r.detail)}</div>
      ${pct != null ? `<div class="bar big"><i style="width:${pct.toFixed(1)}%"></i></div><div class="hint">${fmtBytes(p.done)} of ${fmtBytes(p.total)} · ${pct.toFixed(0)}%</div>` : '<div class="hint"><span class="spinner"></span> working…</div>'}
      </div>${s1line}`;
    return;
  }
  const options = Object.entries(r.models).map(([k, m]) =>
    `<option value="${k}" ${k === (chooserOpen ? r.model : r.recommended) ? 'selected' : ''}>${esc(m.label)} · ${m.size_gb} GB${k === r.recommended ? ' · recommended' : ''}${r.installed[k] ? ' · downloaded' : ''}</option>`).join('');
  card.innerHTML = `<div class="setup">
    ${r.state === 'error' ? `<div class="err">${esc(r.detail)}</div>` : ''}
    ${r.note ? `<div class="hint">${esc(r.note)}</div>` : ''}
    <div class="setup-title">Language model</div>
    <div class="hint">This computer: ${hwLine(r)}</div>
    <div class="hint">Recommended: ${esc(r.models[r.recommended]?.label)}, because ${esc(r.reason)}.</div>
    <select id="model-pick">${options}</select>
    <div class="hint" id="model-about"></div>
    <div class="setup-actions"><button class="btn primary" id="model-go"></button>${chooserOpen ? '<button class="btn" id="model-cancel">Cancel</button>' : ''}</div>
    <div class="hint">Runs entirely on this computer with llama.cpp. Models download once from Hugging Face.</div>
    <details class="custom-model" ${r.custom ? 'open' : ''}>
      <summary>Use another model</summary>
      <div class="hint">Any GGUF model on Hugging Face (runs here with llama.cpp):</div>
      <div class="cm-row"><input id="cm-hf" placeholder="owner/repo or owner/repo:Q4_K_M" value="${r.custom?.kind === 'hf' ? esc(r.custom.value) : ''}"><button class="btn" id="cm-hf-go">Use</button></div>
      <div class="hint">A GGUF file already on this computer:</div>
      <div class="cm-row"><input id="cm-path" placeholder="/path/to/model.gguf" value="${r.custom?.kind === 'path' ? esc(r.custom.value) : ''}"><button class="btn" id="cm-path-go">Use</button></div>
      <div class="hint">Any OpenAI-compatible server (LM Studio, Ollama, vLLM, a hosted API):</div>
      <input id="cm-url" placeholder="http://localhost:11434/v1" value="${r.custom?.kind === 'server' ? esc(r.custom.value) : ''}">
      <div class="cm-row"><input id="cm-model" placeholder="model name, e.g. llama3.1" value="${esc(r.custom?.model || '')}"><input id="cm-key" type="password" placeholder="${r.custom?.has_key ? 'API key saved' : 'API key (optional)'}"></div>
      <div class="cm-row"><button class="btn" id="cm-url-go">Connect</button></div>
      <div class="hint">Saved for this computer only (data/settings), never in git.</div>
    </details>
  </div>${s1line}`;
  const pick = $('model-pick');
  const upd = () => {
    const k = pick.value, m = r.models[k];
    $('model-about').textContent = m.about;
    $('model-go').textContent = (r.state === 'error' ? 'Try again: ' : '') + (r.installed[k] ? 'Start' : `Download (${m.size_gb} GB) and start`);
  };
  pick.onchange = upd;
  upd();
  $('model-go').onclick = () => { chooserOpen = false; send({ cmd: 'model_setup', model: pick.value }); };
  const custom = (kind, value, extra = {}) => { if (!value.trim()) return; chooserOpen = false; send({ cmd: 'model_custom', kind, value, ...extra }); };
  $('cm-hf-go').onclick = () => custom('hf', $('cm-hf').value);
  $('cm-path-go').onclick = () => custom('path', $('cm-path').value);
  $('cm-url-go').onclick = () => custom('server', $('cm-url').value, { model: $('cm-model').value, api_key: $('cm-key').value });
  if (chooserOpen) $('model-cancel').onclick = () => { chooserOpen = false; renderCard(s); };
}

// ------------------------------------------------------------------ composer
const prompt = $('prompt');
const sendBtn = $('send');
function setBusy(b) {
  busy = b;
  sendBtn.textContent = b ? 'Stop' : 'Send';
  sendBtn.classList.toggle('primary', !b);
}
$('composer').onsubmit = e => {
  e.preventDefault();
  if (busy) { send({ cmd: 'stop' }); return; }
  const text = prompt.value.trim();
  if (!text) return;
  send({ cmd: 'chat', text });
  prompt.value = '';
  setBusy(true);
  stickToBottom = true;
};
prompt.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); }
});
$('auto-approve').onchange = e => send({ cmd: 'auto_approve', on: e.target.checked });
$('bridge-on').onchange = e => send({ cmd: 'bridge', on: e.target.checked });
$('new-chat').onclick = () => send({ cmd: 'new_chat' });
$('history').onchange = e => { if (e.target.value) send({ cmd: 'open_chat', id: e.target.value }); e.target.value = ''; };

// ------------------------------------------------------------------ conversations
function loadConversation(c) {
  log.innerHTML = '';
  streams.clear();
  $('chat-title').textContent = c.title;
  for (const it of c.log) renderItem(it);
  emptyState();
  stickToBottom = true;
  scroll();
}
async function refreshConversation() {
  try { loadConversation(await fetch('api/agent/conversation').then(r => r.json())); } catch (e) { /* server restarting */ }
}
async function refreshHistory() {
  try {
    const list = await fetch('api/agent/conversations').then(r => r.json());
    $('history').innerHTML = '<option value="">History</option>' + list.map(c => `<option value="${c.id}">${esc(c.title)} (${c.turns})</option>`).join('');
  } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------ body link feed
const feed = $('bridge-feed');
function onBridge(m) {
  feed.querySelector('.empty')?.remove();
  const d = document.createElement('div');
  d.className = 'bridge-item fresh';
  d.title = m.why;
  d.innerHTML = `<span class="t">${esc(m.what)}</span><span class="r">${fmtN(m.neurons)} neurons</span><div class="why">${esc(m.why)}</div>`;
  feed.prepend(d);
  setTimeout(() => d.classList.remove('fresh'), 1500);
  while (feed.children.length > 6) feed.lastElementChild.remove();
}
