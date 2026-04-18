// ── State ──────────────────────────────────────────────────────────────────────

const state = {
  settings: {},
  poses:    [],
  triggers: [],
  actions:  [],
  dirty:    false,
};

// Live camera state from SSE
let liveFingers = { right: null, left: null };  // each: [T,I,M,R,P] or null
let livePose    = { right: null, left: null };
let liveSpreads = { right: null, left: null };  // each: {thumbIndex, indexMiddle, middleRing, ringPinky}

// Current modal context
let modalCtx = null;   // { type, index, data }

// Which finger the capture button should write to (set when a pose editor is open)
let captureTarget = null;  // function(fingers) or null

// Mutable step list for the sequence trigger editor
let editingSteps = [];

// Mutable prefix step list for the sequenced_continuous trigger editor
let editingPrefixSteps = [];

// Mutable require-pose list for the trigger editor
let editingRequire = [];


// ── Boot ──────────────────────────────────────────────────────────────────────

async function loadConfig() {
  const res  = await fetch("/api/config");
  const data = await res.json();
  state.settings = data.settings || {};
  state.poses    = data.poses    || [];
  state.triggers = data.triggers || [];
  state.actions  = data.actions  || [];
  renderAll();
  loadCameras();
}

function renderAll() {
  renderPoses();
  renderTriggers();
  renderActions();
  renderSettings();
}

function renderSettings() {
  const dwellEl    = document.getElementById("setting-dwell-ms");
  const spreadEl   = document.getElementById("setting-spread-threshold");
  if (dwellEl)  dwellEl.value  = state.settings.dwell_ms          ?? 200;
  if (spreadEl) spreadEl.value = state.settings.spread_threshold   ?? 0.20;
}

function onSettingChange(key, value) {
  if (isNaN(value)) return;
  state.settings[key] = value;
  markDirty();
}

document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  startSSE();
  startFramePolling();
});


// ── Camera frame polling ───────────────────────────────────────────────────────

function startFramePolling() {
  const img = document.getElementById("camera-feed");
  let prevUrl = null;

  async function poll() {
    try {
      const res = await fetch("/api/frame");
      if (res.ok) {
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        img.src = url;
        if (prevUrl) URL.revokeObjectURL(prevUrl);
        prevUrl = url;
      }
    } catch {}
    setTimeout(poll, 50); // ~20 fps
  }

  poll();
}


// ── Camera selector ───────────────────────────────────────────────────────────

async function loadCameras() {
  const sel = document.getElementById("camera-select");
  try {
    const res     = await fetch("/api/cameras");
    const cameras = await res.json();

    const current = state.settings.camera ?? null;
    sel.innerHTML = cameras.length === 0
      ? `<option value="">No cameras found</option>`
      : cameras.map(c =>
          `<option value="${c.index}" ${c.index === current ? "selected" : ""}>${c.name} (${c.path})</option>`
        ).join("");
  } catch {
    sel.innerHTML = `<option value="">Could not load cameras</option>`;
  }
}

async function onCameraChange(value) {
  const index = parseInt(value, 10);
  if (isNaN(index)) return;

  const sel = document.getElementById("camera-select");
  sel.disabled = true;
  const prev = sel.title;
  try {
    const res  = await fetch("/api/set-camera", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ camera: index }),
    });
    const data = await res.json();
    if (data.ok) {
      state.settings.camera = index;
      if (data.restarted) {
        sel.title = "Engine restarting…";
        setTimeout(() => { sel.title = prev; }, 3000);
      }
    } else {
      alert("Failed to switch camera:\n" + data.error);
    }
  } finally {
    sel.disabled = false;
  }
}


// ── SSE: live hand state ───────────────────────────────────────────────────────

function startSSE() {
  const es = new EventSource("/state");
  es.onmessage = (evt) => {
    const data = JSON.parse(evt.data);

    if (data.error) {
      showCameraError(data.error);
      return;
    }

    hideCameraError();

    const hands = data.hands || {};
    liveFingers.right = hands.right ? hands.right.fingers : null;
    liveFingers.left  = hands.left  ? hands.left.fingers  : null;
    livePose.right    = hands.right ? hands.right.pose    : null;
    livePose.left     = hands.left  ? hands.left.pose     : null;
    liveSpreads.right = hands.right ? hands.right.spreads : null;
    liveSpreads.left  = hands.left  ? hands.left.spreads  : null;

    updateFingerDisplay("right", liveFingers.right, livePose.right);
    updateFingerDisplay("left",  liveFingers.left,  livePose.left);

    // Update the capture row inside an open pose modal
    updateCaptureRow();

    const hasHand = !!(liveFingers.right || liveFingers.left);
    document.getElementById("capture-btn").disabled = !hasHand;
    document.getElementById("capture-hint").classList.toggle("hidden", hasHand);
  };
  es.onerror = () => {
    // reconnects automatically
  };

  document.getElementById("capture-btn").addEventListener("click", () => {
    const fingers = liveFingers.right || liveFingers.left;
    if (!fingers) return;

    // If a pose editor modal is open, fill it in directly
    if (captureTarget) {
      captureTarget(fingers);
      return;
    }

    // Otherwise find a matching pose or create a new one
    const spreads = liveSpreads.right || liveSpreads.left || null;
    const matchIndex = state.poses.findIndex(p => poseMatchesState(p, fingers, spreads));
    if (matchIndex !== -1) {
      switchTab("poses");
      editPose(matchIndex);
    } else {
      switchTab("poses");
      captureNewPose(fingers, spreads);
    }
  });
}

function showCameraError(msg) {
  const el = document.getElementById("camera-error");
  document.getElementById("camera-error-text").textContent = msg;
  el.classList.remove("hidden");
}

function hideCameraError() {
  document.getElementById("camera-error").classList.add("hidden");
}

function updateFingerDisplay(side, fingers, pose) {
  const row = document.getElementById(`hand-${side}`);
  if (!fingers) {
    row.classList.remove("active");
    ["thumb","index","middle","ring","pinky"].forEach((f, i) => {
      const dot = document.getElementById(`${side[0]}-${f}`);
      dot.classList.remove("on");
    });
    document.getElementById(`${side[0]}-pose`).textContent = "---";
    document.getElementById(`${side[0]}-pose`).classList.remove("matched");
    return;
  }
  row.classList.add("active");
  const names = ["thumb","index","middle","ring","pinky"];
  names.forEach((f, i) => {
    const dot = document.getElementById(`${side[0]}-${f}`);
    dot.classList.toggle("on", fingers[i]);
  });
  const chip = document.getElementById(`${side[0]}-pose`);
  chip.textContent = pose || "---";
  chip.classList.toggle("matched", !!pose);
}


// ── Tab switching ──────────────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-pane").forEach(p => {
    p.classList.toggle("active", p.id === `tab-${name}`);
  });
}


// ── Dirty state ───────────────────────────────────────────────────────────────

function markDirty() {
  state.dirty = true;
  document.getElementById("dirty-badge").classList.remove("hidden");
  document.getElementById("save-btn").disabled = false;
}

function markClean() {
  state.dirty = false;
  document.getElementById("dirty-badge").classList.add("hidden");
  document.getElementById("save-btn").disabled = true;
}


// ── Save ──────────────────────────────────────────────────────────────────────

async function saveAll() {
  const [r1, r2] = await Promise.all([
    fetch("/api/config/triggers", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ settings: state.settings, poses: state.poses, triggers: state.triggers }),
    }),
    fetch("/api/config/actions", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ actions: state.actions }),
    }),
  ]);
  const d1 = await r1.json();
  const d2 = await r2.json();
  if (d1.ok && d2.ok) {
    markClean();
  } else {
    alert("Save failed:\n" + [d1.error, d2.error].filter(Boolean).join("\n"));
  }
}


// ── Renders ───────────────────────────────────────────────────────────────────

function renderPoses() {
  const list = document.getElementById("poses-list");
  list.innerHTML = "";
  state.poses.forEach((pose, i) => {
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `
      <div class="item-name">${esc(pose.name)}</div>
      <div class="item-fingers">${fingerDots(pose)}</div>
      <div class="item-actions">
        <button class="btn-icon" title="Edit"   onclick="editPose(${i})">✎</button>
        <button class="btn-icon del" title="Delete" onclick="deletePose(${i}, event)">✕</button>
      </div>`;
    list.appendChild(card);
  });
}

function renderTriggers() {
  const list = document.getElementById("triggers-list");
  list.innerHTML = "";
  state.triggers.forEach((t, i) => {
    const trig = t.trigger || {};
    const meta = triggerMeta(trig);
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `
      <div class="item-name">${esc(t.name)}</div>
      <div class="item-meta">${esc(meta)}</div>
      <div class="item-actions">
        <button class="btn-icon" title="Edit"   onclick="editTrigger(${i})">✎</button>
        <button class="btn-icon del" title="Delete" onclick="deleteTrigger(${i}, event)">✕</button>
      </div>`;
    list.appendChild(card);
  });
}

function renderActions() {
  const list = document.getElementById("actions-list");
  list.innerHTML = "";
  state.actions.forEach((a, i) => {
    const act  = a.action || {};
    const meta = [a.context ? `ctx:${a.context}` : null, actionMeta(act)].filter(Boolean).join("  •  ");
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `
      <div class="item-name">${esc(a.signal)}</div>
      <div class="item-meta">${esc(meta)}</div>
      <div class="item-actions">
        <button class="btn-icon" title="Edit"   onclick="editAction(${i})">✎</button>
        <button class="btn-icon del" title="Delete" onclick="deleteAction(${i}, event)">✕</button>
      </div>`;
    list.appendChild(card);
  });
}

function fingerDots(pose) {
  const fingers = ["thumb","index","middle","ring","pinky"];
  const labels  = ["T","I","M","R","P"];
  const dots = fingers.map((f, i) => {
    const val = pose[f];
    const cls = val === true ? "on" : val === false ? "off" : "";
    return `<div class="item-finger ${cls}">${labels[i]}</div>`;
  }).join("");

  const spreadDots = SPREAD_PAIRS.map(({ key, label }) => {
    const val = pose[key];
    if (!val) return "";
    const icon = val === "apart" ? "◀  ▶" : "◀▶";
    return `<div class="item-spread" title="${label}: ${val}">${icon}</div>`;
  }).join("");

  return dots + (spreadDots ? `<span class="item-spread-sep">|</span>${spreadDots}` : "");
}

function triggerMeta(t) {
  if (!t.type) return "";
  const d = TRIGGER_TYPES.find(d => d.id === t.type);
  return d ? d.meta(t) : t.type;
}

function actionMeta(a) {
  if (!a.type) return "";
  const d = ACTION_TYPES.find(d => d.id === a.type);
  return d ? d.meta(a) : a.type;
}

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}


// ── Modal infrastructure ───────────────────────────────────────────────────────

function openModal(html, ctx, onCapture) {
  document.getElementById("modal-body").innerHTML = html;
  document.getElementById("modal-backdrop").classList.remove("hidden");
  modalCtx      = ctx;
  captureTarget = onCapture || null;
}

function closeModal() {
  document.getElementById("modal-backdrop").classList.add("hidden");
  modalCtx      = null;
  captureTarget = null;
}

function maybeCloseModal(evt) {
  if (evt.target === document.getElementById("modal-backdrop")) closeModal();
}

function commitModal() {
  if (!modalCtx) return;
  const { type, index } = modalCtx;
  if (type === "pose")    commitPose(index);
  if (type === "trigger") commitTrigger(index);
  if (type === "action")  commitAction(index);
}

// Update the capture-row inside an open modal with live finger state
function updateCaptureRow() {
  const row = document.querySelector(".capture-live");
  if (!row) return;
  const fingers = liveFingers.right || liveFingers.left || [];
  const names   = ["T","I","M","R","P"];
  row.querySelectorAll(".finger-dot").forEach((dot, i) => {
    dot.classList.toggle("on", !!fingers[i]);
  });
}


// ── Pose editing ──────────────────────────────────────────────────────────────

const FINGER_NAMES  = ["thumb","index","middle","ring","pinky"];
const FINGER_LABELS = ["Thumb","Index","Middle","Ring","Pinky"];

// Adjacent finger pairs — config key, display label, live spread key
const SPREAD_PAIRS = [
  { key: "spread_thumb_index",  label: "T–I", liveKey: "thumbIndex"  },
  { key: "spread_index_middle", label: "I–M", liveKey: "indexMiddle" },
  { key: "spread_middle_ring",  label: "M–R", liveKey: "middleRing"  },
  { key: "spread_ring_pinky",   label: "R–P", liveKey: "ringPinky"   },
];

const SPREAD_THRESHOLD = 0.20;  // mirrors DEFAULT_SPREAD_THRESHOLD in engine

function poseMatchesState(pose, fingers, spreads) {
  const fingerOk = FINGER_NAMES.every((f, i) => {
    const v = pose[f];
    return v === undefined || v === null || v === fingers[i];
  });
  if (!fingerOk) return false;
  if (!spreads) return true;
  return SPREAD_PAIRS.every(({ key, liveKey }) => {
    const constraint = pose[key];
    const value = spreads[liveKey] ?? 0;
    if (constraint === undefined || constraint === null) return true;
    if (typeof constraint === "number") return value >= constraint;
    if (constraint === "apart") return value >= SPREAD_THRESHOLD;
    if (constraint === "close") return value < SPREAD_THRESHOLD;
    return true;
  });
}

function captureNewPose(fingers, spreads) {
  const pose = { name: "" };
  FINGER_NAMES.forEach((f, i) => { pose[f] = fingers[i]; });
  if (spreads) {
    SPREAD_PAIRS.forEach(({ key, liveKey }) => {
      pose[key] = spreads[liveKey] >= SPREAD_THRESHOLD ? "apart" : "close";
    });
  }
  openModal(poseModalHtml(pose), { type: "pose", index: -1 }, applyCaptureToPoseEditor);
}

function fingerToggleHtml(fingers) {
  return FINGER_NAMES.map((f, i) => {
    const val  = fingers?.[f] ?? null;
    const st   = val === true ? "true" : val === false ? "false" : "null";
    const icon = val === true ? "●" : val === false ? "○" : "–";
    return `
      <div class="finger-toggle" data-finger="${f}" data-state="${st}"
           onclick="cycleFingerState(this)" title="Click to cycle: on → off → don't care">
        <div class="finger-toggle-btn">${icon}</div>
        <span>${FINGER_LABELS[i]}</span>
      </div>`;
  }).join("");
}

function spreadToggleHtml(pose) {
  return SPREAD_PAIRS.map(({ key, label }) => {
    const val = pose?.[key] ?? null;
    const st  = val === null ? "null" : String(val);
    const icon = val === "close" ? "◀▶" : val === "apart" ? "◀  ▶" : "–";
    return `
      <div class="spread-toggle" data-spread="${key}" data-state="${st}"
           onclick="cycleSpreadState(this)" title="Click to cycle: don't care → close → apart">
        <div class="spread-toggle-btn">${icon}</div>
        <span>${label}</span>
      </div>`;
  }).join("");
}

function cycleFingerState(el) {
  const next = el.dataset.state === "null" ? "true" : el.dataset.state === "true" ? "false" : "null";
  el.dataset.state = next;
  el.querySelector(".finger-toggle-btn").textContent =
    next === "true" ? "●" : next === "false" ? "○" : "–";
}

function cycleSpreadState(el) {
  const next = el.dataset.state === "null" ? "close" : el.dataset.state === "close" ? "apart" : "null";
  el.dataset.state = next;
  el.querySelector(".spread-toggle-btn").textContent =
    next === "close" ? "◀▶" : next === "apart" ? "◀  ▶" : "–";
}

function applyCaptureToPoseEditor(fingers, spreads) {
  FINGER_NAMES.forEach((f, i) => {
    const el = document.querySelector(`.finger-toggle[data-finger="${f}"]`);
    if (!el) return;
    const st = fingers[i] ? "true" : "false";
    el.dataset.state = st;
    el.querySelector(".finger-toggle-btn").textContent = fingers[i] ? "●" : "○";
  });
  if (spreads) {
    SPREAD_PAIRS.forEach(({ key, liveKey }) => {
      const el = document.querySelector(`.spread-toggle[data-spread="${key}"]`);
      if (!el) return;
      const st = spreads[liveKey] >= SPREAD_THRESHOLD ? "apart" : "close";
      el.dataset.state = st;
      el.querySelector(".spread-toggle-btn").textContent = st === "apart" ? "◀  ▶" : "◀▶";
    });
  }
}

function poseModalHtml(pose) {
  const name    = pose?.name ?? "";
  const fingers = pose ?? {};
  return `
    <div class="modal-title">${pose ? "Edit Pose" : "New Pose"}</div>
    <div class="field">
      <label>Name</label>
      <input id="pose-name" type="text" value="${esc(name)}" placeholder="e.g. FIST">
    </div>
    <div class="field">
      <label>Fingers <span style="font-weight:400;color:var(--text-muted)">— click to cycle on / off / don't care</span></label>
      <div class="finger-toggle-row">${fingerToggleHtml(fingers)}</div>
    </div>
    <div class="field">
      <label>Finger spread <span style="font-weight:400;color:var(--text-muted)">— click to cycle don't care / close / apart</span></label>
      <div class="spread-toggle-row">${spreadToggleHtml(pose)}</div>
    </div>
    <div class="capture-row">
      <div class="capture-live">
        ${["T","I","M","R","P"].map(l => `<div class="finger-dot">${l}</div>`).join("")}
      </div>
      <button class="btn-secondary" style="font-size:12px;padding:5px 12px"
              onclick="applyCaptureToPoseEditor(window._captureFingers || [], window._captureSpreads)">
        Use live pose
      </button>
    </div>`;
}

function newPose()   { openModal(poseModalHtml(null),           { type: "pose", index: -1 }, applyCaptureToPoseEditor); }
function editPose(i) { openModal(poseModalHtml(state.poses[i]), { type: "pose", index: i  }, applyCaptureToPoseEditor); }

function deletePose(i, evt) {
  evt.stopPropagation();
  state.poses.splice(i, 1);
  markDirty();
  renderPoses();
}

function commitPose(index) {
  const name = document.getElementById("pose-name").value.trim();
  if (!name) { alert("Pose name is required."); return; }

  const pose = { name };
  document.querySelectorAll(".finger-toggle").forEach(el => {
    const f  = el.dataset.finger;
    const st = el.dataset.state;
    if (st === "true")  pose[f] = true;
    if (st === "false") pose[f] = false;
    // null = omit
  });
  document.querySelectorAll(".spread-toggle").forEach(el => {
    const st = el.dataset.state;
    if (st !== "null") pose[el.dataset.spread] = st;  // "close" or "apart"
  });

  if (index === -1) state.poses.push(pose);
  else              state.poses[index] = pose;

  markDirty();
  renderPoses();
  closeModal();
}

// Keep live state accessible to the "Use live pose" button
Object.defineProperty(window, "_captureFingers", {
  get: () => liveFingers.right || liveFingers.left || [],
});
Object.defineProperty(window, "_captureSpreads", {
  get: () => liveSpreads.right || liveSpreads.left || null,
});


// ── Trigger editing ───────────────────────────────────────────────────────────

const METRICS = ["pinch_distance","hand_height","hand_x","finger_spread","angle"];

function poseOptions(selected) {
  return state.poses.map(p =>
    `<option value="${esc(p.name)}" ${p.name === selected ? "selected" : ""}>${esc(p.name)}</option>`
  ).join("");
}

function poseOptionsWithNone(selected) {
  return `<option value="">— none —</option>` + poseOptions(selected);
}

// Each entry describes one trigger type: how to render its fields, how to read
// them back from the DOM, and how to summarise it in the list view.
// Adding a new trigger type = add one object here; nothing else changes.
const TRIGGER_TYPES = [
  {
    id: "pose",
    label: "Pose",
    sectionId: "trig-pose-section",
    usesHand: true,
    usesCrossHand: true,
    fieldsHtml(trig) {
      return `
        <div class="field-row">
          <div class="field">
            <label>Shape</label>
            <select id="trig-shape">${poseOptions(trig.shape || "")}</select>
          </div>
          <div class="field" style="max-width:110px">
            <label>Dwell (ms)</label>
            <input id="trig-dwell" type="number" min="0" step="50" value="${trig.dwell_ms ?? ""}">
          </div>
        </div>`;
    },
    readFields() {
      const shape = document.getElementById("trig-shape").value;
      if (!shape) { alert("Shape is required."); return null; }
      const dwell = document.getElementById("trig-dwell").value;
      return { shape, ...(dwell ? { dwell_ms: parseInt(dwell, 10) } : {}) };
    },
    meta(t) { return `pose  •  ${t.hand || "?"}  •  ${t.shape || "?"}`; },
  },
  {
    id: "continuous",
    label: "Continuous",
    sectionId: "trig-cont-section",
    usesHand: true,
    usesCrossHand: true,
    fieldsHtml(trig) {
      return `
        <div class="field-row">
          <div class="field">
            <label>Metric</label>
            <select id="trig-metric">
              ${METRICS.map(m => `<option value="${m}" ${m === trig.metric ? "selected" : ""}>${m}</option>`).join("")}
            </select>
          </div>
        </div>
        <div class="field">
          <label>Range <span style="font-weight:400;color:var(--text-muted)">[min, max] raw sensor value</span></label>
          <div class="range-row">
            <input id="trig-range-lo" type="number" step="0.01" placeholder="0.0" value="${trig.range?.[0] ?? ""}">
            <span class="range-sep">→</span>
            <input id="trig-range-hi" type="number" step="0.01" placeholder="1.0" value="${trig.range?.[1] ?? ""}">
          </div>
        </div>
        <div class="field" style="max-width:160px">
          <label>Hysteresis <span style="font-weight:400;color:var(--text-muted)">slot deadzone</span></label>
          <input id="trig-hysteresis" type="number" step="0.01" min="0" max="0.5"
                 placeholder="0.04" value="${trig.hysteresis ?? ""}">
        </div>`;
    },
    readFields() {
      const metric     = document.getElementById("trig-metric").value;
      const lo         = parseFloat(document.getElementById("trig-range-lo").value);
      const hi         = parseFloat(document.getElementById("trig-range-hi").value);
      const hysteresis = parseFloat(document.getElementById("trig-hysteresis").value);
      return {
        metric,
        ...(!isNaN(lo) && !isNaN(hi) ? { range: [lo, hi] } : {}),
        ...(!isNaN(hysteresis) ? { hysteresis } : {}),
      };
    },
    meta(t) { return `continuous  •  ${t.hand || "?"}  •  ${t.metric || "?"}`; },
  },
  {
    id: "sequence",
    label: "Sequence",
    sectionId: "trig-seq-section",
    usesHand: true,
    usesCrossHand: true,
    onShow() { renderSteps(); },
    fieldsHtml(trig) {
      return `
        <div class="field">
          <label>Steps — in order</label>
          <div id="trig-steps-list"></div>
          <div class="step-add-row">
            <select id="trig-step-select">${poseOptions("")}</select>
            <button class="btn-secondary" style="white-space:nowrap" onclick="addStep()">+ Add Step</button>
          </div>
        </div>
        <div class="field-row">
          <div class="field">
            <label>Window (ms) <span style="font-weight:400;color:var(--text-muted)">max time to complete</span></label>
            <input id="trig-window-ms" type="number" min="100" step="100" value="${trig.window_ms ?? 1500}">
          </div>
          <div class="field">
            <label>Step dwell (ms)</label>
            <input id="trig-step-dwell-ms" type="number" min="0" step="10" value="${trig.step_dwell_ms ?? 100}">
          </div>
        </div>`;
    },
    readFields() {
      if (editingSteps.length < 2) { alert("A sequence needs at least two steps."); return null; }
      return {
        steps:         [...editingSteps],
        window_ms:     parseInt(document.getElementById("trig-window-ms").value, 10)     || 1500,
        step_dwell_ms: parseInt(document.getElementById("trig-step-dwell-ms").value, 10) || 100,
      };
    },
    meta(t) { return `sequence  •  ${t.hand || "?"}  •  [${(t.steps || []).join(" → ")}]`; },
  },
  {
    id: "swipe",
    label: "Swipe",
    sectionId: "trig-swipe-section",
    usesHand: true,
    usesCrossHand: true,
    fieldsHtml(trig) {
      return `
        <div class="field-row">
          <div class="field">
            <label>Direction</label>
            <select id="trig-swipe-direction">
              <option value="left"  ${(trig.direction || "left") === "left"  ? "selected" : ""}>Left</option>
              <option value="right" ${trig.direction === "right" ? "selected" : ""}>Right</option>
            </select>
          </div>
          <div class="field" style="max-width:160px">
            <label>Min displacement <span style="font-weight:400;color:var(--text-muted)">0.0–1.0</span></label>
            <input id="trig-swipe-min-disp" type="number" step="0.05" min="0" max="1"
                   value="${trig.min_displacement ?? 0.3}">
          </div>
        </div>`;
    },
    readFields() {
      const direction = document.getElementById("trig-swipe-direction").value;
      const minDisp   = parseFloat(document.getElementById("trig-swipe-min-disp").value);
      return { direction, ...(!isNaN(minDisp) ? { min_displacement: minDisp } : {}) };
    },
    meta(t) { return `swipe  •  ${t.hand || "?"}  •  ${t.direction || "?"}`; },
  },
  {
    id: "chord",
    label: "Chord",
    sectionId: "trig-chord-section",
    usesHand: false,
    usesCrossHand: false,
    fieldsHtml(trig) {
      return `
        <div class="field-row">
          <div class="field">
            <label>Left hand pose</label>
            <select id="trig-chord-left">${poseOptions(trig.left || "")}</select>
          </div>
          <div class="field">
            <label>Right hand pose</label>
            <select id="trig-chord-right">${poseOptions(trig.right || "")}</select>
          </div>
        </div>
        <div class="field" style="max-width:150px">
          <label>Dwell (ms)</label>
          <input id="trig-chord-dwell" type="number" min="0" step="50" value="${trig.dwell_ms ?? ""}">
        </div>`;
    },
    readFields() {
      const left  = document.getElementById("trig-chord-left").value;
      const right = document.getElementById("trig-chord-right").value;
      if (!left || !right) { alert("Both hand poses are required for a chord."); return null; }
      const dwell = document.getElementById("trig-chord-dwell").value;
      return { left, right, ...(dwell ? { dwell_ms: parseInt(dwell, 10) } : {}) };
    },
    meta(t) { return `chord  •  L:${t.left || "?"}  +  R:${t.right || "?"}`; },
  },
  {
    id: "sequenced_continuous",
    label: "Sequence → Continuous",
    sectionId: "trig-seqcont-section",
    usesHand: true,
    usesCrossHand: true,
    onShow() { renderPrefixSteps(); },
    fieldsHtml(trig) {
      return `
        <div class="field">
          <label>Prefix steps — complete these to start the continuous gesture</label>
          <div id="trig-prefix-steps-list"></div>
          <div class="step-add-row">
            <select id="trig-prefix-step-select">${poseOptions("")}</select>
            <button class="btn-secondary" style="white-space:nowrap" onclick="addPrefixStep()">+ Add Step</button>
          </div>
        </div>
        <div class="field-row">
          <div class="field">
            <label>Prefix window (ms) <span style="font-weight:400;color:var(--text-muted)">max time to complete sequence</span></label>
            <input id="trig-prefix-window-ms" type="number" min="100" step="100" value="${trig.prefix_window_ms ?? 1500}">
          </div>
          <div class="field">
            <label>Prefix step dwell (ms)</label>
            <input id="trig-prefix-dwell-ms" type="number" min="0" step="10" value="${trig.prefix_dwell_ms ?? 100}">
          </div>
        </div>
        <div class="field">
          <label>Metric <span style="font-weight:400;color:var(--text-muted)">(continuous phase)</span></label>
          <select id="trig-metric">
            ${METRICS.map(m => `<option value="${m}" ${m === trig.metric ? "selected" : ""}>${m}</option>`).join("")}
          </select>
        </div>
        <div class="field">
          <label>Range <span style="font-weight:400;color:var(--text-muted)">[min, max] raw sensor value</span></label>
          <div class="range-row">
            <input id="trig-range-lo" type="number" step="0.01" placeholder="0.0" value="${trig.range?.[0] ?? ""}">
            <span class="range-sep">→</span>
            <input id="trig-range-hi" type="number" step="0.01" placeholder="1.0" value="${trig.range?.[1] ?? ""}">
          </div>
        </div>
        <div class="field" style="max-width:160px">
          <label>Hysteresis <span style="font-weight:400;color:var(--text-muted)">slot deadzone</span></label>
          <input id="trig-hysteresis" type="number" step="0.01" min="0" max="0.5"
                 placeholder="0.04" value="${trig.hysteresis ?? ""}">
        </div>`;
    },
    readFields() {
      if (editingPrefixSteps.length < 1) { alert("At least one prefix step is required."); return null; }
      const metric     = document.getElementById("trig-metric").value;
      const lo         = parseFloat(document.getElementById("trig-range-lo").value);
      const hi         = parseFloat(document.getElementById("trig-range-hi").value);
      const hysteresis = parseFloat(document.getElementById("trig-hysteresis").value);
      return {
        prefix_steps:     [...editingPrefixSteps],
        prefix_window_ms: parseInt(document.getElementById("trig-prefix-window-ms").value, 10) || 1500,
        prefix_dwell_ms:  parseInt(document.getElementById("trig-prefix-dwell-ms").value, 10) || 100,
        metric,
        ...(!isNaN(lo) && !isNaN(hi) ? { range: [lo, hi] } : {}),
        ...(!isNaN(hysteresis) ? { hysteresis } : {}),
      };
    },
    meta(t) {
      const prefix = (t.prefix_steps || []).join(" → ");
      return `seq→cont  •  ${t.hand || "?"}  •  [${prefix}] → ${t.metric || "?"}`;
    },
  },
];

function renderRequire() {
  const container = document.getElementById("trig-require-list");
  if (!container) return;
  if (editingRequire.length === 0) {
    container.innerHTML = `<div class="step-empty">No conditions — fires regardless of other poses.</div>`;
    return;
  }
  container.innerHTML = editingRequire.map((r, i) => `
    <div class="step-item">
      <span class="step-name">${esc(r.hand)} hand: ${esc(r.pose)}</span>
      <button class="btn-icon del" title="Remove" onclick="removeRequire(${i})">✕</button>
    </div>`).join("");
}

function addRequire() {
  const hand = document.getElementById("trig-req-hand")?.value;
  const pose = document.getElementById("trig-req-pose")?.value;
  if (!hand || !pose) return;
  editingRequire.push({ hand, pose });
  renderRequire();
}

function removeRequire(i) {
  editingRequire.splice(i, 1);
  renderRequire();
}

function triggerModalHtml(binding) {
  const name = binding?.name ?? "";
  const trig = binding?.trigger ?? {};
  const type = trig.type || "pose";
  const hand = trig.hand || "either";

  const descriptor  = TRIGGER_TYPES.find(t => t.id === type);
  const typeOptions = TRIGGER_TYPES.map(t =>
    `<option value="${t.id}" ${t.id === type ? "selected" : ""}>${t.label}</option>`
  ).join("");
  const sections = TRIGGER_TYPES.map(t =>
    `<div id="${t.sectionId}" class="${t.id !== type ? "hidden" : ""}">${t.fieldsHtml(trig)}</div>`
  ).join("");

  return `
    <div class="modal-title">${binding ? "Edit Trigger" : "New Trigger"}</div>
    <div class="field">
      <label>Name <span style="font-weight:400;color:var(--text-muted)">— used as the D-Bus signal name</span></label>
      <input id="trig-name" type="text" value="${esc(name)}" placeholder="e.g. play_pause">
    </div>
    <div class="field-row">
      <div class="field">
        <label>Trigger type</label>
        <select id="trig-type" onchange="onTriggerTypeChange(this.value)">
          ${typeOptions}
        </select>
      </div>
      <div id="trig-hand-field" class="field ${descriptor?.usesHand ? "" : "hidden"}">
        <label>Hand</label>
        <select id="trig-hand">
          <option value="either" ${hand === "either" ? "selected" : ""}>Either</option>
          <option value="right"  ${hand === "right"  ? "selected" : ""}>Right</option>
          <option value="left"   ${hand === "left"   ? "selected" : ""}>Left</option>
        </select>
      </div>
    </div>
    ${sections}
    <hr class="divider">
    <div id="trig-cross-hand-row" class="${descriptor?.usesCrossHand ? "" : "hidden"}">
      <div class="field">
        <label>Required poses (optional) <span style="font-weight:400;color:var(--text-muted)">— all must be held</span></label>
        <div id="trig-require-list"></div>
        <div class="step-add-row">
          <select id="trig-req-hand">
            <option value="either">Either</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
          <select id="trig-req-pose">${poseOptions("")}</select>
          <button class="btn-secondary" style="white-space:nowrap" onclick="addRequire()">+ Add</button>
        </div>
      </div>
    </div>`;
}

function onTriggerTypeChange(type) {
  const descriptor = TRIGGER_TYPES.find(t => t.id === type);
  TRIGGER_TYPES.forEach(t => {
    document.getElementById(t.sectionId)?.classList.toggle("hidden", t.id !== type);
  });
  document.getElementById("trig-hand-field")?.classList.toggle("hidden",     !descriptor?.usesHand);
  document.getElementById("trig-cross-hand-row")?.classList.toggle("hidden", !descriptor?.usesCrossHand);
  descriptor?.onShow?.();
  renderRequire();
}

// ── Sequence step management ───────────────────────────────────────────────────

function renderSteps() {
  const container = document.getElementById("trig-steps-list");
  if (!container) return;
  if (editingSteps.length === 0) {
    container.innerHTML = `<div class="step-empty">No steps yet — add at least two below.</div>`;
    return;
  }
  container.innerHTML = editingSteps.map((step, i) => `
    <div class="step-item">
      <span class="step-index">${i + 1}</span>
      <span class="step-name">${esc(step)}</span>
      <div class="step-btns">
        <button class="btn-icon" title="Move up"   onclick="moveStep(${i}, -1)" ${i === 0 ? "disabled" : ""}>↑</button>
        <button class="btn-icon" title="Move down" onclick="moveStep(${i},  1)" ${i === editingSteps.length - 1 ? "disabled" : ""}>↓</button>
        <button class="btn-icon del" title="Remove" onclick="removeStep(${i})">✕</button>
      </div>
    </div>`).join("");
}

function addStep() {
  const sel = document.getElementById("trig-step-select");
  if (!sel?.value) return;
  editingSteps.push(sel.value);
  renderSteps();
}

function removeStep(i) {
  editingSteps.splice(i, 1);
  renderSteps();
}

function moveStep(i, dir) {
  const j = i + dir;
  if (j < 0 || j >= editingSteps.length) return;
  [editingSteps[i], editingSteps[j]] = [editingSteps[j], editingSteps[i]];
  renderSteps();
}

function renderPrefixSteps() {
  const container = document.getElementById("trig-prefix-steps-list");
  if (!container) return;
  if (editingPrefixSteps.length === 0) {
    container.innerHTML = `<div class="step-empty">No prefix steps yet — add at least one below.</div>`;
    return;
  }
  container.innerHTML = editingPrefixSteps.map((step, i) => `
    <div class="step-item">
      <span class="step-index">${i + 1}</span>
      <span class="step-name">${esc(step)}</span>
      <div class="step-btns">
        <button class="btn-icon" title="Move up"   onclick="movePrefixStep(${i}, -1)" ${i === 0 ? "disabled" : ""}>↑</button>
        <button class="btn-icon" title="Move down" onclick="movePrefixStep(${i},  1)" ${i === editingPrefixSteps.length - 1 ? "disabled" : ""}>↓</button>
        <button class="btn-icon del" title="Remove" onclick="removePrefixStep(${i})">✕</button>
      </div>
    </div>`).join("");
}

function addPrefixStep() {
  const sel = document.getElementById("trig-prefix-step-select");
  if (!sel?.value) return;
  editingPrefixSteps.push(sel.value);
  renderPrefixSteps();
}

function removePrefixStep(i) {
  editingPrefixSteps.splice(i, 1);
  renderPrefixSteps();
}

function movePrefixStep(i, dir) {
  const j = i + dir;
  if (j < 0 || j >= editingPrefixSteps.length) return;
  [editingPrefixSteps[i], editingPrefixSteps[j]] = [editingPrefixSteps[j], editingPrefixSteps[i]];
  renderPrefixSteps();
}

function newTrigger() {
  editingSteps        = [];
  editingPrefixSteps  = [];
  editingRequire      = [];
  openModal(triggerModalHtml(null), { type: "trigger", index: -1 });
  renderRequire();
}

function editTrigger(i) {
  const binding    = state.triggers[i] ?? {};
  const trig       = binding.trigger ?? {};
  const descriptor = TRIGGER_TYPES.find(t => t.id === trig.type);
  editingSteps       = trig.type === "sequence"              ? [...(trig.steps        || [])] : [];
  editingPrefixSteps = trig.type === "sequenced_continuous" ? [...(trig.prefix_steps || [])] : [];
  editingRequire     = [...(binding.require || [])];
  openModal(triggerModalHtml(binding), { type: "trigger", index: i });
  descriptor?.onShow?.();
  renderRequire();
}

function deleteTrigger(i, evt) {
  evt.stopPropagation();
  state.triggers.splice(i, 1);
  markDirty();
  renderTriggers();
}

function commitTrigger(index) {
  const name = document.getElementById("trig-name").value.trim();
  if (!name) { alert("Trigger name is required."); return; }

  const type       = document.getElementById("trig-type").value;
  const descriptor = TRIGGER_TYPES.find(t => t.id === type);
  const typeFields = descriptor?.readFields();
  if (typeFields == null) return;

  const trigger = { type, ...typeFields };
  if (descriptor.usesHand) trigger.hand = document.getElementById("trig-hand").value;

  const binding = {
    name,
    trigger,
    ...(descriptor.usesCrossHand && editingRequire.length ? { require: [...editingRequire] } : {}),
  };

  if (index === -1) state.triggers.push(binding);
  else              state.triggers[index] = binding;

  markDirty();
  renderTriggers();
  closeModal();
}


// ── Action editing ────────────────────────────────────────────────────────────

// Each entry describes one action type: how to render its fields, how to read
// them back from the DOM, and how to summarise it in the list view.
// Adding a new action type = add one object here; nothing else changes.
const ACTION_TYPES = [
  {
    id: "exec",
    label: "exec — run a command",
    sectionSuffix: "exec-section",
    fieldsHtml(pfx, act, hidden) {
      return `
        <div id="${pfx}-exec-section" class="${hidden ? "hidden" : ""} field">
          <label>Command <span style="font-weight:400;color:var(--text-muted)">— space-separated args</span></label>
          <input id="${pfx}-cmd" type="text" placeholder="playerctl play-pause"
                 value="${esc((act?.cmd || []).join(" "))}">
        </div>`;
    },
    readFields(pfx) {
      const raw = document.getElementById(`${pfx}-cmd`)?.value.trim();
      if (!raw) return null;
      return { type: "exec", cmd: raw.split(/\s+/) };
    },
    meta(a) { return `exec  •  ${(a.cmd || []).join(" ")}`; },
  },
  {
    id: "exec_scaled",
    label: "exec_scaled — command with {value}",
    sectionSuffix: "exec-scaled-section",
    fieldsHtml(pfx, act, hidden) {
      return `
        <div id="${pfx}-exec-scaled-section" class="${hidden ? "hidden" : ""} field">
          <label>Template <span style="font-weight:400;color:var(--text-muted)">— {value} is 0.0–1.0</span></label>
          <input id="${pfx}-template" type="text" placeholder="pactl set-sink-volume @DEFAULT_SINK@ {value:.0%}"
                 value="${esc(act?.template || "")}">
        </div>`;
    },
    readFields(pfx) {
      const tmpl = document.getElementById(`${pfx}-template`)?.value.trim();
      if (!tmpl) return null;
      return { type: "exec_scaled", template: tmpl };
    },
    meta(a) { return `exec_scaled  •  ${a.template || ""}`; },
  },
  {
    id: "key",
    label: "key — synthesize keypress",
    sectionSuffix: "key-section",
    fieldsHtml(pfx, act, hidden) {
      return `
        <div id="${pfx}-key-section" class="${hidden ? "hidden" : ""} field">
          <label>Key name <span style="font-weight:400;color:var(--text-muted)">— xdotool key name</span></label>
          <input id="${pfx}-key" type="text" placeholder="XF86AudioPlay"
                 value="${esc(act?.key || "")}">
        </div>`;
    },
    readFields(pfx) {
      const key = document.getElementById(`${pfx}-key`)?.value.trim();
      if (!key) return null;
      return { type: "key", key };
    },
    meta(a) { return `key  •  ${a.key || ""}`; },
  },
];

function signalOptions(selected) {
  return state.triggers.map(t =>
    `<option value="${esc(t.name)}" ${t.name === selected ? "selected" : ""}>${esc(t.name)}</option>`
  ).join("");
}

function actionFieldsHtml(prefix, act) {
  const type = act?.type || "exec";
  const typeOptions = ACTION_TYPES.map(t =>
    `<option value="${t.id}" ${t.id === type ? "selected" : ""}>${t.label}</option>`
  ).join("");
  const sections = ACTION_TYPES.map(t => t.fieldsHtml(prefix, act, t.id !== type)).join("");
  return `
    <div class="field">
      <label>Action type</label>
      <select id="${prefix}-type" onchange="onActionTypeChange('${prefix}', this.value)">
        ${typeOptions}
      </select>
    </div>
    ${sections}`;
}

function onActionTypeChange(prefix, type) {
  ACTION_TYPES.forEach(t => {
    document.getElementById(`${prefix}-${t.sectionSuffix}`)?.classList.toggle("hidden", t.id !== type);
  });
}

function isContinuousTrigger(signalName) {
  const t = state.triggers.find(t => t.name === signalName);
  return t?.trigger?.type === "continuous";
}

function actionModalHtml(binding) {
  const signal  = binding?.signal  ?? "";
  const act     = binding?.action  ?? {};
  const onEnd   = binding?.on_end  ?? null;
  const context = binding?.context ?? "";
  const isCont  = isContinuousTrigger(signal);

  return `
    <div class="modal-title">${binding ? "Edit Action" : "New Action"}</div>
    <div class="field">
      <label>Signal</label>
      <select id="action-signal" onchange="onActionSignalChange(this.value)">
        ${signalOptions(signal)}
      </select>
    </div>
    <div class="field">
      <label>Context <span style="font-weight:400;color:var(--text-muted)">— WM_CLASS substring; leave empty to always fire</span></label>
      <input id="action-context" type="text" value="${esc(context)}" placeholder="e.g. qutebrowser">
    </div>
    ${actionFieldsHtml("act", act)}
    <div id="on-end-section" class="${!isCont ? "hidden" : ""}">
      <hr class="divider">
      <div style="font-size:12px;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">
        On End (optional — fires when continuous trigger deactivates)
      </div>
      ${actionFieldsHtml("end", onEnd)}
    </div>`;
}

function onActionSignalChange(signal) {
  const cont = isContinuousTrigger(signal);
  document.getElementById("on-end-section")?.classList.toggle("hidden", !cont);
}

function readActionFields(prefix) {
  const type       = document.getElementById(`${prefix}-type`)?.value;
  const descriptor = ACTION_TYPES.find(t => t.id === type);
  return descriptor?.readFields(prefix) ?? null;
}

function newAction()     { openModal(actionModalHtml(null),             { type: "action", index: -1 }); }
function editAction(i)   { openModal(actionModalHtml(state.actions[i]), { type: "action", index: i }); }

function deleteAction(i, evt) {
  evt.stopPropagation();
  state.actions.splice(i, 1);
  markDirty();
  renderActions();
}

function commitAction(index) {
  const signal = document.getElementById("action-signal")?.value;
  if (!signal) { alert("Signal is required."); return; }

  const action = readActionFields("act");
  if (!action) { alert("Action is incomplete."); return; }

  const onEnd   = readActionFields("end");
  const context = document.getElementById("action-context")?.value.trim();

  const binding = {
    signal,
    ...(context ? { context } : {}),
    action,
    ...(onEnd ? { on_end: onEnd } : {}),
  };

  if (index === -1) state.actions.push(binding);
  else              state.actions[index] = binding;

  markDirty();
  renderActions();
  closeModal();
}
