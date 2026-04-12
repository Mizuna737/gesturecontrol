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

// Current modal context
let modalCtx = null;   // { type, index, data }

// Which finger the capture button should write to (set when a pose editor is open)
let captureTarget = null;  // function(fingers) or null

// Mutable step list for the sequence trigger editor
let editingSteps = [];


// ── Boot ──────────────────────────────────────────────────────────────────────

async function loadConfig() {
  const res  = await fetch("/api/config");
  const data = await res.json();
  state.settings = data.settings || {};
  state.poses    = data.poses    || [];
  state.triggers = data.triggers || [];
  state.actions  = data.actions  || [];
  renderAll();
}

function renderAll() {
  renderPoses();
  renderTriggers();
  renderActions();
}

document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  startSSE();
});


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

    updateFingerDisplay("right", liveFingers.right, livePose.right);
    updateFingerDisplay("left",  liveFingers.left,  livePose.left);

    // Update the capture row inside an open pose modal
    updateCaptureRow();

    // Enable capture button when a pose editor is open and a hand is visible
    const captureBtn = document.getElementById("capture-btn");
    if (captureTarget && (liveFingers.right || liveFingers.left)) {
      captureBtn.disabled = false;
    }
  };
  es.onerror = () => {
    // reconnects automatically
  };

  document.getElementById("capture-btn").addEventListener("click", () => {
    if (!captureTarget) return;
    const fingers = liveFingers.right || liveFingers.left;
    if (fingers) captureTarget(fingers);
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
    const act = a.action || {};
    const meta = actionMeta(act);
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
  return fingers.map((f, i) => {
    const val = pose[f];
    const cls = val === true ? "on" : val === false ? "off" : "";
    return `<div class="item-finger ${cls}">${labels[i]}</div>`;
  }).join("");
}

function triggerMeta(t) {
  if (!t.type) return "";
  if (t.type === "pose")       return `pose  •  ${t.hand || "?"}  •  ${t.shape || "?"}`;
  if (t.type === "continuous") return `continuous  •  ${t.hand || "?"}  •  ${t.metric || "?"}`;
  if (t.type === "sequence")   return `sequence  •  ${t.hand || "?"}  •  [${(t.steps || []).join(" → ")}]`;
  if (t.type === "chord")      return `chord  •  L:${t.left || "?"}  +  R:${t.right || "?"}`;
  return t.type;
}

function actionMeta(a) {
  if (!a.type) return "";
  if (a.type === "exec")        return `exec  •  ${(a.cmd || []).join(" ")}`;
  if (a.type === "exec_scaled") return `exec_scaled  •  ${a.template || ""}`;
  if (a.type === "key")         return `key  •  ${a.key || ""}`;
  return a.type;
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
  modalCtx = ctx;

  if (onCapture) {
    captureTarget = onCapture;
    document.getElementById("capture-btn").disabled =
      !(liveFingers.right || liveFingers.left);
    document.getElementById("capture-hint").classList.add("hidden");
  } else {
    captureTarget = null;
    document.getElementById("capture-btn").disabled = true;
    document.getElementById("capture-hint").classList.remove("hidden");
  }
}

function closeModal() {
  document.getElementById("modal-backdrop").classList.add("hidden");
  modalCtx      = null;
  captureTarget = null;
  document.getElementById("capture-btn").disabled = true;
  document.getElementById("capture-hint").classList.remove("hidden");
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
const FINGER_ICONS  = { true: "●", false: "○", null: "–" };

function fingerToggleHtml(id, fingers) {
  return FINGER_NAMES.map((f, i) => {
    const val = fingers?.[f] ?? null;
    const st  = val === true ? "true" : val === false ? "false" : "null";
    const icon = val === true ? "●" : val === false ? "○" : "–";
    return `
      <div class="finger-toggle" data-finger="${f}" data-state="${st}"
           onclick="cycleFingerState(this)" title="Click to cycle: on → off → don't care">
        <div class="finger-toggle-btn">${icon}</div>
        <span>${FINGER_LABELS[i]}</span>
      </div>`;
  }).join("");
}

function cycleFingerState(el) {
  const cur = el.dataset.state;
  const next = cur === "null" ? "true" : cur === "true" ? "false" : "null";
  el.dataset.state = next;
  el.querySelector(".finger-toggle-btn").textContent =
    next === "true" ? "●" : next === "false" ? "○" : "–";
}

function applyCaptureToPoseEditor(fingers) {
  FINGER_NAMES.forEach((f, i) => {
    const el = document.querySelector(`.finger-toggle[data-finger="${f}"]`);
    if (!el) return;
    const st = fingers[i] ? "true" : "false";
    el.dataset.state = st;
    el.querySelector(".finger-toggle-btn").textContent = fingers[i] ? "●" : "○";
  });
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
      <div class="finger-toggle-row">${fingerToggleHtml(null, fingers)}</div>
    </div>
    <div class="capture-row">
      <div class="capture-live">
        ${["T","I","M","R","P"].map(l => `<div class="finger-dot">${l}</div>`).join("")}
      </div>
      <button class="btn-secondary" style="font-size:12px;padding:5px 12px"
              onclick="applyCaptureToPoseEditor(window._captureFingers || [])">
        Use live pose
      </button>
    </div>`;
}

function newPose()      { openModal(poseModalHtml(null), { type: "pose", index: -1 }, applyCaptureToPoseEditor); }
function editPose(i)    { openModal(poseModalHtml(state.poses[i]), { type: "pose", index: i }, applyCaptureToPoseEditor); }

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

  if (index === -1) state.poses.push(pose);
  else              state.poses[index] = pose;

  markDirty();
  renderPoses();
  closeModal();
}

// Keep live fingers accessible to the "Use live pose" button
Object.defineProperty(window, "_captureFingers", {
  get: () => liveFingers.right || liveFingers.left || [],
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

function triggerModalHtml(binding) {
  const name  = binding?.name ?? "";
  const trig  = binding?.trigger ?? {};
  const rL    = binding?.require_left  ?? "";
  const rR    = binding?.require_right ?? "";
  const type  = trig.type || "pose";
  const hand  = trig.hand || "either";

  const poseSection = `
    <div id="trig-pose-section" class="${type !== "pose" ? "hidden" : ""}">
      <div class="field-row">
        <div class="field">
          <label>Shape</label>
          <select id="trig-shape">${poseOptions(trig.shape || "")}</select>
        </div>
        <div class="field" style="max-width:110px">
          <label>Dwell (ms)</label>
          <input id="trig-dwell" type="number" min="0" step="50" value="${trig.dwell_ms ?? ""}">
        </div>
      </div>
    </div>`;

  const contSection = `
    <div id="trig-cont-section" class="${type !== "continuous" ? "hidden" : ""}">
      <div class="field-row">
        <div class="field">
          <label>Metric</label>
          <select id="trig-metric">
            ${METRICS.map(m => `<option value="${m}" ${m === trig.metric ? "selected" : ""}>${m}</option>`).join("")}
          </select>
        </div>
        <div class="field">
          <label>Active while (pose)</label>
          <select id="trig-active-while">${poseOptionsWithNone(trig.active_while || "")}</select>
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
    </div>`;

  const seqSection = `
    <div id="trig-seq-section" class="${type !== "sequence" ? "hidden" : ""}">
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
      </div>
    </div>`;

  const chordSection = `
    <div id="trig-chord-section" class="${type !== "chord" ? "hidden" : ""}">
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
      </div>
    </div>`;

  const crossHandRow = `
    <div id="trig-cross-hand-row" class="${type === "chord" ? "hidden" : ""}">
      <div class="field-row">
        <div class="field">
          <label>Require left hand (optional)</label>
          <select id="trig-req-left">${poseOptionsWithNone(rL)}</select>
        </div>
        <div class="field">
          <label>Require right hand (optional)</label>
          <select id="trig-req-right">${poseOptionsWithNone(rR)}</select>
        </div>
      </div>
    </div>`;

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
          <option value="pose"       ${type === "pose"       ? "selected" : ""}>Pose</option>
          <option value="continuous" ${type === "continuous" ? "selected" : ""}>Continuous</option>
          <option value="sequence"   ${type === "sequence"   ? "selected" : ""}>Sequence</option>
          <option value="chord"      ${type === "chord"      ? "selected" : ""}>Chord</option>
        </select>
      </div>
      <div id="trig-hand-field" class="field ${type === "chord" ? "hidden" : ""}">
        <label>Hand</label>
        <select id="trig-hand">
          <option value="either" ${hand === "either" ? "selected" : ""}>Either</option>
          <option value="right"  ${hand === "right"  ? "selected" : ""}>Right</option>
          <option value="left"   ${hand === "left"   ? "selected" : ""}>Left</option>
        </select>
      </div>
    </div>
    ${poseSection}
    ${contSection}
    ${seqSection}
    ${chordSection}
    <hr class="divider">
    ${crossHandRow}`;
}

function onTriggerTypeChange(type) {
  document.getElementById("trig-pose-section")?.classList.toggle("hidden",  type !== "pose");
  document.getElementById("trig-cont-section")?.classList.toggle("hidden",  type !== "continuous");
  document.getElementById("trig-seq-section")?.classList.toggle("hidden",   type !== "sequence");
  document.getElementById("trig-chord-section")?.classList.toggle("hidden", type !== "chord");
  document.getElementById("trig-hand-field")?.classList.toggle("hidden",    type === "chord");
  document.getElementById("trig-cross-hand-row")?.classList.toggle("hidden", type === "chord");
  if (type === "sequence") renderSteps();
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

function newTrigger() {
  editingSteps = [];
  openModal(triggerModalHtml(null), { type: "trigger", index: -1 });
  renderSteps();
}

function editTrigger(i) {
  const trig = state.triggers[i]?.trigger ?? {};
  editingSteps = trig.type === "sequence" ? [...(trig.steps || [])] : [];
  openModal(triggerModalHtml(state.triggers[i]), { type: "trigger", index: i });
  if (trig.type === "sequence") renderSteps();
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

  const type = document.getElementById("trig-type").value;
  const hand = document.getElementById("trig-hand").value;

  let trigger;

  if (type === "chord") {
    const left  = document.getElementById("trig-chord-left").value;
    const right = document.getElementById("trig-chord-right").value;
    if (!left || !right) { alert("Both hand poses are required for a chord."); return; }
    const dwell = document.getElementById("trig-chord-dwell").value;
    trigger = { type, left, right, ...(dwell ? { dwell_ms: parseInt(dwell, 10) } : {}) };
  } else {
    const hand = document.getElementById("trig-hand").value;
    trigger = { type, hand };

    if (type === "pose") {
      trigger.shape = document.getElementById("trig-shape").value;
      const dwell   = document.getElementById("trig-dwell").value;
      if (dwell) trigger.dwell_ms = parseInt(dwell, 10);
    } else if (type === "continuous") {
      trigger.metric = document.getElementById("trig-metric").value;
      const aw = document.getElementById("trig-active-while").value;
      if (aw) trigger.active_while = aw;
      const lo = parseFloat(document.getElementById("trig-range-lo").value);
      const hi = parseFloat(document.getElementById("trig-range-hi").value);
      if (!isNaN(lo) && !isNaN(hi)) trigger.range = [lo, hi];
    } else if (type === "sequence") {
      if (editingSteps.length < 2) { alert("A sequence needs at least two steps."); return; }
      trigger.steps         = [...editingSteps];
      trigger.window_ms     = parseInt(document.getElementById("trig-window-ms").value, 10)     || 1500;
      trigger.step_dwell_ms = parseInt(document.getElementById("trig-step-dwell-ms").value, 10) || 100;
    }
  }

  const reqL = type !== "chord" ? document.getElementById("trig-req-left")?.value  : "";
  const reqR = type !== "chord" ? document.getElementById("trig-req-right")?.value : "";

  const binding = {
    name,
    trigger,
    ...(reqL ? { require_left: reqL }  : {}),
    ...(reqR ? { require_right: reqR } : {}),
  };

  if (index === -1) state.triggers.push(binding);
  else              state.triggers[index] = binding;

  markDirty();
  renderTriggers();
  closeModal();
}


// ── Action editing ────────────────────────────────────────────────────────────

function signalOptions(selected) {
  return state.triggers.map(t =>
    `<option value="${esc(t.name)}" ${t.name === selected ? "selected" : ""}>${esc(t.name)}</option>`
  ).join("");
}

function actionFieldsHtml(prefix, act) {
  const type = act?.type || "exec";
  return `
    <div class="field">
      <label>Action type</label>
      <select id="${prefix}-type" onchange="onActionTypeChange('${prefix}', this.value)">
        <option value="exec"        ${type === "exec"        ? "selected" : ""}>exec — run a command</option>
        <option value="exec_scaled" ${type === "exec_scaled" ? "selected" : ""}>exec_scaled — command with {value}</option>
        <option value="key"         ${type === "key"         ? "selected" : ""}>key — synthesize keypress</option>
      </select>
    </div>
    <div id="${prefix}-exec-section"        class="${type !== "exec"        ? "hidden" : ""} field">
      <label>Command <span style="font-weight:400;color:var(--text-muted)">— space-separated args</span></label>
      <input id="${prefix}-cmd" type="text" placeholder="playerctl play-pause"
             value="${esc((act?.cmd || []).join(" "))}">
    </div>
    <div id="${prefix}-exec-scaled-section" class="${type !== "exec_scaled" ? "hidden" : ""} field">
      <label>Template <span style="font-weight:400;color:var(--text-muted)">— {value} is 0.0–1.0</span></label>
      <input id="${prefix}-template" type="text" placeholder="pactl set-sink-volume @DEFAULT_SINK@ {value:.0%}"
             value="${esc(act?.template || "")}">
    </div>
    <div id="${prefix}-key-section"         class="${type !== "key"         ? "hidden" : ""} field">
      <label>Key name <span style="font-weight:400;color:var(--text-muted)">— xdotool key name</span></label>
      <input id="${prefix}-key" type="text" placeholder="XF86AudioPlay"
             value="${esc(act?.key || "")}">
    </div>`;
}

function onActionTypeChange(prefix, type) {
  document.getElementById(`${prefix}-exec-section`)?.classList.toggle("hidden",        type !== "exec");
  document.getElementById(`${prefix}-exec-scaled-section`)?.classList.toggle("hidden", type !== "exec_scaled");
  document.getElementById(`${prefix}-key-section`)?.classList.toggle("hidden",         type !== "key");
}

function isContinuousTrigger(signalName) {
  const t = state.triggers.find(t => t.name === signalName);
  return t?.trigger?.type === "continuous";
}

function actionModalHtml(binding) {
  const signal  = binding?.signal  ?? "";
  const act     = binding?.action  ?? {};
  const onEnd   = binding?.on_end  ?? null;
  const isCont  = isContinuousTrigger(signal);

  return `
    <div class="modal-title">${binding ? "Edit Action" : "New Action"}</div>
    <div class="field">
      <label>Signal</label>
      <select id="action-signal" onchange="onActionSignalChange(this.value)">
        ${signalOptions(signal)}
      </select>
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
  const type = document.getElementById(`${prefix}-type`)?.value;
  if (!type) return null;
  if (type === "exec") {
    const raw = document.getElementById(`${prefix}-cmd`)?.value.trim();
    if (!raw) return null;
    return { type: "exec", cmd: raw.split(/\s+/) };
  }
  if (type === "exec_scaled") {
    const tmpl = document.getElementById(`${prefix}-template`)?.value.trim();
    if (!tmpl) return null;
    return { type: "exec_scaled", template: tmpl };
  }
  if (type === "key") {
    const key = document.getElementById(`${prefix}-key`)?.value.trim();
    if (!key) return null;
    return { type: "key", key };
  }
  return null;
}

function newAction()     { openModal(actionModalHtml(null),            { type: "action", index: -1 }); }
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

  const onEnd  = readActionFields("end");

  const binding = { signal, action, ...(onEnd ? { on_end: onEnd } : {}) };

  if (index === -1) state.actions.push(binding);
  else              state.actions[index] = binding;

  markDirty();
  renderActions();
  closeModal();
}
