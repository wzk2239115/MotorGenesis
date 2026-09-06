import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const $ = (id) => document.getElementById(id);
const status = (msg, cls = "") => {
  const el = $("statusLine");
  el.textContent = msg;
  el.className = "status " + cls;
};

// ---------------------------------------------------------------------------
// Three.js scene — product-level PBR rendering
// ---------------------------------------------------------------------------
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

// Studio environment for realistic metal reflections.
const pmrem = new THREE.PMREMGenerator(renderer);
const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1d23);
scene.environment = envTexture;
scene.fog = new THREE.FogExp2(0x1a1d23, 3.5);

const camera = new THREE.PerspectiveCamera(
  38, viewport.clientWidth / viewport.clientHeight, 0.0005, 5
);
camera.position.set(0.09, 0.06, 0.12);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

// Key + fill + rim lighting for metallic surfaces.
const key = new THREE.DirectionalLight(0xfff5e8, 1.8);
key.position.set(0.15, 0.18, 0.12);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.02;
key.shadow.camera.far = 0.5;
key.shadow.camera.left = -0.08;
key.shadow.camera.right = 0.08;
key.shadow.camera.top = 0.08;
key.shadow.camera.bottom = -0.08;
key.shadow.bias = -0.0005;
scene.add(key);

const fill = new THREE.DirectionalLight(0xb0c4e8, 0.5);
fill.position.set(-0.14, 0.05, 0.1);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xffd0a0, 0.7);
rim.position.set(-0.05, -0.08, -0.15);
scene.add(rim);

// Shadow-receiving ground plane.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(0.4, 0.4),
  new THREE.MeshStandardMaterial({
    color: 0x111316,
    metalness: 0.0,
    roughness: 0.92,
  })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.065;
ground.receiveShadow = true;
scene.add(ground);

let currentModel = null;      // raw GLTF scene (kept only for disposal)
let assemblyGroup = null;     // root: statorGroup + rotorGroup
let statorGroup = null;
let rotorGroup = null;
const materialNodes = new Map(); // canonical material -> [mesh, ...]

function disposeTree(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => m.dispose());
    }
  });
}

function clearModel() {
  if (assemblyGroup) {
    scene.remove(assemblyGroup);
    disposeTree(assemblyGroup);
    assemblyGroup = null;
    statorGroup = null;
    rotorGroup = null;
  }
  if (currentModel) { disposeTree(currentModel); currentModel = null; }
  materialNodes.clear();
}

function loadGlb(url) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      url,
      (gltf) => {
        const scene = gltf.scene || (gltf.scenes && gltf.scenes[0]) || new THREE.Group();
        resolve(scene);
      },
      undefined,
      (err) => reject(err)
    );
  });
}

// Canonical material for a node name ("rotor_iron"/"stator_iron" -> iron).
function canonicalMaterial(name) {
  const n = name.toLowerCase();
  if (n.includes("iron")) return "iron";
  if (n.includes("copper")) return "copper";
  if (n.includes("pm")) return "pm";
  if (n.includes("coolant")) return "coolant";
  if (n.includes("insulator")) return "insulator";
  return null;
}

const MAT_STYLE = {
  iron: { color: 0x7a8a9c, metalness: 0.92, roughness: 0.38, env: 1.2 },
  copper: { color: 0xc87533, metalness: 0.95, roughness: 0.28, env: 1.4 },
  pm: { color: 0x8a2042, metalness: 0.5, roughness: 0.45, env: 0.8 },
  coolant: { color: 0x408cde, metalness: 0.1, roughness: 0.15, env: 1.0, opacity: 0.45 },
  insulator: { color: 0xe8e6da, metalness: 0.05, roughness: 0.6, env: 0.7 },
};

// Preserve the WORLD transform when moving a mesh between groups.
function reparentKeepWorld(o, target) {
  const pos = new THREE.Vector3(); o.getWorldPosition(pos);
  const quat = new THREE.Quaternion(); o.getWorldQuaternion(quat);
  const scl = new THREE.Vector3(); o.getWorldScale(scl);
  target.add(o);
  o.position.copy(pos);
  o.quaternion.copy(quat);
  o.scale.copy(scl);
}

async function showCheckpoint(run, step, level) {
  const overlay = $("loadOverlay");
  $("loadText").textContent = `加载第 ${step} 步网格…`;
  overlay.classList.remove("hidden");
  status(`加载第 ${step} 步…`, "busy");
  clearModel();
  resetSimState(); // model changed: old playback must not drive this one
  try {
    const view = viewState.mode === "stator" ? "stator" : "full";
    const url = `/api/runs/${encodeURIComponent(run)}/checkpoint/${step}/glb`
      + `?level=${level}&smoothing=taubin&iterations=5&view=${view}`;
    const model = await loadGlb(url);
    if (!model || typeof model.traverse !== 'function') {
      throw new Error("GLB加载返回空场景 (gltf.scene未定义)");
    }
    currentModel = model;

    assemblyGroup = new THREE.Group();
    assemblyGroup.name = "assembly";
    statorGroup = new THREE.Group();
    statorGroup.name = "stator";
    rotorGroup = new THREE.Group();
    rotorGroup.name = "rotor";
    assemblyGroup.add(statorGroup, rotorGroup);

    // Collect meshes first — reparenting inside traverse would splice
    // the children array mid-iteration, causing children[i] to be undefined.
    const meshes = [];
    model.traverse((o) => { if (o.isMesh) meshes.push(o); });

    for (const o of meshes) {
      o.castShadow = true;
      o.receiveShadow = true;
      const name = (o.name || o.parent?.name || "");
      const mat = canonicalMaterial(name);
      if (mat) {
        if (!materialNodes.has(mat)) materialNodes.set(mat, []);
        materialNodes.get(mat).push(o);
        const st = MAT_STYLE[mat];
        const material = new THREE.MeshStandardMaterial({
          color: st.color, metalness: st.metalness,
          roughness: st.roughness, envMapIntensity: st.env,
        });
        if (st.opacity !== undefined) {
          material.transparent = true;
          material.opacity = st.opacity;
          material.depthWrite = false;
        }
        o.material = material;
        o.visible = materialVisible[mat];
        o.userData.baseVisible = materialVisible[mat];
      } else {
        o.userData.baseVisible = o.visible;
      }
      // Motion group by name prefix — the ONLY thing that rotates.
      const target = name.toLowerCase().includes("rotor") ? rotorGroup : statorGroup;
      reparentKeepWorld(o, target);
    }
    scene.add(assemblyGroup);

    // Frame the camera on the whole assembly.
    const box = new THREE.Box3().setFromObject(assemblyGroup);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360)) * 1.7;
    camera.position
      .set(dist, dist * 0.75, dist * 1.1)
      .add(center);
    controls.update();

    applyViewMode();
    applyExplode();
    overlay.classList.add("hidden");
    status(`第 ${step} 步`, "");
    syncMaterialToggles();
  } catch (err) {
    $("loadText").textContent = "加载失败";
    status(`加载失败: ${parseApiError(err)}`, "error");
    console.error(err);
  }
}

// ---------------------------------------------------------------------------
// View modes: full / stator / rotor / section / explode (display only)
// ---------------------------------------------------------------------------
const viewState = { mode: "full", section: false, explode: 0.0 };

function applyViewMode() {
  if (!assemblyGroup) return;
  const m = viewState.mode;
  statorGroup.visible = (m === "full" || m === "stator");
  rotorGroup.visible = (m === "full" || m === "rotor");
  // Section: a global clipping plane at the axial mid-plane (display only;
  // never affects solver geometry).
  renderer.clippingPlanes = viewState.section
    ? [new THREE.Plane(new THREE.Vector3(0, 0, -1), 0.0)]
    : [];
  syncMaterialToggles();
}

function applyExplode() {
  if (!assemblyGroup) return;
  const e = viewState.explode;
  // Display-only axial separation; rotor rotation stays about its own axis.
  statorGroup.position.z = 0.045 * e;
  rotorGroup.position.z = -0.045 * e;
}

// ---------------------------------------------------------------------------
// Material toggles
// ---------------------------------------------------------------------------
const MATERIAL_COLORS = {
  iron: "#5c748a",
  copper: "#d6662b",
  pm: "#cd2d48",
  coolant: "#408cde",
  insulator: "#e8e6da",
};
// Coolant starts HIDDEN in the solid view (it is fluid inside the coils,
// shown as a translucent blue flow path when toggled on).
const materialVisible = { iron: true, copper: true, pm: true, coolant: false, insulator: true };

function syncMaterialToggles() {
  const host = $("materialToggles");
  host.innerHTML = "";
  for (const mat of ["iron", "copper", "pm", "insulator", "coolant"]) {
    const meshes = materialNodes.get(mat) || [];
    const present = meshes.length > 0;
    const row = document.createElement("label");
    row.className = "toggle";
    row.innerHTML = `
      <input type="checkbox" ${materialVisible[mat] ? "checked" : ""} ${!present ? "disabled" : ""} />
      <span class="swatch" style="background:${MATERIAL_COLORS[mat]}"></span>
      <span>${labelOf(mat)}${present ? "" : " (无)"}</span>`;
    const cb = row.querySelector("input");
    cb.addEventListener("change", () => {
      materialVisible[mat] = cb.checked;
      // act on the meshes actually displayed (no clone/original split)
      for (const m of meshes) m.visible = cb.checked;
    });
    host.appendChild(row);
  }
}
function labelOf(m) {
  return {
    iron: "铁 Iron", copper: "铜 Copper", pm: "磁钢 PM",
    coolant: "冷却 Coolant", insulator: "绝缘 Insulator",
  }[m] || m;
}

// ---------------------------------------------------------------------------
// State + API
// ---------------------------------------------------------------------------
let state = {
  runs: [],
  currentRun: null,
  steps: [],
  stepIndex: -1,
  level: 0.35,
  live: false,
  eventSource: null,
};

async function refreshRuns() {
  const res = await fetch("/api/runs");
  state.runs = await res.json();
  const sel = $("runSelect");
  sel.innerHTML = "";
  if (!state.runs.length) {
    sel.innerHTML = '<option value="">(无运行实例)</option>';
    status("未找到运行实例。先跑一次 motor3d_organic grow。", "error");
    return;
  }
  for (const r of state.runs) {
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = `${r.name}${r.has_checkpoints ? " ✓" : ""}`;
    sel.appendChild(opt);
  }
  // Prefer the most recent run with checkpoints.
  const best = state.runs.find((r) => r.has_checkpoints) || state.runs[0];
  sel.value = best.name;
  await selectRun(best.name);
}

async function selectRun(name) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.currentRun = name;
  status(`加载运行 ${name}…`, "busy");
  const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
  const summary = await res.json();
  state.steps = summary.checkpoint_steps || [];
  const range = $("stepRange");
  range.max = Math.max(0, state.steps.length - 1);
  range.value = range.max;
  state.stepIndex = state.steps.length - 1;
  renderMetrics(summary.latest_metrics || {});
  renderTimeline(summary);
  if (state.steps.length) {
    await showCheckpoint(name, state.steps[state.stepIndex], state.level);
    await loadSlice();
  } else {
    clearModel();
    $("loadOverlay").classList.remove("hidden");
    $("loadText").textContent = "此运行尚无 checkpoint。";
    status("无 checkpoint", "error");
  }
  loadStartupPanel(name);
  if (state.live) startLive(name);
}

async function loadStartupPanel(run) {
  const panel = $("startupPanel");
  const verdict = $("startupVerdict");
  const anglesDiv = $("startupAngles");
  const verdictsDiv = $("startupVerdicts");
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(run)}/startup`);
    if (!res.ok) { panel.style.display = "none"; return; }
    const s = await res.json();
    panel.style.display = "";
    verdict.textContent = s.passed ? "✅ 通过 (六项验证)" : "❌ 未通过";
    verdict.style.color = s.passed ? "#7ee08a" : "#e08a7e";
    anglesDiv.innerHTML = (s.angles || []).map((a) => {
      const cls = a.passed ? "pass" : "fail";
      const rpm = (a.final_speed_rad_s * 60 / (2 * Math.PI)).toFixed(0);
      return `<div class="ang ${cls}"><span>θ₀ ${Math.round(a.initial_angle_rad * 180 / Math.PI)}°</span>` +
             `<span>${a.final_speed_rad_s >= 0 ? "+" : ""}${a.final_speed_rad_s.toFixed(1)} rad/s (${rpm} rpm)</span></div>`;
    }).join("");
    // Six independent verdicts: a green spin cannot cover broken topology.
    if (verdictsDiv && s.verdicts) {
      const marks = { pass: "✅", fail: "❌", none: "➖" };
      verdictsDiv.innerHTML = Object.entries(s.verdicts).map(([key, v]) => {
        const cls = v.passed === true ? "pass" : v.passed === false ? "fail" : "none";
        const label = (s.verdict_labels && s.verdict_labels[key]) || key;
        let extra = "";
        if (key === "manufacturing" && v.detail && v.detail.not_evaluated) {
          extra = `<div class="vnote">未评估: ${v.detail.not_evaluated.join(", ")}</div>`;
        } else if (key === "mesh_convergence" && v.detail) {
          const d = v.detail;
          if (d.torque) {
            extra = `<div class="vnote">T1 Δ${d.torque.t1_amplitude_change_pct?.toFixed?.(1)}%, ` +
                    `T0 Δ${d.torque.t0_rms_change_pct?.toFixed?.(1)}% (${d.physics_shape?.join("×")} → ${d.fine_shape?.join("×")})</div>`;
          } else if (d.topology_physics && d.topology_display) {
            extra = `<div class="vnote">拓扑: 铜 ${d.topology_physics.copper_components} vs ` +
                    `${d.topology_display.copper_components} 网, 相 ${JSON.stringify(d.topology_display.phase_components)}</div>`;
          }
        } else if (key === "winding" && v.detail && v.detail.expected_components) {
          extra = `<div class="vnote">相连通块 ${JSON.stringify(v.detail.expected_components)} · ` +
                  `最小相间隙 ${(v.detail.min_phase_gap_mm || 0).toFixed(2)} mm</div>`;
        } else if (key === "cooling" && v.detail) {
          extra = `<div class="vnote">贯通流道 ${v.detail.through_flow_networks ?? "?"} · 死腔 ${v.detail.trapped_voids ?? "?"}</div>`;
        } else if (key === "structure" && v.detail) {
          extra = `<div class="vnote">浮岛 ${v.detail.floating_islands} · 最小颈 ${((v.detail.min_neck_mm) || 0).toFixed(2)} mm</div>`;
        } else if (key === "electromechanical" && v.detail) {
          extra = `<div class="vnote">${v.detail.n_angles ?? "?"} 初始角 · 最低末速 ${(v.detail.min_final_speed_rad_s || 0).toFixed(1)} rad/s</div>`;
        }
        return `<div class="verdict ${cls}"><span class="mark">${marks[cls]}</span>` +
               `<span class="vlabel">${label}</span>${extra}</div>`;
      }).join("") +
      `<div class="vnote">已评估 ${s.verdicts_evaluated ?? "?"}/6 · 失败: ${s.verdicts_failed?.length ? s.verdicts_failed.join(", ") : "无"}</div>`;
    }
  } catch {
    panel.style.display = "none";
  }
}

function renderTimeline(summary) {
  $("stepLabel").textContent = state.steps.length
    ? `第 ${state.steps[state.stepIndex]} 步 / 共 ${state.steps.length} 步`
    : "—";
  const list = $("metricList");
  const m = summary.latest_metrics || {};
  const order = [
    "torque", "|torque|", "torque_ripple", "mass_kg_per_m",
    "vol_iron", "vol_copper", "vol_pm",
    "loss_W_per_m", "temperature_max_C", "efficiency_proxy",
  ];
  const shown = order.filter((k) => k in m);
  for (const k of Object.keys(m)) if (!shown.includes(k)) shown.push(k);
  list.innerHTML = shown
    .map((k) => `<dt>${k}</dt><dd>${fmt(m[k])}</dd>`)
    .join("");
}
function fmt(v) {
  if (typeof v !== "number") return String(v);
  if (Math.abs(v) >= 1000) return v.toExponential(2);
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
  return v.toFixed(4);
}
function renderMetrics(m) { renderTimeline({ latest_metrics: m }); }

// ---------------------------------------------------------------------------
// Live SSE
// ---------------------------------------------------------------------------
function startLive(run) {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/runs/${encodeURIComponent(run)}/events`);
  status("实时跟随中…", "live");
  state.eventSource.onmessage = async (ev) => {
    let payload;
    try { payload = JSON.parse(ev.data); } catch { return; }
    if (!payload || payload.step === undefined) return;
    if (!state.steps.includes(payload.step)) {
      state.steps.push(payload.step);
      $("stepRange").max = state.steps.length - 1;
    }
    // Jump to newest.
    state.stepIndex = state.steps.length - 1;
    $("stepRange").value = state.stepIndex;
    await showCheckpoint(run, payload.step, state.level);
    const res = await fetch(`/api/runs/${encodeURIComponent(run)}/checkpoint/${payload.step}/metrics`);
    renderMetrics((await res.json()).metrics || {});
    await loadSlice();
  };
  state.eventSource.onerror = () => {
    status("实时连接中断,重试中…", "busy");
  };
}

// ---------------------------------------------------------------------------
// Field slice viewer
// ---------------------------------------------------------------------------
let sliceState = { field: "temperature", axis: 2, index: null };

async function loadSlice() {
  if (!state.currentRun || state.stepIndex < 0) return;
  const step = state.steps[state.stepIndex];
  const idxParam = (sliceState.index != null && !isNaN(sliceState.index))
    ? `&index=${sliceState.index}` : '';
  const url = `/api/runs/${encodeURIComponent(state.currentRun)}/checkpoint/${step}/slice?field=${sliceState.field}&axis=${sliceState.axis}${idxParam}`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      let err = {};
      try { err = await res.json(); } catch { /* non-JSON */ }
      const canvas = $("sliceCanvas");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#666";
      ctx.font = "12px sans-serif";
      const msg = parseApiError(err);
      ctx.fillText(/未计算|not computed/.test(msg) ? "未计算" : "不可用", 12, canvas.height / 2);
      const src = $("sliceSource");
      if (src) {
        const kind = /grid|网格/.test(msg)
          ? "该模型不支持 (网格不匹配)"
          : /未计算|not computed/.test(msg)
            ? "尚未计算 — 该场在此运行中不存在"
            : `获取失败: ${msg}`;
        src.textContent = `⚠ ${kind}`;
      }
      return;
    }
    const data = await res.json();
    drawSlice(data);
    const src = $("sliceSource");
    if (src) {
      src.textContent = data.source_label
        ? `数据来源: ${data.source_label}` : "";
    }
    const si = $("sliceIndex");
    si.max = (data.shape[0] || 1) - 1;
    si.value = data.index;
    sliceState.index = data.index;
  } catch (e) { /* silent */ }
}

function drawSlice(data) {
  const canvas = $("sliceCanvas");
  const ctx = canvas.getContext("2d");
  const [h, w] = data.shape;
  canvas.width = w; canvas.height = h;
  const img = ctx.createImageData(w, h);
  const { vmin, vmax } = data;
  for (let i = 0; i < w * h; i++) {
    const v = data.values[i];
    const t = Math.min(1, Math.max(0, (v - vmin) / (vmax - vmin || 1)));
    const [r, g, b] = turbo(t);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function turbo(t) {
  // Approximate turbo colormap (4 control points + smoothstep).
  const s = t * t * (3 - 2 * t);
  const stops = [
    [30, 8, 90], [40, 120, 210], [220, 230, 70], [220, 60, 30], [120, 5, 30],
  ];
  const x = s * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
$("runSelect").addEventListener("change", (e) => selectRun(e.target.value));

$("stepRange").addEventListener("input", async (e) => {
  state.stepIndex = parseInt(e.target.value, 10);
  $("stepLabel").textContent = `第 ${state.steps[state.stepIndex]} 步 / 共 ${state.steps.length} 步`;
  await showCheckpoint(state.currentRun, state.steps[state.stepIndex], state.level);
  await loadSlice();
});
$("prevBtn").addEventListener("click", () => {
  if (state.stepIndex > 0) { $("stepRange").value = state.stepIndex - 1; $("stepRange").dispatchEvent(new Event("input")); }
});
$("nextBtn").addEventListener("click", () => {
  if (state.stepIndex < state.steps.length - 1) { $("stepRange").value = state.stepIndex + 1; $("stepRange").dispatchEvent(new Event("input")); }
});

$("levelRange").addEventListener("input", (e) => {
  state.level = parseFloat(e.target.value);
  $("levelVal").textContent = state.level.toFixed(2);
});
$("levelRange").addEventListener("change", async () => {
  if (state.stepIndex >= 0) await showCheckpoint(state.currentRun, state.steps[state.stepIndex], state.level);
});

$("liveToggle").addEventListener("change", (e) => {
  state.live = e.target.checked;
  if (state.live && state.currentRun) startLive(state.currentRun);
  else if (state.eventSource) { state.eventSource.close(); state.eventSource = null; status("已停止实时跟随", ""); }
});

$("viewMode").addEventListener("change", (e) => {
  const prev = viewState.mode;
  viewState.mode = e.target.value;
  applyViewMode();
  // The stator view is served WITHOUT rotor meshes from the backend, so
  // crossing the stator/full boundary needs a reload; full<->rotor are
  // pure client-side group toggles.
  const crossed = (prev === "stator") !== (viewState.mode === "stator");
  if (crossed && state.stepIndex >= 0) {
    showCheckpoint(state.currentRun, state.steps[state.stepIndex], state.level);
  }
});
$("sectionToggle").addEventListener("change", (e) => {
  viewState.section = e.target.checked;
  applyViewMode();
});
$("explodeRange").addEventListener("input", (e) => {
  viewState.explode = parseFloat(e.target.value);
  $("explodeVal").textContent = viewState.explode.toFixed(2);
  applyExplode();
});

$("downloadStl").addEventListener("click", () => {
  if (!state.currentRun || state.stepIndex < 0) return;
  const step = state.steps[state.stepIndex];
  const url = `/api/runs/${encodeURIComponent(state.currentRun)}/checkpoint/${step}/stl?level=${state.level}&smoothing=taubin&iterations=5`;
  status("正在生成 STL…", "busy");
  fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.blob();
    })
    .then((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `motor_step${String(step).padStart(6, "0")}.stl`;
      a.click();
      URL.revokeObjectURL(a.href);
      status(`已下载 step ${step} STL`, "");
    })
    .catch((err) => status(`STL 下载失败: ${err.message}`, "error"));
});

$("sliceField").addEventListener("change", (e) => { sliceState.field = e.target.value; sliceState.index = null; loadSlice(); });
document.querySelectorAll(".slice-axes button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".slice-axes button").forEach((b) => b.classList.remove("main"));
    btn.classList.add("main");
    sliceState.axis = parseInt(btn.dataset.axis, 10);
    sliceState.index = null;
    loadSlice();
  });
});
$("sliceIndex").addEventListener("input", () => { sliceState.index = parseInt($("sliceIndex").value, 10); loadSlice(); });

// ---------------------------------------------------------------------------
// Simulation (通电仿真) — physics-time playback, strict input parsing
// ---------------------------------------------------------------------------

// Unified API-error parsing: strings, {detail: str|obj|list}, anything.
function parseApiError(err) {
  if (err == null) return "未知错误";
  if (typeof err === "string") return err;
  if (typeof err === "number") return `HTTP ${err}`;
  if (err.detail !== undefined) return parseApiError(err.detail);
  if (Array.isArray(err)) return err.map(parseApiError).join("; ");
  if (typeof err === "object") {
    if (err.msg) return String(err.msg);
    if (err.message) return String(err.message);
    try { return JSON.stringify(err); } catch { return String(err); }
  }
  return String(err);
}

// Numeric input: EMPTY -> default; legal zero stays zero; garbage -> throw.
function numInput(id, def, label) {
  const raw = $(id).value.trim();
  if (raw === "") return def;
  const v = parseFloat(raw);
  if (!Number.isFinite(v)) throw new Error(`${label}: 无法解析 "${raw}"`);
  return v;
}

let simState = {
  id: null,
  runName: null,        // results only valid for THIS run/model
  status: "idle",
  results: null,
  playing: false,
  playTime: 0.0,        // physical time [s]
  playSpeed: 1.0,
  pollTimer: null,
  progress: null,
};

function resetSimState() {
  if (simState.pollTimer) { clearInterval(simState.pollTimer); }
  simState.id = null;
  simState.runName = null;
  simState.status = "idle";
  simState.results = null;
  simState.playing = false;
  simState.playTime = 0.0;
  simState.pollTimer = null;
  simState.progress = null;
  const play = $("simPlayBtn"), reset = $("simResetBtn"), slider = $("simTimeSlider");
  if (play) { play.disabled = true; play.textContent = "▶ 播放"; }
  if (reset) reset.disabled = true;
  if (slider) { slider.disabled = true; slider.value = 0; }
  const lbl = $("simTimeLabel");
  if (lbl) lbl.textContent = "—";
  const curves = $("curvesPanel");
  if (curves) curves.style.display = "none";
  if (rotorGroup) rotorGroup.rotation.z = 0;
}

async function startSimulation() {
  if (!state.currentRun) { status("请先选择运行实例", "error"); return; }
  let params;
  try {
    const iqRaw = $("simIqRef").value.trim();
    const poRaw = $("simPowerOff").value.trim();
    params = {
      voltage: numInput("simVoltage", 24, "电压"),
      current_limit: numInput("simCurrentLimit", 50, "电流上限"),
      initial_angle: numInput("simInitialAngle", 0, "初始角") * Math.PI / 180,
      load_torque: numInput("simLoadTorque", 0.005, "负载"),
      steps: Math.round(numInput("simSteps", 4000, "步数")),
      control_mode: $("simControlMode").value,
      i_q_ref_A: iqRaw === "" ? null : (() => {
        const v = parseFloat(iqRaw);
        if (!Number.isFinite(v)) throw new Error(`q轴电流: 无法解析 "${iqRaw}"`);
        return v;
      })(),
      power_off_at_s: poRaw === "" ? null : (() => {
        const v = parseFloat(poRaw);
        if (!Number.isFinite(v)) throw new Error(`断电时刻: 无法解析 "${poRaw}"`);
        return v / 1000.0;
      })(),
    };
  } catch (err) {
    $("simStatus").textContent = `输入错误: ${err.message}`;
    return;
  }
  resetSimState();
  simState.runName = state.currentRun;
  $("simRunBtn").disabled = true;
  $("simStatus").textContent = "提交中…";
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(state.currentRun)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      let err = {}; try { err = await res.json(); } catch { /* */ }
      throw new Error(parseApiError(err));
    }
    const data = await res.json();
    simState.id = data.sim_id;
    simState.status = "queued";
    // Echo the settings the backend actually adopted (incl. voltage
    // definition) so the user can verify nothing was silently changed.
    if (data.adopted) {
      const a = data.adopted;
      $("simStatus").textContent =
        `已提交 (${simState.id}) · 后端采用: ${a.control_mode}` +
        ` V相峰值=${a.voltage}V 负载=${a.load_torque}Nm` +
        `${a.i_q_ref_A != null ? ` Iq*=${a.i_q_ref_A}A` : ""}` +
        ` · ${a.voltage_definition}`;
    } else {
      $("simStatus").textContent = `已提交 (${simState.id})`;
    }
    pollSimulation();
  } catch (err) {
    $("simStatus").textContent = `提交失败: ${parseApiError(err)}`;
    $("simRunBtn").disabled = false;
  }
}

function fmtProgress(p) {
  if (!p) return "";
  let s = ` · ${p.phase}`;
  if (p.angles_total) s += ` ${p.angles_done}/${p.angles_total} 角`;
  if (p.elapsed_s != null) s += ` ${p.elapsed_s.toFixed(0)}s`;
  if (p.detail) s += ` (${p.detail})`;
  return s;
}

function pollSimulation() {
  if (simState.pollTimer) clearInterval(simState.pollTimer);
  const myId = simState.id;
  const myRun = simState.runName;
  simState.pollTimer = setInterval(async () => {
    // Stale-response guard: user switched model or resubmitted meanwhile.
    if (!simState.id || simState.id !== myId || simState.runName !== myRun) {
      clearInterval(simState.pollTimer);
      simState.pollTimer = null;
      return;
    }
    try {
      const res = await fetch(`/api/simulations/${myId}/status`);
      if (!res.ok) return;
      const data = await res.json();
      simState.status = data.status;
      simState.progress = data.progress || null;
      if (data.status === "done") {
        const rres = await fetch(`/api/simulations/${myId}/results`);
        const r = await rres.json();
        if (simState.id !== myId) return; // guard again after await
        simState.results = r.results;
        const r2 = r.results;
        const energyNote = r2.energy_valid === false
          ? ` · ⚠ 能量失衡 ${(r2.energy_imbalance_rel * 100).toFixed(0)}% (结果不可信)`
          : "";
        $("simStatus").textContent =
          `完成 · ${r2.time_s.length} 步 · 末速 ${(r2.rpm[r2.rpm.length - 1]).toFixed(0)} rpm${energyNote}`;
        $("simRunBtn").disabled = false;
        $("simPlayBtn").disabled = false;
        $("simResetBtn").disabled = false;
        const slider = $("simTimeSlider");
        slider.disabled = false;
        slider.max = r2.time_s.length - 1;
        slider.value = 0;
        simState.playTime = r2.time_s[0];
        drawCurves(r2);
        $("curvesPanel").style.display = "";
        clearInterval(simState.pollTimer);
        simState.pollTimer = null;
        startPlayback();
      } else if (data.status === "failed" || data.status === "rejected"
                 || data.status === "cancelled") {
        $("simStatus").textContent =
          `${{failed: "求解失败", rejected: "模型被拒绝", cancelled: "已取消"}[data.status]}: ${parseApiError(data.error)}`;
        $("simRunBtn").disabled = false;
        clearInterval(simState.pollTimer);
        simState.pollTimer = null;
      } else {
        $("simStatus").textContent = `运行中${fmtProgress(data.progress)}`;
      }
    } catch (err) { /* keep polling */ }
  }, 1000);
}

// --- physics-time playback -----------------------------------------------
function interpAt(t) {
  const r = simState.results;
  const T = r.time_s;
  if (t <= T[0]) return { i: 0, frac: 0 };
  if (t >= T[T.length - 1]) return { i: T.length - 2, frac: 1 };
  let lo = 0, hi = T.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (T[mid] <= t) lo = mid; else hi = mid;
  }
  return { i: lo, frac: (t - T[lo]) / (T[lo + 1] - T[lo]) };
}

function lerpArr(arr, i, f) {
  return arr[i] * (1 - f) + arr[i + 1] * f;
}

function applyPlaybackState() {
  const r = simState.results;
  if (!r) return;
  const t = simState.playTime;
  const { i, frac } = interpAt(t);
  const angle = lerpArr(r.rotor_angle_rad, i, frac);
  const rpm = lerpArr(r.rpm, i, frac);
  if (rotorGroup) rotorGroup.rotation.z = angle;
  const slider = $("simTimeSlider");
  if (slider && document.activeElement !== slider) slider.value = i + frac;
  $("simTimeLabel").textContent =
    `${(t * 1000).toFixed(2)} ms · ${rpm.toFixed(0)} rpm · ${simState.playSpeed}x`;
  drawCurvesCursor(t);
}

function updatePlayback(dtRender) {
  if (!simState.playing || !simState.results) return;
  const T = simState.results.time_s;
  simState.playTime += dtRender * simState.playSpeed;
  if (simState.playTime >= T[T.length - 1]) {
    simState.playTime = T[T.length - 1];
    pausePlayback();
  }
  applyPlaybackState();
}

function startPlayback() {
  if (!simState.results) return;
  const T = simState.results.time_s;
  if (simState.playTime >= T[T.length - 1]) simState.playTime = T[0];
  simState.playing = true;
  $("simPlayBtn").textContent = "⏸ 暂停";
}

function pausePlayback() {
  simState.playing = false;
  $("simPlayBtn").textContent = "▶ 播放";
}

function resetPlayback() {
  simState.playing = false;
  $("simPlayBtn").textContent = "▶ 播放";
  if (simState.results) {
    // Reset restores the run's ACTUAL initial angle, not zero.
    simState.playTime = simState.results.time_s[0];
    applyPlaybackState();
  } else {
    if (rotorGroup) rotorGroup.rotation.z = 0;
    $("simTimeLabel").textContent = "—";
  }
}

// --- curves with axes, units, legend, time cursor ------------------------
function drawCurves(r) {
  simState._curves = r; // cached for the cursor overlay
  const canvas = $("curvesCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const panels = [
    { data: r.rpm, color: "#7ee08a", label: "转速 [rpm]" },
    { data: r.torque_Nm, color: "#e0a87e", label: "转矩 [N·m]" },
    { data: r.currents_A.map(c => c[0]), color: "#7eaee0", label: "电流 ia [A]" },
  ];
  const n = r.time_s.length;
  const tMax = r.time_s[n - 1] * 1000; // ms
  const padL = 34, padR = 6, padT = 14, padB = 14;
  const plotW = W - padL - padR;
  const panelH = (H - padT - padB) / panels.length;
  ctx.font = "9px sans-serif";
  panels.forEach((s, pi) => {
    let min = Infinity, max = -Infinity;
    for (const v of s.data) { if (v < min) min = v; if (v > max) max = v; }
    if (max - min < 1e-12) { min -= 1; max += 1; }
    const y0 = padT + pi * panelH;
    const yToPx = (v) => y0 + panelH - 8 - ((v - min) / (max - min)) * (panelH - 14);
    // panel frame
    ctx.strokeStyle = "#2a3a48";
    ctx.strokeRect(padL, y0 + 2, plotW, panelH - 8);
    // series
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padL + (r.time_s[i] * 1000 / tMax) * plotW;
      const y = yToPx(s.data[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // scale labels
    ctx.fillStyle = "#7d94a4";
    ctx.fillText(max.toPrecision(3), 2, y0 + 10);
    ctx.fillText(min.toPrecision(3), 2, y0 + panelH - 8);
    // legend
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, padL + 4, y0 + 10);
  });
  // time axis
  ctx.fillStyle = "#7d94a4";
  ctx.fillText("0", padL, H - 3);
  const tLabel = `${tMax.toFixed(1)} ms`;
  ctx.fillText(tLabel, W - padR - ctx.measureText(tLabel).width - 4, H - 3);
}

function drawCurvesCursor(tSec) {
  const r = simState._curves;
  if (!r) return;
  const canvas = $("curvesCanvas");
  const ctx = canvas.getContext("2d");
  // cheap full redraw (cached series) + cursor
  drawCurves(r);
  const W = canvas.width, H = canvas.height;
  const tMax = r.time_s[r.time_s.length - 1] * 1000;
  if (tMax <= 0) return;
  const padL = 34, padR = 6;
  const plotW = W - padL - padR;
  const x = padL + (tSec * 1000 / tMax) * plotW;
  ctx.strokeStyle = "#ffffff88";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, 14);
  ctx.lineTo(x, H - 14);
  ctx.stroke();
}

$("simRunBtn").addEventListener("click", startSimulation);
$("simCancelBtn")?.addEventListener("click", async () => {
  if (!simState.id) return;
  try {
    await fetch(`/api/simulations/${simState.id}/cancel`, { method: "POST" });
    $("simStatus").textContent = "取消请求已发送…";
  } catch { /* */ }
});
$("simPlayBtn").addEventListener("click", () => {
  if (simState.playing) pausePlayback(); else startPlayback();
});
$("simResetBtn").addEventListener("click", resetPlayback);
$("simSpeed")?.addEventListener("change", (e) => {
  simState.playSpeed = parseFloat(e.target.value);
  if (simState.results) applyPlaybackState();
});
$("simTimeSlider").addEventListener("input", () => {
  if (!simState.results) return;
  pausePlayback();
  const r = simState.results;
  const i = Math.min(parseInt($("simTimeSlider").value, 10), r.time_s.length - 2);
  simState.playTime = lerpArr(r.time_s, i,
    parseFloat($("simTimeSlider").value) - i);
  applyPlaybackState();
});

// ---------------------------------------------------------------------------
// Render loop — playback advances by RENDER time so physics time is
// frame-rate independent (120 fps and 60 fps see the same angle at the
// same physical instant).
// ---------------------------------------------------------------------------
let last = performance.now();
let frameCount = 0;
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - last) / 1000.0, 0.1); // clamp tab-switch jumps
  last = now;
  controls.update();
  updatePlayback(dt);
  renderer.render(scene, camera);
  frameCount++;
  if (now - (animate._hud ?? now) > 500) {
    if (animate._hud != null) {
      $("fpsHud").textContent = `${Math.round(frameCount * 1000 / (now - animate._hud))} fps`;
    }
    animate._hud = now;
    frameCount = 0;
  }
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
});

// Go.
refreshRuns();
