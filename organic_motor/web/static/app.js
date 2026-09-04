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

let currentModel = null;
const materialNodes = new Map(); // name -> THREE.Object3D

function clearModel() {
  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((m) => m.dispose());
      }
    });
    currentModel = null;
  }
  materialNodes.clear();
}

function loadGlb(url) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      url,
      (gltf) => resolve(gltf.scene),
      undefined,
      (err) => reject(err)
    );
  });
}

async function showCheckpoint(run, step, level) {
  const overlay = $("loadOverlay");
  $("loadText").textContent = `加载第 ${step} 步网格…`;
  overlay.classList.remove("hidden");
  status(`加载第 ${step} 步…`, "busy");
  clearModel();
  try {
    const statorView = $("statorViewToggle")?.checked ?? false;
    const url = `/api/runs/${encodeURIComponent(run)}/checkpoint/${step}/glb`
      + `?level=${level}&smoothing=taubin&iterations=5&view=${statorView ? "stator" : "full"}`;
    const model = await loadGlb(url);
    currentModel = model;

    // Motor is centred on origin in metres; frame it.
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360)) * 1.7;
    camera.position
      .set(dist, dist * 0.75, dist * 1.1)
      .add(center);
    controls.update();

    // Index material nodes by name for per-material toggling.
    const matStyle = {
      iron: { color: 0x7a8a9c, metalness: 0.92, roughness: 0.38, env: 1.2 },
      copper: { color: 0xc87533, metalness: 0.95, roughness: 0.28, env: 1.4 },
      pm: { color: 0x8a2042, metalness: 0.5, roughness: 0.45, env: 0.8 },
      coolant: { color: 0x408cde, metalness: 0.1, roughness: 0.15, env: 1.0, opacity: 0.45 },
      insulator: { color: 0xe8e6da, metalness: 0.05, roughness: 0.6, env: 0.7 },
    };
    model.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        const name = (o.name || o.parent?.name || "").toLowerCase();
        for (const mat of Object.keys(matStyle)) {
          if (name.includes(mat)) {
            materialNodes.set(mat, o);
            const st = matStyle[mat];
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
          }
        }
      }
    });
    scene.add(model);
    overlay.classList.add("hidden");
    status(`第 ${step} 步`, "");
    syncMaterialToggles();
  } catch (err) {
    $("loadText").textContent = "加载失败";
    status(`加载失败: ${err.message || err}`, "error");
    console.error(err);
  }
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
    const present = materialNodes.has(mat);
    const row = document.createElement("label");
    row.className = "toggle";
    row.innerHTML = `
      <input type="checkbox" ${materialVisible[mat] ? "checked" : ""} ${!present ? "disabled" : ""} />
      <span class="swatch" style="background:${MATERIAL_COLORS[mat]}"></span>
      <span>${labelOf(mat)}${present ? "" : " (无)"}</span>`;
    const cb = row.querySelector("input");
    cb.addEventListener("change", () => {
      materialVisible[mat] = cb.checked;
      const node = materialNodes.get(mat);
      if (node) node.visible = cb.checked;
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
  const idx = sliceState.index ?? "";
  const url = `/api/runs/${encodeURIComponent(state.currentRun)}/checkpoint/${step}/slice?field=${sliceState.field}&axis=${sliceState.axis}&index=${idx}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    drawSlice(data);
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

$("statorViewToggle").addEventListener("change", async () => {
  if (state.stepIndex >= 0) await showCheckpoint(state.currentRun, state.steps[state.stepIndex], state.level);
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
// Render loop
// ---------------------------------------------------------------------------
let last = performance.now();
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  const now = performance.now();
  if (now - last > 500) {
    $("fpsHud").textContent = `${Math.round(1000 / (now - last))} fps`;
    last = now;
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
