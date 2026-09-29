// ========== DOM 工具 ==========
export function byId(id) {
  return document.getElementById(id);
}

// ========== 加载动画 ==========
export function setLoading(visible, text, percent) {
  const overlay = byId("loadingOverlay");
  const t = byId("loadingText");
  const bar = byId("loadingBarInner");
  const pct = byId("loadingPct");

  if (visible) {
    overlay.style.display = "flex";
    if (typeof text === "string") t.textContent = text;
    const p = Math.max(0, Math.min(100, Number(percent ?? 0)));
    bar.style.width = p.toFixed(1) + "%";
    pct.textContent = p.toFixed(0) + "%";
  } else {
    overlay.style.display = "none";
  }
}

// ========== 错误提示 ==========
export function showErr(msg) {
  const el = byId("errBar");
  el.style.display = "block";
  el.textContent = String(msg);
}

// ========== CSV 导出工具 ==========
export function csvEscape(v) {
  const s = String(v ?? "");
  if (/[,"\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function downloadText(filename, text, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export function isoStamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
}

// ========== 随机数生成（确定性） ==========
export function hashToSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  return function () {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function rngFor(key) {
  return mulberry32(hashToSeed(String(key)));
}

// ========== 数值处理 ==========
export function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

export function safeColor(raw, fallback) {
  const s = String(raw || "").trim();
  if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(s)) return s;
  return fallback;
}

export function safeNum(raw, fallback, min, max) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return clamp(n, min, max);
}

export function variance(nums) {
  const arr = nums.filter(x => Number.isFinite(x));
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const v = arr.reduce((s, x) => s + (x - mean) * (x - mean), 0) / arr.length;
  return v;
}

// ========== 下一帧 Promise ==========
export function nextFrame() {
  return new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
}