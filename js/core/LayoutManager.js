import { TARGET_ASPECT } from '../utils/constants.js';
import { rngFor, hashToSeed } from '../utils/helpers.js';
import { baseFeatureIdOf } from './GraphManager.js';

export class LayoutManager {
  constructor(graphManager, dataManager, styleManager) {
    this.graph = graphManager;      // GraphManager实例
    this.data = dataManager;        // DataManager实例
    this.style = styleManager;      // StyleManager实例
    this.featureOwner = new Map();  // 特征归属模型（用于聚类）
  }

  // ======== 新增：估算特征节点最小安全间距 ========
  _featureMinDist() {
    const featureSize = Number(this.style?.styleCfg?.feature?.nodeSize ?? 22);
    const featureFont = Number(this.style?.styleCfg?.feature?.fontSize ?? 12);

    return {
      yGap: Math.max(32, featureSize * 1.25, featureFont * 0.95),
      xyGap: Math.max(34, featureSize * 1.35, featureFont * 1.00),
      ringPush: Math.max(18, featureSize * 0.75)
    };
  }

  // ======== 新增：层级布局，同列内沿Y方向做轻量避碰 ========
  _resolveHierarchicalColumnOverlap(items, minGap) {
    if (!items || items.length <= 1) return items;

    // items: [{id, x, y}]
    const arr = items.slice().sort((a, b) => (a.y - b.y) || (String(a.id) > String(b.id) ? 1 : -1));

    // 前向扫描：如果太近就往下推
    for (let i = 1; i < arr.length; i++) {
      const prev = arr[i - 1];
      const cur = arr[i];
      if (cur.y - prev.y < minGap) {
        cur.y = prev.y + minGap;
      }
    }

    // 保持列中心不漂移太多
    const oldCenter = items.reduce((s, it) => s + it.y, 0) / items.length;
    const newCenter = arr.reduce((s, it) => s + it.y, 0) / arr.length;
    const shift = oldCenter - newCenter;
    for (const it of arr) it.y += shift;

    // 再做一次前向修正，防止回移后再次贴近
    for (let i = 1; i < arr.length; i++) {
      const prev = arr[i - 1];
      const cur = arr[i];
      if (cur.y - prev.y < minGap) {
        cur.y = prev.y + minGap;
      }
    }

    return arr;
  }

  // ======== 新增：环形布局，对同一模型簇的特征节点做轻量避碰 ========
  _resolveClusterOverlapForModel(featureItems, center, minDist, pushStep) {
    if (!featureItems || featureItems.length <= 1) return featureItems;

    // 只在同一模型簇内部做少量迭代，避免破坏原来美观的环状结构
    const arr = featureItems.map(it => ({ ...it }));
    const MAX_ITER = 6;

    for (let iter = 0; iter < MAX_ITER; iter++) {
      let moved = false;

      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const a = arr[i];
          const b = arr[j];

          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.hypot(dx, dy);

          if (d >= minDist || d === 0) continue;

          // 谁更靠里，谁往外推一点，尽量不破坏原有美观
          const ra = Math.hypot(a.x - center.x, a.y - center.y);
          const rb = Math.hypot(b.x - center.x, b.y - center.y);

          const target = ra <= rb ? a : b;

          let vx = target.x - center.x;
          let vy = target.y - center.y;
          let vr = Math.hypot(vx, vy);

          if (vr < 1e-6) {
            // 极端情况给个稳定随机方向
            const rng = rngFor(`push:${target.id}`);
            const ang = rng() * Math.PI * 2;
            vx = Math.cos(ang);
            vy = Math.sin(ang);
            vr = 1;
          }

          vx /= vr;
          vy /= vr;

          const need = Math.min(pushStep, (minDist - d) * 0.7);
          target.x += vx * need;
          target.y += vy * need;
          moved = true;
        }
      }

      if (!moved) break;
    }

    return arr;
  }

  // 确保宽高比
  ensureAspectByScalingX(targetAspect = TARGET_ASPECT) {
    const { nodesDS, network } = this.graph;
    if (!nodesDS || !network) return;

    const ids = nodesDS.getIds();
    const pos = network.getPositions(ids);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    let cx = 0, cy = 0, cnt = 0;

    for (const id of ids) {
      const p = pos[id];
      if (!p) continue;
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
      cx += p.x;
      cy += p.y;
      cnt++;
    }
    if (!cnt) return;
    cx /= cnt;
    cy /= cnt;

    const w = Math.max(1e-6, maxX - minX);
    const h = Math.max(1e-6, maxY - minY);
    if (w / h >= targetAspect) return;

    const scaleX = (targetAspect * h) / w;
    const updates = [];
    for (const id of ids) {
      const p = pos[id];
      if (!p) continue;
      const x = cx + (p.x - cx) * scaleX;
      updates.push({ id, x, y: p.y });
    }
    nodesDS.update(updates);
    network.redraw();
  }

  // 冻结所有节点（固定位置）- 使用graph的方法
  freezeGraphHard() {
    this.graph._freezeGraphHard();
  }

  // 解冻所有节点 - 使用graph的方法
  unfreezeGraph() {
    this.graph._unfreezeGraph();
  }

  // ========== 层级布局 ==========
  layoutHierarchicalStable() {
    const { nodesDS, edgesDS, network } = this.graph;
    if (!nodesDS || !edgesDS || !network) return;

    const nodes = nodesDS.get();
    const edges = edgesDS.get();

    const dbs = nodes.filter(n => n.group === "database").map(n => n.id);
    const models = nodes.filter(n => n.group === "model").map(n => n.id);
    const feats = nodes.filter(n => n.group === "feature").map(n => n.id);

    // X 坐标（列的水平位置，控制列之间的水平间距）
    const X_DB = -4;       // 数据库列的X坐标
    const X_M = -2;         // 模型列的X坐标
    const X_F_NEAR = 0;     // 特征列的基准X坐标
    const F_COLS = 4;       // 特征列的总列数（默认4列）
    const F_COL_GAP2 = 2;   // 特征列之间的水平间距（核心！每列X偏移量）
    const F_GAP2 = 25;      // 特征列内节点的垂直间距

    // Y 间距（行间距）
    const DB_GAP = 250;     // 数据库列内节点的垂直间距
    const M_GAP = 200;      // 模型列内节点的垂直间距

    const updates = [];
    const distCfg = this._featureMinDist();

    // 1) 数据库节点布局
    const dbY = new Map();
    const dbSorted = [...dbs].sort();
    const dbStartY = -(dbSorted.length - 1) * DB_GAP / 2;
    for (let i = 0; i < dbSorted.length; i++) {
      const id = dbSorted[i];
      const y = dbStartY + i * DB_GAP;
      dbY.set(id, y);
      updates.push({ id, x: X_DB, y });
    }

    // 2) 模型节点布局（基于连接的数据库平均Y）
    const modelY0 = new Map();
    for (const mid of models) {
      let s = 0, c = 0;
      for (const e of edges) {
        if (e.type !== "db_model") continue;
        if (e.to !== mid) continue;
        const y = dbY.get(e.from);
        if (Number.isFinite(y)) { s += y; c++; }
      }
      modelY0.set(mid, c ? (s / c) : 0);
    }
    const modelsSorted = [...models].sort((a, b) => (modelY0.get(a) - modelY0.get(b)) || (a > b ? 1 : -1));
    const mStartY = -(modelsSorted.length - 1) * M_GAP / 2;
    const modelY = new Map();
    for (let i = 0; i < modelsSorted.length; i++) {
      const id = modelsSorted[i];
      const y = mStartY + i * M_GAP;
      modelY.set(id, y);
      updates.push({ id, x: X_M, y });
    }

    // 3) 特征节点目标Y（减少交叉）
    const featY0 = new Map();
    const featW = new Map();
    for (const fid of feats) {
      featY0.set(fid, 0);
      featW.set(fid, 0);
    }
    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      const mid = e.from, fid = e.to;
      const my = modelY.get(mid);
      if (!Number.isFinite(my)) continue;
      const w = Math.max(1, Number(e.value) || 1);
      featY0.set(fid, (featY0.get(fid) || 0) + my * w);
      featW.set(fid, (featW.get(fid) || 0) + w);
    }
    const featTarget = feats.map(fid => {
      const w = featW.get(fid) || 0;
      const y = w ? (featY0.get(fid) / w) : 0;
      return { fid, y };
    }).sort((a, b) => (a.y - b.y) || (a.fid > b.fid ? 1 : -1));

    // 特征分列：按重要性（选择DB时用条件化sum，否则用global sum）
    const selectedDbLabel = this.data.getSelectedDbLabel();
    let dStats = null;
    if (selectedDbLabel) {
      const dbId = this.data.findDbNodeIdByLabel(selectedDbLabel, nodesDS);
      if (dbId) dStats = this.data.computeDbConditionalFeatureStats(dbId, edgesDS, nodesDS);
    }

    const scoreOf = (fid) => {
      const base = baseFeatureIdOf(fid);
      if (dStats && selectedDbLabel) {
        return Number(dStats.sum.get(base) || 0);
      }
      return Number(this.data.featScoreSum.get(base) || 0);
    };

    // log压缩归一化
    let maxScore = 0;
    for (const it of featTarget) maxScore = Math.max(maxScore, scoreOf(it.fid));
    const denom = Math.log1p(maxScore) || 1;

    function colByImportance(s) {
      const norm = Math.log1p(s) / denom; // 0..1
      let col = Math.floor((1 - norm) * F_COLS); // 高分 -> col 0
      if (col < 0) col = 0;
      if (col > F_COLS - 1) col = F_COLS - 1;
      return col;
    }

    const perCol = Array.from({ length: F_COLS }, () => []);
    for (const it of featTarget) {
      const col = colByImportance(scoreOf(it.fid));
      perCol[col].push(it.fid);
      this.data.featureColMap.set(it.fid, col);
    }

    // ===== 保留原来的列布局，只在列内做轻量避碰 =====
    for (let col = 0; col < F_COLS; col++) {
      const arr = perCol[col];
      const x = X_F_NEAR + col * F_COL_GAP2;
      const y0 = -(arr.length - 1) * F_GAP2 / 2;

      const colItems = [];
      for (let i = 0; i < arr.length; i++) {
        const fid = arr[i];
        const y = y0 + i * F_GAP2;
        colItems.push({ id: fid, x, y });
      }

      const fixedItems = this._resolveHierarchicalColumnOverlap(colItems, distCfg.yGap);
      for (const it of fixedItems) {
        updates.push({ id: it.id, x: it.x, y: it.y });
      }
    }

    nodesDS.update(updates.map(u => ({ ...u, fixed: { x: true, y: true } })));
    network.redraw();

    this.ensureAspectByScalingX(TARGET_ASPECT);
    this.freezeGraphHard();
  }

  // ========== 聚类环形布局 ==========
  layoutClusterRingStable() {
    const { nodesDS, edgesDS, network } = this.graph;
    if (!nodesDS || !edgesDS || !network) return;

    const nodes = nodesDS.get();
    const edges = edgesDS.get();

    const dbs = nodes.filter(n => n.group === "database").map(n => n.id);
    const models = nodes.filter(n => n.group === "model").map(n => n.id);
    const feats = nodes.filter(n => n.group === "feature").map(n => n.id);

    const m = models.length || 1;
    const RX = 1100 + Math.min(1200, Math.sqrt(m) * 120);
    const RY = 520 + Math.min(900, Math.sqrt(m) * 90);

    const distCfg = this._featureMinDist();

    // 模型位置（椭圆上）
    const modelPos = new Map();
    const modelsSorted = [...models].sort();
    for (let i = 0; i < modelsSorted.length; i++) {
      const mid = modelsSorted[i];
      const ang = (i / m) * Math.PI * 2;
      const x = RX * Math.cos(ang);
      const y = RY * Math.sin(ang);
      modelPos.set(mid, { x, y, ang });
    }

    // 特征归属：最大value的模型
    const perModel = new Map();
    const featBest = new Map();

    this.featureOwner.clear();

    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      const mid = e.from, fid = e.to;
      const w = Math.max(1, Number(e.value) || 1);
      const prev = featBest.get(fid);
      if (!prev || w > prev.w) {
        featBest.set(fid, { mid, w });
      }
    }

    for (const fid of feats) {
      const best = featBest.get(fid);
      if (!best) continue;
      this.featureOwner.set(fid, best.mid);
      if (!perModel.has(best.mid)) perModel.set(best.mid, []);
      perModel.get(best.mid).push({ fid, w: best.w });
    }

    // 数据库位置（围绕其关联的模型）
    const dbPos = new Map();
    const dbModels = new Map();
    for (const e of edges) {
      if (e.type !== "db_model") continue;
      if (!dbModels.has(e.from)) dbModels.set(e.from, new Set());
      dbModels.get(e.from).add(e.to);
    }

    for (const db of dbs) {
      const mids = dbModels.get(db) ? Array.from(dbModels.get(db)) : [];
      if (!mids.length) {
        dbPos.set(db, { x: -RX - 500, y: 0 });
        continue;
      }
      let sx = 0, sy = 0, c = 0;
      for (const mid of mids) {
        const p = modelPos.get(mid);
        if (!p) continue;
        sx += p.x;
        sy += p.y;
        c++;
      }
      sx /= Math.max(1, c);
      sy /= Math.max(1, c);

      const len = Math.hypot(sx, sy) || 1;
      const ux = sx / len, uy = sy / len;
      const x = sx + ux * 420;
      const y = sy + uy * 420;
      dbPos.set(db, { x, y });
    }

    const updates = [];

    // 放置模型
    for (const mid of modelsSorted) {
      const p = modelPos.get(mid);
      updates.push({ id: mid, x: p.x, y: p.y });
    }

    // 放置数据库
    for (const db of dbs) {
      const p = dbPos.get(db) || { x: -RX - 500, y: 0 };
      updates.push({ id: db, x: p.x, y: p.y });
    }

    // ===== 保留原来的环形布局，只在每个模型簇内部做轻量避碰 =====
    const featureSize = this.style.styleCfg.feature?.nodeSize ?? 22;
    const featureFont = this.style.styleCfg.feature?.fontSize ?? 12;
    const MIN_ARC = Math.max(28, featureFont * 2.0 + featureSize * 0.8);
    const BASE_RING_GAP = Math.max(68, featureSize * 2.2);

    for (const mid of modelsSorted) {
      const mp = modelPos.get(mid);
      if (!mp) continue;

      const arr = (perModel.get(mid) || []).slice().sort((a, b) => b.w - a.w);
      if (!arr.length) continue;

      const count = arr.length;
      const ringGap = BASE_RING_GAP + Math.min(90, Math.sqrt(count) * 3.0);
      let r0 = 170 + Math.min(260, Math.sqrt(count) * 9);
      const MAX_RINGS = 14;

      const rings = [];
      let remaining = count;
      let r = r0;
      for (let k = 0; k < MAX_RINGS && remaining > 0; k++) {
        const cap = Math.max(10, Math.floor((2 * Math.PI * r) / MIN_ARC));
        rings.push({ r, cap });
        remaining -= cap;
        r += ringGap;
      }
      if (remaining > 0 && rings.length) rings[rings.length - 1].cap += remaining;

      const phase = mp.ang + ((hashToSeed(mid) % 360) / 360) * Math.PI * 2;

      let idx = 0;
      const placedForThisModel = [];

      for (let ri = 0; ri < rings.length; ri++) {
        const rr = rings[ri].r;
        const take = Math.min(rings[ri].cap, count - idx);
        if (take <= 0) continue;

        for (let i = 0; i < take; i++) {
          const it = arr[idx++];
          const fid = it.fid;

          const t = (i / take) * Math.PI * 2;
          const rng = rngFor(`cluster:${mid}:${fid}`);

          const jitterA = (ri === 0) ? (rng() - 0.5) * (Math.PI / 90) : (rng() - 0.5) * (Math.PI / 18);
          const jitterR = (ri === 0) ? (rng() - 0.5) * (ringGap * 0.08) : (rng() - 0.5) * (ringGap * 0.18);

          const ang = phase + t + jitterA;
          const x = mp.x + (rr + jitterR) * Math.cos(ang);
          const y = mp.y + (rr + jitterR) * Math.sin(ang);

          placedForThisModel.push({ id: fid, x, y });
        }
      }

      // 新增：只对该模型簇内部做轻量避碰，尽量保持原本圆环外观
      const fixedForThisModel = this._resolveClusterOverlapForModel(
        placedForThisModel,
        { x: mp.x, y: mp.y },
        distCfg.xyGap,
        distCfg.ringPush
      );

      for (const it of fixedForThisModel) {
        updates.push({ id: it.id, x: it.x, y: it.y });
      }
    }

    // 未放置的特征
    const placed = new Set(this.featureOwner.keys());
    const unplaced = feats.filter(fid => !placed.has(fid));
    if (unplaced.length) {
      const x = RX + 900;
      const gap = 40;
      const y0 = -(unplaced.length - 1) * gap / 2;
      for (let i = 0; i < unplaced.length; i++) {
        updates.push({ id: unplaced[i], x, y: y0 + i * gap });
      }
    }

    nodesDS.update(updates.map(u => ({ ...u, fixed: { x: true, y: true } })));
    network.redraw();

    this.ensureAspectByScalingX(TARGET_ASPECT);
    this.freezeGraphHard();
  }

  // 应用当前布局
  applyLayout() {
    const { network, nodesDS, edgesDS } = this.graph;
    if (!network || !nodesDS || !edgesDS) return;

    this.unfreezeGraph();
    try { network.setOptions({ physics: { enabled: false } }); } catch (e) { }

    if (this.graph.layoutMode === "cluster") {
      this.layoutClusterRingStable();
    } else {
      this.layoutHierarchicalStable();
    }

    try { network.fit({ animation: { duration: 320 } }); } catch (e) { }
    this.freezeGraphHard();
  }
}
