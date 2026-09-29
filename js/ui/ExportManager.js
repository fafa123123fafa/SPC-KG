import { csvEscape, downloadText, isoStamp, variance } from '../utils/helpers.js';
import { baseFeatureIdOf } from '../core/GraphManager.js';

export class ExportManager {
  constructor(graphManager, dataManager) {
    this.graph = graphManager;
    this.data = dataManager;
  }

  // 导出全图PNG
  exportFullPNG() {
    try {
      const { network, nodesDS, edgesDS } = this.graph;
      if (!network || !nodesDS || !edgesDS) return;

      const W = 6400;
      const H = 3200;
      const SAFE_SCALE = 0.95;

      const ids = nodesDS.getIds();
      const pos = network.getPositions(ids);

      const nodes = nodesDS.get().map(n => {
        const p = pos[n.id];
        return {
          ...n,
          x: p ? p.x : n.x,
          y: p ? p.y : n.y,
          fixed: { x: true, y: true }
        };
      });

      const edges = edgesDS.get().map(e => ({ ...e }));

      const tmp = document.createElement("div");
      tmp.style.position = "fixed";
      tmp.style.left = "-10000px";
      tmp.style.top = "-10000px";
      tmp.style.width = W + "px";
      tmp.style.height = H + "px";
      tmp.style.background = "#ffffff";
      document.body.appendChild(tmp);

      const tmpNet = new vis.Network(
        tmp,
        { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
        {
          layout: { improvedLayout: false, randomSeed: 42 },
          interaction: { hover: false, dragView: false, zoomView: false },
          physics: { enabled: false }
        }
      );

      let stage = 0;
      const cleanup = () => {
        try { tmpNet.destroy(); } catch (e) { }
        try { tmp.remove(); } catch (e) { }
      };

      tmpNet.on("afterDrawing", () => {
        if (stage === 0) {
          stage = 1;
          tmpNet.fit({ animation: false });
          requestAnimationFrame(() => {
            try { tmpNet.moveTo({ scale: tmpNet.getScale() * SAFE_SCALE, animation: false }); } catch (e) { }
            stage = 2;
            tmpNet.redraw();
          });
          return;
        }
        if (stage === 2) {
          stage = 3;
          requestAnimationFrame(() => {
            try {
              const srcCanvas = tmpNet.canvas?.frame?.canvas;
              if (!srcCanvas) {
                cleanup();
                alert("导出失败：找不到 canvas。");
                return;
              }

              const out = document.createElement("canvas");
              out.width = srcCanvas.width;
              out.height = srcCanvas.height;

              const ctx = out.getContext("2d");
              ctx.fillStyle = "#FFFFFF";
              ctx.fillRect(0, 0, out.width, out.height);
              ctx.drawImage(srcCanvas, 0, 0);

              const url = out.toDataURL("image/png");
              const a = document.createElement("a");
              a.href = url;
              a.download = "knowledge-graph-full-2to1.png";
              document.body.appendChild(a);
              a.click();
              a.remove();
            } finally {
              cleanup();
            }
          });
        }
      });

      tmpNet.redraw();
      setTimeout(() => { if (stage < 3) cleanup(); }, 8000);

    } catch (err) {
      console.error(err);
      alert("导出失败：\n" + (err?.message || String(err)));
    }
  }

  // 导出特征列CSV
  exportFeatureColumnCSV(colIndex) {
    const { nodesDS, edgesDS, layoutMode } = this.graph;
    if (!nodesDS || !edgesDS) return;

    if (layoutMode !== "hier") {
      alert("当前不是层级布局（hier），列导出仅在层级布局下可用。");
      return;
    }

    const nodes = nodesDS.get();
    const edges = edgesDS.get();

    const nodeById = new Map(nodes.map(n => [n.id, n]));
    const dbByModel = new Map();   // modelId -> Set(dbId)
    const featByModel = new Map(); // modelId -> Array<{fid, value}>

    for (const e of edges) {
      if (e.type !== "db_model") continue;
      const db = e.from, m = e.to;
      if (!dbByModel.has(m)) dbByModel.set(m, new Set());
      dbByModel.get(m).add(db);
    }

    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      const m = e.from;
      const f = e.to;
      const v = Number(e.value) || 0;
      if (!featByModel.has(m)) featByModel.set(m, []);
      featByModel.get(m).push({ fid: f, value: v });
    }

    const rows = [];
    for (const [m, feats] of featByModel.entries()) {
      const dbs = dbByModel.get(m);
      if (!dbs || dbs.size === 0) continue;

      for (const { fid, value } of feats) {
        const col = this.data.featureColMap.get(fid);
        if (col !== colIndex) continue;

        const mLabel = nodeById.get(m)?.label ?? m;
        const fLabel = nodeById.get(fid)?.label ?? fid;

        for (const db of dbs) {
          const dbLabel = nodeById.get(db)?.label ?? db;
          rows.push({ database: dbLabel, model: mLabel, feature: fLabel, value });
        }
      }
    }

    if (rows.length === 0) {
      alert(`第 ${colIndex + 1} 列没有可导出的记录。`);
      return;
    }

    const header = ["database", "model", "feature", "value"];
    const lines = [header.join(",")];
    for (const r of rows) {
      lines.push([
        csvEscape(r.database),
        csvEscape(r.model),
        csvEscape(r.feature),
        csvEscape(r.value)
      ].join(","));
    }

    downloadText(`features_col${colIndex + 1}_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 导出补充表（明细）
  exportSupplementaryModelFeatureCSV() {
    const { nodesDS, edgesDS } = this.graph;
    if (!nodesDS || !edgesDS) return;

    const nodes = nodesDS.get();
    const edges = edgesDS.get();
    const nodeById = new Map(nodes.map(n => [n.id, n]));

    const dbByModel = new Map();
    for (const e of edges) {
      if (e.type !== "db_model") continue;
      if (!dbByModel.has(e.to)) dbByModel.set(e.to, new Set());
      dbByModel.get(e.to).add(e.from);
    }

    const rows = [];
    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      const m = e.from, f = e.to;
      const v = Number(e.value) || 0;

      const mLabel = nodeById.get(m)?.label ?? m;
      const fLabel = nodeById.get(f)?.label ?? f;

      const dbs = dbByModel.get(m);
      if (!dbs || dbs.size === 0) continue;

      for (const db of dbs) {
        const dbLabel = nodeById.get(db)?.label ?? db;
        rows.push({ database: dbLabel, model: mLabel, feature: fLabel, value: v });
      }
    }

    if (!rows.length) {
      alert("当前视图没有可导出的 model→feature 明细。");
      return;
    }

    const header = ["database", "model", "feature", "value"];
    const lines = [header.join(",")];
    for (const r of rows) {
      lines.push([
        csvEscape(r.database),
        csvEscape(r.model),
        csvEscape(r.feature),
        csvEscape(r.value)
      ].join(","));
    }

    downloadText(`supp_model_feature_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 导出论文表（DB-Feature汇总）
  exportPaperDBFeatureCSV() {
    const { nodesDS, edgesDS } = this.graph;
    if (!nodesDS || !edgesDS) return;

    const dbLabel = this.data.getSelectedDbLabel();
    if (!dbLabel) {
      alert("请先在左侧选择一个数据库（dbSelect），再导出论文表。");
      return;
    }

    const dbId = this.data.findDbNodeIdByLabel(dbLabel, nodesDS);
    if (!dbId) {
      alert("当前视图里找不到所选数据库节点。请应用筛选或重置后再试。");
      return;
    }

    const st = this.data.computeDbConditionalFeatureStats(dbId, edgesDS, nodesDS);
    const sum = st.sum, mx = st.max, cnt = st.modelCount;

    const nodes = nodesDS.get();
    const featureLabelByBase = new Map();
    for (const n of nodes) {
      if (n.group !== "feature") continue;
      const base = baseFeatureIdOf(n.id);
      if (!featureLabelByBase.has(base)) featureLabelByBase.set(base, n.label ?? base);
    }

    const rows = [];
    for (const [baseFid, s] of sum.entries()) {
      rows.push({
        database: dbLabel,
        feature: featureLabelByBase.get(baseFid) ?? baseFid,
        sum: Number(s) || 0,
        max: Number(mx.get(baseFid) || 0),
        model_count: Number(cnt.get(baseFid) || 0)
      });
    }

    if (!rows.length) {
      alert("该数据库在当前视图内没有可导出的 feature 汇总。");
      return;
    }

    rows.sort((a, b) => (b.sum - a.sum) || (b.max - a.max) || String(a.feature).localeCompare(String(b.feature)));

    const header = ["database", "feature", "sum", "max", "model_count"];
    const lines = [header.join(",")];
    for (const r of rows) {
      lines.push([
        csvEscape(r.database),
        csvEscape(r.feature),
        csvEscape(r.sum),
        csvEscape(r.max),
        csvEscape(r.model_count)
      ].join(","));
    }

    downloadText(`paper_db_feature_${dbLabel}_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 导出Global Feature Hierarchy
  exportGlobalFeatureHierarchyCSV() {
    if (!this.data.allNodes || !this.data.allEdges) {
      alert("数据尚未加载完成。");
      return;
    }

    const merged = this.data.mergeNodesEdgesByLabel(this.data.allNodes, this.data.allEdges);
    const edges = merged.edges;
    const nodes = merged.nodes;

    const featureLabel = new Map();
    for (const n of nodes) {
      if (n.group === "feature") {
        const base = baseFeatureIdOf(n.id);
        if (!featureLabel.has(base)) {
          featureLabel.set(base, n.label ?? base);
        }
      }
    }

    const sum = new Map();
    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      const fid = baseFeatureIdOf(e.to);
      const v = Number(e.value) || 0;
      sum.set(fid, (sum.get(fid) || 0) + v);
    }

    const items = Array.from(sum.entries()).map(([fid, s]) => ({
      fid,
      feature: featureLabel.get(fid) ?? fid,
      sum: s
    }));

    items.sort((a, b) => (b.sum - a.sum) || a.feature.localeCompare(b.feature));

    let rank = 0;
    let prev = null;
    for (let i = 0; i < items.length; i++) {
      if (prev === null || items[i].sum !== prev) {
        rank = i + 1;
        prev = items[i].sum;
      }
      items[i].rank = rank;
    }

    const header = ["feature", "global_sum", "global_rank"];
    const lines = [header.join(",")];
    for (const it of items) {
      lines.push([
        csvEscape(it.feature),
        csvEscape(it.sum),
        csvEscape(it.rank)
      ].join(","));
    }

    downloadText(`supp_global_feature_hierarchy_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 导出DB Feature Hierarchy
  exportDBFeatureHierarchyCSV() {
    if (!this.data.allNodes || !this.data.allEdges) {
      alert("数据尚未加载完成。");
      return;
    }

    const merged = this.data.mergeNodesEdgesByLabel(this.data.allNodes, this.data.allEdges);
    const nodes = merged.nodes;
    const edges = merged.edges;

    const dbIdByLabel = new Map();
    for (const n of nodes) {
      if (n.group === "database") {
        dbIdByLabel.set(String(n.label), n.id);
      }
    }

    const featureLabel = new Map();
    for (const n of nodes) {
      if (n.group === "feature") {
        const base = baseFeatureIdOf(n.id);
        if (!featureLabel.has(base)) {
          featureLabel.set(base, n.label ?? base);
        }
      }
    }

    const rows = [];

    for (const [dbLabel, dbId] of dbIdByLabel.entries()) {
      const st = this.computeDbConditionalFeatureStatsOnEdges(edges, dbId);

      const items = Array.from(st.sum.entries()).map(([fid, s]) => ({
        fid,
        feature: featureLabel.get(fid) ?? fid,
        sum: s
      }));

      items.sort((a, b) => (b.sum - a.sum) || a.feature.localeCompare(b.feature));

      let rank = 0;
      let prev = null;
      for (let i = 0; i < items.length; i++) {
        if (prev === null || items[i].sum !== prev) {
          rank = i + 1;
          prev = items[i].sum;
        }
        rows.push({
          database: dbLabel,
          feature: items[i].feature,
          sum_db: items[i].sum,
          rank_db: rank
        });
      }
    }

    if (!rows.length) {
      alert("未生成任何 DB-feature 记录。");
      return;
    }

    const header = ["database", "feature", "sum_db", "rank_db"];
    const lines = [header.join(",")];
    for (const r of rows) {
      lines.push([
        csvEscape(r.database),
        csvEscape(r.feature),
        csvEscape(r.sum_db),
        csvEscape(r.rank_db)
      ].join(","));
    }

    downloadText(`supp_db_feature_hierarchy_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 辅助：在边上计算DB条件化统计
  computeDbConditionalFeatureStatsOnEdges(edges, dbNodeId) {
    const empty = {
      sum: new Map(),
      max: new Map(),
      modelCount: new Map(),
      modelIds: new Set()
    };
    if (!dbNodeId || !Array.isArray(edges)) return empty;

    const modelIds = new Set();
    for (const e of edges) {
      if (e.type !== "db_model") continue;
      if (e.from !== dbNodeId) continue;
      modelIds.add(e.to);
    }

    const sum = new Map();
    const mx = new Map();
    const modelSeenPerFeature = new Map();

    for (const e of edges) {
      if (e.type !== "model_feature") continue;
      if (!modelIds.has(e.from)) continue;

      const baseFid = baseFeatureIdOf(e.to);
      const w = Math.max(0, Number(e.value) || 0);

      sum.set(baseFid, (sum.get(baseFid) || 0) + w);
      mx.set(baseFid, Math.max(mx.get(baseFid) || 0, w));

      if (!modelSeenPerFeature.has(baseFid)) modelSeenPerFeature.set(baseFid, new Set());
      modelSeenPerFeature.get(baseFid).add(e.from);
    }

    const modelCount = new Map();
    for (const [fid] of sum.entries()) {
      modelCount.set(fid, modelSeenPerFeature.get(fid)?.size || 0);
    }

    return { sum, max: mx, modelCount, modelIds };
  }

  // 导出Feature Rank Stability
  exportFeatureRankStabilityCSV() {
    if (!this.data.allNodes || !this.data.allEdges) {
      alert("数据尚未加载完成。");
      return;
    }

    const dbLabels = (this.data.meta?.databases || []).slice();
    if (dbLabels.length < 2) {
      alert("需要至少 2 个数据库才能做 Rank Stability。");
      return;
    }

    const merged = this.data.mergeNodesEdgesByLabel(this.data.allNodes, this.data.allEdges);
    const nodesMerged = merged.nodes;
    const edgesMerged = merged.edges;

    const dbIdByLabel = new Map();
    for (const n of nodesMerged) {
      if (n.group === "database") dbIdByLabel.set(String(n.label), n.id);
    }

    const featureLabelByBase = new Map();
    for (const n of nodesMerged) {
      if (n.group !== "feature") continue;
      const base = baseFeatureIdOf(n.id);
      if (!featureLabelByBase.has(base)) featureLabelByBase.set(base, n.label ?? base);
    }
    const allBaseFeatures = Array.from(featureLabelByBase.keys()).sort((a, b) =>
      String(featureLabelByBase.get(a)).localeCompare(String(featureLabelByBase.get(b)))
    );

    if (allBaseFeatures.length === 0) {
      alert("未找到 feature 节点。");
      return;
    }

    const buildRanksByScore = (items) => {
      items.sort((a, b) => (b.score - a.score) || String(a.label).localeCompare(String(b.label)));

      const rankByFid = new Map();
      let prevScore = null;
      let rank = 0;
      let seen = 0;

      for (const it of items) {
        seen += 1;
        if (prevScore === null || it.score !== prevScore) {
          rank = seen;
          prevScore = it.score;
        }
        rankByFid.set(it.fid, rank);
      }
      return rankByFid;
    };

    const rankByDb = new Map();
    const scoreByDb = new Map();

    for (const dbLabel of dbLabels) {
      const dbId = dbIdByLabel.get(String(dbLabel));
      if (!dbId) {
        const zeroScore = new Map(allBaseFeatures.map(fid => [fid, 0]));
        scoreByDb.set(dbLabel, zeroScore);
        const items0 = allBaseFeatures.map(fid => ({ fid, score: 0, label: featureLabelByBase.get(fid) ?? fid }));
        rankByDb.set(dbLabel, buildRanksByScore(items0));
        continue;
      }

      const st = this.computeDbConditionalFeatureStatsOnEdges(edgesMerged, dbId);
      const sumMap = new Map();

      for (const fid of allBaseFeatures) {
        sumMap.set(fid, Number(st.sum.get(fid) || 0));
      }
      scoreByDb.set(dbLabel, sumMap);

      const items = allBaseFeatures.map(fid => ({
        fid,
        score: sumMap.get(fid) || 0,
        label: featureLabelByBase.get(fid) ?? fid
      }));
      rankByDb.set(dbLabel, buildRanksByScore(items));
    }

    const safeCol = (s) => String(s).trim().replace(/\s+/g, "_").replace(/[^\w\u4e00-\u9fa5]/g, "_");
    const rankCols = dbLabels.map(d => `rank_in_${safeCol(d)}`);

    const header = ["feature", ...rankCols, "rank_variance"];
    const lines = [header.join(",")];

    for (const fid of allBaseFeatures) {
      const featLabel = featureLabelByBase.get(fid) ?? fid;
      const ranks = dbLabels.map(db => Number(rankByDb.get(db)?.get(fid) ?? NaN));
      const v = variance(ranks);

      const row = [
        csvEscape(featLabel),
        ...ranks.map(x => csvEscape(Number.isFinite(x) ? x : "")),
        csvEscape(v)
      ];
      lines.push(row.join(","));
    }

    downloadText(`feature_rank_stability_${isoStamp()}.csv`, lines.join("\n"), "text/csv;charset=utf-8");
  }

  // 保存当前标签状态
  snapshotNodeLabels() {
    const { nodesDS } = this.graph;
    if (!nodesDS) return [];
    return nodesDS.get().map(n => ({ id: n.id, label: n.label ?? "" }));
  }

  // 恢复标签状态
  restoreNodeLabels(snapshot) {
    const { nodesDS, network } = this.graph;
    if (!nodesDS || !Array.isArray(snapshot)) return;
    nodesDS.update(snapshot);
    try { if (network) network.redraw(); } catch (_) { }
  }

  // 导出当前视图为 SVG（矢量）
  exportCurrentSVG(filename = `knowledge-graph-${isoStamp()}.svg`) {
    try {
      const { network, nodesDS, edgesDS } = this.graph;
      if (!network || !nodesDS || !edgesDS) return;

      const ids = nodesDS.getIds();
      const pos = network.getPositions(ids);
      const nodes = nodesDS.get();
      const edges = edgesDS.get();

      const placedNodes = nodes.map(n => {
        const p = pos[n.id];
        return {
          ...n,
          x: p ? p.x : (n.x || 0),
          y: p ? p.y : (n.y || 0)
        };
      });

      if (!placedNodes.length) {
        alert("没有可导出的节点。");
        return;
      }

      const esc = (s) => String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");

      const colorOf = (c, fallback = "#999999") => {
        if (!c) return fallback;
        if (typeof c === "string") return c;
        if (typeof c.color === "string") return c.color;
        if (typeof c.background === "string") return c.background;
        if (typeof c.border === "string") return c.border;
        return fallback;
      };

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

      for (const n of placedNodes) {
        const size = Number(n.size || 20);
        const fontSize = Number(n.font?.size || 16);
        const padX = Math.max(50, size * 2 + fontSize * 2.5);
        const padY = Math.max(40, size * 2 + fontSize * 1.8);

        minX = Math.min(minX, n.x - padX);
        maxX = Math.max(maxX, n.x + padX);
        minY = Math.min(minY, n.y - padY);
        maxY = Math.max(maxY, n.y + padY);
      }

      const margin = 120;
      minX -= margin;
      maxX += margin;
      minY -= margin;
      maxY += margin;

      const W = Math.max(100, maxX - minX);
      const H = Math.max(100, maxY - minY);

      const sx = (x) => x - minX;
      const sy = (y) => y - minY;

      const nodeById = new Map(placedNodes.map(n => [n.id, n]));

      const defs = `
        <defs>
          <marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#AAB7B8"></path>
          </marker>
          <marker id="arrowOrange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#F39C12"></path>
          </marker>
        </defs>
      `;

      const edgeSvg = [];
      for (const e of edges) {
        const a = nodeById.get(e.from);
        const b = nodeById.get(e.to);
        if (!a || !b) continue;

        const x1 = sx(a.x);
        const y1 = sy(a.y);
        const x2 = sx(b.x);
        const y2 = sy(b.y);

        const stroke = colorOf(e.color, e.type === "db_model" ? "#AAB7B8" : "#F39C12");
        const strokeWidth = Number(e.width || 2);
        const marker = e.type === "db_model" ? "url(#arrowGray)" : "url(#arrowOrange)";

        edgeSvg.push(
          `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${esc(stroke)}" stroke-width="${strokeWidth}" stroke-linecap="round" marker-end="${marker}" opacity="0.95" />`
        );
      }

      const nodeSvg = [];
      const textSvg = [];

      for (const n of placedNodes) {
        const x = sx(n.x);
        const y = sy(n.y);

        const fill = colorOf(n.color?.background || n.color, "#999999");
        const stroke = colorOf(n.color?.border || n.color, fill);

        const font = n.font || {};
        const fontColor = font.color || "#000000";
        const fontSize = Number(font.size || 16);
        const label = n.label ?? "";
        const size = Number(n.size || 20);

        if (n.group === "database" || n.shape === "box") {
          const w = Math.max(90, label.length * fontSize * 0.62 + 34);
          const h = Math.max(38, fontSize * 1.55);
          const rx = 8;

          nodeSvg.push(
            `<rect x="${(x - w / 2).toFixed(2)}" y="${(y - h / 2).toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}" rx="${rx}" ry="${rx}" fill="${esc(fill)}" stroke="${esc(stroke)}" stroke-width="2" />`
          );

          if (label) {
            textSvg.push(
              `<text x="${x.toFixed(2)}" y="${(y + fontSize * 0.34).toFixed(2)}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="${fontSize}" font-weight="700" fill="${esc(fontColor)}">${esc(label)}</text>`
            );
          }
        } else {
          nodeSvg.push(
            `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${size.toFixed(2)}" fill="${esc(fill)}" stroke="${esc(stroke)}" stroke-width="1.6" />`
          );

          if (label) {
            let tx = x;
            let ty = y + size + fontSize * 0.95;
            let anchor = "middle";

            if (n.group === "model") {
              ty = y - size - 10;
            }

            textSvg.push(
              `<text x="${tx.toFixed(2)}" y="${ty.toFixed(2)}" text-anchor="${anchor}" font-family="Arial, Helvetica, sans-serif" font-size="${fontSize}" fill="${esc(fontColor)}">${esc(label)}</text>`
            );
          }
        }
      }

      const svg = [
        `<?xml version="1.0" encoding="UTF-8" standalone="no"?>`,
        `<svg xmlns="http://www.w3.org/2000/svg" width="${W.toFixed(0)}" height="${H.toFixed(0)}" viewBox="0 0 ${W.toFixed(2)} ${H.toFixed(2)}">`,
        defs,
        `<rect x="0" y="0" width="${W.toFixed(2)}" height="${H.toFixed(2)}" fill="#ffffff" />`,
        `<g class="edges">${edgeSvg.join("\n")}</g>`,
        `<g class="nodes">${nodeSvg.join("\n")}</g>`,
        `<g class="labels">${textSvg.join("\n")}</g>`,
        `</svg>`
      ].join("\n");

      downloadText(filename, svg, "image/svg+xml;charset=utf-8");
    } catch (err) {
      console.error(err);
      alert("导出 SVG 失败：\n" + (err?.message || String(err)));
    }
  }

  // 导出论文版 SVG
  // mode:
  // - structure: 只保留 database / model 标签
  // - feature-top: 显示 topN 个 feature 标签
  exportPaperSVG({
    filename = `knowledge-graph-paper-${isoStamp()}.svg`,
    mode = "structure",
    topN = 30
  } = {}) {
    const { nodesDS, edgesDS, network } = this.graph;
    if (!nodesDS || !edgesDS || !network) return;

    const snapshot = this.snapshotNodeLabels();

    try {
      const nodes = nodesDS.get();
      const edges = edgesDS.get();

      if (mode === "structure") {
        const ups = [];
        for (const n of nodes) {
          if (n.group === "feature") {
            ups.push({ id: n.id, label: "" });
          } else {
            ups.push({ id: n.id, label: n.label ?? "" });
          }
        }
        nodesDS.update(ups);
      } else if (mode === "feature-top") {
        const scoreMap = new Map();
        for (const e of edges) {
          if (e.type !== "model_feature") continue;
          const fid = baseFeatureIdOf(e.to);
          const v = Number(e.value) || 0;
          scoreMap.set(fid, (scoreMap.get(fid) || 0) + v);
        }

        const topFeatureIds = new Set(
          Array.from(scoreMap.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN)
            .map(([fid]) => fid)
        );

        const ups = [];
        for (const n of nodes) {
          if (n.group !== "feature") {
            ups.push({ id: n.id, label: n.label ?? "" });
            continue;
          }
          const base = baseFeatureIdOf(n.id);
          ups.push({ id: n.id, label: topFeatureIds.has(base) ? (n.label ?? "") : "" });
        }
        nodesDS.update(ups);
      }

      try { network.redraw(); } catch (_) { }
      this.exportCurrentSVG(filename);

    } finally {
      this.restoreNodeLabels(snapshot);
    }
  }
}
