import { baseFeatureIdOf } from './GraphManager.js';

export class DataManager {
  constructor(rawData) {
    this.allNodes = rawData.nodes || [];
    this.allEdges = rawData.edges || [];
    this.meta = rawData.meta || {};
    this.featScoreMax = new Map();
    this.featScoreSum = new Map();
    this.featureColMap = new Map(); // featureId(含shadow) -> colIndex 0..3
    this.featureOwner = new Map();   // (特征/影子特征) -> 归属 model
  }

  // 合并同名节点（模型/特征共用节点）
  mergeNodesEdgesByLabel(nodesArr, edgesArr) {
    const mergedId = (group, label) => `g:${group}:${String(label ?? "").trim()}`;
    const nodeByKey = new Map();
    const idMap = new Map();

    for (const n of nodesArr) {
      const key = mergedId(n.group, n.label);
      if (!nodeByKey.has(key)) {
        nodeByKey.set(key, {
          id: key,
          label: n.label,
          title: n.title,
          group: n.group
        });
      }
      idMap.set(n.id, key);
    }

    const outEdges = [];
    const seen = new Set();
    for (const e of edgesArr) {
      const from = idMap.get(e.from) || e.from;
      const to = idMap.get(e.to) || e.to;
      const type = e.type;

      if (from === to) continue;
      const k = `${from}|${to}|${type}`;
      if (seen.has(k)) {
        const last = outEdges.find(x => x._k === k);
        if (last) {
          const v0 = Number(last.value) || 0, v1 = Number(e.value) || 0;
          last.value = v0 + v1;
          last.label = String(Math.round(last.value));
          last.title = (type === "db_model") ? `使用次数: ${last.value}` : `使用频率: ${last.value}`;
        }
        continue;
      }
      seen.add(k);
      outEdges.push({
        ...e,
        from, to,
        value: Number(e.value) || 0,
        label: String(Math.round(Number(e.value) || 0)),
        title: e.title ?? "",
        _k: k
      });
    }

    const outNodes = Array.from(nodeByKey.values());
    return { nodes: outNodes, edges: outEdges, idMap };
  }

  // 检测共享特征
  detectSharedOriginal(nodesArr, edgesArr) {
    const m = new Map(); // fid -> Set(models)
    for (const e of edgesArr) {
      if (e.type !== "model_feature") continue;
      const fid = e.to, mid = e.from;
      if (!m.has(fid)) m.set(fid, new Set());
      m.get(fid).add(mid);
    }
    const shared = new Set();
    for (const [fid, mids] of m.entries()) {
      if (mids.size >= 2) shared.add(fid);
    }
    return { shared };
  }

  // 影子共享特征（可选）
  prepareRenderData(nodesArr, edgesArr, SHADOW_SHARED_FEATURES) {
    if (!SHADOW_SHARED_FEATURES) {
      return { nodes: nodesArr, edges: edgesArr };
    }

    const { shared } = this.detectSharedOriginal(nodesArr, edgesArr);
    const nodeById = new Map(nodesArr.map(n => [n.id, n]));
    const outNodes = [];
    const outEdges = [];

    for (const n of nodesArr) {
      if (n.group === "feature" && shared.has(n.id)) {
        continue; // 共享 feature 用影子代替
      }
      outNodes.push({ ...n });
    }

    const shadowSet = new Set();

    for (const e of edgesArr) {
      if (e.type !== "model_feature") {
        outEdges.push({ ...e });
        continue;
      }

      const fid = e.to, mid = e.from;

      if (!shared.has(fid)) {
        outEdges.push({ ...e });
        continue;
      }

      const sid = `shadow:${fid}:${mid}`;
      if (!shadowSet.has(sid)) {
        shadowSet.add(sid);
        const base = nodeById.get(fid) || { id: fid, label: String(fid), group: "feature" };
        outNodes.push({
          id: sid,
          label: base.label,
          title: (base.title || base.label || "") + "（共享特征-按模型复制）",
          group: "feature"
        });
      }

      outEdges.push({ ...e, to: sid });
    }

    return { nodes: outNodes, edges: outEdges };
  }

  // 计算特征分数（基于合并后的边）
  computeFeatureScoresFromMergedEdges(edgesMerged) {
    this.featScoreMax = new Map();
    this.featScoreSum = new Map();

    for (const e of edgesMerged) {
      if (e.type !== "model_feature") continue;
      const fid = e.to;
      const w = Math.max(0, Number(e.value) || 0);

      this.featScoreSum.set(fid, (this.featScoreSum.get(fid) || 0) + w);
      this.featScoreMax.set(fid, Math.max(this.featScoreMax.get(fid) || 0, w));
    }
  }

  // 获取选中的数据库标签
  getSelectedDbLabel() {
    const el = document.getElementById("dbSelect");
    return el ? String(el.value || "") : "";
  }

  // 通过标签找数据库节点ID
  findDbNodeIdByLabel(dbLabel, nodesDS) {
    if (!nodesDS || !dbLabel) return null;
    const nodes = nodesDS.get();
    const hit = nodes.find(n => n.group === "database" && String(n.label) === String(dbLabel));
    return hit ? hit.id : null;
  }

  // 计算数据库条件化特征统计
  computeDbConditionalFeatureStats(dbNodeId, edgesDS, nodesDS) {
    const empty = {
      sum: new Map(),
      max: new Map(),
      modelCount: new Map(),
      modelIds: new Set()
    };
    if (!dbNodeId || !edgesDS) return empty;

    const edges = edgesDS.get();
    const modelIds = new Set();
    for (const e of edges) {
      if (e.type !== "db_model") continue;
      if (e.from !== dbNodeId) continue;
      modelIds.add(e.to);
    }

    const sum = new Map();
    const mx = new Map();
    const modelCount = new Map();
    const modelSeenPerFeature = new Map(); // baseFid -> Set(modelId)

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

    for (const [fid] of sum.entries()) {
      modelCount.set(fid, modelSeenPerFeature.get(fid)?.size || 0);
    }

    return { sum, max: mx, modelCount, modelIds };
  }

  // 构建特征标题（带分数）
  featureTitleWithScore(nodeId, label, baseTitle, nodesDS, edgesDS) {
    const baseId = baseFeatureIdOf(nodeId);

    const gMax = Number(this.featScoreMax.get(baseId) || 0);
    const gSum = Number(this.featScoreSum.get(baseId) || 0);

    const title0 = String(baseTitle || label || "");
    const head = title0 ? title0 : String(label || "");

    const dbLabel = this.getSelectedDbLabel();
    if (!dbLabel) {
      return `${head}\n分数（global）：max=${gMax}，sum=${gSum}`;
    }

    const dbId = this.findDbNodeIdByLabel(dbLabel, nodesDS);
    const st = this.computeDbConditionalFeatureStats(dbId, edgesDS, nodesDS);
    const dMax = Number(st.max.get(baseId) || 0);
    const dSum = Number(st.sum.get(baseId) || 0);
    const dCnt = Number(st.modelCount.get(baseId) || 0);

    return `${head}\n分数（global）：max=${gMax}，sum=${gSum}\n分数（DB=${dbLabel}）：max=${dMax}，sum=${dSum}，model_count=${dCnt}`;
  }
}