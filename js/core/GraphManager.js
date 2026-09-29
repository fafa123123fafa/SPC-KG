import { byId, setLoading } from '../utils/helpers.js';
import { enableWheelZoom, enableManualDragForFixedNodes, bindClickHighlight, reorderNodesByLayer } from './InteractionManager.js';

// 基础工具函数（导出供其他模块使用）
export function baseFeatureIdOf(nodeId) {
  const id = String(nodeId || "");
  if (!id.startsWith("shadow:")) return id;
  const last = id.lastIndexOf(":");
  if (last <= 7) return id.slice(7);
  return id.substring(7, last); // shadow:<fid>:<mid> -> <fid>
}

export class GraphManager {
  constructor(containerId) {
    this.container = byId(containerId);
    this.nodesDS = null;
    this.edgesDS = null;
    this.network = null;
    this.physicsEnabled = false;
    this.layoutMode = "hier";
    this.currentOriginalNodes = [];
    this.currentOriginalEdges = [];
  }

  // 设置物理引擎状态
  setPhysicsEnabled(enabled, { userAction = false } = {}) {
    this.physicsEnabled = !!enabled;

    const left = byId("togglePhysics");
    const right = byId("p_enabled");
    if (left) left.checked = this.physicsEnabled;
    if (right) right.checked = this.physicsEnabled;

    if (!this.network) return;

    if (this.physicsEnabled) {
      this._unfreezeGraph();
      const p = this._currentPhysFromUI();
      this.network.setOptions({
        physics: {
          enabled: true,
          stabilization: { iterations: 900, updateInterval: 25 },
          barnesHut: {
            gravitationalConstant: p.gravitationalConstant,
            centralGravity: p.centralGravity,
            springConstant: p.springConstant,
            springLength: p.springLength,
            avoidOverlap: p.avoidOverlap
          }
        }
      });
      try { this.network.startSimulation(); } catch (e) { }
    } else {
      this._freezeGraphHard();
    }
  }

  // 从UI获取当前物理参数
  _currentPhysFromUI() {
    const ge = (id) => document.getElementById(id);
    const grav = ge("p_grav") ? Number(ge("p_grav").value) : -3000;
    const len = ge("p_len") ? Number(ge("p_len").value) : 200;
    const k = ge("p_k") ? Number(ge("p_k").value) : 0.02;
    const cg = ge("p_cg") ? Number(ge("p_cg").value) : 0.10;
    const ao = ge("p_ao") ? Number(ge("p_ao").value) : 0.50;

    return {
      gravitationalConstant: Math.max(-10000, Math.min(-200, grav)),
      springLength: Math.max(50, Math.min(600, len)),
      springConstant: Math.max(0.001, Math.min(0.2, k)),
      centralGravity: Math.max(0, Math.min(1, cg)),
      avoidOverlap: Math.max(0, Math.min(1, ao))
    };
  }

  // 冻结所有节点
  _freezeGraphHard() {
    if (!this.network || !this.nodesDS) return;
    const ids = this.nodesDS.getIds();
    const positions = this.network.getPositions(ids);
    const updates = [];
    for (const id of ids) {
      const p = positions[id];
      if (!p) continue;
      updates.push({ id, x: p.x, y: p.y, fixed: { x: true, y: true } });
    }
    if (updates.length) this.nodesDS.update(updates);
    try { this.network.stopSimulation(); } catch (e) { }
    try { this.network.setOptions({ physics: { enabled: false } }); } catch (e) { }
    try { this.network.redraw(); } catch (e) { }
  }

  // 解冻所有节点
  _unfreezeGraph() {
    if (!this.nodesDS) return;
    const ids = this.nodesDS.getIds();
    this.nodesDS.update(ids.map(id => ({ id, fixed: { x: false, y: false } })));
  }

  //初始化网络
  initNetwork(nodesArr, edgesArr, dataManager, styleManager, layoutManager) {
    setLoading(true, "构建图数据…", 20);

    requestAnimationFrame(() => {
      try {
        // 1) 合并同名节点
        const merged = dataManager.mergeNodesEdgesByLabel(nodesArr, edgesArr);
        let nodesMerged = merged.nodes;
        let edgesMerged = merged.edges;

        // 先算全局特征分数
        dataManager.computeFeatureScoresFromMergedEdges(edgesMerged);

        // 2) shadow（可选）
        const SHADOW_SHARED_FEATURES = document.getElementById("toggleShadowShared")?.checked || false;
        const prepared = dataManager.prepareRenderData(nodesMerged, edgesMerged, SHADOW_SHARED_FEATURES);
        const nodesFinal = prepared.nodes;
        const edgesFinal = prepared.edges;

        // 保存当前视图原始数据
        this.currentOriginalNodes = nodesArr;
        this.currentOriginalEdges = edgesArr;

        // 3) 创建 DataSet
        this.nodesDS = new vis.DataSet(nodesFinal.map(n => {
          let title = n.title;
          if (n.group === "feature") {
            title = dataManager.featureTitleWithScore(n.id, n.label, n.title, this.nodesDS, this.edgesDS);
          }

          return {
            id: n.id,
            label: n.label,
            title,
            group: n.group,
            ...styleManager.nodeStyleByGroup(n.group),
          };
        }));

        this.edgesDS = new vis.DataSet(edgesFinal.map((e, idx) => ({
          id: e.id ?? `${e.from}->${e.to}:${e.type ?? ""}:${idx}`,
          from: e.from,
          to: e.to,
          type: e.type,
          value: e.value,
          label: String(e.label ?? ""),
          title: e.title ?? "",
          ...styleManager.edgeStyle(e),
          smooth: (e.type === "model_feature") ? { enabled: true, type: "continuous", roundness: 0.0 } : undefined
        })));

        setLoading(true, "初始化网络…", 30);

        requestAnimationFrame(() => {
          try {
            const data = { nodes: this.nodesDS, edges: this.edgesDS };
            const options = {
              layout: { improvedLayout: false, randomSeed: 42 },
              interaction: { hover: true, multiselect: true, zoomView: true, dragView: true, zoomSpeed: 0.6 },
              physics: { enabled: this.physicsEnabled }
            };

            this.network = new vis.Network(this.container, data, options);

            enableWheelZoom(this.container, this.network);
            enableManualDragForFixedNodes(this.network, this.nodesDS);
            bindClickHighlight(this.network, this.nodesDS, this.edgesDS);

            this.setPhysicsEnabled(this.physicsEnabled, { userAction: false });
            if (!this.physicsEnabled) this._freezeGraphHard();

            setLoading(true, "应用布局模式…", 70);
            requestAnimationFrame(() => {
              layoutManager.applyLayout();

              // 重新刷新hover
              try {
                const current = this.nodesDS.get();
                const ups = [];
                for (const n of current) {
                  if (n.group !== "feature") continue;
                  ups.push({
                    id: n.id,
                    title: dataManager.featureTitleWithScore(n.id, n.label, n.title, this.nodesDS, this.edgesDS)
                  });
                }
                if (ups.length) this.nodesDS.update(ups);
              } catch (_) { }

              reorderNodesByLayer(this.nodesDS, this.network);
              setLoading(false);
            });

          } catch (err) {
            setLoading(false);
            console.error(err);
            alert("初始化网络失败：\n" + (err?.message || String(err)));
          }
        });

      } catch (err) {
        setLoading(false);
        console.error(err);
        alert("构建图数据失败：\n" + (err?.message || String(err)));
      }
    });
  }

  // 重新渲染（用于切换影子共享等）
  reRender(dataManager, styleManager, layoutManager) {
    if (this.currentOriginalNodes.length && this.currentOriginalEdges.length) {
      this.initNetwork(this.currentOriginalNodes, this.currentOriginalEdges, dataManager, styleManager, layoutManager);
    }
  }
}