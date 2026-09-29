import { byId, setLoading, showErr, nextFrame } from './utils/helpers.js';
import { DEFAULT_STYLE } from './utils/constants.js';
import { DataManager } from './core/DataManager.js';
import { GraphManager } from './core/GraphManager.js';
import { LayoutManager } from './core/LayoutManager.js';
import { StyleManager } from './core/StyleManager.js';
import { PanelManager } from './ui/PanelManager.js';
import { ExportManager } from './ui/ExportManager.js';
import { reorderNodesByLayer } from './core/InteractionManager.js';

// 全局错误捕获
window.addEventListener("error", (e) => {
  try { setLoading(false); } catch (_) { }
  try { showErr("JS 运行错误：\n" + (e?.message || e)); } catch (_) { }
});

window.addEventListener("unhandledrejection", (e) => {
  try { setLoading(false); } catch (_) { }
  try { showErr("Promise 未捕获错误：\n" + (e?.reason?.message || e?.reason || e)); } catch (_) { }
});

// 应用主类
class App {
  constructor() {
    this.dataManager = null;
    this.graphManager = new GraphManager("mynetwork");
    this.styleManager = new StyleManager();
    this.panelManager = new PanelManager();
    this.exportManager = null;
    this.layoutManager = null;

    // 聚类列可见性
    this.clusterVisibleCols = new Set([0, 1, 2, 3]);
  }

  async init() {
    try {
      setLoading(true, "加载 graph_data.json…", 10);
      await nextFrame();

      const url = new URL("./graph_data.json", window.location.href).href;
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`graph_data.json 加载失败: HTTP ${res.status}`);

      setLoading(true, "读取 graph_data.json…", 14);
      await nextFrame();
      const txt = await res.text();

      setLoading(true, "解析 JSON…", 18);
      await nextFrame();
      const rawData = JSON.parse(txt);

      // 初始化数据管理器
      this.dataManager = new DataManager(rawData);

      // 初始化导出管理器
      this.exportManager = new ExportManager(this.graphManager, this.dataManager);

      // 初始化布局管理器
      this.layoutManager = new LayoutManager(this.graphManager, this.dataManager, this.styleManager);

      // 构建下拉框
      this.panelManager.buildSelect(byId("dbSelect"), (rawData.meta?.databases || []), "（不选择）");
      this.panelManager.buildSelect(byId("modelSelect"), (rawData.meta?.models || []), "（不选择）");

      // 显示统计信息
      byId("stats").textContent =
        `nodes=${this.dataManager.allNodes.length}, edges=${this.dataManager.allEdges.length}, ` +
        `databases=${rawData.meta?.databases?.length || 0}, models=${rawData.meta?.models?.length || 0}, ` +
        `features=${rawData.meta?.features_count || 0}`;

      // 绑定事件
      this.bindEvents();

      // 设置样式输入
      this.styleManager.setStyleInputsFromCfg();
      this.styleManager.bindStyleInputGuards();

      // 默认设置
      this.graphManager.layoutMode = "hier";
      byId("layoutMode").value = "hier";
      this.graphManager.physicsEnabled = false;
      byId("togglePhysics").checked = false;
      byId("toggleShadowShared").checked = false;

      // 挂载物理面板
      this.panelManager.mountPhysicsPanel(this.graphManager);

      // 渲染图
      this.graphManager.initNetwork(
        this.dataManager.allNodes,
        this.dataManager.allEdges,
        this.dataManager,
        this.styleManager,
        this.layoutManager
      );

      // 面板控制
      byId("hideLeftBtn").onclick = () => this.panelManager.setLeftVisible(false, this.graphManager.network);
      byId("showLeftBtn").onclick = () => this.panelManager.setLeftVisible(true, this.graphManager.network);
      byId("hideRightBtn").onclick = () => this.panelManager.setRightVisible(false, this.graphManager.network);
      byId("showRightBtn").onclick = () => this.panelManager.setRightVisible(true, this.graphManager.network);

    } catch (err) {
      console.error(err);
      setLoading(false);
      showErr("页面脚本报错：\n" + (err?.message || String(err)));
      alert("页面脚本报错：\n" + (err?.message || String(err)));
    }
  }

  bindEvents() {
    // 筛选
    byId("applyFilterBtn").onclick = () => this.applyFilter();
    byId("resetBtn").onclick = () => this.resetAll();

    // 搜索
    byId("searchBtn").onclick = () => this.searchAndFocus();
    byId("searchInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.searchAndFocus();
    });

    // 视图控制
    byId("fitBtn").onclick = () => this.graphManager.network && this.graphManager.network.fit({ animation: { duration: 500 } });
    byId("toggleLabels").onchange = () => this.toggleLabels();

    // 物理引擎
    byId("togglePhysics").onchange = (e) => {
      this.graphManager.setPhysicsEnabled(!!e.target.checked, { userAction: true });
      reorderNodesByLayer(this.graphManager.nodesDS, this.graphManager.network);
    };

    // 影子共享
    byId("toggleShadowShared").onchange = () => {
      this.graphManager.reRender(this.dataManager, this.styleManager, this.layoutManager);
    };

    // 布局模式
    byId("layoutMode").onchange = (e) => {
      this.graphManager.layoutMode = e.target.value || "hier";
      if (this.graphManager.network && this.graphManager.nodesDS && this.graphManager.edgesDS) {
        this.layoutManager.applyLayout();
      }
    };
    byId("applyLayoutBtn").onclick = () => {
      this.graphManager.layoutMode = byId("layoutMode").value || "hier";
      if (this.graphManager.network && this.graphManager.nodesDS && this.graphManager.edgesDS) {
        this.layoutManager.applyLayout();
      }
    };

    // 导出按钮
    byId("exportBtn").onclick = () => this.exportManager.exportFullPNG();

    if (byId("exportSvgBtn")) {
      byId("exportSvgBtn").onclick = () => {
        const mode = this.graphManager.layoutMode || "hier";
        this.exportManager.exportCurrentSVG(`knowledge-graph-${mode}-${Date.now()}.svg`);
      };
    }

    if (byId("exportPaperSvgBtn")) {
      byId("exportPaperSvgBtn").onclick = () => {
        const mode = this.graphManager.layoutMode || "hier";
        this.exportManager.exportPaperSVG({
          filename: `knowledge-graph-paper-${mode}-${Date.now()}.svg`,
          mode: "structure"
        });
      };
    }

    if (byId("exportPaperTopSvgBtn")) {
      byId("exportPaperTopSvgBtn").onclick = () => {
        const mode = this.graphManager.layoutMode || "hier";
        this.exportManager.exportPaperSVG({
          filename: `knowledge-graph-paper-top-${mode}-${Date.now()}.svg`,
          mode: "feature-top",
          topN: 30
        });
      };
    }

    byId("exportPaperDBBtn").onclick = () => this.exportManager.exportPaperDBFeatureCSV();
    byId("exportSuppModelBtn").onclick = () => this.exportManager.exportSupplementaryModelFeatureCSV();
    byId("exportGlobalFeatureBtn").onclick = () => this.exportManager.exportGlobalFeatureHierarchyCSV();
    byId("exportDBFeatureBtn").onclick = () => this.exportManager.exportDBFeatureHierarchyCSV();
    byId("exportRankStabilityBtn").onclick = () => this.exportManager.exportFeatureRankStabilityCSV();

    // 列导出
    byId("exportCol1Btn").onclick = () => this.exportManager.exportFeatureColumnCSV(0);
    byId("exportCol2Btn").onclick = () => this.exportManager.exportFeatureColumnCSV(1);
    byId("exportCol3Btn").onclick = () => this.exportManager.exportFeatureColumnCSV(2);
    byId("exportCol4Btn").onclick = () => this.exportManager.exportFeatureColumnCSV(3);

    // 样式按钮
    byId("applyStyleBtn").onclick = () => {
      this.styleManager.readStyleCfgFromInputs();
      this.styleManager.applyNodeStylesToAll(this.graphManager.nodesDS, this.graphManager.network);
    };
    byId("resetStyleBtn").onclick = () => {
      this.styleManager.styleCfg = JSON.parse(JSON.stringify(DEFAULT_STYLE));
      this.styleManager.setStyleInputsFromCfg();
      this.styleManager.applyNodeStylesToAll(this.graphManager.nodesDS, this.graphManager.network);
    };

    // 聚类列可见性
    document.querySelectorAll(".clusterCol").forEach(cb => {
      cb.onchange = () => {
        this.clusterVisibleCols = new Set(
          Array.from(document.querySelectorAll(".clusterCol"))
            .filter(x => x.checked)
            .map(x => Number(x.value))
        );
        if (this.graphManager.layoutMode === "cluster") this.applyClusterFeatureLabelVisibility();
      };
    });
  }

  applyFilter() {
    const db = byId("dbSelect").value;
    const model = byId("modelSelect").value;

    if (!db && !model) {
      this.graphManager.initNetwork(
        this.dataManager.allNodes,
        this.dataManager.allEdges,
        this.dataManager,
        this.styleManager,
        this.layoutManager
      );
      return;
    }

    const keep = new Set();

    if (db) {
      const dbNode = this.dataManager.allNodes.find(n => n.group === "database" && n.label === db);
      const dbId = dbNode?.id;
      if (dbId) {
        keep.add(dbId);
        this.dataManager.allEdges.forEach(e => { if (e.from === dbId) keep.add(e.to); });
        const keepModels = new Set([...keep]);
        this.dataManager.allEdges.forEach(e => {
          if (keepModels.has(e.from) && e.type === "model_feature") keep.add(e.to);
        });
      }
    }

    if (model) {
      const mNode = this.dataManager.allNodes.find(n => n.group === "model" && n.label === model);
      const mId = mNode?.id;
      if (mId) {
        keep.add(mId);
        this.dataManager.allEdges.forEach(e => {
          if (e.to === mId && e.type === "db_model") keep.add(e.from);
          if (e.from === mId && e.type === "model_feature") keep.add(e.to);
        });
      }
    }

    const nodesFiltered = this.dataManager.allNodes.filter(n => keep.has(n.id));
    const edgesFiltered = this.dataManager.allEdges.filter(e => keep.has(e.from) && keep.has(e.to));

    this.graphManager.initNetwork(nodesFiltered, edgesFiltered, this.dataManager, this.styleManager, this.layoutManager);
  }

  resetAll() {
    byId("dbSelect").value = "";
    byId("modelSelect").value = "";
    byId("searchInput").value = "";
    this.graphManager.initNetwork(
      this.dataManager.allNodes,
      this.dataManager.allEdges,
      this.dataManager,
      this.styleManager,
      this.layoutManager
    );
  }

  searchAndFocus() {
    const q = byId("searchInput").value.trim().toLowerCase();
    if (!q) return;

    const all = this.graphManager.nodesDS.get();
    const hit = all.find(n => (n.label || "").toLowerCase().includes(q));
    if (!hit) { alert("未找到匹配节点（当前筛选范围内）"); return; }

    this.graphManager.network.selectNodes([hit.id]);
    this.graphManager.network.focus(hit.id, {
      scale: Math.max(this.graphManager.network.getScale(), 1.8),
      animation: { duration: 500 }
    });
  }

  toggleLabels() {
    const show = byId("toggleLabels").checked;
    const ups = [];
    for (const n of this.graphManager.nodesDS.get()) {
      ups.push({ id: n.id, label: show ? (n.label ?? "") : "" });
    }
    this.graphManager.nodesDS.update(ups);
    reorderNodesByLayer(this.graphManager.nodesDS, this.graphManager.network);
  }

  applyClusterFeatureLabelVisibility() {
    if (!this.graphManager.nodesDS) return;
    const ups = [];
    for (const n of this.graphManager.nodesDS.get()) {
      if (n.group !== "feature") continue;
      const col = this.dataManager.featureColMap.get(n.id);
      if (col === undefined) continue;
      ups.push({ id: n.id, label: this.clusterVisibleCols.has(col) ? n.label : "" });
    }
    if (ups.length) {
      this.graphManager.nodesDS.update(ups);
      reorderNodesByLayer(this.graphManager.nodesDS, this.graphManager.network);
    }
  }
}

// 启动应用
const app = new App();
app.init();
