import { DEFAULT_STYLE } from '../utils/constants.js';
import { safeColor, safeNum, clamp } from '../utils/helpers.js';
import { byId } from '../utils/helpers.js';

export class StyleManager {
  constructor() {
    this.styleCfg = JSON.parse(JSON.stringify(DEFAULT_STYLE));
  }

  // 节点颜色辅助
  fullColor(bg) {
    return {
      background: bg,
      border: bg,
      highlight: { background: bg, border: bg },
      hover: { background: bg, border: bg }
    };
  }

  // 根据分组获取节点样式
  nodeStyleByGroup(group) {
    const g = (group === "database" || group === "model") ? group : "feature";
    const st0 = this.styleCfg[g] || DEFAULT_STYLE[g];

    const nodeSize = (Number.isFinite(st0.nodeSize) && st0.nodeSize > 0) ? st0.nodeSize : DEFAULT_STYLE[g].nodeSize;
    const fontSize = (Number.isFinite(st0.fontSize) && st0.fontSize > 0) ? st0.fontSize : DEFAULT_STYLE[g].fontSize;

    const nodeColor = (typeof st0.nodeColor === "string" && st0.nodeColor.startsWith("#")) ? st0.nodeColor : DEFAULT_STYLE[g].nodeColor;
    const fontColor = (typeof st0.fontColor === "string" && st0.fontColor.startsWith("#")) ? st0.fontColor : DEFAULT_STYLE[g].fontColor;

    const font = { color: fontColor, size: fontSize, strokeWidth: 2, strokeColor: "#FFFFFF" };

    if (g === "database") {
      const pad = Math.max(4, Math.min(40, Math.round(nodeSize / 2)));
      return {
        shape: "box",
        color: this.fullColor(nodeColor),
        font,
        margin: pad,
        shapeProperties: { borderRadius: 6 }
      };
    }
    if (g === "model") {
      return {
        shape: "dot",
        color: this.fullColor(nodeColor),
        size: nodeSize,
        font: { ...font, vadjust: -40 }
      };
    }
    return {
      shape: "dot",
      color: this.fullColor(nodeColor),
      size: nodeSize,
      font
    };
  }

  // 边样式
  edgeStyle(e) {
    if (e.type === "db_model") return { color: "#AAB7B8", arrows: "to", width: 2 };
    const w = Math.max(1, Math.min(8, (e.value || 0) / 15));
    return { color: "#F39C12", arrows: "to", width: w };
  }

  // 应用样式到所有节点
  applyNodeStylesToAll(nodesDS, network) {
    if (!nodesDS) return;
    try {
      const ids = nodesDS.getIds();
      const updates = [];
      for (const id of ids) {
        const n = nodesDS.get(id);
        if (!n) continue;
        const st = this.nodeStyleByGroup(n.group);

        const upd = { id };
        if (st.shape !== undefined) upd.shape = st.shape;
        if (st.color !== undefined) upd.color = st.color;
        if (st.font !== undefined) upd.font = st.font;
        if (st.size !== undefined) upd.size = st.size;
        if (st.margin !== undefined) upd.margin = st.margin;
        if (st.shapeProperties !== undefined) upd.shapeProperties = st.shapeProperties;

        updates.push(upd);
      }
      nodesDS.update(updates);
      if (network) network.redraw();
    } catch (err) {
      console.error(err);
    }
  }

  // 从UI读取样式配置
  readStyleCfgFromInputs() {
    this.styleCfg.database.nodeColor = this.normalizeColorInput("c_db_node", this.styleCfg.database.nodeColor);
    this.styleCfg.database.nodeSize = this.normalizeNumberInput("s_db_node", this.styleCfg.database.nodeSize, 8, 80);
    this.styleCfg.database.fontColor = this.normalizeColorInput("c_db_font", this.styleCfg.database.fontColor);
    this.styleCfg.database.fontSize = this.normalizeNumberInput("s_db_font", this.styleCfg.database.fontSize, 8, 40);

    this.styleCfg.model.nodeColor = this.normalizeColorInput("c_m_node", this.styleCfg.model.nodeColor);
    this.styleCfg.model.nodeSize = this.normalizeNumberInput("s_m_node", this.styleCfg.model.nodeSize, 8, 80);
    this.styleCfg.model.fontColor = this.normalizeColorInput("c_m_font", this.styleCfg.model.fontColor);
    this.styleCfg.model.fontSize = this.normalizeNumberInput("s_m_font", this.styleCfg.model.fontSize, 8, 40);

    this.styleCfg.feature.nodeColor = this.normalizeColorInput("c_f_node", this.styleCfg.feature.nodeColor);
    this.styleCfg.feature.nodeSize = this.normalizeNumberInput("s_f_node", this.styleCfg.feature.nodeSize, 8, 80);
    this.styleCfg.feature.fontColor = this.normalizeColorInput("c_f_font", this.styleCfg.feature.fontColor);
    this.styleCfg.feature.fontSize = this.normalizeNumberInput("s_f_font", this.styleCfg.feature.fontSize, 8, 40);
  }

  // 设置UI输入框的值
  setStyleInputsFromCfg() {
    byId("c_db_node").value = this.styleCfg.database.nodeColor;
    byId("s_db_node").value = this.styleCfg.database.nodeSize;
    byId("c_db_font").value = this.styleCfg.database.fontColor;
    byId("s_db_font").value = this.styleCfg.database.fontSize;

    byId("c_m_node").value = this.styleCfg.model.nodeColor;
    byId("s_m_node").value = this.styleCfg.model.nodeSize;
    byId("c_m_font").value = this.styleCfg.model.fontColor;
    byId("s_m_font").value = this.styleCfg.model.fontSize;

    byId("c_f_node").value = this.styleCfg.feature.nodeColor;
    byId("s_f_node").value = this.styleCfg.feature.nodeSize;
    byId("c_f_font").value = this.styleCfg.feature.fontColor;
    byId("s_f_font").value = this.styleCfg.feature.fontSize;
  }

  // 辅助：规范化数字输入
  normalizeNumberInput(id, fallback, min, max) {
    const el = byId(id);
    if (!el) return fallback;
    const val = safeNum(el.value, fallback, min, max);
    el.value = String(Math.round(val * 1000) / 1000);
    return val;
  }

  // 辅助：规范化颜色输入
  normalizeColorInput(id, fallback) {
    const el = byId(id);
    if (!el) return fallback;
    const val = safeColor(el.value, fallback);
    el.value = val;
    return val;
  }

  // 绑定样式输入防护
  bindStyleInputGuards() {
    const numCfg = [
      ["s_db_node", () => this.styleCfg.database.nodeSize, 8, 80],
      ["s_db_font", () => this.styleCfg.database.fontSize, 8, 40],
      ["s_m_node", () => this.styleCfg.model.nodeSize, 8, 80],
      ["s_m_font", () => this.styleCfg.model.fontSize, 8, 40],
      ["s_f_node", () => this.styleCfg.feature.nodeSize, 8, 80],
      ["s_f_font", () => this.styleCfg.feature.fontSize, 8, 40],
    ];

    const colorCfg = [
      ["c_db_node", () => this.styleCfg.database.nodeColor],
      ["c_db_font", () => this.styleCfg.database.fontColor],
      ["c_m_node", () => this.styleCfg.model.nodeColor],
      ["c_m_font", () => this.styleCfg.model.fontColor],
      ["c_f_node", () => this.styleCfg.feature.nodeColor],
      ["c_f_font", () => this.styleCfg.feature.fontColor],
    ];

    numCfg.forEach(([id, fbFn, min, max]) => {
      const el = byId(id);
      if (!el) return;
      const normalize = () => this.normalizeNumberInput(id, fbFn(), min, max);
      el.addEventListener("blur", normalize);
      el.addEventListener("change", normalize);
      el.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); el.blur(); } });
    });

    colorCfg.forEach(([id, fbFn]) => {
      const el = byId(id);
      if (!el) return;
      const normalize = () => this.normalizeColorInput(id, fbFn());
      el.addEventListener("change", normalize);
      el.addEventListener("blur", normalize);
    });
  }
}