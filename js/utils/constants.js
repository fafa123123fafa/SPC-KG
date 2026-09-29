// ========== 常量配置 ==========
export const DEFAULT_STYLE = {
  database: {
    nodeColor: "#0099FF",
    nodeSize: 58,
    fontColor: "#FFFFFF",
    fontSize: 42,
    fontWeight: 'bold'
  },
  model: {
    nodeColor: "#E74C3C",
    nodeSize: 38,
    fontColor: "#000000",
    fontSize: 34,
    fontWeight: 'bold'
  },
  feature: {
  nodeColor: "#2ECC71",
  nodeSize: 18,
  fontColor: "#000000",
  fontSize: 16
  }
};

export const DEFAULT_PHYS = {
  enabled: false,
  gravitationalConstant: -3000,
  springLength: 200,
  springConstant: 0.02,
  centralGravity: 0.10,
  avoidOverlap: 0.50
};

export const MIN_SCALE = 0.35;
export const MAX_SCALE = 3.50;
export const WHEEL_ZOOM_K = 0.0012;
export const TARGET_ASPECT = 1.0;