import { MIN_SCALE, MAX_SCALE, WHEEL_ZOOM_K } from '../utils/constants.js';

// 滚轮缩放
export function enableWheelZoom(container, network) {
  container.addEventListener("wheel", (e) => {
    if (!network) return;
    e.preventDefault();

    const oldScale = network.getScale();
    const factor = Math.exp(-e.deltaY * WHEEL_ZOOM_K);
    let newScale = oldScale * factor;
    newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));

    const rect = container.getBoundingClientRect();
    const pointer = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    const canvasPos = network.DOMtoCanvas(pointer);

    const viewPos = network.getViewPosition();
    const ratio = oldScale / newScale;
    const newViewPos = {
      x: canvasPos.x - (canvasPos.x - viewPos.x) * ratio,
      y: canvasPos.y - (canvasPos.y - viewPos.y) * ratio
    };

    network.moveTo({ position: newViewPos, scale: newScale, animation: false });
  }, { passive: false });
}

// 允许拖动固定节点
export function enableManualDragForFixedNodes(network, nodesDS) {
  if (!network || !nodesDS) return;

  try { network.setOptions({ interaction: { dragNodes: true } }); } catch (e) { }

  network.on("dragStart", (params) => {
    if (!params || !params.nodes || params.nodes.length === 0) return;
    nodesDS.update(params.nodes.map(id => ({ id, fixed: { x: false, y: false } })));
  });

  network.on("dragEnd", (params) => {
    if (!params || !params.nodes || params.nodes.length === 0) return;
    const pos = network.getPositions(params.nodes);
    const updates = params.nodes.map(id => ({
      id,
      x: pos[id]?.x,
      y: pos[id]?.y,
      fixed: { x: true, y: true }
    }));
    nodesDS.update(updates);
  });
}

// 点击高亮邻居
export function bindClickHighlight(network, nodesDS, edgesDS) {
  if (!network || !nodesDS || !edgesDS) return;

  network.on("click", (params) => {
    if (!params.nodes || params.nodes.length === 0) {
      nodesDS.forEach(node => nodesDS.update({ id: node.id, opacity: 1 }));
      edgesDS.forEach(edge => edgesDS.update({ id: edge.id, opacity: 1 }));
      return;
    }
    const selected = params.nodes[0];
    const connected = network.getConnectedNodes(selected);
    const keep = new Set([selected, ...connected]);

    nodesDS.forEach(node => {
      if (node.id) nodesDS.update({ id: node.id, opacity: keep.has(node.id) ? 1 : 0.15 });
    });

    edgesDS.forEach(edge => {
      const on = keep.has(edge.from) && keep.has(edge.to);
      edgesDS.update({ id: edge.id, opacity: on ? 1 : 0.10 });
    });
  });
}

// 节点重新排序（确保渲染层级：database > model > feature）
export function reorderNodesByLayer(nodesDS, network) {
  if (!nodesDS || !network) return;

  const rank = (g) => (g === "database" ? 3 : (g === "model" ? 2 : 1));

  const arr = nodesDS.get();
  arr.sort((a, b) => {
    const ra = rank(a.group), rb = rank(b.group);
    if (ra !== rb) return ra - rb;
    const la = String(a.label ?? a.id), lb = String(b.label ?? b.id);
    return la.localeCompare(lb);
  });

  nodesDS.clear();
  nodesDS.add(arr);

  try { network.redraw(); } catch (e) { }
}