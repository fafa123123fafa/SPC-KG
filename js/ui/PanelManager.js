import { byId } from '../utils/helpers.js';

export class PanelManager {
  constructor() {
    this.leftVisible = true;
    this.rightVisible = true;
  }

  // 刷新布局（窗口调整后）
  refreshLayoutAfterResize(network) {
    if (!network) return;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        try { network.setSize('100%', '100%'); } catch (e) { }
        try { network.redraw(); } catch (e) { }
      });
    });
  }

  // 设置左侧面板可见性
  setLeftVisible(visible, network) {
    this.leftVisible = !!visible;
    const wrap = byId("wrapRoot");
    const showBtn = byId("showLeftBtn");
    wrap.classList.toggle("left-hidden", !this.leftVisible);
    showBtn.style.display = this.leftVisible ? "none" : "block";
    this.refreshLayoutAfterResize(network);
  }

  // 设置右侧面板可见性
  setRightVisible(visible, network) {
    this.rightVisible = !!visible;
    const box = byId("configBox");
    const showBtn = byId("showRightBtn");
    box.style.display = this.rightVisible ? "block" : "none";
    showBtn.style.display = this.rightVisible ? "none" : "block";
    this.refreshLayoutAfterResize(network);
  }

  // 构建下拉框
  buildSelect(selectEl, items, placeholder) {
    selectEl.innerHTML = "";
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = placeholder;
    selectEl.appendChild(opt0);
    items.forEach(x => {
      const opt = document.createElement("option");
      opt.value = x;
      opt.textContent = x;
      selectEl.appendChild(opt);
    });
  }

  // 挂载物理引擎面板
  mountPhysicsPanel(graphManager) {
    const box = byId("configInner");
    if (!box) return;

    box.innerHTML = `
      <div class="cfgRow">
        <div class="topline">
          <span>Physics Enabled</span>
          <input id="p_enabled" type="checkbox" />
        </div>
        <div class="mini">开启仅用于拖动排版；布局模式切换是确定性算法。</div>
      </div>

      <div class="sep"></div>

      <div class="cfgRow">
        <div class="topline"><span>吸引/排斥（grav）</span><span id="v_grav"></span></div>
        <input id="p_grav" type="range" min="-10000" max="-200" step="50" />
        <div class="mini">越负越“聚拢”（太聚会挤）。</div>
      </div>

      <div class="cfgRow">
        <div class="topline"><span>节点间距（springLength）</span><span id="v_len"></span></div>
        <input id="p_len" type="range" min="50" max="600" step="10" />
        <div class="mini">越大越疏，越小越紧。</div>
      </div>

      <div class="cfgRow">
        <div class="topline"><span>弹性强度（springConst）</span><span id="v_k"></span></div>
        <input id="p_k" type="range" min="0.001" max="0.2" step="0.001" />
        <div class="mini">越大越“硬”，收敛更快但更抖。</div>
      </div>

      <div class="cfgRow">
        <div class="topline"><span>中心引力（centralGravity）</span><span id="v_cg"></span></div>
        <input id="p_cg" type="range" min="0" max="1" step="0.01" />
        <div class="mini">越大越往中心收拢（太大容易堆）。</div>
      </div>

      <div class="cfgRow">
        <div class="topline"><span>避免重叠（avoidOverlap）</span><span id="v_ao"></span></div>
        <input id="p_ao" type="range" min="0" max="1" step="0.05" />
        <div class="mini">越大越不压在一起（可能更散）。</div>
      </div>

      <div class="cfgBtns">
        <button id="p_apply">应用</button>
        <button id="p_stabilize">重新稳定化</button>
      </div>
    `;

    const $ = (id) => document.getElementById(id);

    // 设置默认值
    $("p_grav").value = -3000;
    $("p_len").value = 200;
    $("p_k").value = 0.02;
    $("p_cg").value = 0.10;
    $("p_ao").value = 0.50;

    function refreshValueLabels() {
      $("v_grav").textContent = String($("p_grav").value);
      $("v_len").textContent = String($("p_len").value);
      $("v_k").textContent = Number($("p_k").value).toFixed(3);
      $("v_cg").textContent = Number($("p_cg").value).toFixed(2);
      $("v_ao").textContent = Number($("p_ao").value).toFixed(2);
    }

    ["p_grav", "p_len", "p_k", "p_cg", "p_ao"].forEach(id => $(id).addEventListener("input", refreshValueLabels));
    refreshValueLabels();

    $("p_enabled").checked = graphManager.physicsEnabled;
    $("p_enabled").onchange = () => graphManager.setPhysicsEnabled($("p_enabled").checked, { userAction: true });

    $("p_apply").onclick = () => {
      if (graphManager.physicsEnabled) graphManager.setPhysicsEnabled(true, { userAction: false });
      if (graphManager.network) graphManager.network.redraw();
    };

    $("p_stabilize").onclick = () => {
      graphManager.setPhysicsEnabled(true, { userAction: false });
      try { graphManager.network.stabilize(); } catch (e) { }
    };
  }
}