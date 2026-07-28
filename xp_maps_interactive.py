"""Interactive 12x8 maps: aggregate pool view + per-player O→D routes."""

from __future__ import annotations

import json

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

XP_COLORSCALE: tuple[tuple[float, str], ...] = (
    (0.00, "#4b5563"),
    (0.25, "#9ca3af"),
    (0.55, "#f87171"),
    (0.80, "#ef4444"),
    (1.00, "#b91c1c"),
)
VOLUME_COLORSCALE: tuple[tuple[float, str], ...] = (
    (0.00, "#132033"),
    (0.35, "#166534"),
    (0.70, "#22c55e"),
    (1.00, "#bbf7d0"),
)

_SHARED_CSS = """
  * { box-sizing: border-box; }
  body { margin: 0; background: #0f172a; color: #e2e8f0;
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .qmap-wrap { border: 1px solid rgba(148,163,184,0.18); border-radius: 16px;
    background: linear-gradient(180deg, rgba(30,41,59,0.55) 0%, rgba(15,23,42,0.35) 100%);
    padding: 0.75rem 0.9rem 0.85rem 0.9rem; margin-bottom: 0.85rem; }
  .qmap-section-title { font-size: 0.82rem; font-weight: 800; color: #bae6fd; margin: 0 0 0.55rem 0;
    letter-spacing: 0.03em; text-transform: uppercase; }
  .qmap-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.6rem; }
  .qmap-toolbar-label { color: #93a4bc; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-right: 0.25rem; }
  .qmap-toolbar-sep { flex: 0 0 auto; width: 1px; height: 18px; background: rgba(148,163,184,0.22); margin: 0 0.3rem; }
  .qmap-btn { background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.28); color: #cbd5e1;
    border-radius: 999px; padding: 0.24rem 0.85rem; font-size: 0.79rem; font-weight: 700; cursor: pointer; }
  .qmap-btn:hover { border-color: rgba(56,189,248,0.5); color: #e2e8f0; }
  .qmap-btn.is-active { background: rgba(56,189,248,0.16); border-color: rgba(56,189,248,0.6); color: #bae6fd; }
  .qmap-body { display: grid; grid-template-columns: minmax(300px, 1.55fr) minmax(250px, 1fr); gap: 0.75rem; align-items: start; }
  @media (max-width: 640px) { .qmap-body { grid-template-columns: 1fr; } }
  #qmap-pl-wrap .qmap-body { grid-template-columns: 1fr; }
  .qmap-plot { width: 100%; border-radius: 12px; overflow: visible; min-height: 0; }
  .qmap-panel { background: rgba(15,23,42,0.55); border: 1px solid rgba(148,163,184,0.16); border-radius: 12px;
    padding: 0.7rem 0.8rem; overflow-y: auto; scrollbar-width: thin; }
  .qp-title { display: block; font-size: 0.95rem; font-weight: 800; color: #e2e8f0; margin: 0 0 0.15rem 0; }
  .qp-sub { display: block; color: #94a3b8; font-size: 0.75rem; line-height: 1.4; margin-bottom: 0.55rem; }
  .qp-pin { display: inline-block; margin-left: 0.35rem; padding: 0.05rem 0.4rem; border-radius: 999px;
    background: rgba(56,189,248,0.18); border: 1px solid rgba(56,189,248,0.45); color: #bae6fd;
    font-size: 0.62rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; vertical-align: middle; }
  .qp-warn { display: block; margin-bottom: 0.5rem; padding: 0.3rem 0.45rem; border-radius: 7px;
    background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); color: #fcd34d; font-size: 0.7rem; }
  .qp-section-label { display: block; font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.06em; color: #93a4bc; margin: 0.55rem 0 0.35rem 0; }
  .qp-section-label:first-of-type { margin-top: 0; }
  .qp-row { border: 1px solid rgba(148,163,184,0.14); border-radius: 8px; padding: 0.3rem 0.45rem;
    margin-bottom: 0.26rem; background: rgba(2,6,23,0.35); font-size: 0.73rem; line-height: 1.35; }
  .qp-row b { color: #e2e8f0; }
  .qp-row-route { color: #cbd5e1; font-weight: 700; display: block; margin-bottom: 0.12rem; }
  .qp-row-meta { color: #94a3b8; font-size: 0.7rem; }
  .qp-quads { display: flex; gap: 0.25rem; margin-top: 0.1rem; }
  .qp-quad { flex: 1 1 0; min-width: 0; border: 1px solid rgba(148,163,184,0.14); border-radius: 7px;
    padding: 0.24rem 0.3rem; background: rgba(2,6,23,0.35); text-align: center; }
  .qp-quad-name { display: block; font-size: 0.58rem; font-weight: 800; text-transform: uppercase;
    color: #93a4bc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .qp-quad-val { display: block; font-size: 0.82rem; font-weight: 800; color: #38bdf8; }
  .qp-hint { color: #94a3b8; font-size: 0.76rem; line-height: 1.5; margin: 0.3rem 0 0 0; }
  .qp-player-select { width: 100%; margin-bottom: 0.55rem; padding: 0.42rem 0.55rem; border-radius: 8px;
    border: 1px solid rgba(148,163,184,0.28); background: rgba(2,6,23,0.55); color: #e2e8f0;
    font-size: 0.8rem; font-weight: 600; outline: none; }
"""

_MAP_ENGINE_JS = """
function qmapEngine(prefix, DATA, XP_SCALE, VOL_SCALE, PLOT_HEIGHT, opts) {
  opts = opts || {};
  var originsMode = !!opts.originsMode;
  var hidePanel = !!opts.hidePanel;
  var FIELD_X = DATA.field_x, FIELD_Y = DATA.field_y, COLS = DATA.cols, ROWS = DATA.rows;
  var NCELLS = COLS * ROWS, CELL_W = FIELD_X / COLS, CELL_H = FIELD_Y / ROWS;
  var SPLIT_X = FIELD_X / 2, SPLIT_Y = FIELD_Y / 2;
  var plotEl = document.getElementById(prefix + '-plot');
  var panelEl = document.getElementById(prefix + '-panel');
  var metric = 'xp', scaleMode = 'fixed', activeCell = null, pinnedCell = null, pendingFrame = null;
  var origins = DATA.origins || {}, overall = DATA.overall;

  function fmtInt(v) { return Number(v).toLocaleString('pt-BR'); }
  function fmtPct(v) { return Number(v).toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%'; }
  function fmtXp(v) { return Number(v).toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }

  function currentState() {
    if (originsMode && activeCell !== null && origins[String(activeCell)]) return origins[String(activeCell)];
    return overall;
  }
  function volumeValues(state) {
    var total = Math.max(state.passes, 1);
    return state.counts.map(function (v) { return v > 0 ? (v / total) * 100 : null; });
  }
  function stateValues(state) { return metric === 'xp' ? state.xp : volumeValues(state); }
  function toGrid(flat) {
    var out = []; for (var r = 0; r < ROWS; r++) out.push(flat.slice(r * COLS, (r + 1) * COLS)); return out;
  }
  function scaleMax(state) {
    if (scaleMode === 'fixed') return metric === 'xp' ? DATA.xp_scale_max : DATA.volume_scale_max;
    var best = 0; stateValues(state).forEach(function (v) { if (v !== null && v > best) best = v; }); return best || 1;
  }
  function stopsToScale(stops) { return stops.map(function (s) { return [s[0], s[1]]; }); }
  function colorAt(stops, t) {
    t = Math.max(0, Math.min(1, t)); var lo = stops[0], hi = stops[stops.length - 1];
    for (var i = 0; i < stops.length - 1; i++) { if (t >= stops[i][0] && t <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; } }
    var span = (hi[0] - lo[0]) || 1, k = (t - lo[0]) / span;
    function rgb(hex) { var h = hex.replace('#', ''); return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; }
    var a = rgb(lo[1]), b = rgb(hi[1]);
    return 'rgb(' + Math.round(a[0]+(b[0]-a[0])*k) + ',' + Math.round(a[1]+(b[1]-a[1])*k) + ',' + Math.round(a[2]+(b[2]-a[2])*k) + ')';
  }
  function xpDot(value) { return colorAt(XP_SCALE, value / DATA.xp_scale_max); }
  function cellCenters(n, size) { var out = []; for (var i = 0; i < n; i++) out.push((i + 0.5) * size); return out; }
  function cellIndexAt(x, y) {
    var c = Math.max(0, Math.min(COLS - 1, Math.floor(x / CELL_W)));
    var r = Math.max(0, Math.min(ROWS - 1, Math.floor(y / CELL_H)));
    return r * COLS + c;
  }
  function cellBounds(idx) {
    var c = idx % COLS, r = Math.floor(idx / COLS);
    return [c * CELL_W, r * CELL_H, (c + 1) * CELL_W, (r + 1) * CELL_H];
  }
  function quadrantKeyAt(idx) {
    var c = idx % COLS, r = Math.floor(idx / COLS);
    var x = (c + 0.5) * CELL_W, y = (r + 0.5) * CELL_H;
    if (x < SPLIT_X) return y < SPLIT_Y ? 'def_left' : 'def_right';
    return y < SPLIT_Y ? 'att_left' : 'att_right';
  }
  function quadrantSplit(state) {
    var total = Math.max(state.passes, 1), acc = { def_left: 0, def_right: 0, att_left: 0, att_right: 0 };
    for (var i = 0; i < NCELLS; i++) { if (!state.counts[i]) continue; acc[quadrantKeyAt(i)] += state.counts[i]; }
    return ['def_left','def_right','att_left','att_right'].map(function (k) {
      return { key: k, label: DATA.quadrant_labels[k], share: (acc[k] / total) * 100 };
    });
  }
  function quadBlocks(state) {
    return '<div class="qp-quads">' + quadrantSplit(state).map(function (q) {
      return '<div class="qp-quad"><span class="qp-quad-name">' + q.label.replace(' · ', '<br>') + '</span>'
        + '<span class="qp-quad-val">' + fmtPct(q.share) + '</span></div>';
    }).join('') + '</div>';
  }
  function pitchShapes() {
    var line = 'rgba(248,250,252,0.72)', mk = function (x0,y0,x1,y1) {
      return { type: 'rect', x0: x0, y0: y0, x1: x1, y1: y1, line: { color: line, width: 1.1 }, layer: 'above' };
    };
    var shapes = [mk(0,0,FIELD_X,FIELD_Y), mk(0,18,18,62), mk(FIELD_X-18,18,FIELD_X,62), mk(0,30,6,50), mk(FIELD_X-6,30,FIELD_X,50),
      { type: 'line', x0: SPLIT_X, y0: 0, x1: SPLIT_X, y1: FIELD_Y, line: { color: line, width: 1.1 }, layer: 'above' },
      { type: 'circle', x0: SPLIT_X-10, y0: SPLIT_Y-10, x1: SPLIT_X+10, y1: SPLIT_Y+10, line: { color: line, width: 1.1 }, layer: 'above' }];
    if (originsMode && activeCell !== null) {
      var b = cellBounds(activeCell);
      shapes.push({ type: 'rect', x0: b[0], y0: b[1], x1: b[2], y1: b[3],
        line: { color: 'rgba(56,189,248,0.95)', width: 2.6 }, fillcolor: 'rgba(0,0,0,0)', layer: 'above' });
    }
    return shapes;
  }
  function hoverText(state) {
    var total = Math.max(state.passes, 1), values = stateValues(state), out = [];
    for (var r = 0; r < ROWS; r++) {
      var row = [];
      for (var c = 0; c < COLS; c++) {
        var i = r * COLS + c, origin = origins[String(i)], originLine;
        if (originsMode && i === activeCell) originLine = '<i>origem atual · ' + fmtInt(state.passes) + ' passes</i>';
        else if (originsMode && origin) originLine = '<i>' + fmtInt(origin.passes) + ' passes saem daqui</i>';
        else originLine = '<i>destino dos passes</i>';
        if (values[i] === null) row.push('<b>' + DATA.cell_labels[i] + '</b><br>Sem passes<br>' + originLine);
        else row.push('<b>' + DATA.cell_labels[i] + '</b><br>' + fmtInt(state.counts[i]) + ' passes · ' + fmtPct((state.counts[i]/total)*100)
          + '<br>xP ' + fmtXp(state.xp[i]||0) + '<br>' + originLine);
      }
      out.push(row);
    }
    return out;
  }
  function heatmapTrace(state) {
    var zmax = scaleMax(state);
    return { type: 'heatmap', x: cellCenters(COLS, CELL_W), y: cellCenters(ROWS, CELL_H),
      z: toGrid(stateValues(state)), text: hoverText(state), hoverinfo: 'text',
      colorscale: stopsToScale(metric === 'xp' ? XP_SCALE : VOL_SCALE), zmin: 0, zmax: zmax, xgap: 1, ygap: 1, showscale: true,
      colorbar: { title: { text: metric === 'xp' ? 'xP médio' : '% dos passes', font: { size: 10, color: '#94a3b8' } },
        thickness: 10, len: 0.82, outlinewidth: 0, tickfont: { size: 9, color: '#94a3b8' } }, hoverongaps: false };
  }
  function layout() {
    return { height: PLOT_HEIGHT, margin: { l: 6, r: 6, t: 6, b: 6 }, paper_bgcolor: '#0f172a', plot_bgcolor: '#0d1526',
      shapes: pitchShapes(), hovermode: 'closest',
      xaxis: { range: [-2, FIELD_X + 2], visible: false, fixedrange: true }, yaxis: { range: [FIELD_Y + 2, -2], visible: false, fixedrange: true, scaleanchor: 'x', scaleratio: 1 } };
  }
  function renderPanel(state) {
    if (hidePanel || !panelEl) return;
    var head = DATA.title || 'Mapa', sub = DATA.subtitle || '';
    var warn = '';
    if (originsMode && activeCell !== null) {
      head = state.label; sub = fmtInt(state.passes) + ' passes saindo daqui · xP médio ' + fmtXp(state.mean_xp);
      if (pinnedCell !== null) head += '<span class="qp-pin">fixado</span>';
      if (state.passes < 150) warn = '<span class="qp-warn">Amostra pequena para esta célula.</span>';
    }
    panelEl.innerHTML = '<span class="qp-title">' + head + '</span><span class="qp-sub">' + sub + '</span>' + warn
      + '<span class="qp-section-label">Distribuição por quadrante</span>' + quadBlocks(state);
  }
  function draw() {
    var state = currentState();
    Plotly.react(plotEl, [heatmapTrace(state)], layout(), { displayModeBar: false, responsive: true });
    renderPanel(state);
  }
  function scheduleDraw() {
    if (pendingFrame) return;
    pendingFrame = requestAnimationFrame(function () { pendingFrame = null; draw(); });
  }
  function setActive(idx) {
    if (!originsMode || idx === activeCell) return;
    if (idx !== null && !origins[String(idx)]) return;
    activeCell = idx; scheduleDraw();
  }
  Plotly.newPlot(plotEl, [heatmapTrace(currentState())], layout(), { displayModeBar: false, responsive: true }).then(function () {
    renderPanel(currentState());
    if (originsMode) {
      plotEl.on('plotly_hover', function (ev) {
        if (pinnedCell !== null) return;
        var pt = ev.points && ev.points[0];
        if (!pt) return;
        setActive(cellIndexAt(pt.x, pt.y));
      });
      plotEl.on('plotly_click', function (ev) {
        var pt = ev.points && ev.points[0]; if (!pt) return;
        var idx = cellIndexAt(pt.x, pt.y);
        if (!origins[String(idx)]) return;
        pinnedCell = pinnedCell === idx ? null : idx; activeCell = idx; draw();
      });
    }
  });
  document.querySelectorAll('#' + prefix + '-wrap .qmap-btn[data-metric]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('#' + prefix + '-wrap .qmap-btn[data-metric]').forEach(function (b) { b.classList.remove('is-active'); });
      btn.classList.add('is-active'); metric = btn.getAttribute('data-metric'); draw();
    });
  });
  document.querySelectorAll('#' + prefix + '-wrap .qmap-btn[data-scale]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('#' + prefix + '-wrap .qmap-btn[data-scale]').forEach(function (b) { b.classList.remove('is-active'); });
      btn.classList.add('is-active'); scaleMode = btn.getAttribute('data-scale'); draw();
    });
  });
  var resetBtn = document.getElementById(prefix + '-reset');
  if (resetBtn) resetBtn.addEventListener('click', function () { activeCell = null; pinnedCell = null; draw(); });
  return { setData: function (newData) {
    origins = newData.origins || {}; overall = newData.overall;
    DATA.title = newData.title; DATA.subtitle = newData.subtitle;
    activeCell = null; pinnedCell = null; draw();
  }};
}
"""

_PLAYER_ROUTES_JS = """
function initPlayerRoutes(DATA, XP_SCALE, VOL_SCALE, PLOT_HEIGHT) {
  var PLAYERS = DATA.players || [];
  var selectedId = DATA.default_player_id || (PLAYERS[0] && PLAYERS[0].id) || null;
  var engine = null;
  var panelEl = document.getElementById('qmap-pl-panel');

  function fmtInt(v) { return Number(v).toLocaleString('pt-BR'); }
  function fmtPct(v) { return Number(v).toLocaleString('pt-BR', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%'; }
  function fmtXp(v) { return Number(v).toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }

  function selectedPlayer() {
    for (var i = 0; i < PLAYERS.length; i++) if (PLAYERS[i].id === selectedId) return PLAYERS[i];
    return null;
  }
  function routeRow(route, highlightXp) {
    return '<div class="qp-row"><span class="qp-row-route">' + route.route_label + '</span>'
      + '<span class="qp-row-meta">' + fmtInt(route.count) + ' passes · ' + fmtPct(route.share_pct)
      + (highlightXp ? ' · xP total <b>' + fmtXp(route.total_xp) + '</b>' : ' · xP médio <b>' + fmtXp(route.mean_xp) + '</b>')
      + '</span></div>';
  }
  function renderRoutesPanel(player) {
    if (!player) {
      panelEl.innerHTML = '<p class="qp-hint">Sem atletas disponíveis.</p>'; return;
    }
    var routes = player.routes || {};
    var common = routes.common || [], highXp = routes.high_xp || [];
    panelEl.innerHTML = '<span class="qp-title">' + player.name + '</span>'
      + '<span class="qp-sub">' + player.team + ' · ' + fmtInt(player.passes) + ' passes · xP médio ' + fmtXp(player.mean_xp) + '</span>'
      + '<span class="qp-section-label">Top 5 O→D mais comuns</span>'
      + (common.length ? common.map(function (r) { return routeRow(r, false); }).join('') : '<p class="qp-hint">Sem rotas com amostra suficiente.</p>')
      + '<span class="qp-section-label">Top 5 O→D com maior xP gerado</span>'
      + (highXp.length ? highXp.map(function (r) { return routeRow(r, true); }).join('') : '<p class="qp-hint">Sem rotas com amostra suficiente.</p>');
  }
  function playerMapData(player) {
    return {
      cols: DATA.cols, rows: DATA.rows, field_x: DATA.field_x, field_y: DATA.field_y,
      xp_scale_max: DATA.xp_scale_max, volume_scale_max: DATA.volume_scale_max,
      cell_labels: DATA.cell_labels, quadrant_labels: DATA.quadrant_labels,
      origins: {}, overall: player.overall,
      title: player.name, subtitle: player.team + ' · destinos dos passes completados'
    };
  }
  function refresh() {
    var player = selectedPlayer();
    renderRoutesPanel(player);
    if (!player) return;
    if (!engine) {
      engine = qmapEngine('qmap-pl', playerMapData(player), XP_SCALE, VOL_SCALE, PLOT_HEIGHT, { originsMode: false, hidePanel: true });
    } else {
      engine.setData(playerMapData(player));
    }
  }
  var selectEl = document.getElementById('qmap-pl-player-select');
  if (selectEl) {
    selectEl.innerHTML = PLAYERS.map(function (p) {
      return '<option value="' + p.id + '"' + (p.id === selectedId ? ' selected' : '') + '>' + p.name + ' · ' + p.team + '</option>';
    }).join('');
    selectEl.onchange = function () { selectedId = selectEl.value; refresh(); };
  }
  refresh();
}
"""

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="__PLOTLY_CDN__"></script>
<style>
__SHARED_CSS__
  #qmap-agg-panel { height: __AGG_PLOT_HEIGHT__px; }
  #qmap-agg-plot, #qmap-pl-plot { min-height: __AGG_PLOT_HEIGHT__px; }
  #qmap-pl-plot { min-height: __PL_PLOT_HEIGHT__px; }
  #qmap-pl-panel { max-height: 340px; height: auto; }
</style>
</head>
<body>
<div class="qmap-wrap" id="qmap-agg-wrap">
  <p class="qmap-section-title">Todos os atletas</p>
  <div class="qmap-toolbar">
    <span class="qmap-toolbar-label">Cor</span>
    <button class="qmap-btn is-active" data-metric="xp">xP médio</button>
    <button class="qmap-btn" data-metric="volume">Volume (%)</button>
    <span class="qmap-toolbar-sep"></span>
    <span class="qmap-toolbar-label">Escala</span>
    <button class="qmap-btn is-active" data-scale="fixed">Fixa</button>
    <button class="qmap-btn" data-scale="relative">Relativa</button>
    <button class="qmap-btn" id="qmap-agg-reset" style="margin-left:auto">Ver todos</button>
  </div>
  <div class="qmap-body">
    <div id="qmap-agg-plot" class="qmap-plot"></div>
    <aside class="qmap-panel" id="qmap-agg-panel"></aside>
  </div>
</div>

<div class="qmap-wrap" id="qmap-pl-wrap">
  <p class="qmap-section-title">Por atleta — rotas O→D</p>
  <div class="qmap-toolbar">
    <span class="qmap-toolbar-label">Atleta</span>
    <select class="qp-player-select" id="qmap-pl-player-select" style="flex:1;max-width:360px;margin-bottom:0"></select>
  </div>
  <div class="qmap-body">
    <div id="qmap-pl-plot" class="qmap-plot"></div>
    <aside class="qmap-panel" id="qmap-pl-panel"></aside>
  </div>
</div>

<script>
__MAP_ENGINE_JS__
__PLAYER_ROUTES_JS__
(function () {
  var DATA = __DATA__;
  var XP_SCALE = __XP_SCALE__;
  var VOL_SCALE = __VOL_SCALE__;
  var aggData = Object.assign({}, DATA.aggregate, {
    title: 'Todos os atletas',
    subtitle: (DATA.aggregate.total_passes || 0).toLocaleString('pt-BR') + ' passes agregados · passe o mouse por uma célula de origem'
  });
  qmapEngine('qmap-agg', aggData, XP_SCALE, VOL_SCALE, __AGG_PLOT_HEIGHT__, { originsMode: true });
})();
(function () {
  initPlayerRoutes(__PLAYER_DATA__, __XP_SCALE__, __VOL_SCALE__, __PL_PLOT_HEIGHT__);
})();
</script>
</body>
</html>
"""


def _aggregate_payload(analysis: dict) -> dict:
    return {
        "origins": analysis.get("origins") or {},
        "overall": analysis.get("overall"),
        "total_passes": analysis.get("total_passes", 0),
        "cols": analysis.get("cols", 12),
        "rows": analysis.get("rows", 8),
        "field_x": analysis.get("field_x", 120.0),
        "field_y": analysis.get("field_y", 80.0),
        "xp_scale_max": analysis.get("xp_scale_max", 1.0),
        "volume_scale_max": analysis.get("volume_scale_max", 1.0),
        "cell_labels": analysis.get("cell_labels") or [],
        "quadrant_labels": analysis.get("quadrant_labels") or {},
    }


def _players_payload(analysis: dict) -> dict:
    players = []
    for player in analysis.get("players") or []:
        players.append({
            "id": player.get("id"),
            "name": player.get("name"),
            "team": player.get("team"),
            "passes": player.get("passes"),
            "mean_xp": player.get("mean_xp"),
            "overall": player.get("overall"),
            "routes": player.get("routes") or {"common": [], "high_xp": []},
        })
    return {
        "players": players,
        "default_player_id": analysis.get("default_player_id"),
        "cols": analysis.get("cols", 12),
        "rows": analysis.get("rows", 8),
        "field_x": analysis.get("field_x", 120.0),
        "field_y": analysis.get("field_y", 80.0),
        "xp_scale_max": analysis.get("xp_scale_max", 1.0),
        "volume_scale_max": analysis.get("volume_scale_max", 1.0),
        "cell_labels": analysis.get("cell_labels") or [],
        "quadrant_labels": analysis.get("quadrant_labels") or {},
    }


def build_cell_map_html(
    analysis: dict,
    *,
    plot_height: int = 500,
    player_plot_height: int = 420,
) -> str:
    """Self-contained Plotly page: aggregate hover map + per-player O→D routes."""
    html = (
        _TEMPLATE
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
        .replace("__SHARED_CSS__", _SHARED_CSS)
        .replace("__MAP_ENGINE_JS__", _MAP_ENGINE_JS)
        .replace("__PLAYER_ROUTES_JS__", _PLAYER_ROUTES_JS)
        .replace("__DATA__", json.dumps({"aggregate": _aggregate_payload(analysis)}))
        .replace("__PLAYER_DATA__", json.dumps(_players_payload(analysis)))
        .replace("__XP_SCALE__", json.dumps([[s, c] for s, c in XP_COLORSCALE]))
        .replace("__VOL_SCALE__", json.dumps([[s, c] for s, c in VOLUME_COLORSCALE]))
        .replace("__AGG_PLOT_HEIGHT__", str(int(plot_height)))
        .replace("__PL_PLOT_HEIGHT__", str(int(player_plot_height)))
    )
    return html


build_quadrant_map_html = build_cell_map_html
