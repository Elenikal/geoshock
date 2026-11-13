/* GeoShock Monitor — renders docs/data/dashboard.json */
"use strict";

const QS = ["0.05", "0.10", "0.25", "0.50", "0.75", "0.90", "0.95"];
const SPECS = { baseline: "Baseline", enhanced: "With channels", enhanced_l0: "Layer 0 scenario" };
let D = null, SIG = [], iarH = "6";

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2, sign = false) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : (sign && x > 0 ? "+" : "") + Number(x).toFixed(d);
const pct = (x, d = 0) => (x === null || x === undefined ? "—" : (100 * x).toFixed(d) + "%");
const monthName = (ym) => {
  if (!ym) return "—";
  const [y, m] = ym.slice(0, 7).split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-GB", { month: "short", year: "numeric", timeZone: "UTC" });
};
const el = (id) => document.getElementById(id);

function layout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: css("--ink"), size: 12, family: getComputedStyle(document.body).fontFamily },
    margin: { l: 44, r: 12, t: 8, b: 36 },
    xaxis: { gridcolor: css("--grid"), zerolinecolor: css("--line"), linecolor: css("--line") },
    yaxis: { gridcolor: css("--grid"), zerolinecolor: css("--line"), linecolor: css("--line") },
    legend: { orientation: "h", y: 1.12, x: 0, bgcolor: "rgba(0,0,0,0)" },
    hovermode: "x unified",
  }, extra);
}
const CFG = { displayModeBar: false, responsive: true };

function regimeColor(r) {
  return r === "CRISIS" ? css("--crit") : r === "ELEVATED" ? css("--warn") : css("--ok");
}

/* ── KPIs ─────────────────────────────────────────────────────────────── */
function renderKPIs() {
  const l0 = D.layer0 || {}, st = D.state, L = D.latest;
  const e6 = D.iar.enhanced["6"], b6 = D.iar.baseline["6"];
  const items = [
    {
      lab: "Threat nowcast (Layer 0)",
      val: l0.ok ? fmt(l0.gpr_nowcast, 2) : "—",
      sub: l0.ok ? `<span class="pill" style="color:${regimeColor(l0.regime)}">${l0.regime}</span>` +
        (l0.stale ? ` <span class="stale">stale</span>` : "") : "unavailable",
    },
    { lab: `Official GPRT z (${monthName(L.gprt_z?.month)})`, val: fmt(L.gprt_z?.value, 2),
      sub: `Acts GPRA z: ${fmt(L.gpra_z?.value, 2)}` },
    { lab: `Channel state, GIPI (${monthName(st.month)})`, val: fmt(st.gipi, 2, true),
      sub: `<span class="pill" style="color:${st.open ? css("--open") : css("--closed")}">${st.open ? "OPEN" : "CLOSED"}</span> ${st.months_in_state} mo` },
    { lab: "P(CPI > 3%) in 6 months", val: pct(e6.p_gt3),
      sub: `without channels: ${pct(b6.p_gt3)}` },
  ];
  el("kpis").innerHTML = items.map((k) =>
    `<div class="kpi"><div class="lab">${k.lab}</div><div class="val">${k.val}</div><div class="sub">${k.sub}</div></div>`).join("");
}

function renderVerdict() {
  const st = D.state, l0 = D.layer0 || {}, lp = D.lp_state.cpi_inflation;
  const b12 = st.open ? lp.b_hi[12] : lp.b_lo[12];
  const e6 = D.iar.enhanced["6"];
  const threat = l0.ok ? `Real-time threats are <b>${l0.regime.toLowerCase()}</b> (ẑ = ${fmt(l0.gpr_nowcast)})` :
    `The latest official threat reading is z = ${fmt(D.latest.gprt_z?.value)}`;
  const chan = st.open
    ? `channels are <b>open</b>, so a one-standard-deviation threat shock has historically raised headline inflation by about ${fmt(b12, 2)} pp after a year`
    : `channels are <b>closed</b>, so a one-standard-deviation threat shock has historically moved headline inflation by ${fmt(b12, 2, true)} pp after a year, i.e. no upward pressure`;
  el("verdict").innerHTML = `${threat}, and ${chan}. The model puts the probability of CPI inflation above 3% six months ahead at <b>${pct(e6.p_gt3)}</b>` +
    (D.iar.enhanced_l0 ? ` (${pct(D.iar.enhanced_l0["6"].p_gt3)} under the Layer 0 scenario).` : ".");
}

/* ── Charts ───────────────────────────────────────────────────────────── */
function renderGPR() {
  const H = D.history;
  const x = H.dates.map((d) => d + "-01");
  const lines = [1.5, 2.5].map((y, i) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
    line: { color: i ? css("--crit") : css("--warn"), width: 1, dash: "dot" } }));
  Plotly.react("ch-gpr", [
    { x, y: H.gprt_z, name: "Threats (GPRT)", type: "scatter", mode: "lines", line: { color: css("--accent"), width: 2 } },
    { x, y: H.gpra_z, name: "Acts (GPRA)", type: "scatter", mode: "lines", line: { color: css("--muted"), width: 1.5, dash: "dash" } },
  ], layout({ shapes: lines, yaxis: { title: "z", gridcolor: css("--grid") } }), CFG);
}

function renderGIPI() {
  const H = D.history, med = D.state.median;
  const x = H.dates.map((d) => d + "-01");
  const base = H.gipi.map((v) => (v === null ? null : med));
  const up = H.gipi.map((v) => (v === null ? null : Math.max(v, med)));
  const dn = H.gipi.map((v) => (v === null ? null : Math.min(v, med)));
  const area = (y, c) => ({ x, y, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
    fillcolor: c, hoverinfo: "skip", showlegend: false });
  const ref = { x, y: base, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false };
  Plotly.react("ch-gipi", [
    ref, area(up, css("--open") + "55"), ref, area(dn, css("--closed") + "40"),
    { x, y: H.gipi, name: "GIPI", type: "scatter", mode: "lines", line: { color: css("--ink"), width: 1.8 } },
  ], layout({
    showlegend: false,
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: med, y1: med, line: { color: css("--muted"), dash: "dash", width: 1 } }],
  }), CFG);
}

function band(b, se, k = 1.645) {
  return { lo: b.map((v, i) => v - k * se[i]), hi: b.map((v, i) => v + k * se[i]) };
}

function renderLP() {
  const lp = D.lp_state.cpi_inflation, h = lp.b_hi.map((_, i) => i), open = D.state.open;
  const tr = [];
  [["hi", "Channels open", css("--hi"), open], ["lo", "Channels closed", css("--lo"), !open]].forEach(([k, nm, c, cur]) => {
    const B = band(lp["b_" + k], lp["se_" + k]);
    tr.push({ x: h.concat([...h].reverse()), y: B.hi.concat([...B.lo].reverse()), fill: "toself", type: "scatter",
      mode: "lines", line: { width: 0 }, fillcolor: c + (cur ? "33" : "14"), hoverinfo: "skip", showlegend: false });
    tr.push({ x: h, y: lp["b_" + k], name: nm + (cur ? " (current)" : ""), type: "scatter", mode: "lines+markers",
      marker: { size: cur ? 5 : 3 }, line: { color: c, width: cur ? 3 : 1.5, dash: cur ? "solid" : "dot" } });
  });
  Plotly.react("ch-lp", tr, layout({ xaxis: { title: "months after shock", gridcolor: css("--grid") },
    yaxis: { title: "pp", gridcolor: css("--grid"), zeroline: true }, hovermode: "x" }), CFG);
  const p = lp.p_diff[12];
  el("lp-text").textContent = `Difference between states at 12 months: ${fmt(lp.diff[12], 2, true)} pp (p = ${fmt(p, 3)}). ` +
    `Core CPI and industrial production show no state-dependence (see paper, Table 6). Re-estimated on the current data vintage.`;
}

function renderIARTabs() {
  el("iar-tabs").innerHTML = ["3", "6", "12"].map((h) =>
    `<button role="tab" aria-selected="${h === iarH}" data-h="${h}">${h} months</button>`).join("");
  el("iar-tabs").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    iarH = b.dataset.h; renderIARTabs(); renderIAR();
  }));
}

function renderIAR() {
  const specs = Object.keys(SPECS).filter((s) => D.iar[s]);
  const colors = { baseline: css("--muted"), enhanced: css("--hi"), enhanced_l0: css("--accent") };
  const tr = specs.map((s, i) => {
    const q = D.iar[s][iarH].q;
    return { type: "box", name: SPECS[s], orientation: "h", y: [SPECS[s]],
      q1: [q["0.25"]], median: [q["0.50"]], q3: [q["0.75"]], lowerfence: [q["0.05"]], upperfence: [q["0.95"]],
      marker: { color: colors[s] }, line: { color: colors[s] }, fillcolor: colors[s] + "33", hoverinfo: "x" };
  });
  Plotly.react("ch-iar", tr, layout({
    showlegend: false, height: 60 + 60 * specs.length, margin: { l: 120, r: 12, t: 8, b: 36 },
    xaxis: { title: "CPI inflation YoY, % (5th–95th percentile; box = 25th–75th)", gridcolor: css("--grid") },
    shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: 3, x1: 3, line: { color: css("--crit"), dash: "dot", width: 1 } }],
    hovermode: "closest",
  }), CFG);
  const rows = specs.map((s) => {
    const r = D.iar[s][iarH];
    return `<tr><td>${SPECS[s]}</td>${["0.05", "0.50", "0.90", "0.95"].map((k) => `<td>${fmt(r.q[k])}</td>`).join("")}
      <td>${pct(r.p_gt3, 1)}</td><td>${pct(r.p_gt4, 1)}</td><td>${monthName(r.origin)}</td></tr>`;
  }).join("");
  el("tb-iar").innerHTML = `<thead><tr><th>${iarH}-month horizon</th><th>Q05</th><th>Median</th><th>Q90</th><th>Q95</th>
    <th>P(&gt;3%)</th><th>P(&gt;4%)</th><th>Data as of</th></tr></thead><tbody>${rows}</tbody>`;
}

function renderChannels() {
  const H = D.history, n = H.dates.length, start = Math.max(0, n - 60);
  const x = H.dates.slice(start).map((d) => d + "-01");
  const series = [["energy", "Global energy YoY %"], ["imports", "Import prices YoY %"],
                  ["food", "Global food YoY %"], ["gscpi", "GSCPI (s.d.)"]];
  const tr = series.map(([k, nm], i) => ({ x, y: H[k].slice(start), name: nm, type: "scatter", mode: "lines",
    xaxis: "x" + (i + 1), yaxis: "y" + (i + 1), line: { color: css("--ink"), width: 1.6 }, showlegend: false }));
  const L = layout({ grid: { rows: 2, columns: 2, pattern: "independent" }, hovermode: "closest",
    margin: { l: 36, r: 8, t: 22, b: 28 },
    annotations: series.map(([, nm], i) => ({ text: nm, xref: `x${i + 1} domain`, yref: `y${i + 1} domain`,
      x: 0, y: 1.14, showarrow: false, xanchor: "left", font: { size: 11, color: css("--muted") } })) });
  for (let i = 1; i <= 4; i++) {
    L["xaxis" + i] = { gridcolor: css("--grid"), linecolor: css("--line"), nticks: 4 };
    L["yaxis" + i] = { gridcolor: css("--grid"), zerolinecolor: css("--muted") };
  }
  Plotly.react("ch-channels", tr, L, CFG);
}

function renderGAR() {
  const specs = ["baseline", "enhanced", "enhanced_l0"].filter((s) => D.gar[s]);
  const rows = [];
  ["3", "6", "12"].forEach((h) => specs.forEach((s) => {
    const r = D.gar[s][h];
    rows.push(`<tr><td>${h}m · ${SPECS[s]}</td><td>${fmt(r.q["0.05"])}</td><td>${fmt(r.q["0.50"])}</td>
      <td>${pct(r.p_lt0)}</td><td>${pct(r.p_ltm2)}</td></tr>`);
  }));
  el("tb-gar").innerHTML = `<thead><tr><th>IP growth YoY</th><th>GaR 5%</th><th>Median</th><th>P(&lt;0)</th>
    <th>P(&lt;−2%)</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

function renderL0() {
  const l0 = D.layer0 || {};
  if (!l0.ok) {
    el("l0-detail").innerHTML = `<p class="small">Layer 0 unavailable on the last build${l0.error ? ` (${l0.error.slice(0, 80)}…)` : ""}.</p>`;
  } else {
    const a = l0.ais_signal || {};
    const heads = (l0.top_headlines || []).slice(0, 5).map((t) => `<li>${t.replace(/</g, "&lt;")}</li>`).join("");
    el("l0-detail").innerHTML = `
      <table><tbody>
        <tr><td>Signal time (UTC)</td><td>${(l0.timestamp || "").replace("T", " ").slice(0, 16)}${l0.stale ? ' <span class="stale">stale</span>' : ""}</td></tr>
        <tr><td>Composite severity S</td><td>${fmt(l0.severity_score, 1)} / 10</td></tr>
        <tr><td>Anchor (GPRT ${monthName(l0.anchor_month)})</td><td>${fmt(l0.gpr_anchor_z)}</td></tr>
        <tr><td>CAMEO codes</td><td>${(l0.cameo_codes || []).join(", ") || "—"}</td></tr>
        <tr><td>Articles (Middle East share)</td><td>${l0.article_count ?? "—"} (${pct(l0.location_hit_rate)})</td></tr>
        <tr><td>Tanker / freight / Brent–WTI z</td><td>${fmt(a.tanker_z)} / ${fmt(a.freight_z)} / ${fmt(a.brent_wti_z)}</td></tr>
        <tr><td>AIS anomaly</td><td>${l0.ais_anomaly ? "yes" : "no"}</td></tr>
        <tr><td>Coding</td><td>${l0.llm_used ? "LLM" : "rule-based"}</td></tr>
      </tbody></table>
      ${heads ? `<ul class="headlines">${heads}</ul>` : ""}`;
  }
  const s = SIG.filter((r) => r.gpr_nowcast !== null && r.gpr_nowcast !== undefined);
  if (!s.length) {
    el("ch-l0").innerHTML = `<p class="small">The signal archive starts with the first automated build.</p>`;
    return;
  }
  Plotly.react("ch-l0", [{ x: s.map((r) => r.timestamp), y: s.map((r) => r.gpr_nowcast), type: "scatter",
    mode: "lines+markers", name: "ẑ", line: { color: css("--accent") } }],
  layout({ showlegend: false, yaxis: { title: "ẑ", gridcolor: css("--grid") },
    shapes: [1.5, 2.5].map((y, i) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
      line: { color: i ? css("--crit") : css("--warn"), dash: "dot", width: 1 } })) }), CFG);
}

function renderAll() {
  const built = new Date(D.built_at);
  el("meta").innerHTML = `Updated ${built.toUTCString().replace(" GMT", " UTC")}` +
    (D.offline_test ? ` · <span class="stale">test build on pinned April-2026 data</span>` : "") +
    ` · CPI through ${monthName(D.latest.cpi?.month)} · GIPI through ${monthName(D.state.month)}`;
  if ((D.notes || []).length) {
    const ul = document.querySelector(".notes ul");
    D.notes.forEach((n) => { const li = document.createElement("li"); li.textContent = n; ul.appendChild(li); });
  }
  renderKPIs(); renderVerdict(); renderGPR(); renderGIPI(); renderLP();
  renderIARTabs(); renderIAR(); renderChannels(); renderGAR(); renderL0();
}

async function boot() {
  try {
    const [d, s] = await Promise.all([
      fetch("data/dashboard.json", { cache: "no-store" }).then((r) => r.json()),
      fetch("data/signals.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : [])).catch(() => []),
    ]);
    D = d; SIG = s;
    renderAll();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  } catch (e) {
    el("meta").textContent = "Could not load dashboard data: " + e;
  }
}
window.addEventListener("DOMContentLoaded", () => (window.Plotly ? boot() : window.addEventListener("load", boot)));
