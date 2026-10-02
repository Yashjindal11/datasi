/* DataSI offline report renderer. Data is only ever inserted as text nodes. */
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("datasi-data").textContent);
  var SVGNS = "http://www.w3.org/2000/svg";
  var SEV_ORDER = { critical: 3, high: 2, warning: 1, info: 0 };
  var CATEGORY_TITLES = {
    schema: "Schema & structure", missingness: "Missingness", duplicates: "Duplicates",
    types: "Types & rules", numeric: "Numeric", categorical: "Categorical", strings: "Text hygiene",
    temporal: "Temporal", relationships: "Relationships", distribution: "Distribution", target: "Target"
  };

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "class") node.className = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), attrs[k]);
        else node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined || c === false) continue;
      if (Array.isArray(c)) c.forEach(function (x) { if (x) node.appendChild(typeof x === "string" ? document.createTextNode(x) : x); });
      else node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }
  function svg(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }
  function stext(x, y, s, anchor) {
    var t = svg("text", { x: x, y: y, "text-anchor": anchor || "start" });
    t.textContent = s;
    return t;
  }
  function fmt(v, digits) {
    if (v === null || v === undefined) return "–";
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v)) return v.toLocaleString();
    var a = Math.abs(v);
    if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(2);
    return v.toLocaleString(undefined, { maximumFractionDigits: digits === undefined ? 4 : digits });
  }
  function pct(v) { return v === null || v === undefined ? "–" : (v * 100).toFixed(v < 0.001 && v > 0 ? 3 : 1) + "%"; }
  function bytes(n) {
    var u = ["B", "KB", "MB", "GB"], i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i ? 1 : 0) + " " + u[i];
  }
  function sevBadge(s) { return el("span", { class: "sev " + s, text: s }); }

  /* ---------- charts ---------- */
  function barChart(items, opts) {
    opts = opts || {};
    var w = opts.width || 520, rowH = 18, labelW = opts.labelWidth || 160;
    var h = items.length * rowH + 8;
    var max = opts.max || Math.max.apply(null, items.map(function (d) { return d.value || 0; }).concat([1e-12]));
    var root = svg("svg", { width: "100%", viewBox: "0 0 " + w + " " + h, role: "img" });
    items.forEach(function (d, i) {
      var y = i * rowH + 4;
      var label = String(d.label);
      root.appendChild(stext(labelW - 6, y + 12, label.length > 26 ? label.slice(0, 25) + "…" : label, "end"));
      var bw = Math.max(1, ((d.value || 0) / max) * (w - labelW - 70));
      var r = svg("rect", { x: labelW, y: y + 2, width: bw, height: rowH - 6, class: "bar" + (d.alt ? " alt" : "") });
      var title = svg("title", {}); title.textContent = label + ": " + (opts.fmt || fmt)(d.value); r.appendChild(title);
      root.appendChild(r);
      root.appendChild(stext(labelW + bw + 4, y + 12, (opts.fmt || fmt)(d.value)));
    });
    return root;
  }
  function histogram(h, title) {
    if (!h || !h.counts || !h.counts.length) return el("div", { class: "empty", text: "No histogram" });
    var w = 340, ht = 120, pad = 18;
    var max = Math.max.apply(null, h.counts);
    var bw = (w - 2 * pad) / h.counts.length;
    var root = svg("svg", { width: "100%", viewBox: "0 0 " + w + " " + (ht + 20), role: "img" });
    h.counts.forEach(function (c, i) {
      var bh = max ? (c / max) * ht : 0;
      var r = svg("rect", { x: pad + i * bw, y: ht - bh, width: Math.max(bw - 1, 1), height: bh, class: "bar" });
      var t = svg("title", {}); t.textContent = fmt(h.edges[i]) + " – " + fmt(h.edges[i + 1]) + ": " + fmt(c); r.appendChild(t);
      root.appendChild(r);
    });
    root.appendChild(svg("line", { x1: pad, x2: w - pad, y1: ht, y2: ht, class: "axis" }));
    root.appendChild(stext(pad, ht + 14, fmt(h.edges[0])));
    root.appendChild(stext(w - pad, ht + 14, fmt(h.edges[h.edges.length - 1]), "end"));
    if (title) root.appendChild(stext(w / 2, ht + 14, title, "middle"));
    return root;
  }
  function boxplot(s) {
    if (!s || s.q1 === undefined) return null;
    var w = 340, h = 40, pad = 18, lo = s.min, hi = s.max;
    var span = hi - lo || 1;
    function x(v) { return pad + ((v - lo) / span) * (w - 2 * pad); }
    var iqr = s.q3 - s.q1, wl = Math.max(lo, s.q1 - 1.5 * iqr), wh = Math.min(hi, s.q3 + 1.5 * iqr);
    var root = svg("svg", { width: "100%", viewBox: "0 0 " + w + " " + (h + 14), role: "img" });
    root.appendChild(svg("line", { x1: x(wl), x2: x(wh), y1: h / 2, y2: h / 2, class: "axis" }));
    root.appendChild(svg("rect", { x: x(s.q1), y: 8, width: Math.max(1, x(s.q3) - x(s.q1)), height: h - 16, class: "bar" }));
    root.appendChild(svg("line", { x1: x(s.median), x2: x(s.median), y1: 6, y2: h - 6, stroke: "currentColor", "stroke-width": 2 }));
    root.appendChild(stext(pad, h + 12, fmt(lo)));
    root.appendChild(stext(w - pad, h + 12, fmt(hi), "end"));
    root.appendChild(stext(x(s.median), h + 12, "median " + fmt(s.median), "middle"));
    return root;
  }
  function lineChart(points, opts) {
    opts = opts || {};
    var pts = points.filter(function (p) { return p.v !== null && p.v !== undefined; });
    if (pts.length < 2) return el("div", { class: "empty", text: "Not enough points" });
    var w = 560, h = 150, pad = 30;
    var vals = pts.map(function (p) { return p.v; });
    var min = Math.min.apply(null, vals.concat(opts.zero ? [0] : [])), max = Math.max.apply(null, vals);
    if (max === min) max = min + 1;
    var n = points.length;
    function x(i) { return pad + (i / Math.max(n - 1, 1)) * (w - 2 * pad); }
    function y(v) { return h - pad + 10 - ((v - min) / (max - min)) * (h - pad); }
    var root = svg("svg", { width: "100%", viewBox: "0 0 " + w + " " + (h + 10), role: "img" });
    var d = "";
    points.forEach(function (p, i) {
      if (p.v === null || p.v === undefined) return;
      d += (d ? " L " : "M ") + x(i).toFixed(1) + " " + y(p.v).toFixed(1);
    });
    root.appendChild(svg("line", { x1: pad, x2: w - pad, y1: h - pad + 10, y2: h - pad + 10, class: "axis" }));
    root.appendChild(svg("path", { d: d, class: "line" }));
    if (opts.marker) {
      var mi = points.findIndex(function (p) { return String(opts.marker).indexOf(p.t) === 0; });
      if (mi >= 0) root.appendChild(svg("line", { x1: x(mi), x2: x(mi), y1: 4, y2: h - pad + 10, stroke: "#d4572a", "stroke-dasharray": "4 3" }));
    }
    root.appendChild(stext(pad, h + 6, points[0].t));
    root.appendChild(stext(w - pad, h + 6, points[n - 1].t, "end"));
    root.appendChild(stext(4, 12, (opts.fmt || fmt)(max)));
    root.appendChild(stext(4, h - pad + 10, (opts.fmt || fmt)(min)));
    return root;
  }
  function heatmap(labels, matrix) {
    var n = labels.length, cell = Math.max(10, Math.min(28, Math.floor(520 / Math.max(n, 1)))), lab = 120;
    var size = lab + n * cell;
    var root = svg("svg", { width: "100%", viewBox: "0 0 " + (size + 90) + " " + size, role: "img", style: "max-width:" + (size + 90) + "px" });
    for (var i = 0; i < n; i++) {
      root.appendChild(stext(lab - 4, lab + i * cell + cell * 0.7, labels[i].slice(0, 18), "end"));
      var t = stext(lab + i * cell + cell * 0.7, lab - 4, labels[i].slice(0, 18));
      t.setAttribute("transform", "rotate(-60 " + (lab + i * cell + cell * 0.7) + " " + (lab - 4) + ")");
      root.appendChild(t);
      for (var j = 0; j < n; j++) {
        var v = matrix[i][j];
        var color = v === null ? "#ccc" : v >= 0 ? "rgba(47,91,234," + Math.abs(v) + ")" : "rgba(212,87,42," + Math.abs(v) + ")";
        var r = svg("rect", { x: lab + j * cell, y: lab + i * cell, width: cell - 1, height: cell - 1, fill: color });
        var tt = svg("title", {}); tt.textContent = labels[i] + " × " + labels[j] + ": " + fmt(v, 3); r.appendChild(tt);
        root.appendChild(r);
      }
    }
    return root;
  }

  /* ---------- tables ---------- */
  function table(columns, rows) {
    var state = { key: null, asc: true };
    var tbody = el("tbody");
    var ths = columns.map(function (c) {
      return el("th", { text: c.title, onclick: function () {
        state.asc = state.key === c.key ? !state.asc : true; state.key = c.key;
        ths.forEach(function (t) { t.className = ""; });
        this.className = "sorted" + (state.asc ? " asc" : "");
        draw();
      } });
    });
    function draw() {
      var data = rows.slice();
      if (state.key) {
        data.sort(function (a, b) {
          var x = a[state.key], y = b[state.key];
          if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
          return (x < y ? -1 : 1) * (state.asc ? 1 : -1);
        });
      }
      tbody.textContent = "";
      data.forEach(function (r) {
        tbody.appendChild(el("tr", null, columns.map(function (c) {
          var v = r[c.key];
          var cell = c.render ? c.render(v, r) : (typeof v === "number" ? fmt(v) : (v === null || v === undefined ? "–" : String(v)));
          return el("td", { class: typeof v === "number" ? "num" : "" }, cell);
        })));
      });
    }
    draw();
    return el("table", null, el("thead", null, el("tr", null, ths)), tbody);
  }

  /* ---------- finding cards ---------- */
  function findingCard(f, open) {
    var docs = (DATA.detector_docs || {})[f.detector] || {};
    var body = el("div", { class: "body" },
      el("div", { class: "label", text: "Observation" }), el("div", { text: f.observation }),
      f.interpretation ? el("div", { class: "label", text: "Possible interpretation" + (f.confidence !== null && f.confidence !== undefined ? " · confidence " + f.confidence.toFixed(2) : "") }) : null,
      f.interpretation ? el("div", { text: f.interpretation }) : null,
      f.suggestion ? el("div", { class: "label", text: "Suggested investigation" }) : null,
      f.suggestion ? el("div", { text: f.suggestion }) : null,
      f.rule ? el("div", { class: "label", text: "Why this severity" }) : null,
      f.rule ? el("div", { class: "small", text: f.rule }) : null
    );
    if (f.evidence && f.evidence.series) {
      body.appendChild(el("div", { class: "label", text: "Over time" }));
      body.appendChild(lineChart(f.evidence.series, { marker: f.evidence.change_at || undefined }));
    }
    var ev = {};
    Object.keys(f.evidence || {}).forEach(function (k) { if (k !== "series") ev[k] = f.evidence[k]; });
    body.appendChild(el("div", { class: "label", text: "Evidence" }));
    body.appendChild(el("pre", { text: JSON.stringify(ev, null, 2) }));
    if (docs.description) {
      body.appendChild(el("div", { class: "label", text: "About the " + f.detector + " detector" }));
      body.appendChild(el("div", { class: "small muted", text: docs.description + (docs.limitations ? " Limitations: " + docs.limitations : "") }));
    }
    body.appendChild(el("div", { class: "small muted", text: "code " + f.code + " · id " + f.id }));
    var d = el("details", { class: "finding " + f.severity },
      el("summary", null, sevBadge(f.severity), el("span", { class: "title", text: f.title }),
        el("span", { class: "cols", text: (f.columns || []).join(", ") })),
      body);
    if (open) d.open = true;
    return d;
  }

  /* ---------- views ---------- */
  var S = DATA.summary, D = DATA.dataset, SEC = DATA.sections || {}, PROF = DATA.profiles || {};
  var findings = DATA.findings || [];

  function overview() {
    var root = el("div");
    var cards = [["Rows", fmt(D.rows)], ["Columns", fmt(D.columns)], ["Memory", bytes(D.memory_bytes)], ["Findings", fmt(S.findings)]];
    ["critical", "high", "warning", "info"].forEach(function (s) { cards.push([s[0].toUpperCase() + s.slice(1), fmt(S.by_severity[s] || 0)]); });
    root.appendChild(el("div", { class: "cards" }, cards.map(function (c) {
      return el("div", { class: "card" }, el("div", { class: "k", text: c[0] }), el("div", { class: "v", text: c[1] }));
    })));
    if (DATA.reference) root.appendChild(el("div", { class: "note", text: "Comparison: current '" + D.name + "' (" + fmt(D.rows) + " rows) against reference '" + DATA.reference.name + "' (" + fmt(DATA.reference.rows) + " rows)." }));
    if (D.sampling) root.appendChild(el("div", { class: "note", text: "Only " + fmt(D.sampling.rows_read) + " rows were read from the source (" + D.sampling.method + ")." }));
    if (D.analysis_sample_rows) root.appendChild(el("div", { class: "note", text: "Pairwise and multivariate analyses used a deterministic sample of " + fmt(D.analysis_sample_rows) + " rows; counts and per-column statistics use all rows." }));
    root.appendChild(el("h2", { text: "Executive summary" }));
    var top = findings.filter(function (f) { return SEV_ORDER[f.severity] >= 2; });
    var warn = findings.filter(function (f) { return f.severity === "warning"; });
    root.appendChild(el("p", { text: S.findings + " findings: " + (S.by_severity.critical || 0) + " critical, " + (S.by_severity.high || 0) + " high, " + (S.by_severity.warning || 0) + " warnings and " + (S.by_severity.info || 0) + " informational, affecting " + S.columns_affected + " column(s)." }));
    root.appendChild(el("h2", { text: "Critical and high findings" }));
    if (!top.length) root.appendChild(el("div", { class: "empty", text: "None." }));
    top.forEach(function (f) { root.appendChild(findingCard(f, false)); });
    root.appendChild(el("h2", { text: "Warnings" }));
    if (!warn.length) root.appendChild(el("div", { class: "empty", text: "None." }));
    warn.forEach(function (f) { root.appendChild(findingCard(f, false)); });
    var cats = Object.keys(S.by_category || {});
    if (cats.length) {
      root.appendChild(el("h2", { text: "Findings by section" }));
      root.appendChild(el("div", { class: "panel" }, barChart(cats.map(function (c) { return { label: CATEGORY_TITLES[c] || c, value: S.by_category[c] }; }))));
    }
    return root;
  }

  function findingsView() {
    var root = el("div");
    var state = { sev: {}, cat: "", q: "" };
    var list = el("div");
    var sevChips = ["critical", "high", "warning", "info"].map(function (s) {
      var b = el("button", { class: "chip", text: s + " (" + (S.by_severity[s] || 0) + ")", onclick: function () {
        state.sev[s] = !state.sev[s]; b.className = "chip" + (state.sev[s] ? " on" : ""); draw();
      } });
      return b;
    });
    var catSel = el("select", { onchange: function () { state.cat = this.value; draw(); } },
      el("option", { value: "", text: "All sections" }),
      Object.keys(CATEGORY_TITLES).map(function (c) { return el("option", { value: c, text: CATEGORY_TITLES[c] }); }));
    var search = el("input", { type: "search", placeholder: "Search title, column, code…", oninput: function () { state.q = this.value.toLowerCase(); draw(); } });
    function draw() {
      var anySev = Object.keys(state.sev).some(function (k) { return state.sev[k]; });
      var shown = findings.filter(function (f) {
        if (anySev && !state.sev[f.severity]) return false;
        if (state.cat && f.category !== state.cat) return false;
        if (state.q) {
          var hay = (f.title + " " + f.code + " " + (f.columns || []).join(" ") + " " + f.detector).toLowerCase();
          if (hay.indexOf(state.q) < 0) return false;
        }
        return true;
      });
      list.textContent = "";
      list.appendChild(el("div", { class: "muted small", text: shown.length + " of " + findings.length + " findings" }));
      shown.forEach(function (f) { list.appendChild(findingCard(f, false)); });
    }
    root.appendChild(el("div", { class: "filters" }, sevChips, catSel, search));
    root.appendChild(list);
    draw();
    return root;
  }

  function schemaView() {
    var fields = (DATA.schema && DATA.schema.fields) || [];
    var counts = {};
    findings.forEach(function (f) { (f.columns || []).forEach(function (c) { counts[c] = (counts[c] || 0) + (f.severity === "info" ? 0 : 1); }); });
    var rows = fields.map(function (c) {
      return { name: c.name, type: c.type, dtype: c.dtype, missing_ratio: c.missing_ratio, unique: c.unique, unique_ratio: c.unique_ratio, memory_bytes: c.memory_bytes, issues: counts[c.name] || 0, hints: (c.hints || []).join(", "), why: (c.reasons || [])[0] || "" };
    });
    return el("div", null,
      el("div", { class: "note", text: "Types are inferred from values, not just dtypes. 'Why' explains each inference. Sort by clicking a header." }),
      el("div", { class: "panel" }, table([
        { key: "name", title: "Column" }, { key: "type", title: "Type" }, { key: "dtype", title: "dtype" },
        { key: "missing_ratio", title: "Missing", render: pct }, { key: "unique", title: "Unique" },
        { key: "unique_ratio", title: "Unique %", render: pct }, { key: "memory_bytes", title: "Memory", render: bytes },
        { key: "issues", title: "Warnings+" }, { key: "hints", title: "Storage hints" }, { key: "why", title: "Why" }
      ], rows)));
  }

  function sectionFindings(cat) {
    var fs = findings.filter(function (f) { return f.category === cat; });
    var box = el("div");
    box.appendChild(el("h2", { text: "Findings" }));
    if (!fs.length) box.appendChild(el("div", { class: "empty", text: "No findings in this section." }));
    fs.forEach(function (f) { box.appendChild(findingCard(f, false)); });
    return box;
  }

  function missingnessView() {
    var m = (SEC.missingness && SEC.missingness.columns) || {};
    var items = Object.keys(m).map(function (c) { return { label: c, value: m[c].ratio }; })
      .filter(function (d) { return d.value > 0; }).sort(function (a, b) { return b.value - a.value; });
    var root = el("div");
    if (SEC.missingness) root.appendChild(el("p", { text: fmt(SEC.missingness.missing_cells) + " of " + fmt(SEC.missingness.total_cells) + " cells (" + pct(SEC.missingness.missing_cells / Math.max(SEC.missingness.total_cells, 1)) + ") are missing." }));
    root.appendChild(el("div", { class: "panel" }, items.length ? barChart(items, { max: 1, fmt: pct }) : el("div", { class: "empty", text: "No missing values." })));
    root.appendChild(sectionFindings("missingness"));
    return root;
  }

  function numericView() {
    var root = el("div", { class: "grid2" });
    Object.keys(PROF).forEach(function (c) {
      var p = PROF[c];
      if (!p.numeric || !p.numeric.count) return;
      var s = p.numeric;
      root.appendChild(el("div", { class: "panel" },
        el("h3", { text: c }),
        histogram(p.histogram),
        boxplot(s),
        el("div", { class: "small muted", text: "n=" + fmt(s.count) + " · mean " + fmt(s.mean) + " · sd " + fmt(s.std) + " · median " + fmt(s.median) + " · IQR " + fmt(s.iqr) + " · skew " + fmt(s.skewness, 2) + " · excess kurtosis " + fmt(s.excess_kurtosis, 2) })
      ));
    });
    var wrap = el("div", null, root.childNodes.length ? root : el("div", { class: "empty", text: "No numeric columns." }));
    wrap.appendChild(sectionFindings("numeric"));
    return wrap;
  }

  function categoricalView() {
    var root = el("div", { class: "grid2" });
    Object.keys(PROF).forEach(function (c) {
      var p = PROF[c].categorical;
      if (!p) return;
      root.appendChild(el("div", { class: "panel" },
        el("h3", { text: c }),
        barChart(p.top.map(function (t) { return { label: t.label, value: t.share }; }), { fmt: pct, max: 1, labelWidth: 140 }),
        el("div", { class: "small muted", text: fmt(p.distinct) + " distinct · entropy " + fmt(p.entropy_bits, 2) + " bits (normalised " + fmt(p.normalized_entropy, 2) + ") · " + fmt(p.rare_levels) + " rare levels" })
      ));
    });
    var wrap = el("div", null, root.childNodes.length ? root : el("div", { class: "empty", text: "No categorical columns." }));
    wrap.appendChild(sectionFindings("categorical"));
    wrap.appendChild(sectionFindings("strings"));
    return wrap;
  }

  function temporalView() {
    var t = SEC.temporal;
    var root = el("div");
    if (!t) root.appendChild(el("div", { class: "note", text: DATA.time_column ? "The time range of '" + DATA.time_column + "' is too short for period analysis." : "No suitable time column was found. Set time_column to enable temporal analysis." }));
    else {
      root.appendChild(el("p", { text: "Primary time column: '" + t.time_column + "', analysed per " + t.period + "." }));
      root.appendChild(el("div", { class: "panel" }, el("h3", { text: "Rows per " + t.period }), lineChart(t.volume, { zero: true })));
    }
    root.appendChild(sectionFindings("temporal"));
    return root;
  }

  function relationshipsView() {
    var c = SEC.correlation || {};
    var root = el("div");
    if (c.numeric_columns && c.numeric_columns.length >= 2) {
      var holder = el("div");
      var which = "pearson";
      var toggle = el("div", { class: "filters" }, ["pearson", "spearman"].map(function (m) {
        var b = el("button", { class: "chip" + (m === which ? " on" : ""), text: m, onclick: function () {
          which = m; Array.prototype.forEach.call(toggle.children, function (x) { x.className = "chip" + (x.textContent === m ? " on" : ""); }); draw();
        } });
        return b;
      }));
      var draw = function () { holder.textContent = ""; holder.appendChild(heatmap(c.numeric_columns, c[which])); };
      draw();
      root.appendChild(el("div", { class: "panel" }, el("h3", { text: "Correlation heatmap" }), toggle, holder,
        el("div", { class: "small muted", text: "Blue = positive, orange = negative. Correlation does not imply causation." + (c.sampled_rows ? " Computed on " + fmt(c.sampled_rows) + " rows." : "") })));
    }
    root.appendChild(sectionFindings("relationships"));
    return root;
  }

  function distributionView() {
    var d = SEC.drift;
    var root = el("div");
    if (d && d.columns) {
      root.appendChild(el("div", { class: "panel" }, table([
        { key: "column", title: "Column" }, { key: "kind", title: "Kind" },
        { key: "ks_statistic", title: "KS D" }, { key: "psi", title: "PSI" }, { key: "jsd", title: "JSD" },
        { key: "wasserstein_iqr", title: "W1 / IQR" }, { key: "missing_reference", title: "Missing (ref)", render: pct },
        { key: "missing_current", title: "Missing (cur)", render: pct }, { key: "severity", title: "Severity" }
      ], d.columns)));
      if (d.domain_classifier_auc !== undefined && d.domain_classifier_auc !== null) root.appendChild(el("p", { text: "Domain classifier AUC (can a model tell the datasets apart?): " + fmt(d.domain_classifier_auc, 3) + " — 0.5 means indistinguishable." }));
    } else root.appendChild(el("div", { class: "note", text: "Distribution comparison runs with `datasi compare` / inspect_train_test()." }));
    root.appendChild(sectionFindings("distribution"));
    return root;
  }

  function targetView() {
    var t = SEC.target;
    var root = el("div");
    if (!t) { root.appendChild(el("div", { class: "note", text: "No target configured. Pass target='column' to enable target analysis." })); return root; }
    root.appendChild(el("p", { text: "Target '" + t.column + "' (" + t.task + "), " + fmt(t.missing) + " missing." }));
    if (t.classes) root.appendChild(el("div", { class: "panel" }, el("h3", { text: "Class distribution" }), barChart(t.classes.map(function (c) { return { label: c.label, value: c.share }; }), { fmt: pct, max: 1 })));
    if (t.feature_scores) root.appendChild(el("div", { class: "panel" }, el("h3", { text: "Single-feature predictive power (held-out " + (t.task === "regression" ? "R²" : "AUC") + ")" }),
      barChart(t.feature_scores.filter(function (s) { return s.score !== null; }).map(function (s) { return { label: s.column, value: s.score, alt: s.score >= 0.95 }; }), { max: 1, fmt: function (v) { return fmt(v, 3); } })));
    root.appendChild(sectionFindings("target"));
    return root;
  }

  function recommendationsView() {
    var recs = DATA.recommendations || [];
    var root = el("div");
    root.appendChild(el("div", { class: "note", text: "Suggestions derived from warning-or-worse findings. Each cites the evidence that motivates it." }));
    if (!recs.length) root.appendChild(el("div", { class: "empty", text: "No recommendations." }));
    root.appendChild(el("ol", { class: "recs" }, recs.map(function (r) {
      return el("li", null, sevBadge(r.severity), " ", el("strong", { text: r.suggestion }), el("div", { class: "small muted", text: "Because: " + r.because + (r.columns.length ? " (" + r.columns.join(", ") + ")" : "") }));
    })));
    return root;
  }

  function detectorsView() {
    var docs = DATA.detector_docs || {};
    var rows = (DATA.detectors || []).map(function (d) { return { name: d.name, status: d.status, seconds: d.seconds, findings: d.findings, error: d.error || "" }; });
    var root = el("div", null, el("div", { class: "panel" }, table([
      { key: "name", title: "Detector" }, { key: "status", title: "Status" }, { key: "seconds", title: "Seconds" }, { key: "findings", title: "Findings" }, { key: "error", title: "Error" }
    ], rows)));
    Object.keys(docs).forEach(function (k) {
      var d = docs[k];
      root.appendChild(el("div", { class: "panel" }, el("h3", { text: d.name }),
        el("div", { text: d.description }),
        d.assumptions ? el("div", { class: "small" }, el("strong", { text: "Assumptions: " }), d.assumptions) : null,
        d.limitations ? el("div", { class: "small" }, el("strong", { text: "Limitations: " }), d.limitations) : null));
    });
    return root;
  }

  var VIEWS = [
    ["Overview", overview], ["Findings", findingsView], ["Schema", schemaView], ["Missingness", missingnessView],
    ["Numeric", numericView], ["Categorical", categoricalView], ["Temporal", temporalView],
    ["Relationships", relationshipsView], ["Distribution", distributionView], ["Target", targetView],
    ["Recommendations", recommendationsView], ["Detectors", detectorsView]
  ];

  var app = document.getElementById("app");
  app.textContent = "";
  var main = el("main");
  var buttons = VIEWS.map(function (v, i) {
    return el("button", { text: v[0], onclick: function () { show(i); } });
  });
  function show(i) {
    buttons.forEach(function (b, j) { b.className = i === j ? "active" : ""; });
    main.textContent = "";
    main.appendChild(VIEWS[i][1]());
    if (history.replaceState) history.replaceState(null, "", "#" + VIEWS[i][0].toLowerCase());
  }
  var title = (DATA.kind === "comparison" ? "DataSI comparison: " : "DataSI investigation: ") + D.name;
  app.appendChild(el("header", null,
    el("h1", { text: title }),
    el("div", { class: "meta", text: fmt(D.rows) + " rows × " + D.columns + " columns · " + (D.format || "dataframe") + " · DataSI " + DATA.datasi_version + " · " + DATA.created_at + " · generated locally" }),
    el("nav", null, buttons)));
  app.appendChild(main);
  var hash = (location.hash || "").slice(1);
  var start = VIEWS.findIndex(function (v) { return v[0].toLowerCase() === hash; });
  show(start >= 0 ? start : 0);
})();
