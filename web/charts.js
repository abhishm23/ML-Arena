/* ==========================================================================
   ML Arena — charts.js
   Hand-written inline SVG charts. No libraries, no network, fully offline.
   Every function returns an SVG *string* scaled by viewBox (width:100%).
   Inserted with innerHTML into an HTML document, so no XML namespace needed.
   ========================================================================== */
(function (global) {
  'use strict';

  var PALETTE = ['#5B8DEF', '#7C5CFF', '#2FBF71', '#F2A33C', '#EF5350', '#F5C451', '#4DD0E1', '#B388FF'];
  var GRID = 'rgba(147,161,184,0.15)';
  var AXIS = '#5F6E85';
  var TEXT = '#93A1B8';
  var STRONG = '#E8EDF5';

  /* ------------------------------------------------------------ helpers -- */

  function esc(v) {
    if (v === null || v === undefined) return '';
    return String(v)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function arr(v) { return Array.isArray(v) ? v : []; }

  function toNum(v) {
    if (typeof v === 'number') return isFinite(v) ? v : null;
    if (typeof v === 'string' && v.trim() !== '') {
      var n = Number(v);
      return isFinite(n) ? n : null;
    }
    return null;
  }

  function clamp(v, lo, hi) {
    if (!isFinite(v)) return lo;
    return v < lo ? lo : (v > hi ? hi : v);
  }

  function trim1(n) {
    var s = n.toFixed(1);
    return s.slice(-2) === '.0' ? s.slice(0, -2) : s;
  }

  function fmtValue(v) {
    var n = toNum(v);
    if (n === null) return '\u2014';
    var a = Math.abs(n);
    if (a === 0) return '0';
    if (a >= 1e12) return trim1(n / 1e12) + 'T';
    if (a >= 1e9) return trim1(n / 1e9) + 'B';
    if (a >= 1e6) return trim1(n / 1e6) + 'M';
    if (a >= 1e4) return trim1(n / 1e3) + 'k';
    if (a >= 100) return n.toFixed(0);
    if (a >= 1) return trim1(n);
    if (a >= 0.01) return n.toFixed(2);
    return n.toPrecision(2);
  }

  function clip(s, n) {
    var t = String(s === null || s === undefined ? '' : s);
    return t.length > n ? t.slice(0, n - 1) + '\u2026' : t;
  }

  function longest(list) {
    var m = 0;
    for (var i = 0; i < list.length; i++) {
      var l = String(list[i] === null || list[i] === undefined ? '' : list[i]).length;
      if (l > m) m = l;
    }
    return m;
  }

  function open(w, h, label, cls) {
    return '<svg class="chart-svg' + (cls ? ' ' + cls : '') + '" viewBox="0 0 ' + w + ' ' + h +
      '" width="100%" preserveAspectRatio="xMidYMid meet" role="img"' +
      (label ? ' aria-label="' + esc(label) + '"' : ' aria-hidden="true"') +
      ' focusable="false">';
  }

  function close() { return '</svg>'; }

  function emptyBox(w, h, msg) {
    return open(w, h, msg || 'No data', 'chart-empty') +
      '<text x="' + (w / 2) + '" y="' + (h / 2) + '" text-anchor="middle" dominant-baseline="middle" ' +
      'fill="' + AXIS + '" font-size="13" font-family="inherit">' +
      esc(msg || 'Nothing to plot yet') + '</text>' + close();
  }

  function niceScale(min, max, want) {
    want = want || 4;
    if (!isFinite(min) || !isFinite(max)) { min = 0; max = 1; }
    if (min === max) { max = min + 1; }
    var span = max - min;
    var raw = span / want;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = mag * (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1);
    var lo = Math.floor(min / step) * step;
    var hi = Math.ceil(max / step) * step;
    var ticks = [];
    for (var v = lo; v <= hi + step * 0.5; v += step) {
      ticks.push(Math.round(v * 1e6) / 1e6);
      if (ticks.length > 40) break;
    }
    return { min: lo, max: hi, ticks: ticks, step: step };
  }

  function poly(points) {
    var out = [];
    for (var i = 0; i < points.length; i++) {
      out.push(points[i][0].toFixed(1) + ',' + points[i][1].toFixed(1));
    }
    return out.join(' ');
  }

  /* ------------------------------------------------------ horizontal bar -- */
  /* {labels, values, title, color, max, width, rowHeight, valueFmt} */
  function hbar(o) {
    o = o || {};
    var labels = arr(o.labels);
    var values = arr(o.values).map(toNum);
    var rowH = o.rowHeight || 28;
    var w = o.width || 660;
    var h = o.height || Math.max(104, labels.length * rowH + 20);
    if (!labels.length) return emptyBox(w, h, o.empty);

    var padL = Math.round(clamp(longest(labels) * 7.3 + 16, 86, 210));
    var padR = 70;
    var padT = 4;
    var plotW = Math.max(24, w - padL - padR);

    var mx = toNum(o.max);
    if (mx === null) {
      var vals = values.filter(function (v) { return v !== null; });
      mx = vals.length ? Math.max.apply(null, vals) : 1;
    }
    if (!isFinite(mx) || mx <= 0) mx = 1;

    var s = open(w, h, o.title || 'Bar chart', o.className);
    for (var i = 0; i < labels.length; i++) {
      var y = padT + i * rowH;
      var bh = Math.max(8, rowH - 12);
      var by = y + (rowH - bh) / 2;
      var v = values[i];
      var frac = v === null ? 0 : clamp(Math.abs(v) / mx, 0, 1);
      var bw = v === null ? 0 : Math.max(2, frac * plotW);
      var color = o.color || (o.colors && o.colors[i]) || PALETTE[i % PALETTE.length];
      s += '<text x="' + (padL - 10) + '" y="' + (by + bh / 2 + 4).toFixed(1) + '" text-anchor="end" ' +
        'fill="' + STRONG + '" font-size="12.5" font-family="inherit">' + esc(clip(labels[i], 28)) + '</text>';
      s += '<rect x="' + padL + '" y="' + by.toFixed(1) + '" width="' + plotW + '" height="' + bh +
        '" rx="4" fill="rgba(147,161,184,0.10)"/>';
      s += '<rect x="' + padL + '" y="' + by.toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + bh +
        '" rx="4" fill="' + color + '"><title>' + esc(labels[i] + ': ' + fmtValue(v)) + '</title></rect>';
      s += '<text x="' + (padL + bw + 8).toFixed(1) + '" y="' + (by + bh / 2 + 4).toFixed(1) + '" ' +
        'fill="' + TEXT + '" font-size="12" font-family="inherit">' +
        esc(o.valueFmt ? o.valueFmt(v) : fmtValue(v)) + '</text>';
    }
    return s + close();
  }

  /* -------------------------------------------------------- vertical bar -- */
  /* {labels, values, title, color, width, height, maxBarWidth, colors} */
  function bar(o) {
    o = o || {};
    var labels = arr(o.labels);
    var values = arr(o.values).map(toNum);
    var w = o.width || 660;
    var h = o.height || 260;
    if (!labels.length) return emptyBox(w, h, o.empty);

    var padL = 56, padR = 16, padT = 16, padB = 58;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    var nums = values.filter(function (v) { return v !== null; });
    var lo = Math.min(0, nums.length ? Math.min.apply(null, nums) : 0);
    var hi = Math.max(0, nums.length ? Math.max.apply(null, nums) : 1);
    var sc = niceScale(lo, hi, 4);
    var yOf = function (v) { return padT + plotH - ((v - sc.min) / (sc.max - sc.min)) * plotH; };
    var zeroY = yOf(0);

    var s = open(w, h, o.title || 'Bar chart', o.className);
    for (var t = 0; t < sc.ticks.length; t++) {
      var ty = yOf(sc.ticks[t]);
      s += '<line x1="' + padL + '" y1="' + ty.toFixed(1) + '" x2="' + (padL + plotW) + '" y2="' + ty.toFixed(1) +
        '" stroke="' + GRID + '" stroke-width="1"/>';
      s += '<text x="' + (padL - 9) + '" y="' + (ty + 4).toFixed(1) + '" text-anchor="end" fill="' + AXIS +
        '" font-size="11" font-family="inherit">' + esc(fmtValue(sc.ticks[t])) + '</text>';
    }

    var slot = plotW / labels.length;
    var bw = Math.max(4, Math.min(o.maxBarWidth || 56, slot * 0.62));
    for (var i = 0; i < labels.length; i++) {
      var v = values[i];
      var cx = padL + slot * i + slot / 2;
      var x = cx - bw / 2;
      var yv = v === null ? zeroY : yOf(v);
      var top = Math.min(yv, zeroY);
      var bh = Math.max(2, Math.abs(zeroY - yv));
      var color = o.color || (o.colors && o.colors[i]) || '#5B8DEF';
      s += '<rect x="' + x.toFixed(1) + '" y="' + top.toFixed(1) + '" width="' + bw.toFixed(1) +
        '" height="' + bh.toFixed(1) + '" rx="4" fill="' + color + '"><title>' +
        esc(labels[i] + ': ' + fmtValue(v)) + '</title></rect>';
      var ly = padT + plotH + 18;
      if (slot < 66) {
        s += '<text transform="translate(' + cx.toFixed(1) + ',' + ly + ') rotate(-38)" text-anchor="end" ' +
          'fill="' + TEXT + '" font-size="11" font-family="inherit">' + esc(clip(labels[i], 16)) + '</text>';
      } else {
        s += '<text x="' + cx.toFixed(1) + '" y="' + ly + '" text-anchor="middle" fill="' + TEXT +
          '" font-size="11.5" font-family="inherit">' + esc(clip(labels[i], 14)) + '</text>';
      }
    }
    s += '<line x1="' + padL + '" y1="' + zeroY.toFixed(1) + '" x2="' + (padL + plotW) + '" y2="' +
      zeroY.toFixed(1) + '" stroke="' + AXIS + '" stroke-opacity="0.55" stroke-width="1"/>';
    return s + close();
  }

  /* ----------------------------------------------------------- histogram -- */
  /* {bins:[edges|centres], counts:[], title, width, height} */
  function histogram(o) {
    o = o || {};
    var counts = arr(o.counts).map(function (v) { var n = toNum(v); return n === null ? 0 : n; });
    var bins = arr(o.bins).map(toNum);
    var w = o.width || 660;
    var h = o.height || 250;
    if (!counts.length) return emptyBox(w, h, o.empty);

    var edges = [];
    var i;
    if (bins.length === counts.length + 1) {
      edges = bins.slice();
    } else if (bins.length === counts.length && bins.length > 1) {
      var stepv = (bins[bins.length - 1] - bins[0]) / (bins.length - 1);
      if (!isFinite(stepv) || stepv === 0) stepv = 1;
      edges.push(bins[0] - stepv / 2);
      for (i = 0; i < bins.length; i++) edges.push(bins[i] + stepv / 2);
    } else {
      for (i = 0; i <= counts.length; i++) edges.push(i);
    }
    for (i = 0; i < edges.length; i++) if (edges[i] === null) edges[i] = i;

    var padL = 50, padR = 16, padT = 16, padB = 44;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;
    var maxC = Math.max.apply(null, counts.concat([1]));
    var sc = niceScale(0, maxC, 4);
    var yOf = function (v) { return padT + plotH - (v / sc.max) * plotH; };
    var x0 = edges[0];
    var x1 = edges[edges.length - 1];
    if (x1 === x0) x1 = x0 + 1;
    var xOf = function (v) { return padL + ((v - x0) / (x1 - x0)) * plotW; };

    var s = open(w, h, o.title || 'Histogram', o.className);
    for (var t = 0; t < sc.ticks.length; t++) {
      var ty = yOf(sc.ticks[t]);
      s += '<line x1="' + padL + '" y1="' + ty.toFixed(1) + '" x2="' + (padL + plotW) + '" y2="' + ty.toFixed(1) +
        '" stroke="' + GRID + '"/>';
      s += '<text x="' + (padL - 9) + '" y="' + (ty + 4).toFixed(1) + '" text-anchor="end" fill="' + AXIS +
        '" font-size="11" font-family="inherit">' + esc(fmtValue(sc.ticks[t])) + '</text>';
    }

    var bw = plotW / counts.length;
    for (var b = 0; b < counts.length; b++) {
      var bx = xOf(edges[b]);
      var bwid = Math.max(1, xOf(edges[b + 1]) - bx - 1);
      var bh = Math.max(1, plotH - (yOf(counts[b]) - padT));
      var by = padT + plotH - bh;
      var frac = counts[b] / maxC;
      var op = (0.42 + 0.58 * frac).toFixed(2);
      s += '<rect x="' + bx.toFixed(1) + '" y="' + by.toFixed(1) + '" width="' + bwid.toFixed(1) +
        '" height="' + bh.toFixed(1) + '" rx="2.5" fill="#5B8DEF" fill-opacity="' + op + '"><title>' +
        esc(fmtValue(edges[b]) + ' \u2013 ' + fmtValue(edges[b + 1]) + ': ' + fmtValue(counts[b])) +
        '</title></rect>';
    }
    s += '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (padL + plotW) + '" y2="' + (padT + plotH) +
      '" stroke="' + AXIS + '" stroke-opacity="0.5"/>';

    var tickCount = Math.min(6, counts.length);
    for (var k = 0; k <= tickCount; k++) {
      var xv = x0 + ((x1 - x0) * k) / tickCount;
      var tx = xOf(xv);
      var anchor = k === 0 ? 'start' : (k === tickCount ? 'end' : 'middle');
      s += '<text x="' + tx.toFixed(1) + '" y="' + (padT + plotH + 18) + '" text-anchor="' + anchor +
        '" fill="' + TEXT + '" font-size="11" font-family="inherit">' + esc(fmtValue(xv)) + '</text>';
    }
    s += '<text x="' + (padL + plotW / 2) + '" y="' + (h - 6) + '" text-anchor="middle" fill="' + AXIS +
      '" font-size="11" font-family="inherit">' + esc(o.x_label || 'Score') + '</text>';
    return s + close();
  }

  /* ---------------------------------------------------------------- line -- */
  /* {x:[], series:[{name, values, color}], area, title, width, height} */
  function line(o) {
    o = o || {};
    var xs = arr(o.x);
    var series = arr(o.series).filter(function (sr) { return sr && arr(sr.values).length; });
    var w = o.width || 660;
    var h = o.height || 280;
    if (!series.length || !xs.length) return emptyBox(w, h, o.empty);

    var padL = 58, padR = 18, padT = 18, padB = 52;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;
    var n = xs.length;

    var all = [];
    series.forEach(function (sr) {
      arr(sr.values).forEach(function (v) { var num = toNum(v); if (num !== null) all.push(num); });
    });
    if (!all.length) return emptyBox(w, h, o.empty);

    var sc = niceScale(Math.min.apply(null, all), Math.max.apply(null, all), 4);
    var xOf = function (i) { return padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW); };
    var yOf = function (v) { return padT + plotH - ((v - sc.min) / (sc.max - sc.min)) * plotH; };

    var s = open(w, h, o.title || 'Line chart', o.className);
    for (var t = 0; t < sc.ticks.length; t++) {
      var ty = yOf(sc.ticks[t]);
      s += '<line x1="' + padL + '" y1="' + ty.toFixed(1) + '" x2="' + (padL + plotW) + '" y2="' + ty.toFixed(1) +
        '" stroke="' + GRID + '"/>';
      s += '<text x="' + (padL - 9) + '" y="' + (ty + 4).toFixed(1) + '" text-anchor="end" fill="' + AXIS +
        '" font-size="11" font-family="inherit">' + esc(fmtValue(sc.ticks[t])) + '</text>';
    }

    series.forEach(function (sr, si) {
      var color = sr.color || PALETTE[si % PALETTE.length];
      var vals = arr(sr.values);
      var segments = [];
      var cur = [];
      for (var i = 0; i < n; i++) {
        var v = toNum(vals[i]);
        if (v === null) {
          if (cur.length > 1) segments.push(cur);
          cur = [];
        } else {
          cur.push([xOf(i), yOf(v)]);
        }
      }
      if (cur.length > 1) segments.push(cur);

      if (o.area && segments.length) {
        var first = segments[0];
        var lastSeg = segments[segments.length - 1];
        var areaPts = first.concat(lastSeg.slice().reverse());
        s += '<polygon points="' + poly(areaPts) + '" fill="' + color + '" fill-opacity="0.12"/>';
      }
      segments.forEach(function (seg) {
        var d = 'M' + seg.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L');
        s += '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
      });
      if (n <= 60) {
        for (var j = 0; j < n; j++) {
          var vv = toNum(vals[j]);
          if (vv === null) continue;
          s += '<circle cx="' + xOf(j).toFixed(1) + '" cy="' + yOf(vv).toFixed(1) + '" r="2.4" fill="' + color +
            '"><title>' + esc(String(xs[j]) + ' \u00b7 ' + (sr.name || 'series') + ': ' + fmtValue(vv)) + '</title></circle>';
        }
      }
    });

    var tickCount = Math.min(6, Math.max(1, n - 1));
    for (var k = 0; k <= tickCount; k++) {
      var idx = Math.round((n - 1) * (k / tickCount));
      var anchor = k === 0 ? 'start' : (k === tickCount ? 'end' : 'middle');
      s += '<text x="' + xOf(idx).toFixed(1) + '" y="' + (padT + plotH + 18) + '" text-anchor="' + anchor +
        '" fill="' + TEXT + '" font-size="11" font-family="inherit">' + esc(clip(xs[idx], 14)) + '</text>';
    }

    var legend = series.length > 1 ? series : [];
    var lx = padL;
    legend.forEach(function (sr, si) {
      var color = sr.color || PALETTE[si % PALETTE.length];
      var y = h - 12;
      s += '<rect x="' + lx + '" y="' + (y - 8) + '" width="9" height="9" rx="2.5" fill="' + color + '"/>';
      s += '<text x="' + (lx + 14) + '" y="' + y + '" fill="' + TEXT + '" font-size="11.5" font-family="inherit">' +
        esc(clip(sr.name || ('series ' + (si + 1)), 22)) + '</text>';
      lx += 14 + clip(sr.name || 'series', 22).length * 6.6 + 22;
    });
    return s + close();
  }

  /* --------------------------------------------------------------- donut -- */
  /* {labels, values, colors, centerLabel, centerValue, title, width, height} */
  function donut(o) {
    o = o || {};
    var labels = arr(o.labels);
    var values = arr(o.values).map(function (v) { var n = toNum(v); return n === null ? 0 : Math.abs(n); });
    var w = o.width || 660;
    var h = o.height || 240;
    if (!labels.length) return emptyBox(w, h, o.empty);

    var total = values.reduce(function (a, b) { return a + b; }, 0);
    if (!(total > 0)) return emptyBox(w, h, o.empty);

    var size = Math.min(h - 30, 176);
    var r = size / 2;
    var cx = 24 + r;
    var cy = h / 2;
    var sw = Math.max(14, r * 0.36);
    var rr = r - sw / 2;
    var circ = 2 * Math.PI * rr;

    var s = open(w, h, o.title || 'Donut chart', o.className);
    var offset = 0;
    for (var i = 0; i < labels.length; i++) {
      var frac = values[i] / total;
      var len = frac * circ;
      var color = (o.colors && o.colors[i]) || PALETTE[i % PALETTE.length];
      if (len > 0.5) {
        s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rr.toFixed(1) + '" fill="none" stroke="' + color +
          '" stroke-width="' + sw.toFixed(1) + '" stroke-dasharray="' + len.toFixed(2) + ' ' + (circ - len).toFixed(2) +
          '" stroke-dashoffset="' + (-offset).toFixed(2) + '" transform="rotate(-90 ' + cx + ' ' + cy + ')"><title>' +
          esc(labels[i] + ': ' + fmtValue(values[i]) + ' (' + (frac * 100).toFixed(1) + '%)') + '</title></circle>';
      }
      offset += len;
    }
    s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rr.toFixed(1) + '" fill="none" stroke="#243044" stroke-width="1"/>';
    s += '<text x="' + cx + '" y="' + (cy - 1) + '" text-anchor="middle" fill="' + STRONG +
      '" font-size="21" font-weight="600" font-family="inherit">' +
      esc(o.centerValue !== undefined && o.centerValue !== null ? o.centerValue : fmtValue(total)) + '</text>';
    s += '<text x="' + cx + '" y="' + (cy + 18) + '" text-anchor="middle" fill="' + TEXT +
      '" font-size="11.5" font-family="inherit">' + esc(o.centerLabel || 'Total') + '</text>';

    var lx = cx + r + 40;
    var startY = cy - ((labels.length * 24) / 2) + 10;
    for (var j = 0; j < labels.length; j++) {
      var y = startY + j * 24;
      var c2 = (o.colors && o.colors[j]) || PALETTE[j % PALETTE.length];
      s += '<rect x="' + lx + '" y="' + (y - 9) + '" width="10" height="10" rx="3" fill="' + c2 + '"/>';
      s += '<text x="' + (lx + 18) + '" y="' + y + '" fill="' + STRONG + '" font-size="12" font-family="inherit">' +
        esc(clip(labels[j], 24)) + '</text>';
      s += '<text x="' + (w - 16) + '" y="' + y + '" text-anchor="end" fill="' + TEXT +
        '" font-size="12" font-family="inherit">' + esc(fmtValue(values[j])) + '</text>';
    }
    return s + close();
  }

  /* ------------------------------------------------------------- scatter -- */
  /* {x:[], y:[], color:[], x_label, y_label, title, width, height} */
  function scatter(o) {
    o = o || {};
    var xs = arr(o.x).map(toNum);
    var ys = arr(o.y).map(toNum);
    var cs = arr(o.color);
    var w = o.width || 660;
    var h = o.height || 300;

    var pts = [];
    for (var i = 0; i < xs.length; i++) {
      if (xs[i] === null || ys[i] === null) continue;
      pts.push([xs[i], ys[i], cs.length ? cs[i] : 0]);
    }
    if (!pts.length) return emptyBox(w, h, o.empty);

    var cap = o.maxPoints || 1400;
    if (pts.length > cap) {
      var stride = pts.length / cap;
      var thinned = [];
      for (var k = 0; k < cap; k++) thinned.push(pts[Math.floor(k * stride)]);
      pts = thinned;
    }

    var padL = 60, padR = 18, padT = 18, padB = 48;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    var xsv = pts.map(function (p) { return p[0]; });
    var ysv = pts.map(function (p) { return p[1]; });
    var sx = niceScale(Math.min.apply(null, xsv), Math.max.apply(null, xsv), 4);
    var sy = niceScale(Math.min.apply(null, ysv), Math.max.apply(null, ysv), 4);
    var X = function (v) { return padL + ((v - sx.min) / (sx.max - sx.min)) * plotW; };
    var Y = function (v) { return padT + plotH - ((v - sy.min) / (sy.max - sy.min)) * plotH; };

    var s = open(w, h, o.title || 'Scatter plot', o.className);
    for (var t = 0; t < sy.ticks.length; t++) {
      var ty = Y(sy.ticks[t]);
      s += '<line x1="' + padL + '" y1="' + ty.toFixed(1) + '" x2="' + (padL + plotW) + '" y2="' + ty.toFixed(1) +
        '" stroke="' + GRID + '"/>';
      s += '<text x="' + (padL - 9) + '" y="' + (ty + 4).toFixed(1) + '" text-anchor="end" fill="' + AXIS +
        '" font-size="11" font-family="inherit">' + esc(fmtValue(sy.ticks[t])) + '</text>';
    }
    for (var t2 = 0; t2 < sx.ticks.length; t2++) {
      var tx = X(sx.ticks[t2]);
      s += '<line x1="' + tx.toFixed(1) + '" y1="' + padT + '" x2="' + tx.toFixed(1) + '" y2="' + (padT + plotH) +
        '" stroke="' + GRID + '"/>';
      s += '<text x="' + tx.toFixed(1) + '" y="' + (padT + plotH + 18) + '" text-anchor="middle" fill="' + AXIS +
        '" font-size="11" font-family="inherit">' + esc(fmtValue(sx.ticks[t2])) + '</text>';
    }

    var groups = {};
    for (var p = 0; p < pts.length; p++) {
      var key = String(pts[p][2]);
      if (!groups[key]) groups[key] = [];
      groups[key].push(pts[p]);
    }
    var keys = Object.keys(groups).sort();
    var multi = keys.length > 1;
    keys.forEach(function (key, ki) {
      var color = multi ? (Number(key) === 1 ? '#EF5350' : PALETTE[ki % PALETTE.length]) : '#5B8DEF';
      var op = multi ? 0.72 : 0.55;
      var dots = '';
      groups[key].forEach(function (pt) {
        dots += '<circle cx="' + X(pt[0]).toFixed(1) + '" cy="' + Y(pt[1]).toFixed(1) + '" r="' +
          (multi ? 3 : 2.4) + '" fill="' + color + '" fill-opacity="' + op + '"/>';
      });
      s += '<g>' + dots + '</g>';
    });

    s += '<text x="' + (padL + plotW / 2) + '" y="' + (h - 8) + '" text-anchor="middle" fill="' + AXIS +
      '" font-size="11" font-family="inherit">' + esc(o.x_label || 'x') + '</text>';
    s += '<text x="14" y="' + (padT + plotH / 2) + '" text-anchor="middle" fill="' + AXIS +
      '" font-size="11" font-family="inherit" transform="rotate(-90 14 ' + (padT + plotH / 2) + ')">' +
      esc(o.y_label || 'y') + '</text>';
    return s + close();
  }

  /* --------------------------------------------------------------- gauge -- */
  /* {value 0..1, label, sublabel, size, width, height, color} */
  function gauge(o) {
    o = o || {};
    var size = o.size || 148;
    var v = clamp(toNum(o.value) === null ? 0 : toNum(o.value), 0, 1);
    var w = o.width || size + 20;
    var h = o.height || size + 20;
    var cx = w / 2;
    var cy = h / 2;
    var r = size / 2 - 8;
    var sw = Math.max(9, r * 0.2);
    var rr = r - sw / 2;
    var circ = 2 * Math.PI * rr;
    var color = o.color || (v >= 0.75 ? '#2FBF71' : v >= 0.5 ? '#5B8DEF' : '#F2A33C');

    var s = open(w, h, o.title || o.label || 'Progress', o.className);
    s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rr.toFixed(1) + '" fill="none" stroke="#243044" stroke-width="' + sw.toFixed(1) + '"/>';
    s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + rr.toFixed(1) + '" fill="none" stroke="' + color +
      '" stroke-width="' + sw.toFixed(1) + '" stroke-linecap="round" stroke-dasharray="' + (v * circ).toFixed(2) +
      ' ' + circ.toFixed(2) + '" transform="rotate(-90 ' + cx + ' ' + cy + ')"/>';
    s += '<text x="' + cx + '" y="' + (cy + 2) + '" text-anchor="middle" fill="' + STRONG +
      '" font-size="' + Math.round(size * 0.22) + '" font-weight="680" font-family="inherit">' +
      esc(o.display || Math.round(v * 100) + '%') + '</text>';
    if (o.label) {
      s += '<text x="' + cx + '" y="' + (cy + Math.round(size * 0.22) + 6) + '" text-anchor="middle" fill="' + TEXT +
        '" font-size="11.5" font-family="inherit">' + esc(clip(o.label, 22)) + '</text>';
    }
    return s + close();
  }

  /* --------------------------------------------------------------- radar -- */
  /* {axes:[], series:[{name, values}], max, title, width, height} */
  function radar(o) {
    o = o || {};
    var axes = arr(o.axes);
    var series = arr(o.series).filter(function (sr) { return sr && arr(sr.values).length; });
    var w = o.width || 660;
    var h = o.height || 340;
    if (axes.length < 3 || !series.length) return emptyBox(w, h, o.empty || 'Not enough measures to compare');

    var cx = w / 2;
    var cy = h / 2 + 4;
    var R = Math.max(60, Math.min(w / 2 - 118, h / 2 - 40));
    var count = axes.length;
    var ang = function (i) { return -Math.PI / 2 + (i * 2 * Math.PI) / count; };

    var maxes = [];
    for (var a = 0; a < count; a++) {
      var m = 0;
      for (var si = 0; si < series.length; si++) {
        var n = toNum(arr(series[si].values)[a]);
        if (n !== null) m = Math.max(m, Math.abs(n));
      }
      maxes.push(m > 0 ? m : 1);
    }

    var s = open(w, h, o.title || 'Radar chart', o.className);
    var ringF = [0.25, 0.5, 0.75, 1];
    ringF.forEach(function (f) {
      var pts = [];
      for (var i = 0; i < count; i++) pts.push([cx + Math.cos(ang(i)) * R * f, cy + Math.sin(ang(i)) * R * f]);
      s += '<polygon points="' + poly(pts) + '" fill="none" stroke="' + GRID + '" stroke-width="1"/>';
    });
    for (var i2 = 0; i2 < count; i2++) {
      var ex = cx + Math.cos(ang(i2)) * R;
      var ey = cy + Math.sin(ang(i2)) * R;
      s += '<line x1="' + cx + '" y1="' + cy + '" x2="' + ex.toFixed(1) + '" y2="' + ey.toFixed(1) +
        '" stroke="' + GRID + '"/>';
      var lx = cx + Math.cos(ang(i2)) * (R + 20);
      var ly = cy + Math.sin(ang(i2)) * (R + 20);
      var anchor = Math.abs(lx - cx) < 8 ? 'middle' : (lx > cx ? 'start' : 'end');
      s += '<text x="' + lx.toFixed(1) + '" y="' + (ly + 4).toFixed(1) + '" text-anchor="' + anchor +
        '" fill="' + TEXT + '" font-size="11.5" font-family="inherit">' + esc(clip(axes[i2], 20)) + '</text>';
    }

    series.forEach(function (sr, sIdx) {
      var color = sr.color || PALETTE[sIdx % PALETTE.length];
      var pts = [];
      for (var i3 = 0; i3 < count; i3++) {
        var val = toNum(arr(sr.values)[i3]);
        var frac = val === null ? 0 : clamp(Math.abs(val) / maxes[i3], 0, 1);
        pts.push([cx + Math.cos(ang(i3)) * R * frac, cy + Math.sin(ang(i3)) * R * frac]);
      }
      s += '<polygon points="' + poly(pts) + '" fill="' + color + '" fill-opacity="0.16" stroke="' + color +
        '" stroke-width="2" stroke-linejoin="round"/>';
      pts.forEach(function (pt, i4) {
        s += '<circle cx="' + pt[0].toFixed(1) + '" cy="' + pt[1].toFixed(1) + '" r="2.6" fill="' + color +
          '"><title>' + esc((sr.name || 'series') + ' \u00b7 ' + axes[i4] + ': ' + fmtValue(arr(sr.values)[i4])) +
          '</title></circle>';
      });
    });

    if (series.length > 1) {
      var lx2 = 16;
      series.forEach(function (sr, sIdx2) {
        var color = sr.color || PALETTE[sIdx2 % PALETTE.length];
        s += '<rect x="' + lx2 + '" y="' + (h - 16) + '" width="9" height="9" rx="2.5" fill="' + color + '"/>';
        s += '<text x="' + (lx2 + 14) + '" y="' + (h - 8) + '" fill="' + TEXT + '" font-size="11.5" font-family="inherit">' +
          esc(clip(sr.name || ('model ' + (sIdx2 + 1)), 20)) + '</text>';
        lx2 += 14 + clip(sr.name || 'model', 20).length * 6.6 + 20;
      });
    }
    return s + close();
  }

  /* ----------------------------------------------------------- sparkline -- */
  /* {values, width, height, color} */
  function sparkline(o) {
    o = o || {};
    var vals = arr(o.values).map(toNum).filter(function (v) { return v !== null; });
    var w = o.width || 120;
    var h = o.height || 30;
    if (vals.length < 2) return '';
    var lo = Math.min.apply(null, vals);
    var hi = Math.max.apply(null, vals);
    var span = hi - lo || 1;
    var pts = vals.map(function (v, i) {
      return [ (i / (vals.length - 1)) * (w - 4) + 2, h - 3 - ((v - lo) / span) * (h - 6) ];
    });
    var color = o.color || '#5B8DEF';
    return open(w, h, null, o.className) +
      '<path d="M' + pts.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L') +
      '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' +
      close();
  }

  global.Charts = {
    palette: PALETTE,
    bar: bar,
    hbar: hbar,
    line: line,
    histogram: histogram,
    donut: donut,
    scatter: scatter,
    gauge: gauge,
    radar: radar,
    sparkline: sparkline,
    fmtValue: fmtValue,
    escape: esc
  };
})(typeof window !== 'undefined' ? window : this);
