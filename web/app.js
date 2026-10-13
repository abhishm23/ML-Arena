/* ==========================================================================
   ML Arena — app.js
   Vanilla JS six-step wizard. Talks only to the local FastAPI server.
   No frameworks, no build step, no network calls beyond this machine.
   ========================================================================== */
(function () {
  'use strict';

  /* ======================================================== tiny helpers == */

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

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

  function get(obj, path, fallback) {
    var cur = obj;
    var parts = String(path || '').split('.');
    for (var i = 0; i < parts.length; i++) {
      if (cur === null || cur === undefined) return fallback;
      cur = cur[parts[i]];
    }
    return cur === null || cur === undefined ? fallback : cur;
  }

  function clamp(n, lo, hi) { return n < lo ? lo : (n > hi ? hi : n); }

  function fmtInt(v) {
    var n = toNum(v);
    if (n === null) return '\u2014';
    return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function fmtNum(v, digits) {
    var n = toNum(v);
    if (n === null) return '\u2014';
    var d = digits === undefined ? 2 : digits;
    return n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: d });
  }

  function fmtBytes(b) {
    var n = toNum(b);
    if (n === null) return '';
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var i = 0;
    while (n >= 1024 && i < units.length - 1) { n = n / 1024; i++; }
    return (i === 0 ? n.toFixed(0) : n.toFixed(n < 10 ? 1 : 0)) + ' ' + units[i];
  }

  function fmtMs(ms) {
    var n = toNum(ms);
    if (n === null) return '';
    if (n < 1000) return Math.round(n) + ' ms';
    if (n < 60000) return (n / 1000).toFixed(n < 10000 ? 1 : 0) + ' s';
    return Math.floor(n / 60000) + 'm ' + Math.round((n % 60000) / 1000) + 's';
  }

  function pad2(n) { return n < 10 ? '0' + n : String(n); }

  function fmtClock(ts) {
    var d = ts ? new Date(Number(ts)) : new Date();
    if (isNaN(d.getTime())) d = new Date();
    return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds());
  }

  function basename(p) {
    var s = String(p || '');
    var parts = s.split(/[\\/]/);
    return parts[parts.length - 1] || s;
  }

  function titleCase(s) {
    return String(s || '').replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function humanMetric(k) {
    return String(k || '').replace(/_/g, ' ').replace(/\b(roc|auc|r2|rmse|mae|mape|wape|f1|pr)\b/gi, function (m) {
      return m.toUpperCase();
    });
  }

  /* ================================================================ icons == */

  var ICONS = {
    crown: '<path d="M3 7.5l4.5 3.5L12 4.5l4.5 6.5L21 7.5 19 19H5z"/><path d="M5.5 21h13"/>',
    check: '<path d="M4.5 12.5l5 5L19.5 6.5"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    info: '<circle cx="12" cy="12" r="8.8"/><path d="M12 11v5.5"/><path d="M12 7.8h.01"/>',
    warn: '<path d="M12 4.2l9 15.6H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    error: '<circle cx="12" cy="12" r="8.8"/><path d="M12 7.8v5"/><path d="M12 16.2h.01"/>',
    spark: '<path d="M12 3l1.7 4.6L18.3 9l-4.6 1.4L12 15l-1.7-4.6L5.7 9l4.6-1.4z"/><path d="M18.6 15.2l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>',
    upload: '<path d="M12 16V4"/><path d="M7.5 8.5L12 4l4.5 4.5"/><path d="M4 16.5v1.8A2.7 2.7 0 0 0 6.7 21h10.6A2.7 2.7 0 0 0 20 18.3v-1.8"/>',
    download: '<path d="M12 4v12"/><path d="M7.5 11.5L12 16l4.5-4.5"/><path d="M4 20h16"/>',
    refresh: '<path d="M20 11.5A8 8 0 1 0 17.6 17"/><path d="M20.5 4.5v7h-7"/>',
    table: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M9.5 9.5v10"/>',
    target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/>',
    clock: '<circle cx="12" cy="12" r="8.6"/><path d="M12 7.4V12l3.2 2"/>',
    bolt: '<path d="M13 2.5L4.8 13.5H11l-1 8 8.2-11H12z"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>',
    db: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    arrowRight: '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    arrowLeft: '<path d="M19 12H5"/><path d="M11 18l-6-6 6-6"/>',
    eye: '<path d="M2.5 12S6 5.8 12 5.8 21.5 12 21.5 12 18 18.2 12 18.2 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.6"/>',
    shield: '<path d="M12 3l7.5 3v6c0 4.4-3.1 7.9-7.5 9.2C7.6 19.9 4.5 16.4 4.5 12V6z"/>',
    chart: '<path d="M3.5 20.5h17"/><rect x="5" y="11" width="3.4" height="7"/><rect x="10.3" y="6.5" width="3.4" height="11.5"/><rect x="15.6" y="13.5" width="3.4" height="4.5"/>',
    wand: '<path d="M5 19L17.5 6.5"/><path d="M15.5 4.5l4 4"/><path d="M6 5l1 2 2 1-2 1-1 2-1-2-2-1 2-1z"/>',
    dot: '<circle cx="12" cy="12" r="4"/>'
  };

  function icon(name, size, cls) {
    var body = ICONS[name] || ICONS.dot;
    return '<svg class="icon' + (cls ? ' ' + cls : '') + '" width="' + (size || 16) + '" height="' + (size || 16) +
      '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" ' +
      'stroke-linejoin="round" aria-hidden="true" focusable="false">' + body + '</svg>';
  }

  /* =============================================================== toasts == */

  var TOAST_ICON = { info: 'info', success: 'check', warn: 'warn', error: 'error' };
  var TOAST_TITLE = { info: 'Heads up', success: 'Done', warn: 'Careful', error: 'Something went wrong' };

  function toast(message, kind, title) {
    var k = kind || 'info';
    var host = $('#toasts');
    if (!host) return;
    var node = document.createElement('div');
    node.className = 'toast kind-' + k;
    node.setAttribute('role', k === 'error' ? 'alert' : 'status');
    node.innerHTML =
      '<span class="t-ico">' + icon(TOAST_ICON[k] || 'info', 16) + '</span>' +
      '<span class="toast-body">' +
        '<span class="toast-title">' + esc(title || TOAST_TITLE[k] || 'Notice') + '</span>' +
        '<span class="toast-msg">' + esc(message || '') + '</span>' +
      '</span>' +
      '<button type="button" class="toast-close" aria-label="Dismiss">' + icon('close', 13) + '</button>';
    var closeBtn = $('.toast-close', node);
    var remove = function () {
      if (!node.parentNode) return;
      node.classList.add('leaving');
      setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 200);
    };
    closeBtn.addEventListener('click', remove);
    host.appendChild(node);
    setTimeout(remove, k === 'error' ? 11000 : 6000);
  }

  /* ==================================================== working indicator == */

  var workingDepth = 0;
  var workingTimer = null;

  function workingStart(label) {
    workingDepth++;
    var box = $('#working');
    if (!box) return;
    box.hidden = false;
    var txt = $('#working-text');
    if (txt) txt.textContent = label || 'Working\u2026';
    if (workingTimer) clearTimeout(workingTimer);
    workingTimer = setTimeout(function () {
      if (workingDepth > 0 && txt) txt.textContent = 'Still working \u2014 large data takes a moment\u2026';
    }, 4000);
  }

  function workingStop() {
    workingDepth = Math.max(0, workingDepth - 1);
    if (workingDepth === 0) {
      var box = $('#working');
      if (box) box.hidden = true;
      if (workingTimer) { clearTimeout(workingTimer); workingTimer = null; }
    }
  }

  /* =========================================================== API client == */

  function ApiError(message, status) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function api(path, opts) {
    opts = opts || {};
    var label = opts.label || 'Talking to the local engine\u2026';
    if (!opts.quiet) workingStart(label);
    return fetch(path, {
      method: opts.method || 'GET',
      headers: opts.headers || undefined,
      body: opts.body || undefined
    }).then(function (res) {
      return res.text().then(function (text) {
        var data = null;
        if (text) {
          try { data = JSON.parse(text); } catch (e) { data = null; }
        }
        if (!res.ok) {
          var detail = '';
          if (data && typeof data === 'object') {
            detail = data.detail || data.error || data.message || '';
            if (detail && typeof detail === 'object') {
              detail = Array.isArray(detail)
                ? detail.map(function (d) { return d.msg || JSON.stringify(d); }).join('; ')
                : (detail.msg || JSON.stringify(detail));
            }
          }
          if (!detail) detail = text ? String(text).slice(0, 300) : ('Request failed with status ' + res.status);
          throw new ApiError(detail, res.status);
        }
        return data !== null ? data : {};
      });
    }).catch(function (err) {
      if (!opts.silent) {
        var msg = err && err.message ? err.message : 'The request could not be completed.';
        if (err && err.name === 'TypeError') msg = 'Could not reach the local engine. Is the ML Arena server still running?';
        toast(msg, 'error', err && err.name === 'ApiError' ? 'Request failed' : 'Connection problem');
      }
      throw err;
    }).then(function (data) {
      if (!opts.quiet) workingStop();
      return data;
    }, function (err) {
      if (!opts.quiet) workingStop();
      throw err;
    });
  }

  function apiJson(path, payload, opts) {
    opts = opts || {};
    return api(path, {
      method: opts.method || 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload === undefined ? {} : payload),
      label: opts.label,
      silent: opts.silent
    });
  }

  /* ================================================================ state == */

  var state = {
    step: 1,
    health: null,
    samples: [],
    datasetId: null,
    dataset: null,
    profile: null,
    usecases: [],
    functions: [],
    operators: [],
    suggestions: [],
    usecaseKey: null,
    usecase: null,
    mapping: {},
    staged: [],
    runId: null,
    run: null,
    events: [],
    seenSeq: {},
    models: {},
    championKey: null,
    streaming: false,
    finished: false,
    polling: false,
    pollTimer: null,
    idleTimer: null,
    lastEventAt: 0,
    ws: null,
    autoScroll: true
  };

  /* ============================================================ navigation == */

  var STEP_TITLES = ['Start', 'Your data', 'The goal', 'The columns', 'Watch it work', 'Your answer'];

  function canGo(step) {
    if (step === 1) return true;
    if (step === 2 || step === 3) return !!state.datasetId;
    if (step === 4) return !!state.usecaseKey;
    if (step === 5) return !!state.runId;
    if (step === 6) return !!state.run;
    return false;
  }

  function go(step) {
    var target = clamp(Number(step) || 1, 1, 6);
    if (!canGo(target)) {
      toast('Finish the earlier step first \u2014 then this one unlocks.', 'warn', 'Not yet');
      return;
    }
    state.step = target;

    for (var i = 1; i <= 6; i++) {
      var sec = document.getElementById('step-' + i);
      if (sec) sec.hidden = i !== target;
    }

    $$('#rail .rail-step').forEach(function (btn) {
      var n = Number(btn.getAttribute('data-goto'));
      var reachable = canGo(n);
      btn.disabled = !reachable;
      if (n === target) btn.setAttribute('aria-current', 'step');
      else btn.removeAttribute('aria-current');
      if (n < target) btn.classList.add('done'); else btn.classList.remove('done');
    });

    var head = $('#step-' + target + ' .step-title') || $('#step-' + target + ' .hero-title');
    if (head) {
      try { head.focus({ preventScroll: true }); } catch (e) { /* noop */ }
    }
    window.scrollTo({ top: 0, behavior: state.reducedMotion ? 'auto' : 'smooth' });
  }

  /* ============================================================ step 1: start */

  function renderHealth() {
    var h = state.health || {};
    var cpu = toNum(get(h, 'cpu_count', get(h, 'cpus', null)));
    var mem = toNum(get(h, 'memory_gb', null));
    var gpu = get(h, 'gpu', null);
    var version = get(h, 'version', null);

    if (version) {
      var vp = $('#version-pill');
      if (vp) vp.textContent = 'v' + version;
      var fv = $('#footer-version');
      if (fv) fv.textContent = 'v' + version;
    }

    var items = [];
    items.push({
      ico: 'cpu', val: cpu !== null ? fmtInt(cpu) + ' cores' : 'CPU',
      key: 'processor cores available', cls: ''
    });
    items.push({
      ico: 'bolt', val: mem !== null ? fmtNum(mem, 1) + ' GB' : 'Memory',
      key: 'of memory to work with', cls: ''
    });
    if (gpu === true) {
      items.push({ ico: 'spark', val: 'GPU found', key: 'will be used where it helps', cls: '' });
    } else {
      items.push({ ico: 'shield', val: 'CPU only', key: 'no GPU needed \u2014 we use methods built for that', cls: 'cap-warn' });
    }
    var libs = [];
    if (get(h, 'polars', null)) libs.push('polars ' + h.polars);
    if (get(h, 'sklearn', null)) libs.push('scikit-learn ' + h.sklearn);
    if (get(h, 'xgboost', null)) libs.push('xgboost ' + h.xgboost);
    if (get(h, 'lightgbm', null)) libs.push('lightgbm ' + h.lightgbm);

    var html = items.map(function (it) {
      return '<div class="cap-item ' + it.cls + '">' +
        '<span class="cap-ico">' + icon(it.ico, 18) + '</span>' +
        '<span><span class="cap-val">' + esc(it.val) + '</span><br><span class="cap-key">' + esc(it.key) + '</span></span>' +
        '</div>';
    }).join('');
    var host = $('#cap-strip');
    if (host) host.innerHTML = html;

    var foot = $('#cap-foot');
    if (foot) {
      foot.textContent = libs.length
        ? 'Ready to go with ' + libs.join(', ') + '. Everything below is computed on this machine \u2014 your data is never sent anywhere.'
        : 'Everything below is computed on this machine \u2014 your data is never sent anywhere.';
    }
    var fcap = $('#footer-cap');
    if (fcap) fcap.textContent = cpu !== null ? (cpu + ' CPU cores \u00b7 ' + (mem !== null ? fmtNum(mem, 1) + ' GB RAM' : 'local only')) : 'CPU only \u2014 no GPU needed';
  }

  function loadHealth() {
    return api('/api/health', { quiet: true, silent: true, label: 'Checking this computer\u2026' })
      .then(function (h) { state.health = h || {}; renderHealth(); })
      .catch(function () {
        var host = $('#cap-strip');
        if (host) host.innerHTML = '<span class="cap-skeleton">Could not read machine details \u2014 that is fine, we can still run.</span>';
      });
  }

  function loadSamples() {
    return api('/api/samples', { quiet: true, silent: true })
      .then(function (res) {
        state.samples = arr(get(res, 'samples', res));
        renderSamples();
      })
      .catch(function () {
        var host = $('#sample-grid');
        if (host) host.innerHTML = '<p class="empty-note">No bundled samples were found. Drop your own file above to get started.</p>';
      });
  }

  function renderSamples() {
    var host = $('#sample-grid');
    if (!host) return;
    if (!state.samples.length) {
      host.innerHTML = '<p class="empty-note">No bundled samples were found. Drop your own file above to get started.</p>';
      return;
    }
    host.innerHTML = state.samples.map(function (s, i) {
      var name = String(get(s, 'name', 'sample ' + (i + 1)));
      var pretty = basename(name).replace(/\.(csv|tsv|xlsx|xls|parquet|json)$/i, '').replace(/[_-]+/g, ' ');
      var size = fmtBytes(get(s, 'size_bytes', null));
      var uc = get(s, 'usecase', null);
      return '<button type="button" class="sample-card" data-sample="' + esc(name) + '">' +
        '<span class="sample-top">' +
          '<span class="sample-name">' + esc(titleCase(pretty)) + '</span>' +
          '<span class="sample-ico">' + icon('db', 15) + '</span>' +
        '</span>' +
        '<span class="sample-meta">' +
          (uc ? '<span class="sample-uc">' + esc(titleCase(uc)) + '</span>' : '') +
          (size ? '<span>' + esc(size) + '</span>' : '') +
          '<span>' + esc(name.split('.').pop().toUpperCase()) + '</span>' +
        '</span>' +
        '</button>';
    }).join('');
  }

  function useSample(name, btn) {
    if (!name) return;
    if (btn) btn.disabled = true;
    apiJson('/api/samples/' + encodeURIComponent(name) + '/load', {}, { label: 'Reading the sample\u2026' })
      .then(function (res) { adoptDataset(res, 'Sample loaded'); })
      .catch(function () { /* toast already shown */ })
      .then(function () { if (btn) btn.disabled = false; });
  }

  function uploadFile(file) {
    if (!file) return;
    var fd = new FormData();
    fd.append('file', file, file.name);
    api('/api/datasets', { method: 'POST', body: fd, label: 'Reading ' + file.name + '\u2026' })
      .then(function (res) { adoptDataset(res, 'File read'); })
      .catch(function () { /* toast already shown */ });
  }

  function adoptDataset(loadRes, successWord) {
    var id = get(loadRes, 'dataset_id', get(loadRes, 'id', get(loadRes, 'dataset.dataset_id', null)));
    if (!id) {
      toast('The engine did not return a dataset id, so we cannot continue.', 'error');
      return Promise.resolve();
    }
    state.datasetId = id;
    state.dataset = get(loadRes, 'dataset', loadRes) || {};
    state.profile = get(loadRes, 'profile', null);
    state.staged = [];
    state.mapping = {};
    state.usecaseKey = null;
    state.usecase = null;
    state.run = null;
    state.runId = null;
    renderStaged();
    hidePreview();

    var msg = successWord ? successWord + '.' : 'Data loaded.';
    var rows = get(loadRes, 'rows', get(loadRes, 'dataset.rows', null));
    var ms = get(loadRes, 'load_ms', get(loadRes, 'dataset.load_ms', null));
    toast(msg + (rows !== null ? ' ' + fmtInt(rows) + ' rows' : '') + (ms !== null ? ' in ' + fmtMs(ms) + '.' : ''), 'success');

    return refreshDataset(id).then(function () { go(2); });
  }

  function refreshDataset(id) {
    return api('/api/datasets/' + encodeURIComponent(id), { label: 'Working out what your columns mean\u2026' })
      .then(function (res) {
        var ds = get(res, 'dataset', res) || {};
        var prof = get(res, 'profile', null);
        state.dataset = ds;
        state.datasetId = get(ds, 'dataset_id', id);
        state.profile = prof;
        renderDatasetSummary();
        renderColumnTable();
        state.suggestions = arr(get(state.profile, 'suggestions', []));
        renderSuggestions();
        renderUseCases();
        renderRoles();
        loadSuggestions(state.datasetId);
        return res;
      });
  }

  /* ========================================================= step 2: data == */

  function kindInfo(col) {
    var kind = String(get(col, 'kind', '')).toLowerCase();
    var sem = String(get(col, 'semantic', '')).toLowerCase();
    var label = 'Column', tone = 'neutral';
    if (kind === 'measure') {
      if (sem === 'currency') { label = 'Money'; tone = 'money'; }
      else if (sem === 'percent') { label = 'Percent'; tone = 'percent'; }
      else if (sem === 'count') { label = 'Count'; tone = 'count'; }
      else if (sem === 'quantity') { label = 'Quantity'; tone = 'count'; }
      else { label = 'Number'; tone = 'number'; }
    } else if (kind === 'dimension') {
      if (sem === 'flag') { label = 'Yes / No'; tone = 'flag'; }
      else { label = 'Category'; tone = 'category'; }
    } else if (kind === 'datetime') {
      label = 'Date'; tone = 'date';
    } else if (kind === 'identifier') {
      if (sem === 'code') { label = 'Code'; tone = 'code'; } else { label = 'ID'; tone = 'id'; }
    } else if (kind === 'text') {
      label = 'Free text'; tone = 'text';
    }
    return { label: label, tone: tone, kind: kind, semantic: sem };
  }

  function kindBadge(col) {
    var info = kindInfo(col);
    var hint = get(col, 'role_hint', '');
    var title = info.label + (info.semantic ? ' \u00b7 ' + info.semantic : '') + (hint ? ' \u2014 ' + hint : '');
    return '<span class="badge badge-' + info.tone + '" title="' + esc(title) + '">' + esc(info.label) + '</span>';
  }

  function fmtSample(v) {
    if (v === null || v === undefined) return '\u2014';
    if (typeof v === 'number') return fmtNum(v, 4);
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    return String(v);
  }

  function qualityTone(score) {
    var s = toNum(score);
    if (s === null) return '';
    if (s >= 80) return 'good';
    if (s >= 55) return '';
    return 'warn';
  }

  function renderDatasetSummary() {
    var host = $('#dataset-summary');
    if (!host) return;
    var ds = state.dataset || {};
    var prof = state.profile || {};
    var rows = toNum(get(prof, 'rows', get(ds, 'rows', null)));
    var cols = toNum(get(prof, 'cols', get(ds, 'cols', null)));
    var loadMs = toNum(get(ds, 'load_ms', null));
    var engine = get(ds, 'engine', 'polars');
    var fmtName = String(get(ds, 'fmt', '')).toUpperCase();
    var fname = basename(get(ds, 'path', get(ds, 'filename', 'your data')));
    var size = toNum(get(ds, 'size_bytes', null));
    var profiledMs = toNum(get(prof, 'profiled_ms', null));
    var derivedFrom = get(ds, 'derived_from', null);

    var fast = loadMs !== null && loadMs < 800;
    var summary = '' +
      '<div class="summary-card">' +
        '<div class="summary-cell">' +
          '<div class="summary-label">File</div>' +
          '<div class="summary-value sm" title="' + esc(fname) + '">' + esc(fname) + '</div>' +
          '<div class="summary-sub">' + esc(fmtName || 'data') + (size !== null ? ' \u00b7 ' + esc(fmtBytes(size)) : '') + '</div>' +
        '</div>' +
        '<div class="summary-cell">' +
          '<div class="summary-label">Rows</div>' +
          '<div class="summary-value">' + esc(fmtInt(rows)) + '</div>' +
          '<div class="summary-sub">records to analyse</div>' +
        '</div>' +
        '<div class="summary-cell">' +
          '<div class="summary-label">Columns</div>' +
          '<div class="summary-value">' + esc(fmtInt(cols)) + '</div>' +
          '<div class="summary-sub">things we can look at</div>' +
        '</div>' +
        '<div class="summary-cell">' +
          '<div class="summary-label">Read in</div>' +
          '<div class="summary-value">' + esc(loadMs !== null ? fmtMs(loadMs) : '\u2014') + '</div>' +
          '<div class="summary-sub">using ' + esc(engine) + (fast ? ' \u2014 that was quick' : '') + '</div>' +
        '</div>' +
      '</div>';

    var q = get(prof, 'quality', {}) || {};
    var headline = get(q, 'headline', '');
    var notes = arr(get(ds, 'notes', []));
    var qHtml = '';
    if (headline || notes.length) {
      qHtml = '<div class="quality-line ' + qualityTone(get(q, 'score', null)) + '">' +
        '<span class="q-ico">' + icon('info', 17) + '</span>' +
        '<span>' + esc(headline || 'Here is what your data looks like.') +
        (notes.length ? '<span class="notes-list">' + notes.map(function (n) {
          return '<span class="note-line">' + esc(n) + '</span>';
        }).join('') + '</span>' : '') +
        '</span></div>';
    }
    if (derivedFrom) {
      qHtml += '<div class="quality-line good">' + icon('check', 17) +
        '<span>This is a fresh copy with your new columns added. Your original file was never changed' +
        (profiledMs !== null ? ' \u2014 we re-read it in ' + esc(fmtMs(profiledMs)) : '') + '.</span></div>';
    }
    host.innerHTML = summary + qHtml;
  }

  function renderColumnTable() {
    var host = $('#column-table');
    if (!host) return;
    var prof = state.profile || {};
    var cols = arr(get(prof, 'columns', []));
    var note = $('#cols-note');
    if (note) note.textContent = cols.length + ' columns';

    if (!cols.length) {
      host.innerHTML = '<p class="empty-note">We could not read any columns from this file.</p>';
      return;
    }

    var rows = cols.map(function (c) {
      var name = String(get(c, 'name', ''));
      var nullPct = toNum(get(c, 'null_pct', 0)) || 0;
      var nullCls = nullPct > 20 ? 'high' : (nullPct > 1 ? 'mid' : '');
      var samples = arr(get(c, 'samples', [])).slice(0, 5);
      var dtype = String(get(c, 'dtype', ''));
      return '<tr>' +
        '<td class="cell-name">' + esc(name) +
          (dtype ? '<div class="faint" style="font-size:11px;font-family:var(--mono)">' + esc(dtype) + '</div>' : '') +
        '</td>' +
        '<td>' + kindBadge(c) + '</td>' +
        '<td class="num"><span class="nullbar"><span class="nullbar-track">' +
          '<span class="nullbar-fill ' + nullCls + '" style="width:' + clamp(nullPct, 0, 100).toFixed(1) + '%"></span>' +
        '</span>' + nullPct.toFixed(nullPct < 1 && nullPct > 0 ? 1 : 0) + '%</span></td>' +
        '<td class="num">' + esc(fmtInt(get(c, 'distinct', null))) + '</td>' +
        '<td><span class="cell-samples">' +
          (samples.length
            ? samples.map(function (s) {
                return '<span class="sample-chip" title="' + esc(fmtSample(s)) + '">' + esc(fmtSample(s)) + '</span>';
              }).join('')
            : '<span class="faint">no sample values</span>') +
        '</span></td>' +
        '</tr>';
    }).join('');

    host.innerHTML = '<table class="data"><thead><tr>' +
      '<th>Column</th><th>What it looks like</th><th class="num">Missing</th><th class="num">Distinct</th>' +
      '<th>Example values</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }

  /* ------------------------------------------------------- column builder -- */

  function renderSuggestions() {
    var host = $('#suggestion-chips');
    if (!host) return;
    var list = arr(state.suggestions);
    if (!list.length) { host.innerHTML = ''; return; }
    host.innerHTML = list.map(function (s, i) {
      var name = String(get(s, 'name', 'new_column_' + i));
      var expr = String(get(s, 'expression', ''));
      var reason = String(get(s, 'reason', ''));
      return '<button type="button" class="suggest-chip" data-suggest="' + i + '" title="' + esc(reason) + '">' +
        '<span class="spark">' + icon('wand', 14) + '</span>' +
        esc(titleCase(name)) +
        '<span class="mono">' + esc(expr.length > 40 ? expr.slice(0, 39) + '\u2026' : expr) + '</span>' +
        '</button>';
    }).join('');
  }

  /* The profile already carries suggestions, but the dedicated endpoint is the
     source of truth once the column engine has had a proper look. */
  function loadSuggestions(id) {
    return api('/api/datasets/' + encodeURIComponent(id) + '/suggestions', { quiet: true, silent: true })
      .then(function (res) {
        var list = arr(get(res, 'suggestions', null));
        if (list.length) {
          state.suggestions = list;
          renderSuggestions();
        }
      })
      .catch(function () { /* the profile suggestions stay on screen */ });
  }

  function renderHelper() {
    var host = $('#fn-helper');
    if (!host) return;
    var fns = arr(state.functions);
    var ops = arr(state.operators);

    function fnChip(f) {
      var name = String(get(f, 'name', ''));
      var sig = String(get(f, 'signature', name));
      var desc = String(get(f, 'description', ''));
      var ex = String(get(f, 'example', ''));
      var title = desc + (ex ? '\nExample: ' + ex : '');
      return '<button type="button" class="chip" data-insert="' + esc(name + '(') + '" title="' + esc(title) + '">' +
        esc(sig || name) + '</button>';
    }

    function opChip(o) {
      var name = String(get(o, 'name', get(o, 'symbol', '')));
      var desc = String(get(o, 'description', ''));
      var sym = String(get(o, 'symbol', name));
      return '<button type="button" class="chip op" data-insert="' + esc(sym) + '" title="' + esc(desc) + '">' +
        esc(sym) + '</button>';
    }

    var html = '';
    if (fns.length) {
      html += '<div class="helper-group"><div class="helper-label">Functions \u2014 click to add</div>' +
        '<div class="chip-row">' + fns.map(fnChip).join('') + '</div></div>';
    }
    if (ops.length) {
      html += '<div class="helper-group"><div class="helper-label">Operators</div>' +
        '<div class="chip-row">' + ops.map(opChip).join('') + '</div></div>';
    }
    if (!html) {
      html = '<div class="helper-group"><div class="helper-label">Operators</div><div class="chip-row">' +
        ['+', '-', '*', '/', '(', ')'].map(function (s) {
          return '<button type="button" class="chip op" data-insert="' + esc(s) + '">' + esc(s) + '</button>';
        }).join('') + '</div></div>';
    }
    host.innerHTML = html;
  }

  function insertIntoExpr(text) {
    var input = $('#derived-expr');
    if (!input || !text) return;
    var start = input.selectionStart === undefined ? input.value.length : input.selectionStart;
    var end = input.selectionEnd === undefined ? input.value.length : input.selectionEnd;
    var before = input.value.slice(0, start);
    var after = input.value.slice(end);
    var needsClose = text.charAt(text.length - 1) === '(';
    input.value = before + text + (needsClose ? ')' : '') + after;
    var caret = start + text.length + (needsClose ? 0 : 0);
    input.focus();
    try { input.setSelectionRange(caret, caret); } catch (e) { /* noop */ }
  }

  function hidePreview() {
    var box = $('#preview-out');
    if (box) { box.hidden = true; box.innerHTML = ''; }
  }

  function previewDerived(name, expression) {
    if (!state.datasetId) { toast('Load a file first.', 'warn'); return; }
    if (!expression || !String(expression).trim()) {
      toast('Type how the new column should be worked out.', 'warn', 'Nothing to preview');
      var exprInput = $('#derived-expr');
      if (exprInput) exprInput.focus();
      return;
    }
    apiJson('/api/datasets/' + encodeURIComponent(state.datasetId) + '/columns/preview',
      { name: name || 'preview_column', expression: expression },
      { label: 'Testing your formula\u2026' })
      .then(function (res) { renderPreview(res, name, expression); })
      .catch(function () { hidePreview(); });
  }

  function renderPreview(res, name, expression) {
    var box = $('#preview-out');
    if (!box) return;
    box.hidden = false;

    if (!get(res, 'ok', false)) {
      var err = String(get(res, 'error', 'That formula could not be worked out.'));
      box.innerHTML = '<div class="preview-head"><span class="preview-title">Preview</span></div>' +
        '<div class="role-empty">' + icon('warn', 16) + '<span>' + esc(err) + '</span></div>' +
        '<p class="field-help" style="margin-top:10px">Check the column names and make sure every opening bracket is closed.</p>';
      return;
    }

    var values = arr(get(res, 'values', []));
    var stats = get(res, 'stats', {}) || {};
    var dtype = get(res, 'dtype', '');
    var rowCount = toNum(get(res, 'row_count', null));

    var head = '<div class="preview-head">' +
      '<span class="preview-title">' + esc(name || 'preview_column') + ' \u2014 first ' + values.length + ' values</span>' +
      '<span class="badge badge-number">' + esc(dtype || 'value') + '</span>' +
      '</div>';

    var table = '<div class="table-scroll" style="max-height:220px"><table class="data"><thead><tr>' +
      '<th class="num">Row</th><th>' + esc(name || 'new column') + '</th></tr></thead><tbody>' +
      values.slice(0, 60).map(function (v, i) {
        return '<tr><td class="num faint">' + (i + 1) + '</td><td>' + esc(fmtSample(v)) + '</td></tr>';
      }).join('') + '</tbody></table></div>';

    var statKeys = ['min', 'max', 'mean', 'nulls'];
    var statsHtml = '<div class="stat-row">' + statKeys.map(function (k) {
      if (get(stats, k, undefined) === undefined) return '';
      var v = stats[k];
      return '<div class="stat-chip"><div class="k">' + esc(k) + '</div><div class="v">' +
        esc(typeof v === 'number' ? fmtNum(v, 3) : fmtSample(v)) + '</div></div>';
    }).join('') +
      (rowCount !== null ? '<div class="stat-chip"><div class="k">rows</div><div class="v">' + esc(fmtInt(rowCount)) + '</div></div>' : '') +
      '</div>';

    box.innerHTML = head + table + statsHtml +
      '<p class="field-help" style="margin-top:10px">Formula: <span class="mono">' + esc(expression) + '</span></p>';
  }

  function stageCurrent() {
    var nameInput = $('#derived-name');
    var exprInput = $('#derived-expr');
    var name = nameInput ? nameInput.value.trim() : '';
    var expr = exprInput ? exprInput.value.trim() : '';
    if (!name) { toast('Give the new column a name so you can find it later.', 'warn'); if (nameInput) nameInput.focus(); return; }
    if (!/^[A-Za-z_][A-Za-z0-9_ ]*$/.test(name)) {
      toast('Use letters, numbers and underscores only \u2014 for example discount_pct.', 'warn', 'Column name');
      return;
    }
    if (!expr) { toast('Type how the new column should be worked out.', 'warn'); if (exprInput) exprInput.focus(); return; }
    if (state.staged.some(function (s) { return s.name === name; })) {
      toast('You already have a column called ' + name + ' waiting.', 'warn');
      return;
    }
    state.staged.push({ name: name, expression: expr });
    if (nameInput) nameInput.value = '';
    if (exprInput) exprInput.value = '';
    renderStaged();
    hidePreview();
    toast(name + ' is waiting to be added. Add more, or apply them to your data.', 'info', 'Added to the list');
  }

  function renderStaged() {
    var box = $('#staged-box');
    var list = $('#staged-list');
    var btn = $('#btn-apply');
    if (!box || !list) return;
    if (!state.staged.length) {
      box.hidden = true;
      list.innerHTML = '';
      if (btn) btn.disabled = true;
      return;
    }
    box.hidden = false;
    if (btn) btn.disabled = false;
    list.innerHTML = state.staged.map(function (s, i) {
      return '<li class="staged-item">' +
        '<span class="grow"><strong>' + esc(s.name) + '</strong>' +
        '<div class="expr">' + esc(s.expression) + '</div></span>' +
        '<button type="button" class="remove-btn" data-unstage="' + i + '">Remove</button>' +
        '</li>';
    }).join('');
  }

  function applyStaged() {
    if (!state.staged.length || !state.datasetId) return;
    var payload = { columns: state.staged.map(function (s) { return { name: s.name, expression: s.expression }; }) };
    apiJson('/api/datasets/' + encodeURIComponent(state.datasetId) + '/columns/apply', payload,
      { label: 'Adding your new columns\u2026' })
      .then(function (res) {
        state.staged = [];
        renderStaged();
        hidePreview();
        var id = get(res, 'dataset_id', get(res, 'id', get(res, 'dataset.dataset_id', null)));
        if (!id) { toast('The engine did not return the new dataset.', 'error'); return; }
        state.datasetId = id;
        state.dataset = get(res, 'dataset', res) || {};
        toast('Your new columns are ready. The original file is untouched.', 'success', 'Columns added');
        return refreshDataset(id);
      })
      .catch(function () { /* toast already shown */ });
  }

  /* ======================================================== step 3: goals == */

  function renderUseCases() {
    var host = $('#usecase-groups');
    if (!host) return;
    var list = arr(state.usecases);
    if (!list.length) {
      host.innerHTML = '<p class="empty-note">No analysis goals were available from the engine.</p>';
      return;
    }

    var bySector = {};
    var order = [];
    list.forEach(function (uc) {
      var sector = String(get(uc, 'sector', 'Other'));
      if (!bySector[sector]) { bySector[sector] = []; order.push(sector); }
      bySector[sector].push(uc);
    });

    host.innerHTML = order.map(function (sector) {
      var cards = bySector[sector].map(function (uc) {
        var key = String(get(uc, 'key', ''));
        var selected = key === state.usecaseKey ? ' selected' : '';
        var roles = arr(get(uc, 'roles', []));
        var required = roles.filter(function (r) { return get(r, 'required', false); }).length;
        return '<button type="button" class="uc-card' + selected + '" data-usecase="' + esc(key) + '" ' +
          'aria-pressed="' + (key === state.usecaseKey ? 'true' : 'false') + '">' +
          '<span class="uc-top">' +
            '<span class="uc-emoji" aria-hidden="true">' + esc(get(uc, 'emoji', '\u25c6')) + '</span>' +
            '<span>' +
              '<span class="uc-name">' + esc(get(uc, 'name', titleCase(key))) + '</span>' +
              '<div class="uc-tag">' + esc(get(uc, 'tagline', '')) + '</div>' +
            '</span>' +
          '</span>' +
          '<span class="uc-story">' + esc(get(uc, 'story', '')) + '</span>' +
          '<span class="uc-foot">' +
            '<span class="uc-metric">' + roles.length + ' column' + (roles.length === 1 ? '' : 's') + ' to pick' +
              (required ? ' \u00b7 ' + required + ' required' : '') + '</span>' +
            '<span class="uc-tick">' + icon('check', 16) + '</span>' +
          '</span>' +
          '</button>';
      }).join('');
      return '<section class="sector"><div class="sector-head">' +
        '<h3 class="sector-title">' + esc(sector) + '</h3><span class="sector-rule"></span></div>' +
        '<div class="uc-grid">' + cards + '</div></section>';
    }).join('');

    var hint = $('#usecase-hint');
    var hintText = $('#usecase-hint-text');
    var starter = list.filter(function (u) { return String(get(u, 'key', '')) === 'discount_leakage'; })[0];
    if (hint && hintText && starter) {
      hint.hidden = false;
      hintText.innerHTML = '<strong>Not sure? Start here.</strong> ' +
        esc(get(starter, 'name', 'Discount leakage')) + ' is the easiest one to see working \u2014 ' +
        esc(String(get(starter, 'tagline', '')).toLowerCase()) + '. ' +
        '<button type="button" class="btn btn-sm btn-soft" data-usecase="discount_leakage">Pick it for me</button>';
    } else if (hint) {
      hint.hidden = true;
    }
  }

  function selectUseCase(key) {
    var uc = arr(state.usecases).filter(function (u) { return String(get(u, 'key', '')) === key; })[0];
    if (!uc) { toast('That goal is not available.', 'warn'); return; }
    state.usecaseKey = key;
    state.usecase = uc;
    state.mapping = {};
    renderUseCases();
    renderRoles();
    var btn = $('#to-step-4');
    if (btn) btn.disabled = false;
    go(4);
  }

  /* ====================================================== step 4: columns == */

  function eligibleColumns(role) {
    var accepts = arr(get(role, 'accepts', [])).map(function (a) { return String(a).toLowerCase(); });
    var cols = arr(get(state.profile, 'columns', []));
    if (!accepts.length) return cols;
    return cols.filter(function (c) {
      return accepts.indexOf(String(get(c, 'kind', '')).toLowerCase()) >= 0;
    });
  }

  function renderRoles() {
    var host = $('#role-pickers');
    if (!host) return;
    var uc = state.usecase;
    if (!uc) {
      host.innerHTML = '<p class="empty-note">Pick a goal first.</p>';
      renderConfirm();
      return;
    }

    var roles = arr(get(uc, 'roles', []));
    var sub = $('#step-4-sub');
    if (sub) {
      sub.textContent = 'For \u201c' + String(get(uc, 'name', '')) + '\u201d we need ' + roles.length +
        ' thing' + (roles.length === 1 ? '' : 's') + ' from you. We only show columns that fit each slot.';
    }

    // sensible defaults: a single required role with exactly one eligible column
    roles.forEach(function (role) {
      var rk = String(get(role, 'key', ''));
      if (arr(state.mapping[rk]).length) return;
      if (!get(role, 'required', false) || get(role, 'multi', false)) return;
      var el = eligibleColumns(role);
      if (el.length === 1) state.mapping[rk] = [String(get(el[0], 'name', ''))];
    });

    host.innerHTML = roles.map(function (role) {
      var key = String(get(role, 'key', ''));
      var label = String(get(role, 'label', titleCase(key)));
      var help = String(get(role, 'help', ''));
      var required = !!get(role, 'required', false);
      var multi = !!get(role, 'multi', false);
      var elig = eligibleColumns(role);
      var picked = arr(state.mapping[key]);
      var filled = picked.length > 0;

      var badge = required
        ? '<span class="badge badge-required">Needed</span>'
        : '<span class="badge badge-optional">Optional</span>';
      var multiBadge = multi ? '<span class="badge badge-neutral">Pick as many as you like</span>' : '';

      var body;
      if (!elig.length) {
        body = '<div class="role-empty">' +
          icon('warn', 16) +
          '<span>None of your columns look like what this needs. You can build one in step 2 \u2014 we will bring you straight back here.</span>' +
          '<button type="button" class="btn btn-sm btn-soft" data-make-column="' + esc(key) + '">' +
          icon('plus', 14) + ' Create a column</button>' +
          '</div>';
      } else {
        var groups = {};
        var gOrder = [];
        elig.forEach(function (c) {
          var k = String(get(c, 'kind', 'other'));
          if (!groups[k]) { groups[k] = []; gOrder.push(k); }
          groups[k].push(c);
        });
        var opts = gOrder.map(function (k) {
          return '<optgroup label="' + esc(titleCase(k)) + '">' + groups[k].map(function (c) {
            var name = String(get(c, 'name', ''));
            var isSel = picked.indexOf(name) >= 0;
            var info = kindInfo(c);
            return '<option value="' + esc(name) + '"' + (isSel ? ' selected' : '') + '>' +
              esc(name) + '  \u2014  ' + esc(info.label) + '</option>';
          }).join('') + '</optgroup>';
        }).join('');

        var sizeAttr = multi ? ' multiple size="' + Math.min(7, Math.max(3, elig.length)) + '"' : '';
        body = '<div class="field">' +
          '<label for="role-' + esc(key) + '">' +
            (multi ? 'Which columns should we use?' : 'Which column?') +
          '</label>' +
          '<select id="role-' + esc(key) + '" data-role="' + esc(key) + '"' + sizeAttr + '>' +
            (multi ? '' : '<option value="">\u2014 choose a column \u2014</option>') +
            opts +
          '</select>' +
          '<p class="field-help">' + elig.length + ' suitable column' + (elig.length === 1 ? '' : 's') +
            (multi ? ' \u00b7 hold Ctrl (or Cmd) to pick more than one' : '') + '</p>' +
          '</div>';
      }

      var cls = 'role-card' + (filled ? ' filled' : (required && !elig.length ? ' missing' : ''));
      return '<div class="' + cls + '" data-role-card="' + esc(key) + '">' +
        '<div class="role-head">' +
          '<span class="role-label">' + esc(label) + '</span>' + badge + multiBadge +
        '</div>' +
        (help ? '<p class="role-help">' + esc(help) + '</p>' : '') +
        body +
        (filled ? '<div class="role-picked">' + picked.map(function (p) {
          return '<span class="badge badge-neutral">' + esc(p) + '</span>';
        }).join('') + '</div>' : '') +
        '</div>';
    }).join('');

    // sensible defaults: a single required role with exactly one eligible column

    renderConfirm();
  }

  function readRolePickers() {
    $$('#role-pickers select[data-role]').forEach(function (sel) {
      var key = sel.getAttribute('data-role');
      var values = Array.prototype.slice.call(sel.selectedOptions || [])
        .map(function (o) { return o.value; })
        .filter(function (v) { return v !== ''; });
      state.mapping[key] = values;
    });
    $$('#role-pickers .role-card').forEach(function (card) {
      var key = card.getAttribute('data-role-card');
      if (arr(state.mapping[key]).length) card.classList.add('filled');
      else card.classList.remove('filled');
    });
    renderConfirm();
  }

  function roleFilled(role) {
    var key = String(get(role, 'key', ''));
    var picked = arr(state.mapping[key]);
    if (!get(role, 'required', false)) return true;
    var min = toNum(get(role, 'min_count', 1)) || 1;
    return picked.length >= min;
  }

  function renderConfirm() {
    var host = $('#role-confirm');
    if (!host) return;
    var uc = state.usecase;
    if (!uc) {
      host.innerHTML = '<p class="empty-note">Pick a goal first.</p>';
      var runBtn0 = $('#btn-run');
      if (runBtn0) runBtn0.disabled = true;
      return;
    }
    var roles = arr(get(uc, 'roles', []));
    var missing = roles.filter(function (r) { return !roleFilled(r); });
    var parts = [];
    roles.forEach(function (r) {
      var key = String(get(r, 'key', ''));
      var picked = arr(state.mapping[key]);
      if (!picked.length) return;
      parts.push(picked.map(function (p) { return '<strong>' + esc(p) + '</strong>'; }).join(', ') +
        ' <span class="faint">(' + esc(String(get(r, 'label', titleCase(key))).toLowerCase()) + ')</span>');
    });

    var sentence = parts.length
      ? 'We\u2019ll look at: ' + parts.join(', and ') + '.'
      : 'We still need a few columns from you before we can start.';

    var rows = roles.map(function (r) {
      var key = String(get(r, 'key', ''));
      var picked = arr(state.mapping[key]);
      var ok = roleFilled(r);
      return '<div class="confirm-row">' +
        '<span class="k">' + esc(get(r, 'label', titleCase(key))) + '</span>' +
        '<span class="v">' + (picked.length
          ? picked.map(function (p) { return esc(p); }).join(', ')
          : (ok ? '<span class="faint">not used</span>' : '<span style="color:#F5A9A7">still needed</span>')) + '</span>' +
        '</div>';
    }).join('');

    host.innerHTML = '<p class="confirm-sentence">' + sentence + '</p>' +
      (rows ? '<div class="confirm-list">' + rows + '</div>' : '') +
      (missing.length
        ? '<p class="field-help" style="margin-top:12px">' + icon('info', 14) +
          ' Still to choose: ' + esc(missing.map(function (r) { return get(r, 'label', get(r, 'key', '')); }).join(', ')) + '.</p>'
        : '<p class="field-help" style="margin-top:12px">' + icon('check', 14) +
          ' Everything we need is here. The whole run stays on this computer.</p>');

    var runBtn = $('#btn-run');
    if (runBtn) runBtn.disabled = missing.length > 0;
  }

  function jumpToBuilder(roleKey) {
    go(2);
    setTimeout(function () {
      var panel = $('#builder-panel');
      if (panel && panel.scrollIntoView) panel.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth', block: 'start' });
      var nameInput = $('#derived-name');
      if (nameInput) nameInput.focus();
      if (roleKey) {
        var role = arr(get(state.usecase, 'roles', [])).filter(function (r) {
          return String(get(r, 'key', '')) === roleKey;
        })[0];
        if (role) {
          toast('Build a column for \u201c' + get(role, 'label', roleKey) + '\u201d, then come back to step 4.', 'info', 'Create a column');
        }
      }
    }, 120);
  }

  /* ========================================================= step 5: run == */

  function startRun() {
    if (!state.datasetId || !state.usecaseKey) return;
    readRolePickers();
    var runBtn = $('#btn-run');
    if (runBtn) runBtn.disabled = true;

    resetRunUi();

    apiJson('/api/runs', {
      dataset_id: state.datasetId,
      usecase: state.usecaseKey,
      mapping: state.mapping,
      options: {}
    }, { label: 'Starting the competition\u2026' })
      .then(function (res) {
        var id = get(res, 'run_id', get(res, 'id', null));
        if (!id) { toast('The engine did not return a run id.', 'error'); if (runBtn) runBtn.disabled = false; return; }
        state.runId = id;
        go(5);
        openStream(id);
      })
      .catch(function () {
        stopTimers();
        state.streaming = false;
        setBanner('Could not start the run.', 'Nothing was changed. Check the message and try again.', 'error');
        if (runBtn) runBtn.disabled = false;
      });
  }

  function resetRunUi() {
    state.run = null;
    state.events = [];
    state.seenSeq = {};
    state.models = {};
    state.championKey = null;
    state.streaming = true;
    state.finished = false;
    state.polling = false;
    state.lastEventAt = Date.now();
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
    if (state.ws) { try { state.ws.close(); } catch (e) { /* noop */ } state.ws = null; }

    var console_ = $('#console');
    if (console_) console_.innerHTML = '';
    var board = $('#scoreboard');
    if (board) board.innerHTML = '<p class="empty-note">The scoreboard fills in as each model finishes its turn.</p>';
    var count = $('#console-count');
    if (count) count.textContent = '0 events';
    var bcount = $('#board-count');
    if (bcount) bcount.textContent = '0 models';
    setProgress(0.02, 'Starting');
    setBanner('Warming up\u2026', 'Getting your data ready.', 'info');
    var idle = $('#banner-idle');
    if (idle) idle.hidden = true;

    if (state.idleTimer) clearInterval(state.idleTimer);
    state.idleTimer = setInterval(tickIdle, 1000);
  }

  function tickIdle() {
    if (!state.streaming || state.finished) return;
    var quietFor = Date.now() - state.lastEventAt;
    var idle = $('#banner-idle');
    if (idle) idle.hidden = quietFor < 3000;
  }

  function stopTimers() {
    if (state.idleTimer) { clearInterval(state.idleTimer); state.idleTimer = null; }
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
    state.polling = false;
    state.streaming = false;
  }

  function wsUrl(runId) {
    var scheme = (window.location.protocol === 'https:') ? 'wss' : 'ws';
    return scheme + ':' + '//' + window.location.host + '/ws/runs/' + encodeURIComponent(runId);
  }

  function openStream(runId) {
    var socket = null;
    try {
      socket = new WebSocket(wsUrl(runId));
    } catch (e) {
      startPolling();
      return;
    }
    state.ws = socket;

    socket.onopen = function () {
      state.lastEventAt = Date.now();
    };

    socket.onmessage = function (evt) {
      var data = null;
      try { data = JSON.parse(evt.data); } catch (e) { data = null; }
      if (!data || typeof data !== 'object') return;
      var seq = data.seq;
      if (seq !== undefined && seq !== null) {
        if (state.seenSeq[seq]) return;
        state.seenSeq[seq] = true;
      }
      handleEvent(data);
      if (String(data.phase) === 'done') finishRun(null);
    };

    socket.onerror = function () {
      if (!state.finished) startPolling();
    };

    socket.onclose = function () {
      if (state.finished) return;
      if (!state.polling) startPolling();
    };
  }

  function startPolling() {
    if (state.polling || state.finished || !state.runId) return;
    state.polling = true;
    if (state.ws) { try { state.ws.close(); } catch (e) { /* noop */ } state.ws = null; }
    setBanner('Still working\u2026', 'The live link dropped, so we are checking in every second instead.', 'warn');
    pollOnce();
    state.pollTimer = setInterval(pollOnce, 1400);
  }

  function pollOnce() {
    if (!state.runId || state.finished) return;
    api('/api/runs/' + encodeURIComponent(state.runId), { quiet: true, silent: true })
      .then(function (res) {
        var run = get(res, 'run', res) || {};
        arr(get(res, 'events', get(run, 'log', []))).forEach(function (e) {
          if (!e || typeof e !== 'object') return;
          var seq = e.seq;
          if (seq !== undefined && seq !== null) {
            if (state.seenSeq[seq]) return;
            state.seenSeq[seq] = true;
          }
          handleEvent(e);
        });
        var status = String(get(run, 'status', 'running'));
        if (status && status !== 'running' && status !== 'queued' && status !== 'pending') {
          finishRun(run);
        }
      })
      .catch(function () { /* keep trying quietly */ });
  }

  function handleEvent(evt) {
    state.lastEventAt = Date.now();
    state.events.push(evt);
    var idle = $('#banner-idle');
    if (idle) idle.hidden = true;

    var progress = toNum(get(evt, 'progress', null));
    if (progress !== null) {
      var pct = progress <= 1.0001 ? progress * 100 : progress;
      setProgress(pct / 100, String(get(evt, 'title', 'Working\u2026')));
    }
    setBanner(
      String(get(evt, 'title', get(evt, 'phase', 'Working'))),
      String(get(evt, 'detail', '')),
      String(get(evt, 'level', 'info'))
    );
    appendLogRow(evt);
    upsertModel(evt);
  }

  function setProgress(frac, label) {
    var pct = clamp(frac, 0, 1) * 100;
    var fill = $('#progress-fill');
    if (fill) fill.style.width = pct.toFixed(1) + '%';
    var pctEl = $('#progress-pct');
    if (pctEl) pctEl.textContent = Math.round(pct) + '%';
    var bar = $('#progress-bar');
    if (bar) bar.setAttribute('aria-valuenow', String(Math.round(pct)));
    var lab = $('#progress-label');
    if (lab && label) lab.textContent = label.length > 96 ? label.slice(0, 95) + '\u2026' : label;
  }

  function setBanner(title, detail, level) {
    var t = $('#banner-title');
    var d = $('#banner-detail');
    if (t) t.textContent = title || 'Working\u2026';
    if (d) d.textContent = detail || '';
    var banner = $('#now-banner');
    if (banner) {
      banner.className = 'banner' + (level && level !== 'info' ? ' level-' + level : '');
    }
    var ico = $('#now-banner .banner-icon');
    if (ico) {
      var name = level === 'champion' ? 'crown' : level === 'success' ? 'check' : level === 'warn' ? 'warn'
        : level === 'error' ? 'error' : level === 'story' ? 'spark' : 'clock';
      ico.innerHTML = icon(name, 18);
    }
  }

  function appendLogRow(evt) {
    var host = $('#console');
    if (!host) return;
    var level = String(get(evt, 'level', 'info'));
    var row = document.createElement('div');
    row.className = 'log-row level-' + (['info', 'success', 'warn', 'error', 'champion', 'story'].indexOf(level) >= 0 ? level : 'info');

    var metrics = get(evt, 'metrics', null);
    var metaBits = [];
    if (metrics && typeof metrics === 'object') {
      Object.keys(metrics).slice(0, 3).forEach(function (k) {
        var v = metrics[k];
        if (v === null || v === undefined || typeof v === 'object') return;
        metaBits.push(esc(humanMetric(k)) + ' ' + esc(typeof v === 'number' ? fmtNum(v, 3) : String(v)));
      });
    }
    var elapsed = toNum(get(evt, 'elapsed_ms', null));
    if (elapsed !== null) metaBits.push(esc(fmtMs(elapsed)) + ' elapsed');

    row.innerHTML =
      '<span class="log-time">' + esc(fmtClock(get(evt, 'ts', Date.now()))) + '</span>' +
      '<span class="log-dot"></span>' +
      '<span class="log-body">' +
        '<span class="log-title">' + esc(get(evt, 'title', get(evt, 'phase', 'Working'))) + '</span>' +
        (get(evt, 'detail', '') ? '<span class="log-detail">' + esc(evt.detail) + '</span>' : '') +
        (metaBits.length ? '<span class="log-meta">' + metaBits.join(' \u00b7 ') + '</span>' : '') +
      '</span>';

    var atBottom = host.scrollHeight - host.scrollTop - host.clientHeight < 60;
    host.appendChild(row);
    if (state.autoScroll || atBottom) {
      host.scrollTop = host.scrollHeight;
      state.autoScroll = true;
      var jump = $('#console-jump');
      if (jump) jump.hidden = true;
    } else {
      var jump2 = $('#console-jump');
      if (jump2) jump2.hidden = false;
    }
    var count = $('#console-count');
    if (count) count.textContent = state.events.length + ' event' + (state.events.length === 1 ? '' : 's');
  }

  function metricValue(metrics, primary) {
    if (!metrics || typeof metrics !== 'object') return null;
    var keys = Object.keys(metrics);
    if (primary && metrics[primary] !== undefined) return toNum(metrics[primary]);
    var order = ['roc_auc', 'r2', 'silhouette', 'accuracy', 'f1', 'pr_auc', 'consensus', 'agreement'];
    for (var i = 0; i < order.length; i++) {
      if (metrics[order[i]] !== undefined) return toNum(metrics[order[i]]);
    }
    if (keys.length) return toNum(metrics[keys[0]]);
    return null;
  }

  function upsertModel(evt) {
    var model = get(evt, 'model', null);
    if (!model) return;
    var phase = String(get(evt, 'phase', ''));
    if (phase && phase !== 'evaluate' && phase !== 'train' && phase !== 'select' && phase !== 'score') return;

    var payload = get(evt, 'payload', null);
    var p = payload && typeof payload === 'object' ? payload : {};
    var key = String(model);
    var entry = state.models[key] || { key: key, display: key };
    entry.display = get(p, 'display', entry.display || key);
    entry.persona = get(p, 'persona', entry.persona || '');
    entry.family = get(p, 'family', entry.family || '');
    entry.status = get(p, 'status', entry.status || 'ok');
    entry.error = get(p, 'error', entry.error || null);
    if (p.metrics && typeof p.metrics === 'object') entry.metrics = p.metrics;
    else if (evt.metrics && typeof evt.metrics === 'object' && Object.keys(evt.metrics).length) entry.metrics = evt.metrics;
    entry.metrics = entry.metrics || {};

    var primary = toNum(get(p, 'primary', null));
    if (primary !== null) entry.primary = primary;
    else if (entry.primary === undefined) entry.primary = metricValue(entry.metrics, null);

    var score = toNum(get(p, 'score', null));
    if (score !== null) entry.score = score;

    var trainMs = toNum(get(p, 'train_ms', null));
    if (trainMs !== null) entry.train_ms = trainMs;
    if (get(p, 'why', null)) entry.why = p.why;

    state.models[key] = entry;
    renderScoreboard();
  }

  function renderScoreboard() {
    var host = $('#scoreboard');
    if (!host) return;
    var entries = Object.keys(state.models).map(function (k) { return state.models[k]; });
    var count = $('#board-count');
    if (count) count.textContent = entries.length + ' model' + (entries.length === 1 ? '' : 's');
    if (!entries.length) {
      host.innerHTML = '<p class="empty-note">The scoreboard fills in as each model finishes its turn.</p>';
      return;
    }

    var maxPrimary = 0;
    entries.forEach(function (e) {
      var v = Math.abs(toNum(e.primary) === null ? 0 : toNum(e.primary));
      if (v > maxPrimary) maxPrimary = v;
    });

    entries.sort(function (a, b) {
      var sa = scorePct(a, maxPrimary), sb = scorePct(b, maxPrimary);
      if (sb !== sa) return sb - sa;
      return String(a.display).localeCompare(String(b.display));
    });

    var rows = entries.map(function (e) {
      var isChamp = state.championKey && e.key === state.championKey;
      var pct = scorePct(e, maxPrimary);
      var failed = String(e.status) === 'failed';
      var statusPill = failed
        ? '<span class="badge badge-fail">Did not finish</span>'
        : '<span class="badge badge-ok">Finished</span>';
      var metricLabel = humanMetric(e.primaryMetricKey || '');
      var shown = toNum(e.primary);
      var metricText = shown !== null ? (metricLabel ? metricLabel + ' ' : '') + fmtNum(shown, 3) : '\u2014';
      return '<tr' + (isChamp ? ' class="is-champion"' : '') + '>' +
        '<td>' +
          '<div class="m-name">' + (isChamp ? '<span class="m-crown">' + icon('crown', 15) + '</span>' : '') +
            esc(e.display || e.key) + '</div>' +
          (e.persona ? '<div class="m-persona">' + esc(e.persona) + '</div>' : '') +
          (failed && e.error ? '<div class="m-persona" style="color:#F5A9A7">' + esc(String(e.error).slice(0, 90)) + '</div>' : '') +
        '</td>' +
        '<td class="nowrap" style="font-family:var(--mono);font-size:11.5px;color:var(--muted)">' + esc(metricText) + '</td>' +
        '<td><span class="score-cell"><span class="score-track"><span class="score-fill" style="width:' +
          clamp(pct, 0, 100).toFixed(1) + '%"></span></span><span class="score-num">' + Math.round(pct) + '</span></span></td>' +
        '<td class="nowrap" style="color:var(--muted)">' + esc(e.train_ms !== undefined ? fmtMs(e.train_ms) : '\u2014') + '</td>' +
        '<td>' + statusPill + '</td>' +
        '</tr>';
    }).join('');

    host.innerHTML = '<table class="board"><thead><tr>' +
      '<th>Model</th><th>Result</th><th>Score</th><th>Train time</th><th>Status</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table>';
  }

  function scorePct(entry, maxPrimary) {
    var s = toNum(entry.score);
    if (s !== null) return clamp(s, 0, 100);
    var p = toNum(entry.primary);
    if (p === null) return 0;
    if (maxPrimary > 0) return clamp((Math.abs(p) / maxPrimary) * 100, 0, 100);
    return 0;
  }

  /* ======================================================== step 6: answer == */

  function finishRun(runFromPoll) {
    if (state.finished) return;
    state.finished = true;
    stopTimers();
    if (state.ws) { try { state.ws.close(); } catch (e) { /* noop */ } state.ws = null; }

    var finish = function (run) {
      if (!run || !Object.keys(run).length) {
        toast('The run finished but we could not read its results.', 'error');
        return;
      }
      state.run = run;
      var champ = get(run, 'champion', null);
      state.championKey = champ ? String(get(champ, 'key', get(champ, 'display', ''))) : null;
      setProgress(1, 'Finished');
      setBanner('All done \u2014 here is what we found.',
        String(get(champ, 'display', 'The winning model')) + ' came out on top.',
        'champion');
      if (String(get(run, 'status', '')) === 'failed') {
        toast(String(get(run, 'error', 'The run failed before it could finish.')), 'error', 'Run failed');
      }
      // make sure the scoreboard reflects the authoritative leaderboard
      var board = arr(get(run, 'leaderboard', []));
      if (board.length) {
        board.forEach(function (entry) {
          var key = String(get(entry, 'key', get(entry, 'display', '')));
          if (!key) return;
          var existing = state.models[key] || { key: key };
          existing.display = get(entry, 'display', existing.display || key);
          existing.persona = get(entry, 'persona', existing.persona || '');
          existing.metrics = get(entry, 'metrics', existing.metrics || {});
          existing.primary = toNum(get(entry, 'primary', null));
          existing.primaryMetricKey = primaryKeyOf(run);
          existing.score = toNum(get(entry, 'score', null));
          existing.train_ms = toNum(get(entry, 'train_ms', null));
          existing.status = get(entry, 'status', 'ok');
          existing.error = get(entry, 'error', null);
          state.models[key] = existing;
        });
        renderScoreboard();
      }
      renderResults(run);
      go(6);
    };

    if (runFromPoll && Object.keys(runFromPoll).length) { finish(runFromPoll); return; }
    api('/api/runs/' + encodeURIComponent(state.runId), { label: 'Collecting the results\u2026' })
      .then(function (res) { finish(get(res, 'run', res) || {}); })
      .catch(function () {
        // last resort: build a minimal result from what we streamed
        if (state.events.length) {
          finish({ status: 'succeeded', log: state.events, narration: [], leaderboard: [], distributions: {} });
        }
      });
  }

  function primaryKeyOf(run) {
    var m = get(run, 'metrics', null);
    if (!m || typeof m !== 'object') return '';
    var order = ['roc_auc', 'r2', 'silhouette', 'consensus'];
    for (var i = 0; i < order.length; i++) if (m[order[i]] !== undefined) return order[i];
    return Object.keys(m)[0] || '';
  }

  var METRIC_INFO = {
    roc_auc: { label: 'Separation power', tip: 'How reliably the model ranks a real case above a normal one. 0.5 is a coin flip; 1.0 is perfect.' },
    accuracy: { label: 'How often it is right', tip: 'Out of every 100 rows, how many the model got right.' },
    precision: { label: 'Of the ones we flagged, how many were right', tip: 'When the model says "look at this", how often it is correct. High precision means fewer wasted checks.' },
    recall: { label: 'Of all the real cases, how many we caught', tip: 'Of everything that actually happened, how much the model found. High recall means fewer misses.' },
    f1: { label: 'Balance of hits and misses', tip: 'A single number that balances precision and recall. Useful when neither alone tells the story.' },
    pr_auc: { label: 'Ranking quality on the rare cases', tip: 'Like separation power, but focused on the uncommon outcome you actually care about.' },
    log_loss: { label: 'Penalty for confident mistakes', tip: 'Lower is better. It punishes the model for being sure and wrong.' },
    r2: { label: 'Explained variation', tip: 'How much of the ups and downs the model explains. 1.0 is perfect; 0 means no better than always guessing the average.' },
    mape: { label: 'Average % error', tip: 'On a typical row, how far off the prediction is, as a percentage. Lower is better.' },
    rmse: { label: 'Typical error size', tip: 'The usual size of the miss, in the same units as the thing you are predicting. Lower is better.' },
    mae: { label: 'Average miss', tip: 'The average distance between the prediction and the truth. Lower is better.' },
    wape: { label: 'Total error vs total volume', tip: 'All the misses added up, as a share of the total. Lower is better.' },
    silhouette: { label: 'How cleanly the groups separate', tip: 'Near 1 means tidy, well-separated groups; near 0 means they blur together.' },
    davies_bouldin: { label: 'How tight the groups are', tip: 'Lower is better \u2014 smaller numbers mean tighter, better-separated groups.' },
    n_clusters: { label: 'Groups found', tip: 'How many distinct groups the model decided your data falls into.' },
    contamination: { label: 'Share flagged as unusual', tip: 'The fraction of rows the model considered odd enough to look at.' },
    agreement: { label: 'How much the detectors agree', tip: 'When several different methods flag the same rows, confidence goes up.' },
    consensus: { label: 'Agreement between methods', tip: 'How strongly the different methods point at the same rows.' },
    tp: { label: 'Correctly flagged', tip: 'Rows the model flagged that really were the thing you were looking for.' },
    fp: { label: 'Flagged by mistake', tip: 'Rows the model flagged that turned out to be fine.' },
    tn: { label: 'Correctly left alone', tip: 'Ordinary rows the model correctly ignored.' },
    fn: { label: 'Missed cases', tip: 'Real cases the model did not flag.' }
  };

  function metricInfo(key) {
    return METRIC_INFO[key] || { label: humanMetric(key), tip: 'A measure of how well the model did.' };
  }

  function formatImpact(value, unit) {
    var n = toNum(value);
    if (n === null) return '\u2014';
    var u = String(unit || '').toLowerCase();
    if (u === 'currency') {
      // We never print a currency symbol. The file's currency is not something this
      // tool can know, and guessing it would mislabel every money figure on the page.
      var abs = Math.abs(n);
      var sign = n < 0 ? '-' : '';
      if (abs >= 1e9) return sign + (abs / 1e9).toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' billion';
      if (abs >= 1e6) return sign + (abs / 1e6).toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' million';
      if (abs >= 1e3) return sign + (abs / 1e3).toLocaleString(undefined, { maximumFractionDigits: 1 }) + ' thousand';
      return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    if (u === 'records' || u === 'customers' || u === 'rows' || u === 'units') {
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function severityClass(sev) {
    var s = String(sev || '').toLowerCase();
    if (s === 'success' || s === 'good' || s === 'ok') return 'sev-success';
    if (s === 'warn' || s === 'warning' || s === 'caution') return 'sev-warn';
    if (s === 'error' || s === 'critical' || s === 'high') return 'sev-error';
    return 'sev-info';
  }

  function severityIcon(sev) {
    var s = String(sev || '').toLowerCase();
    if (s === 'success' || s === 'good' || s === 'ok') return 'check';
    if (s === 'warn' || s === 'warning' || s === 'caution') return 'warn';
    if (s === 'error' || s === 'critical' || s === 'high') return 'error';
    return 'info';
  }

  function chartCard(title, sub, svg, wide) {
    return '<figure class="chart-card' + (wide ? ' wide' : '') + '">' +
      '<figcaption><div class="chart-title">' + esc(title) + '</div>' +
      (sub ? '<div class="chart-sub">' + esc(sub) + '</div>' : '') + '</figcaption>' +
      '<div class="chart-body">' + svg + '</div></figure>';
  }

  function renderResults(run) {
    var host = $('#results');
    if (!host) return;
    var C = window.Charts;

    var uc = get(run, 'usecase', {}) || {};
    var ds = get(run, 'dataset', {}) || {};
    var plan = get(run, 'plan', {}) || {};
    var champion = get(run, 'champion', {}) || {};
    var impact = get(run, 'impact', null);
    var metrics = get(run, 'metrics', {}) || {};
    var drivers = arr(get(run, 'drivers', []));
    var insights = arr(get(run, 'insights', []));
    var narration = arr(get(run, 'narration', []));
    var distributions = get(run, 'distributions', {}) || {};
    var preview = get(run, 'preview', null);
    var leaderboard = arr(get(run, 'leaderboard', []));
    lastExplanations = arr(get(run, 'explanations', []));

    var sub = $('#step-6-sub');
    if (sub) {
      sub.textContent = String(get(uc, 'name', 'Your analysis')) + ' \u00b7 ' +
        fmtInt(get(ds, 'rows', null)) + ' rows scanned \u00b7 finished in ' +
        fmtMs(get(run, 'duration_ms', null)) + '. Everything below was computed on this computer.';
    }

    var html = '';

    /* ---- run meta chips ---- */
    html += '<div class="run-meta">' +
      '<span class="meta-chip">Goal: <strong>' + esc(get(uc, 'name', '\u2014')) + '</strong></span>' +
      '<span class="meta-chip">Rows scored: <strong>' + esc(fmtInt(get(ds, 'scored_rows', get(ds, 'rows', null)))) + '</strong></span>' +
      '<span class="meta-chip">Models tried: <strong>' + esc(fmtInt(get(plan, 'model_count', leaderboard.length))) + '</strong></span>' +
      '<span class="meta-chip">How we tested: <strong>' + esc(titleCase(get(plan, 'test_strategy', 'hold-out'))) + '</strong></span>' +
      (get(plan, 'sample_reason', null) ? '<span class="meta-chip">' + esc(String(get(plan, 'sample_reason'))) + '</span>' : '') +
      '</div>';

    /* ---- champion ---- */
    if (Object.keys(champion).length) {
      var tradeoff = get(champion, 'tradeoff', '');
      html += '<section class="champion">' +
        '<div class="champion-head"><span class="champion-crown">' + icon('crown', 18) + '</span>' +
        '<span class="kicker">Champion</span></div>' +
        '<h2 class="champion-name">' + esc(get(champion, 'display', get(champion, 'key', 'The winner'))) + '</h2>' +
        (get(champion, 'persona', '') ? '<p class="champion-persona">' + esc(champion.persona) + '</p>' : '') +
        (get(champion, 'why_won', '') ? '<p class="champion-why">' + esc(champion.why_won) + '</p>' : '') +
        (get(champion, 'plain_summary', '') ? '<p class="champion-plain">' + esc(champion.plain_summary) + '</p>' : '') +
        (tradeoff ? '<div class="champion-tradeoff"><span class="to-ico">' + icon('warn', 15) + '</span>' +
          '<span><strong>Worth knowing: </strong>' + esc(tradeoff) + '</span></div>' : '') +
        '</section>';
    }

    /* ---- impact ---- */
    if (impact && typeof impact === 'object' && get(impact, 'value', null) !== null) {
      var unit = String(get(impact, 'unit', '') || '');
      var unitLow = unit.toLowerCase();
      var unitText = unitLow === 'currency' ? "in your data's currency" : unit;
      var showUnit = unitLow && unitLow !== 'none';
      html += '<section class="impact">' +
        '<div>' +
          '<div class="impact-num">' + esc(formatImpact(get(impact, 'value', null), unit)) +
            (showUnit ? '<span style="font-size:0.45em;font-weight:600;color:var(--muted);letter-spacing:0"> ' +
              esc(unitText) + '</span>' : '') +
          '</div>' +
          '<div class="impact-label">' + esc(get(impact, 'label', 'Estimated impact')) + '</div>' +
        '</div>' +
        '<div>' +
          (get(impact, 'note', '') ? '<p class="impact-note">' + esc(impact.note) + '</p>' : '') +
          (get(impact, 'formula', '') ? '<p class="impact-formula">' + esc(impact.formula) + '</p>' : '') +
        '</div>' +
        '</section>';
    }

    /* ---- metric cards ---- */
    var metricKeys = Object.keys(metrics).filter(function (k) {
      var v = metrics[k];
      return v !== null && v !== undefined && typeof v !== 'object';
    });
    if (metricKeys.length) {
      var order = ['roc_auc', 'r2', 'silhouette', 'accuracy', 'precision', 'recall', 'f1', 'pr_auc',
        'rmse', 'mae', 'mape', 'wape', 'log_loss', 'davies_bouldin', 'n_clusters', 'contamination', 'agreement', 'consensus'];
      metricKeys.sort(function (a, b) {
        var ia = order.indexOf(a), ib = order.indexOf(b);
        if (ia < 0) ia = 999;
        if (ib < 0) ib = 999;
        return ia - ib;
      });
      var cards = metricKeys.slice(0, 8).map(function (k, i) {
        var info = metricInfo(k);
        var v = metrics[k];
        var shown = typeof v === 'number' ? (Math.abs(v) >= 1000 ? fmtNum(v, 0) : fmtNum(v, 4)) : String(v);
        return '<div class="metric-card' + (i === 0 ? ' is-primary' : '') + '" title="' + esc(info.tip) + '">' +
          '<div class="metric-value">' + esc(shown) + '</div>' +
          '<div class="metric-label">' + esc(info.label) + '</div>' +
          '<div class="metric-raw">' + esc(humanMetric(k)) + '</div>' +
          '</div>';
      }).join('');
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">How well did it do?</h2>' +
        '<p class="section-sub">Hover any card for a plain-English explanation.</p></div></div>' +
        '<div class="metric-grid">' + cards + '</div></section>';
    }

    /* ---- charts ---- */
    var chartBits = [];
    if (C) {
      if (distributions.score_histogram) {
        chartBits.push(chartCard('How the scores are spread',
          'Most rows sit on the left. The ones on the right are the ones worth your time.',
          C.histogram({
            bins: arr(get(distributions, 'score_histogram.bins', [])),
            counts: arr(get(distributions, 'score_histogram.counts', [])),
            x_label: 'Score', height: 250
          })));
      }
      if (distributions.class_balance) {
        chartBits.push(chartCard('How the two groups compare',
          'The balance between ordinary rows and the ones we were hunting for.',
          C.donut({
            labels: arr(get(distributions, 'class_balance.labels', [])),
            values: arr(get(distributions, 'class_balance.values', [])),
            centerLabel: 'rows',
            colors: ['#5B8DEF', '#EF5350', '#F5C451'],
            height: 230
          })));
      }
      if (distributions.leaderboard) {
        chartBits.push(chartCard('How the models ranked',
          'Every method was judged on the same held-back data. Longer bar means better.',
          C.hbar({
            labels: arr(get(distributions, 'leaderboard.labels', [])),
            values: arr(get(distributions, 'leaderboard.values', [])),
            height: 250
          })));
      }
      if (distributions.feature_impact) {
        chartBits.push(chartCard('What moved the needle',
          'The columns the winning model leaned on most.',
          C.hbar({
            labels: arr(get(distributions, 'feature_impact.labels', [])),
            values: arr(get(distributions, 'feature_impact.values', [])),
            height: 250
          })));
      }
      if (distributions.segments) {
        chartBits.push(chartCard('How the groups break down',
          'How many records landed in each group the model found.',
          C.hbar({
            labels: arr(get(distributions, 'segments.labels', [])),
            values: arr(get(distributions, 'segments.values', [])),
            height: 240
          })));
      }
      if (distributions.metric_radar) {
        chartBits.push(chartCard('Every model across every measure',
          'A bigger shape means a stronger all-round model.',
          C.radar({
            axes: arr(get(distributions, 'metric_radar.axes', [])),
            series: arr(get(distributions, 'metric_radar.series', [])),
            height: 340
          }), true));
      }
      if (distributions.scatter) {
        chartBits.push(chartCard(
          String(get(distributions, 'scatter.y_label', 'Predicted')) + ' vs ' + String(get(distributions, 'scatter.x_label', 'actual')),
          'Points close to a diagonal line mean accurate predictions. Outliers sit far from the crowd.',
          C.scatter({
            x: arr(get(distributions, 'scatter.x', [])),
            y: arr(get(distributions, 'scatter.y', [])),
            color: arr(get(distributions, 'scatter.color', [])),
            x_label: String(get(distributions, 'scatter.x_label', 'actual')),
            y_label: String(get(distributions, 'scatter.y_label', 'predicted')),
            height: 320
          }), true));
      }
      if (distributions.timeseries) {
        chartBits.push(chartCard('What happened vs what we expected',
          'The solid line is what really happened; the dashed one is what the model expected.',
          C.line({
            x: arr(get(distributions, 'timeseries.x', [])),
            series: [
              { name: 'Actually happened', values: arr(get(distributions, 'timeseries.actual', [])), color: '#5B8DEF' },
              { name: 'Model expected', values: arr(get(distributions, 'timeseries.predicted', [])), color: '#F5C451' }
            ],
            height: 300
          }), true));
      }
    }
    if (chartBits.length) {
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">The picture</h2>' +
        '<p class="section-sub">All charts are drawn locally as plain shapes \u2014 nothing is downloaded.</p></div></div>' +
        '<div class="chart-grid">' + chartBits.join('') + '</div></section>';
    }

    /* ---- leaderboard table ---- */
    if (leaderboard.length) {
      var maxP = 0;
      leaderboard.forEach(function (e) { var v = Math.abs(toNum(get(e, 'primary', 0)) || 0); if (v > maxP) maxP = v; });
      var lbRows = leaderboard.map(function (e) {
        var isChamp = state.championKey && String(get(e, 'key', '')) === state.championKey;
        var status = String(get(e, 'status', 'ok'));
        var p = toNum(get(e, 'primary', null));
        var pct = toNum(get(e, 'score', null));
        if (pct === null) pct = maxP > 0 && p !== null ? (Math.abs(p) / maxP) * 100 : 0;
        return '<tr' + (isChamp ? ' class="is-champion"' : '') + '>' +
          '<td class="num">' + esc(get(e, 'rank', '')) + '</td>' +
          '<td><div class="m-name">' + (isChamp ? '<span class="m-crown">' + icon('crown', 15) + '</span>' : '') +
            esc(get(e, 'display', get(e, 'key', ''))) + '</div>' +
            (get(e, 'persona', '') ? '<div class="m-persona">' + esc(e.persona) + '</div>' : '') + '</td>' +
          '<td class="num">' + esc(p !== null ? fmtNum(p, 4) : '\u2014') + '</td>' +
          '<td><span class="score-cell"><span class="score-track"><span class="score-fill" style="width:' +
            clamp(pct, 0, 100).toFixed(1) + '%"></span></span></span></td>' +
          '<td class="nowrap" style="color:var(--muted)">' + esc(fmtMs(get(e, 'train_ms', null))) + '</td>' +
          '<td>' + (status === 'failed'
            ? '<span class="badge badge-fail">Did not finish</span>'
            : '<span class="badge badge-ok">Finished</span>') + '</td>' +
          '<td style="color:var(--muted);max-width:320px">' + esc(get(e, 'why', '')) + '</td>' +
          '</tr>';
      }).join('');
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">The full scoreboard</h2>' +
        '<p class="section-sub">Every model that took part, including the ones that did badly \u2014 that is how you know the winner earned it.</p></div></div>' +
        '<div class="table-scroll"><table class="data"><thead><tr>' +
        '<th class="num">#</th><th>Model</th><th class="num">Score</th><th>Relative</th><th>Train time</th><th>Status</th><th>What happened</th>' +
        '</tr></thead><tbody>' + lbRows + '</tbody></table></div></section>';
    }

    /* ---- drivers ---- */
    if (drivers.length) {
      var maxImp = 0;
      drivers.forEach(function (d) { var v = Math.abs(toNum(get(d, 'importance', 0)) || 0); if (v > maxImp) maxImp = v; });
      var dRows = drivers.map(function (d) {
        var imp = Math.abs(toNum(get(d, 'importance', 0)) || 0);
        var pct = maxImp > 0 ? (imp / maxImp) * 100 : 0;
        var dir = String(get(d, 'direction', '')).toLowerCase();
        var dirCls = dir.indexOf('up') >= 0 || dir.indexOf('high') >= 0 || dir.indexOf('positive') >= 0 ? 'dir-up'
          : dir.indexOf('down') >= 0 || dir.indexOf('low') >= 0 || dir.indexOf('negative') >= 0 ? 'dir-down' : 'dir-flat';
        var dirIco = dirCls === 'dir-up' ? 'arrowRight' : dirCls === 'dir-down' ? 'arrowLeft' : 'dot';
        return '<tr>' +
          '<td class="cell-name">' + esc(get(d, 'feature', '')) + '</td>' +
          '<td><span class="driver-bar"><span class="driver-track"><span class="driver-fill" style="width:' +
            clamp(pct, 0, 100).toFixed(1) + '%"></span></span>' +
            '<span class="driver-pct">' + Math.round(pct) + '</span></span></td>' +
          '<td class="nowrap ' + dirCls + '">' + icon(dirIco, 14) + ' ' + esc(get(d, 'direction', '\u2014')) + '</td>' +
          '<td style="color:var(--muted);max-width:420px">' + esc(get(d, 'plain', '')) + '</td>' +
          '</tr>';
      }).join('');
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">Key drivers</h2>' +
        '<p class="section-sub">What actually moves the number you care about, in order of importance.</p></div></div>' +
        '<div class="table-scroll"><table class="data"><thead><tr>' +
        '<th>Column</th><th>How much it matters</th><th>Direction</th><th>In plain words</th>' +
        '</tr></thead><tbody>' + dRows + '</tbody></table></div></section>';
    }

    /* ---- insights ---- */
    if (insights.length) {
      var iCards = insights.map(function (ins) {
        var sev = get(ins, 'severity', 'info');
        var ic = get(ins, 'icon', null) || severityIcon(sev);
        var icoName = ICONS[ic] ? ic : severityIcon(sev);
        return '<div class="insight ' + severityClass(sev) + '">' +
          '<span class="in-ico">' + icon(icoName, 17) + '</span>' +
          '<span><div class="insight-title">' + esc(get(ins, 'title', 'Insight')) + '</div>' +
          '<div class="insight-body">' + esc(get(ins, 'body', '')) + '</div></span>' +
          '</div>';
      }).join('');
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">What we noticed</h2>' +
        '<p class="section-sub">Things worth acting on, in plain language.</p></div></div>' +
        '<div class="insight-grid">' + iCards + '</div></section>';
    }

    /* ---- segments ---- */
    var segments = arr(get(run, 'segments', []));
    if (segments.length) {
      var sRows = segments.map(function (s) {
        if (s && typeof s === 'object' && !Array.isArray(s)) {
          var keys = Object.keys(s);
          return '<tr>' + keys.map(function (k, i) {
            var v = s[k];
            return '<td' + (typeof v === 'number' ? ' class="num"' : '') + (i === 0 ? ' class="cell-name"' : '') + '>' +
              esc(typeof v === 'number' ? fmtNum(v, 3) : (v === null || v === undefined ? '\u2014' : String(v))) + '</td>';
          }).join('') + '</tr>';
        }
        return '<tr><td>' + esc(String(s)) + '</td></tr>';
      }).join('');
      var sCols = (segments[0] && typeof segments[0] === 'object' && !Array.isArray(segments[0]))
        ? Object.keys(segments[0]) : ['Group'];
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">The groups we found</h2>' +
        '<p class="section-sub">Records that behave alike, grouped together.</p></div></div>' +
        '<div class="table-scroll"><table class="data"><thead><tr>' +
        sCols.map(function (c) { return '<th>' + esc(humanMetric(c)) + '</th>'; }).join('') +
        '</tr></thead><tbody>' + sRows + '</tbody></table></div></section>';
    }

    /* ---- top flagged rows ---- */
    var pCols = arr(get(preview, 'columns', []));
    var pRows = arr(get(preview, 'rows', []));
    if (pCols.length && pRows.length) {
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">The rows to look at first</h2>' +
        '<p class="section-sub">The highest-scoring records \u2014 start here. The full list is in the exports below.</p></div></div>' +
        '<div class="table-scroll"><table class="data"><thead><tr>' +
        pCols.map(function (c, i) { return '<th' + (i > 0 && typeof pRows[0][i] === 'number' ? ' class="num"' : '') + '>' + esc(c) + '</th>'; }).join('') +
        '</tr></thead><tbody>' +
        pRows.slice(0, 50).map(function (r) {
          return '<tr>' + arr(r).map(function (v, i) {
            return '<td' + (i > 0 && typeof v === 'number' ? ' class="num"' : '') + '>' +
              esc(v === null || v === undefined ? '\u2014' : (typeof v === 'number' ? fmtNum(v, 3) : String(v))) + '</td>';
          }).join('') + '</tr>';
        }).join('') + '</tbody></table></div></section>';
    }

    /* ---- why these rows? ---- */
    if (lastExplanations.length) {
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">Why these rows?</h2>' +
        '<p class="section-sub">For each of the highest-scoring records, the handful of things about it ' +
        'that pushed it up the list. Every line is a plain difference you can check against the record yourself.</p>' +
        '</div></div>' +
        '<div class="why-list">' +
        lastExplanations.slice(0, 12).map(function (e, i) {
          var whyDrivers = arr(get(e, 'drivers', []));
          return '<div class="why-card">' +
            '<div class="why-head">' +
              '<span class="why-label">' + esc(get(e, 'label', 'record')) + '</span>' +
              '<span class="why-score">' + esc(fmtNum(get(e, 'score', null), 2)) + '</span>' +
            '</div>' +
            '<p class="why-summary">' + esc(get(e, 'summary', '')) + '</p>' +
            '<ul class="why-drivers">' +
            whyDrivers.slice(0, 3).map(function (d) {
              var c = toNum(get(d, 'contribution', 0)) || 0;
              return '<li class="why-driver ' + (c > 0 ? 'up' : 'down') + '">' +
                '<span class="why-arrow" aria-hidden="true">' + (c > 0 ? '\u25b2' : '\u25bc') + '</span>' +
                '<span class="why-text">' + esc(get(d, 'plain', '')) + '</span></li>';
            }).join('') +
            '</ul>' +
            (whyDrivers.length > 3
              ? '<button type="button" class="why-more" data-why="' + i + '">' +
                'Show all ' + whyDrivers.length + ' things we checked</button>'
              : '') +
            '</div>';
        }).join('') +
        '</div></section>';
    }

    /* ---- story ---- */
    if (narration.length) {
      html += '<section class="section">' +
        '<div class="section-head"><div><h2 class="section-title">The story so far</h2>' +
        '<p class="section-sub">What we did, in the order we did it \u2014 no jargon.</p></div></div>' +
        '<div class="story">' + narration.map(function (b, i) {
          return '<div class="story-item">' +
            '<span class="story-num">' + (i + 1) + '</span>' +
            '<span>' +
              (get(b, 'beat', '') ? '<div class="story-beat">' + esc(b.beat) + '</div>' : '') +
              '<div class="story-title">' + esc(get(b, 'title', 'Step ' + (i + 1))) + '</div>' +
              '<div class="story-text">' + esc(get(b, 'text', '')) + '</div>' +
            '</span></div>';
        }).join('') + '</div></section>';
    }

    /* ---- exports ---- */
    html += '<section class="section"><div class="section-head"><div>' +
      '<h2 class="section-title">Take it with you</h2>' +
      '<p class="section-sub">Reports and data files are generated on this machine.</p></div></div>' +
      '<div class="export-bar">' +
        '<button type="button" class="btn btn-primary" data-export="html">' + icon('download', 15) + ' HTML report</button>' +
        '<button type="button" class="btn btn-soft" data-export="csv">' + icon('table', 15) + ' CSV of scored rows</button>' +
        '<button type="button" class="btn btn-soft" data-export="xlsx">' + icon('table', 15) + ' Excel workbook</button>' +
        '<button type="button" class="btn btn-soft" data-export="json">' + icon('db', 15) + ' JSON summary</button>' +
        '<span class="export-note">Opens in a new download \u2014 nothing is uploaded.</span>' +
      '</div></section>';

    host.innerHTML = html;
  }

  /* ============================================================== modals == */

  var lastFocused = null;
  var lastExplanations = [];

  function openWhy(index) {
    var e = lastExplanations[index];
    if (!e) return;
    var drivers = arr(get(e, 'drivers', []));
    var body = '<p class="why-summary">' + esc(get(e, 'summary', '')) + '</p>' +
      '<ul class="why-drivers">' + drivers.map(function (d) {
        var c = toNum(get(d, 'contribution', 0)) || 0;
        return '<li class="why-driver ' + (c > 0 ? 'up' : 'down') + '">' +
          '<span class="why-arrow" aria-hidden="true">' + (c > 0 ? '\u25b2' : '\u25bc') + '</span>' +
          '<span class="why-text">' + esc(get(d, 'plain', '')) + '</span></li>';
      }).join('') + '</ul>' +
      '<p class="why-foot">We worked this out by taking the record and swapping one column at a ' +
      'time for the value a typical record has, then seeing how far the score moved. Whatever ' +
      'moves it most is what drove the decision. Columns that overlap in meaning can each look ' +
      'important on their own, so read these as strong hints rather than an exact split.</p>';
    openModal('Why ' + get(e, 'label', 'this record') + ' was flagged', body);
  }

  function openModal(title, bodyHtml) {
    var modal = $('#modal');
    if (!modal) return;
    lastFocused = document.activeElement;
    $('#modal-title').textContent = title;
    $('#modal-body').innerHTML = bodyHtml;
    modal.hidden = false;
    var closeBtn = $('#modal-close');
    if (closeBtn) closeBtn.focus();
  }

  function closeModal() {
    var modal = $('#modal');
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    if (lastFocused && lastFocused.focus) { try { lastFocused.focus(); } catch (e) { /* noop */ } }
  }

  function showFunctionReference() {
    var fns = arr(state.functions);
    var ops = arr(state.operators);
    var body = '';
    if (fns.length) {
      body += '<h3 class="sub-title">Functions</h3>' + fns.map(function (f) {
        return '<div class="fn-row">' +
          '<div class="fn-sig">' + esc(get(f, 'signature', get(f, 'name', ''))) + '</div>' +
          '<div><div class="fn-desc">' + esc(get(f, 'description', '')) + '</div>' +
          (get(f, 'example', '') ? '<div class="fn-ex">e.g. ' + esc(f.example) + '</div>' : '') + '</div>' +
          '</div>';
      }).join('');
    }
    if (ops.length) {
      body += '<h3 class="sub-title" style="margin-top:20px">Operators</h3>' + ops.map(function (o) {
        return '<div class="fn-row">' +
          '<div class="fn-sig">' + esc(get(o, 'symbol', get(o, 'name', ''))) + '</div>' +
          '<div><div class="fn-desc">' + esc(get(o, 'description', '')) + '</div></div>' +
          '</div>';
      }).join('');
    }
    if (!body) {
      body = '<p class="muted">The engine did not send a function list. You can still use the basic operators ' +
        '<span class="mono">+ - * / ( )</span> and plain column names.</p>';
    }
    openModal('Function reference', body);
  }

  /* ================================================================ wiring == */

  function bindGlobal() {
    document.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || !t.closest) return;

      var gotoBtn = t.closest('[data-goto]');
      if (gotoBtn && !gotoBtn.disabled) { go(Number(gotoBtn.getAttribute('data-goto'))); return; }

      var sampleBtn = t.closest('[data-sample]');
      if (sampleBtn) { useSample(sampleBtn.getAttribute('data-sample'), sampleBtn); return; }

      var ucBtn = t.closest('[data-usecase]');
      if (ucBtn) { selectUseCase(ucBtn.getAttribute('data-usecase')); return; }

      var whyBtn = t.closest('[data-why]');
      if (whyBtn) { openWhy(Number(whyBtn.getAttribute('data-why'))); return; }

      var sugBtn = t.closest('[data-suggest]');
      if (sugBtn) {
        var idx = Number(sugBtn.getAttribute('data-suggest'));
        var s = arr(state.suggestions)[idx];
        if (s) {
          var nameInput = $('#derived-name');
          var exprInput = $('#derived-expr');
          if (nameInput) nameInput.value = String(get(s, 'name', ''));
          if (exprInput) exprInput.value = String(get(s, 'expression', ''));
          previewDerived(String(get(s, 'name', '')), String(get(s, 'expression', '')));
        }
        return;
      }

      var insertBtn = t.closest('[data-insert]');
      if (insertBtn) { insertIntoExpr(insertBtn.getAttribute('data-insert')); return; }

      var unstageBtn = t.closest('[data-unstage]');
      if (unstageBtn) {
        var ui = Number(unstageBtn.getAttribute('data-unstage'));
        state.staged.splice(ui, 1);
        renderStaged();
        return;
      }

      var makeColBtn = t.closest('[data-make-column]');
      if (makeColBtn) { jumpToBuilder(makeColBtn.getAttribute('data-make-column')); return; }

      var exportBtn = t.closest('[data-export]');
      if (exportBtn) {
        var fmt = exportBtn.getAttribute('data-export');
        if (!state.runId) { toast('No finished run to export yet.', 'warn'); return; }
        window.location.href = '/api/runs/' + encodeURIComponent(state.runId) + '/export?fmt=' + encodeURIComponent(fmt);
        toast('Your ' + fmt.toUpperCase() + ' file is being prepared for download.', 'info', 'Export started');
        return;
      }

      if (t.closest('#btn-fn-ref')) { showFunctionReference(); return; }
      if (t.closest('#modal-close')) { closeModal(); return; }
      if (t.closest('#btn-start-over')) { startOver(); return; }
      if (t.closest('#console-jump')) {
        var c = $('#console');
        if (c) { c.scrollTop = c.scrollHeight; state.autoScroll = true; }
        var j = $('#console-jump');
        if (j) j.hidden = true;
        return;
      }
      if (t.closest('#btn-stage')) { stageCurrent(); return; }
      if (t.closest('#btn-apply')) { applyStaged(); return; }
      if (t.closest('#btn-run')) { startRun(); return; }

      var modal = $('#modal');
      if (modal && !modal.hidden && t === modal) { closeModal(); }
    });

    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') {
        closeModal();
        return;
      }
      if (ev.key === 'Tab') {
        var modal = $('#modal');
        if (!modal || modal.hidden) return;
        var focusables = $$('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])', modal)
          .filter(function (el) { return !el.disabled && el.offsetParent !== null; });
        if (!focusables.length) return;
        var first = focusables[0], last = focusables[focusables.length - 1];
        if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last.focus(); }
        else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first.focus(); }
      }
    });

    var dz = $('#dropzone');
    var fileInput = $('#file-input');
    if (dz) {
      dz.addEventListener('click', function (ev) {
        var target = ev.target;
        // the label forwards a synthetic click to the input; don't re-open the dialog
        if (target === fileInput) return;
        if (target && target.closest && target.closest('label')) return;
        if (fileInput) fileInput.click();
      });
      ['dragenter', 'dragover'].forEach(function (name) {
        dz.addEventListener(name, function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          dz.classList.add('dragging');
        });
      });
      ['dragleave', 'drop'].forEach(function (name) {
        dz.addEventListener(name, function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          if (name === 'dragleave' && dz.contains(ev.relatedTarget)) return;
          dz.classList.remove('dragging');
        });
      });
      dz.addEventListener('drop', function (ev) {
        var files = ev.dataTransfer && ev.dataTransfer.files;
        if (files && files.length) uploadFile(files[0]);
      });
    }
    if (fileInput) {
      fileInput.addEventListener('change', function () {
        if (fileInput.files && fileInput.files.length) uploadFile(fileInput.files[0]);
        fileInput.value = '';
      });
    }

    var consoleHost = $('#console');
    if (consoleHost) {
      consoleHost.addEventListener('scroll', function () {
        var atBottom = consoleHost.scrollHeight - consoleHost.scrollTop - consoleHost.clientHeight < 60;
        state.autoScroll = atBottom;
        var jump = $('#console-jump');
        if (jump) jump.hidden = atBottom;
      }, { passive: true });
    }

    var builderForm = $('#builder-form');
    if (builderForm) {
      builderForm.addEventListener('submit', function (ev) {
        ev.preventDefault();
        var nameInput = $('#derived-name');
        var exprInput = $('#derived-expr');
        previewDerived(nameInput ? nameInput.value.trim() : '', exprInput ? exprInput.value.trim() : '');
      });
    }

    var roleHost = $('#role-pickers');
    if (roleHost) {
      roleHost.addEventListener('change', function () { readRolePickers(); });
    }

    window.addEventListener('beforeunload', function () {
      if (state.ws) { try { state.ws.close(); } catch (e) { /* noop */ } }
    });

    window.addEventListener('error', function (ev) {
      if (ev && ev.message) toast(String(ev.message).slice(0, 200), 'error', 'Unexpected problem');
    });
    window.addEventListener('unhandledrejection', function (ev) {
      var r = ev && ev.reason;
      if (r && r.name === 'ApiError') return; // already surfaced
      toast(r && r.message ? String(r.message).slice(0, 200) : 'Something unexpected happened.', 'error');
    });
  }

  function startOver() {
    stopTimers();
    if (state.ws) { try { state.ws.close(); } catch (e) { /* noop */ } state.ws = null; }
    state.datasetId = null;
    state.dataset = null;
    state.profile = null;
    state.usecaseKey = null;
    state.usecase = null;
    state.mapping = {};
    state.staged = [];
    state.runId = null;
    state.run = null;
    state.events = [];
    state.seenSeq = {};
    state.models = {};
    state.championKey = null;
    state.finished = false;
    state.streaming = false;

    var results = $('#results');
    if (results) results.innerHTML = '';
    var staged = $('#staged-box');
    if (staged) staged.hidden = true;
    hidePreview();
    var builderForm = $('#builder-form');
    if (builderForm) builderForm.reset();
    var ucHost = $('#usecase-groups');
    if (ucHost) ucHost.innerHTML = '<div class="skeleton-card tall"></div><div class="skeleton-card tall"></div>';
    var roleHost = $('#role-pickers');
    if (roleHost) roleHost.innerHTML = '';
    var dsHost = $('#dataset-summary');
    if (dsHost) dsHost.innerHTML = '';
    var colHost = $('#column-table');
    if (colHost) colHost.innerHTML = '';
    var sugHost = $('#suggestion-chips');
    if (sugHost) sugHost.innerHTML = '';
    var confirmHost = $('#role-confirm');
    if (confirmHost) confirmHost.innerHTML = '';
    var to3 = $('#to-step-3');
    if (to3) to3.disabled = false;

    go(1);
    toast('Cleared. Drop a new file whenever you are ready.', 'info', 'Ready for a new dataset');
    loadSamples();
  }

  function init() {
    try {
      state.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) { state.reducedMotion = false; }

    bindGlobal();
    loadHealth();
    loadSamples();

    api('/api/usecases', { quiet: true, silent: true })
      .then(function (res) {
        state.usecases = arr(get(res, 'usecases', []));
        state.functions = arr(get(res, 'functions', []));
        state.operators = arr(get(res, 'operators', []));
        renderUseCases();
        renderHelper();
      })
      .catch(function () {
        var host = $('#usecase-groups');
        if (host) host.innerHTML = '<p class="empty-note">Could not load the list of goals. Reload the page to try again.</p>';
      });

    go(1);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Small handle so the lead can drive the wizard from the console while wiring
  // the server up, and so tests can render a fixture without a browser.
  window.MLArena = {
    state: state,
    go: go,
    renderResults: renderResults,
    renderScoreboard: renderScoreboard,
    handleEvent: handleEvent
  };
})();
