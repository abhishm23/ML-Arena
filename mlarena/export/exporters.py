"""ML Arena - export module (contract section 13).

Four exporters turn a finished run into something a user can hand to their boss:

* :func:`export_html`          - one self-contained, offline, dark-theme report
* :func:`export_csv`           - every scored row, UTF-8 with BOM (Excel-safe)
* :func:`export_xlsx`          - a formatted multi-sheet workbook
* :func:`export_summary_json`  - the run result as JSON, minus the noisy log

``result`` may be the ``RunResult`` dataclass from ``mlarena.engine.runner`` *or*
a plain ``dict`` (the API layer serialises it).  Every field is read through
:func:`_get`, so a missing key never raises - it just renders as a dash.

The HTML report must work by double-clicking it on a laptop with no network:
there is no CDN, no external font, no external image, and no ``<script>``.  Every
chart is inline SVG built right here.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mlarena import config

__all__ = ["export_html", "export_csv", "export_xlsx", "export_summary_json"]

MAX_XLSX_ROWS = 100_000
MAX_HTML_ROWS = 100

# ---------------------------------------------------------------------------
# tiny generic helpers
# ---------------------------------------------------------------------------


def _get(result: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dataclass *or* a dict, never raising."""
    if result is None:
        return default
    if isinstance(result, Mapping):
        try:
            return result.get(key, default)
        except Exception:
            return default
    try:
        return getattr(result, key, default)
    except Exception:
        return default


def _as_dict(obj: Any) -> dict:
    """Normalise a leaderboard/driver/insight entry to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass
    try:
        return dict(vars(obj))
    except Exception:
        return {}


def _seq(value: Any) -> list:
    """Coerce something list-ish into a real list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except Exception:
        return []


def _num(value: Any, default: float | None = None) -> float | None:
    """Best-effort float conversion that tolerates None/NaN/garbage."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _esc(value: Any) -> str:
    """HTML-escape any user-supplied string.

    Also neutralises literal ``http://`` / ``https://`` substrings (rendered
    identically by the browser) so the report provably never references anything
    off the machine, even if the user's own data happens to contain a URL.
    """
    if value is None:
        return ""
    text = html.escape(str(value), quote=True)
    return text.replace("https://", "https&#58;//").replace("http://", "http&#58;//")


def _abs_path(out_path: str | Path) -> str:
    """Resolve to an absolute path, creating the parent directory if needed."""
    path = Path(out_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
DASH = "\u2014"


def _indian_group(value: float) -> str:
    """1234567 -> '12,34,567' (how rupee amounts are actually written)."""
    negative = value < 0
    digits = str(int(round(abs(value))))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        chunks: list[str] = []
        while len(head) > 2:
            chunks.insert(0, head[-2:])
            head = head[:-2]
        if head:
            chunks.insert(0, head)
        digits = ",".join(chunks + [tail])
    return ("-" if negative else "") + digits


def _fmt_money(value: float, compact: bool = True) -> str:
    """Currency-neutral formatting.

    We deliberately never print a currency symbol. The currency of the user's file
    is not something this tool can know, and guessing one (as Rs or $) would
    mislabel every money figure in the report. Large amounts get a word suffix so
    they stay readable at a glance.
    """
    if value is None:
        return DASH
    sign = "-" if value < 0 else ""
    amount = abs(float(value))
    if compact and amount >= 1e9:
        return f"{sign}{amount / 1e9:,.2f} billion"
    if compact and amount >= 1e6:
        return f"{sign}{amount / 1e6:,.2f} million"
    if compact and amount >= 1e3:
        return f"{sign}{amount / 1e3:,.1f} thousand"
    return f"{sign}{amount:,.0f}"


def _fmt_int(value: float) -> str:
    if value is None:
        return DASH
    # Standard thousands grouping. The tool is currency- and locale-agnostic, so it
    # uses the grouping that reads the same to everyone.
    return f"{float(value):,.0f}"


def _fmt_float(value: float, digits: int = 2) -> str:
    if value is None:
        return DASH
    if abs(value) >= 1000:
        return _fmt_int(value)
    return f"{value:,.{digits}f}"


def _fmt_impact(value: Any, unit: Any) -> str:
    """The big number in the impact callout."""
    if value is None:
        return DASH
    if isinstance(value, str):
        return value
    number = _num(value)
    if number is None:
        return str(value)
    unit = (unit or "").lower()
    if unit == "currency":
        return _fmt_money(number)
    if unit in ("records", "customers", "rows", "orders"):
        return _fmt_int(number)
    if abs(number) >= 1000:
        return _fmt_int(number)
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _fmt_ms(value: Any) -> str:
    number = _num(value)
    if number is None:
        return DASH
    if number >= 1000:
        return f"{number / 1000:.1f} s"
    return f"{number:.0f} ms"


def _fmt_ts(value: Any) -> str:
    """ISO timestamp -> '18 Sep 2026, 14:32' (falls back to the raw string)."""
    if not value:
        return _dt.datetime.now().strftime("%d %b %Y, %H:%M")
    text = str(value)
    try:
        parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y, %H:%M")
    except Exception:
        return text


def _fmt_metric_value(key: str, value: Any) -> str:
    number = _num(value)
    if number is None:
        return DASH
    if key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
               "silhouette", "contamination", "agreement", "r2", "davies_bouldin"):
        return f"{number:.3f}"
    if key in ("mape", "wape"):
        return f"{number * 100:.1f}%" if number <= 1.5 else f"{number:.1f}%"
    if key in ("rmse", "mae", "log_loss"):
        return f"{number:,.2f}"
    if key == "n_clusters":
        return f"{int(number)}"
    return f"{number:,.4f}"


# Plain-English names for every metric the engine can emit (contract section 10).
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "roc_auc": ("Separation power", "How well the model tells the two groups apart. 0.5 is a coin toss, 1.0 is perfect."),
    "accuracy": ("Overall hit rate", "Out of every 100 records, how many the model got right."),
    "precision": ("Flag accuracy", "Of the records it flagged, what share really were problems."),
    "recall": ("Catch rate", "Of all the real problems in the data, what share the model caught."),
    "f1": ("Balance score", "One number that balances flag accuracy against catch rate."),
    "pr_auc": ("Flag quality", "How reliable the flagged list is from the top of the list down."),
    "log_loss": ("Confidence penalty", "Lower is better. It punishes confident mistakes."),
    "r2": ("Explained variation", "How much of the ups and downs the model explains. 1.0 is perfect."),
    "mape": ("Average % error", "On average, how far off each prediction is, as a percentage. Lower is better."),
    "rmse": ("Typical error", "The typical size of the miss, in the units of what you predicted."),
    "mae": ("Average error", "The average size of the miss. Lower is better."),
    "wape": ("Weighted % error", "Total error as a share of the total actual. Lower is better."),
    "silhouette": ("Group clarity", "How cleanly separated the groups are. Higher is better."),
    "davies_bouldin": ("Group overlap", "How much the groups overlap. Lower is better."),
    "n_clusters": ("Groups found", "How many distinct groups the model settled on."),
    "contamination": ("Share flagged", "What share of records were marked unusual."),
    "agreement": ("Detector agreement", "How much the different detectors agree with each other."),
}
METRIC_CARD_ORDER = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy",
                     "r2", "mape", "wape", "rmse", "mae", "silhouette",
                     "davies_bouldin", "n_clusters", "contamination", "agreement"]


def _metric_label(key: str) -> tuple[str, str]:
    return METRIC_LABELS.get(key, (key.replace("_", " ").title(), "How the model scored on this measure."))


# ---------------------------------------------------------------------------
# inline SVG charts (no external dependency, no xmlns needed inside HTML)
# ---------------------------------------------------------------------------
def _svg_histogram(hist: Any, title: str = "Distribution of risk scores") -> str:
    data = _as_dict(hist)
    counts = [_num(c, 0.0) or 0.0 for c in _seq(data.get("counts"))]
    edges = [_num(b, 0.0) or 0.0 for b in _seq(data.get("bins"))]
    if not counts or max(counts) <= 0:
        return ""
    width, height = 640.0, 230.0
    left, right, top, bottom = 52.0, 14.0, 22.0, 40.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max(counts)
    bar_w = plot_w / len(counts)
    gap = max(1.0, bar_w * 0.10)

    parts: list[str] = []
    # horizontal grid lines
    for frac in (0.0, 0.5, 1.0):
        y = top + plot_h * (1.0 - frac)
        parts.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_w:.1f}" y2="{y:.1f}" stroke="#243044" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" fill="#8A97AC" font-size="11" text-anchor="end">'
            f'{_esc(_fmt_int(peak * frac))}</text>'
        )
    for i, count in enumerate(counts):
        bar_h = plot_h * (count / peak) if peak else 0.0
        x = left + i * bar_w + gap / 2
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - gap):.1f}" height="{max(0.0, bar_h):.1f}" '
            f'fill="#5B8DEF" rx="2"/>'
        )
    # x axis labels
    if len(edges) >= 2:
        for frac, anchor in ((0.0, "start"), (0.5, "middle"), (1.0, "end")):
            idx = min(len(edges) - 1, int(round(frac * (len(edges) - 1))))
            x = left + plot_w * frac
            parts.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" fill="#8A97AC" font-size="11" '
                f'text-anchor="{anchor}">{_esc(f"{edges[idx]:.2f}")}</text>'
            )
    parts.append(
        f'<text x="{left:.1f}" y="{top + plot_h + 34:.1f}" fill="#6C7A90" font-size="10">'
        f'low risk</text>'
    )
    parts.append(
        f'<text x="{left + plot_w:.1f}" y="{top + plot_h + 34:.1f}" fill="#6C7A90" font-size="10" '
        f'text-anchor="end">high risk</text>'
    )
    return (
        f'<figure class="chart"><figcaption>{_esc(title)}</figcaption>'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" height="auto" role="img" '
        f'aria-label="{_esc(title)}">{"".join(parts)}</svg></figure>'
    )


def _svg_leaderboard(lb: Any, title: str = "How each model scored") -> str:
    data = _as_dict(lb)
    labels = [str(x) for x in _seq(data.get("labels"))]
    values = [_num(v) for v in _seq(data.get("values"))]
    pairs = [(lab, val) for lab, val in zip(labels, values) if val is not None]
    if not pairs:
        return ""
    pairs = pairs[:12]
    row_h = 26.0
    width = 640.0
    height = 34.0 + row_h * len(pairs)
    label_w = 176.0
    value_w = 62.0
    plot_w = width - label_w - value_w - 16.0
    peak = max((abs(v) for _, v in pairs), default=1.0) or 1.0
    best = max(v for _, v in pairs)

    parts: list[str] = []
    for i, (label, value) in enumerate(pairs):
        y = 12.0 + i * row_h
        bar_w = plot_w * (abs(value) / peak)
        colour = "#2FBF71" if value == best else "#5B8DEF"
        parts.append(
            f'<text x="{label_w - 10:.1f}" y="{y + 13:.1f}" fill="#C7D2E3" font-size="12" '
            f'text-anchor="end">{_esc(label)}</text>'
        )
        parts.append(f'<rect x="{label_w:.1f}" y="{y + 3:.1f}" width="{plot_w:.1f}" height="14" fill="#1A2433" rx="3"/>')
        parts.append(
            f'<rect x="{label_w:.1f}" y="{y + 3:.1f}" width="{max(2.0, bar_w):.1f}" height="14" fill="{colour}" rx="3"/>'
        )
        parts.append(
            f'<text x="{label_w + plot_w + 10:.1f}" y="{y + 14:.1f}" fill="#E8EDF5" font-size="12">'
            f'{_esc(f"{value:.3f}")}</text>'
        )
    return (
        f'<figure class="chart"><figcaption>{_esc(title)}</figcaption>'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" height="auto" role="img" '
        f'aria-label="{_esc(title)}">{"".join(parts)}</svg></figure>'
    )


# ---------------------------------------------------------------------------
# CSS (inline, dark theme matching the app)
# ---------------------------------------------------------------------------
_CSS = """
:root{
  --bg:#0B0F17; --panel:#121A27; --panel2:#16202F; --line:#22304A;
  --text:#E8EDF5; --muted:#93A2B8; --accent:#5B8DEF; --good:#2FBF71;
  --gold:#F5C451; --bad:#EF6A6A;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text)}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px}
h1,h2,h3{line-height:1.25;margin:0}
h1{font-size:30px;letter-spacing:-.4px}
h2{font-size:19px;margin-bottom:14px}
h3{font-size:15px}
p{margin:0 0 10px}
a{color:var(--accent)}
.muted{color:var(--muted)}
.small{font-size:13px}
.brand{font-size:12px;letter-spacing:2.6px;text-transform:uppercase;color:var(--accent);font-weight:700}
.hero{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px}
.hero .sub{color:var(--muted);font-size:14px;margin-top:8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:999px;
  padding:5px 13px;font-size:12.5px;color:var(--muted)}
.chip b{color:var(--text);font-weight:600}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin-bottom:20px}
.lede{font-size:16px;color:#DCE4F0}
.impact{background:linear-gradient(135deg,#16233A 0%,#121A27 70%);border:1px solid #2B4370;
  border-radius:14px;padding:26px 28px;margin-bottom:20px}
.impact .value{font-size:44px;font-weight:700;color:var(--gold);letter-spacing:-1px;line-height:1.1}
.impact .label{font-size:15px;color:var(--text);margin-top:6px;font-weight:600}
.impact .note{font-size:13px;color:var(--muted);margin-top:8px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.metric{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:15px 16px}
.metric .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px}
.metric .v{font-size:24px;font-weight:700;margin-top:5px;color:var(--text)}
.metric .g{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.45}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.6px;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid #1A2333;vertical-align:top}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{background:#1A2433;border-radius:4px;height:9px;min-width:70px;overflow:hidden}
.bar > span{display:block;height:100%;background:var(--accent);border-radius:4px}
.bar.good > span{background:var(--good)}
.bar.gold > span{background:var(--gold)}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);background:var(--panel2)}
.tag.ok{color:var(--good);border-color:#1F5C3C}
.tag.failed{color:var(--bad);border-color:#5C2626}
.rank{color:var(--muted);font-variant-numeric:tabular-nums}
.champ{background:linear-gradient(135deg,#132A21 0%,#121A27 75%);border-color:#1F5C3C}
.champ .who{font-size:22px;font-weight:700;color:var(--good)}
.kv{margin:0}
.kv dt{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.6px;margin-top:14px}
.kv dd{margin:4px 0 0}
.insight{border-left:3px solid var(--accent);background:var(--panel2);border-radius:0 10px 10px 0;
  padding:14px 16px;margin-bottom:12px}
.insight.high{border-left-color:var(--bad)}
.insight.medium{border-left-color:var(--gold)}
.insight.low{border-left-color:var(--good)}
.insight .t{font-weight:600;margin-bottom:4px}
.insight .b{color:#C7D2E3;font-size:14px}
ol.story{margin:0;padding-left:0;list-style:none;counter-reset:beat}
ol.story li{counter-increment:beat;position:relative;padding-left:44px;padding-bottom:16px;
  border-left:1px solid var(--line);margin-left:14px}
ol.story li:last-child{border-left-color:transparent;padding-bottom:0}
ol.story li:before{content:counter(beat);position:absolute;left:-15px;top:0;width:28px;height:28px;
  border-radius:50%;background:var(--panel2);border:1px solid var(--line);color:var(--accent);
  font-size:12.5px;font-weight:700;display:flex;align-items:center;justify-content:center}
ol.story .bt{font-weight:600;margin-bottom:3px}
ol.story .bx{color:var(--muted);font-size:14px}
.chart{margin:0;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 16px 8px}
.chart figcaption{font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid2 > .card{margin-bottom:0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:11px}
.scroll table{font-size:12.5px}
.scroll th{position:sticky;top:0;background:var(--panel2)}
footer{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:18px;margin-top:30px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}.wrap{padding:24px 14px 48px}.impact .value{font-size:34px}}
@media print{
  :root{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7fa;--line:#d6dce6;--text:#111827;--muted:#5b6675}
  body{background:#fff;color:#111827}
  .card,.impact,.chart,.insight{break-inside:avoid;box-shadow:none}
  .bar{background:#e5e9f0}
}
"""


# ---------------------------------------------------------------------------
# narrative helpers
# ---------------------------------------------------------------------------
def _executive_summary(result: Any) -> str:
    """A 3-5 sentence paragraph a CEO could read and act on."""
    dataset = _as_dict(_get(result, "dataset"))
    usecase = _as_dict(_get(result, "usecase"))
    champion = _as_dict(_get(result, "champion"))
    impact = _as_dict(_get(result, "impact"))
    drivers = [_as_dict(d) for d in _seq(_get(result, "drivers"))]
    insights = [_as_dict(i) for i in _seq(_get(result, "insights"))]
    target = _as_dict(_get(result, "target"))

    sentences: list[str] = []

    rows = _num(dataset.get("rows"))
    question = usecase.get("business_question") or usecase.get("name")
    if rows:
        sentences.append(
            f"We analysed {_fmt_int(rows)} records to answer one question: "
            f"{str(question).rstrip('?') if question else 'what is really driving this number'}."
        )
    elif question:
        sentences.append(f"We set out to answer one question: {str(question).rstrip('?')}.")

    display = champion.get("display") or champion.get("key")
    if display:
        persona = champion.get("persona")
        reason = champion.get("why_won") or champion.get("plain_summary") or champion.get("why")
        line = f"The winning approach was {display}"
        if persona:
            line += f" ({persona})"
        line += "."
        if reason:
            line += f" {str(reason).rstrip('.')}."
        sentences.append(line)

    primary = _num(champion.get("primary"))
    primary_key = str(usecase.get("primary_metric") or "")
    if primary is not None:
        label, gloss = _metric_label(primary_key) if primary_key else ("score", "")
        sentence = f"It scored {_fmt_metric_value(primary_key, primary)} on {label.lower()}"
        sentence += f" ({gloss.rstrip('.')})." if gloss else "."
        sentences.append(sentence)

    if drivers:
        top = drivers[0]
        feature = top.get("feature") or top.get("name")
        plain = top.get("plain")
        if feature:
            sentences.append(
                f"The single strongest signal was {feature}"
                + (f": {str(plain).rstrip('.')}." if plain else ".")
            )

    impact_label = impact.get("label")
    impact_value = _fmt_impact(impact.get("value"), impact.get("unit"))
    if impact_label and impact_value != DASH:
        sentences.append(f"Put in money terms, {str(impact_label).lower()} comes to {impact_value}.")

    if len(sentences) < 3 and insights:
        first = insights[0]
        if first.get("title"):
            sentences.append(f"{str(first['title']).rstrip('.')}.")

    if not sentences:
        sentences.append(
            "This run did not produce a champion model, so there is no recommendation to report. "
            "Check the run log for the reason."
        )
    return " ".join(sentences[:5])


def _confusion_sentence(result: Any) -> str:
    """Contract section 10 asks for an explicit 'review the top N' sentence."""
    impact = _as_dict(_get(result, "impact"))
    note = impact.get("note")
    if note:
        return str(note)
    metrics = _as_dict(_get(result, "metrics"))
    matrix = _as_dict(metrics.get("confusion_matrix")) or _as_dict(metrics.get("all_rows"))
    if matrix:
        tn, fp, fn, tp = (matrix.get(k) for k in ("tn", "fp", "fn", "tp"))
        if all(isinstance(v, (int, float)) for v in (tn, fp, fn, tp)):
            total_pos = (tp or 0) + (fn or 0)
            caught = (tp or 0) / total_pos if total_pos else 0.0
            return (
                f"Of the {_fmt_int(total_pos)} real cases in the test set, the model caught "
                f"{caught * 100:.0f}% and raised {_fmt_int(fp or 0)} false alarms."
            )
    return ""


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------
def _render_metric_cards(champion: dict, usecase: dict) -> str:
    metrics = _as_dict(champion.get("metrics"))
    primary_key = str(usecase.get("primary_metric") or "")
    ordered: list[str] = []
    if primary_key and primary_key in metrics:
        ordered.append(primary_key)
    for key in METRIC_CARD_ORDER:
        if key in metrics and key not in ordered:
            ordered.append(key)
    for key in metrics:
        if key not in ordered:
            ordered.append(key)

    cards: list[str] = []
    for key in ordered[:6]:
        label, gloss = _metric_label(key)
        cards.append(
            f'<div class="metric"><div class="k">{_esc(label)}</div>'
            f'<div class="v">{_esc(_fmt_metric_value(key, metrics[key]))}</div>'
            f'<div class="g">{_esc(gloss)}</div></div>'
        )
    train_ms = _num(champion.get("train_ms"))
    if train_ms is not None:
        cards.append(
            '<div class="metric"><div class="k">Training time</div>'
            f'<div class="v">{_esc(_fmt_ms(train_ms))}</div>'
            '<div class="g">How long this model took to learn from your data.</div></div>'
        )
    if not cards:
        return '<p class="muted">No metrics were reported for this run.</p>'
    return f'<div class="metrics">{"".join(cards)}</div>'


def _render_leaderboard(leaderboard: list[dict], primary_metric: str = "") -> str:
    if not leaderboard:
        return '<p class="muted">No models were run.</p>'
    rows: list[str] = []
    for i, raw in enumerate(leaderboard):
        entry = _as_dict(raw)
        rank = entry.get("rank") or (i + 1)
        display = entry.get("display") or entry.get("key") or DASH
        persona = entry.get("persona") or ""
        status = str(entry.get("status") or "ok").lower()
        score = _num(entry.get("score"))
        primary = _num(entry.get("primary"))
        metric_key = str(entry.get("primary_metric") or primary_metric or "")
        why = entry.get("why") or entry.get("error") or ""
        train = _fmt_ms(entry.get("train_ms"))

        if status == "failed":
            score_cell = '<span class="tag failed">did not finish</span>'
            metric_cell = DASH
        else:
            bar_class = "bar good" if i == 0 else "bar"
            width = max(0.0, min(100.0, score if score is not None else 0.0))
            score_cell = f'<div class="{bar_class}"><span style="width:{width:.1f}%"></span></div>'
            metric_cell = _esc(_fmt_metric_value(metric_key, primary)) if primary is not None else DASH

        rows.append(
            "<tr>"
            f'<td class="rank">{_esc(rank)}</td>'
            f"<td><b>{_esc(display)}</b>"
            + (f'<div class="small muted">{_esc(persona)}</div>' if persona else "")
            + "</td>"
            f'<td class="num">{metric_cell}</td>'
            f'<td style="min-width:110px">{score_cell}</td>'
            f'<td class="num">{_esc(train)}</td>'
            f'<td class="small">{_esc(why)}</td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>#</th><th>Model</th><th>Score</th><th>Relative</th><th>Train time</th><th>What happened</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _render_drivers(drivers: list[dict]) -> str:
    if not drivers:
        return '<p class="muted">No key drivers were reported for this run.</p>'
    values = [abs(_num(_as_dict(d).get("importance"), 0.0) or 0.0) for d in drivers]
    peak = max(values) if values else 0.0
    rows: list[str] = []
    for i, raw in enumerate(drivers):
        d = _as_dict(raw)
        feature = d.get("feature") or d.get("name") or DASH
        importance = _num(d.get("importance"))
        direction = d.get("direction") or ""
        plain = d.get("plain") or ""
        width = (abs(importance or 0.0) / peak * 100.0) if peak else 0.0
        rows.append(
            "<tr>"
            f'<td class="rank">{i + 1}</td>'
            f"<td><b>{_esc(feature)}</b>"
            + (f'<div class="small muted">{_esc(direction)}</div>' if direction else "")
            + "</td>"
            f'<td style="min-width:130px"><div class="bar gold"><span style="width:{width:.1f}%"></span></div></td>'
            f'<td class="num">{_esc(_fmt_float(importance, 3) if importance is not None else DASH)}</td>'
            f'<td class="small">{_esc(plain)}</td>'
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>#</th><th>What drives it</th><th>Strength</th><th>Score</th><th>In plain words</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _render_insights(insights: list[dict]) -> str:
    if not insights:
        return '<p class="muted">No extra insights were reported for this run.</p>'
    out: list[str] = []
    for raw in insights:
        item = _as_dict(raw)
        severity = str(item.get("severity") or "info").lower()
        css = severity if severity in ("high", "medium", "low") else ""
        icon = item.get("icon") or ""
        title = item.get("title") or "Finding"
        body = item.get("body") or ""
        out.append(
            f'<div class="insight {css}"><div class="t">{_esc(icon)} {_esc(title)}</div>'
            f'<div class="b">{_esc(body)}</div></div>'
        )
    return "".join(out)


def _render_story(narration: list[dict]) -> str:
    if not narration:
        return '<p class="muted">No narration was recorded for this run.</p>'
    out: list[str] = []
    for raw in narration:
        beat = _as_dict(raw)
        title = beat.get("title") or beat.get("beat") or "Step"
        text = beat.get("text") or ""
        out.append(
            f'<li><div class="bt">{_esc(title)}</div>'
            + (f'<div class="bx">{_esc(text)}</div>' if text else "")
            + "</li>"
        )
    return f'<ol class="story">{"".join(out)}</ol>'


def _render_explanations(explanations: list) -> str:
    """One card per high-scoring record: why the model picked it out."""
    blocks = []
    for entry in explanations[:12]:
        item = _as_dict(entry)
        label = _esc(item.get("label") or "record")
        score = _fmt_float(_num(item.get("score")), 2)
        rows_html = []
        for driver in _seq(item.get("drivers"))[:5]:
            d = _as_dict(driver)
            contribution = _num(d.get("contribution"), 0.0) or 0.0
            colour = "#EF5350" if contribution > 0 else "#2FBF71"
            arrow = "&#9650;" if contribution > 0 else "&#9660;"
            rows_html.append(
                "<li style='margin:5px 0;font-size:13px;line-height:1.55'>"
                f"<span style='color:{colour};font-size:10px'>{arrow}</span> "
                f"{_esc(d.get('plain', ''))}</li>"
            )
        blocks.append(
            "<div style='border:1px solid #243044;border-radius:12px;"
            "padding:12px 14px;margin-bottom:10px'>"
            "<div style='display:flex;justify-content:space-between;gap:12px;margin-bottom:6px'>"
            "<span style='font-family:ui-monospace,Consolas,monospace;font-size:12.5px;"
            f"word-break:break-word'>{label}</span>"
            "<span style='color:#F2A33C;font-weight:650;font-size:12px;white-space:nowrap'>"
            f"score {score}</span></div>"
            f"<div class='small muted' style='margin-bottom:6px'>{_esc(item.get('summary', ''))}</div>"
            f"<ul style='list-style:none;padding:0;margin:0'>{''.join(rows_html)}</ul>"
            "</div>"
        )
    return (
        "<p class='small muted'>For each of the highest-scoring records, the handful of things "
        "about it that pushed it up the list. We worked this out by swapping one column at a time "
        "for the value a typical record has and seeing how far the score moved &mdash; so every line "
        "is a plain difference you can check against the record yourself. Columns that overlap in "
        "meaning can each look important on their own, so read these as strong hints rather than "
        "an exact split.</p>"
        + "".join(blocks)
    )


def _render_rows_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    columns = [str(c) for c in columns]
    if not columns:
        return '<p class="muted">No scored rows were available to show.</p>'
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for value in row:
            if isinstance(value, float):
                text = _fmt_float(value, 4)
            elif value is None:
                text = DASH
            else:
                text = str(value)
            cells.append(f"<td>{_esc(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    if not body:
        body.append(f'<tr><td colspan="{len(columns)}" class="muted">No rows.</td></tr>')
    return (
        '<div class="scroll" style="max-height:520px"><table><thead><tr>'
        + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def _top_rows(result: Any, limit: int = MAX_HTML_ROWS) -> tuple[list[str], list[list]]:
    """Prefer the real scored file; fall back to the run's preview sample."""
    scored_path = _get(result, "scored_path")
    if scored_path and Path(str(scored_path)).exists():
        try:
            columns, rows = _load_scored(str(scored_path), limit)
            if columns:
                return list(columns), [list(r) for r in rows]
        except Exception:
            pass
    preview = _as_dict(_get(result, "preview"))
    columns = [str(c) for c in _seq(preview.get("columns"))]
    rows = [list(r) for r in _seq(preview.get("rows"))][:limit]
    return columns, rows


def export_html(result: Any, out_path: str) -> str:
    """Write a single self-contained HTML report and return its absolute path."""
    path = _abs_path(out_path)

    dataset = _as_dict(_get(result, "dataset"))
    usecase = _as_dict(_get(result, "usecase"))
    champion = _as_dict(_get(result, "champion"))
    impact = _as_dict(_get(result, "impact"))
    plan = _as_dict(_get(result, "plan"))
    distributions = _as_dict(_get(result, "distributions"))
    leaderboard = [_as_dict(x) for x in _seq(_get(result, "leaderboard"))]
    drivers = [_as_dict(x) for x in _seq(_get(result, "drivers"))]
    insights = [_as_dict(x) for x in _seq(_get(result, "insights"))]
    narration = [_as_dict(x) for x in _seq(_get(result, "narration"))]

    dataset_name = (
        dataset.get("name")
        or dataset.get("dataset_id")
        or _get(result, "dataset_id")
        or "Your data"
    )
    run_id = _get(result, "run_id") or DASH
    finished = _fmt_ts(_get(result, "finished_at"))
    status = str(_get(result, "status") or "succeeded")
    duration = _num(_get(result, "duration_ms"))
    error = _get(result, "error")

    chips: list[str] = []
    if dataset.get("rows") is not None:
        chips.append(f"<span class='chip'><b>{_esc(_fmt_int(_num(dataset.get('rows'))))}</b> rows</span>")
    if dataset.get("cols") is not None:
        chips.append(f"<span class='chip'><b>{_esc(_fmt_int(_num(dataset.get('cols'))))}</b> columns</span>")
    if dataset.get("scored_rows") is not None:
        chips.append(f"<span class='chip'><b>{_esc(_fmt_int(_num(dataset.get('scored_rows'))))}</b> rows scored</span>")
    chips.append(f"<span class='chip'><b>{len(leaderboard)}</b> models raced</span>")
    if duration is not None:
        chips.append(f"<span class='chip'><b>{_esc(_fmt_ms(duration))}</b> total</span>")
    if plan.get("test_strategy"):
        chips.append(f"<span class='chip'>tested by <b>{_esc(str(plan['test_strategy']).replace('_', ' '))}</b></span>")

    question = usecase.get("business_question") or usecase.get("name") or "What is driving this number?"
    story = usecase.get("story") or ""

    impact_value = _fmt_impact(impact.get("value"), impact.get("unit"))
    impact_label = impact.get("label") or "Impact"
    impact_note = impact.get("formula") or impact.get("note") or ""

    champion_name = champion.get("display") or champion.get("key") or DASH
    champion_persona = champion.get("persona") or ""
    champion_plain = champion.get("plain_summary") or champion.get("why") or ""
    champion_why = champion.get("why_won") or ""
    champion_runner = champion.get("runner_up") or ""
    champion_tradeoff = champion.get("tradeoff") or ""

    hist_svg = _svg_histogram(distributions.get("score_histogram"))
    lb_svg = _svg_leaderboard(distributions.get("leaderboard"))
    charts = "".join(x for x in (hist_svg, lb_svg) if x)
    if not charts:
        charts = '<p class="muted">No chart data was available for this run.</p>'

    confusion = _confusion_sentence(result)
    columns, rows = _top_rows(result, MAX_HTML_ROWS)

    parts: list[str] = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>ML Arena report {_esc(str(dataset_name))}</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    # --- header -------------------------------------------------------------
    parts.append(
        "<header class='hero'>"
        "<div class='brand'>ML Arena</div>"
        f"<h1>{_esc(str(dataset_name))}</h1>"
        f"<p class='sub'>{_esc(usecase.get('name') or 'Analysis')} &middot; run {_esc(str(run_id))} "
        f"&middot; finished {_esc(finished)}"
        + ("" if status == "succeeded" else f" &middot; <b>{_esc(status)}</b>")
        + "</p>"
        f"<div class='chips'>{''.join(chips)}</div>"
        "</header>"
    )

    if error:
        parts.append(
            f"<section class='card'><h2>Something went wrong</h2><p>{_esc(error)}</p></section>"
        )

    # --- business question --------------------------------------------------
    parts.append(
        "<section class='card'><h2>The business question</h2>"
        f"<p class='lede'>{_esc(question)}</p>"
        + (f"<p class='muted small'>{_esc(story)}</p>" if story else "")
        + "</section>"
    )

    # --- executive summary --------------------------------------------------
    parts.append(
        "<section class='card'><h2>Executive summary</h2>"
        f"<p class='lede'>{_esc(_executive_summary(result))}</p>"
        + (f"<p class='small muted'>{_esc(confusion)}</p>" if confusion else "")
        + "</section>"
    )

    # --- impact callout -----------------------------------------------------
    parts.append(
        "<section class='impact'>"
        f"<div class='value'>{_esc(impact_value)}</div>"
        f"<div class='label'>{_esc(impact_label)}</div>"
        + (f"<div class='note'>{_esc(impact_note)}</div>" if impact_note else "")
        + "</section>"
    )

    # --- metric cards -------------------------------------------------------
    parts.append(
        "<section class='card'><h2>How the winner performed</h2>"
        + _render_metric_cards(champion, usecase)
        + "</section>"
    )

    # --- leaderboard --------------------------------------------------------
    parts.append(
        "<section class='card'><h2>The model leaderboard</h2>"
        "<p class='small muted'>Every model we tried is listed, including the ones that did not finish. "
        "The bar shows how each model scored relative to the others.</p>"
        + _render_leaderboard(leaderboard, str(usecase.get("primary_metric") or ""))
        + "</section>"
    )

    # --- champion + charts --------------------------------------------------
    champ_card = (
        "<section class='card champ'><h2>Why we picked this model</h2>"
        f"<div class='who'>{_esc(champion_name)}</div>"
        + (f"<p class='muted small'>{_esc(champion_persona)}</p>" if champion_persona else "")
        + (f"<p>{_esc(champion_plain)}</p>" if champion_plain else "")
        + "<dl class='kv'>"
        + (f"<dt>Why it won</dt><dd>{_esc(champion_why)}</dd>" if champion_why else "")
        + (f"<dt>Runner-up</dt><dd>{_esc(champion_runner)}</dd>" if champion_runner else "")
        + (f"<dt>The trade-off</dt><dd>{_esc(champion_tradeoff)}</dd>" if champion_tradeoff else "")
        + "</dl></section>"
    )
    chart_card = f"<section class='card'><h2>What the results look like</h2>{charts}</section>"
    parts.append(f"<div class='grid2'>{champ_card}{chart_card}</div>")
    parts.append("<div style='height:20px'></div>")

    # --- drivers ------------------------------------------------------------
    parts.append(
        "<section class='card'><h2>What is driving the result</h2>"
        "<p class='small muted'>Ranked from strongest signal to weakest. "
        "The longer the bar, the more this factor moves the outcome.</p>"
        + _render_drivers(drivers)
        + "</section>"
    )

    # --- insights -----------------------------------------------------------
    parts.append(
        "<section class='card'><h2>What we found</h2>"
        + _render_insights(insights)
        + "</section>"
    )

    # --- story --------------------------------------------------------------
    parts.append(
        "<section class='card'><h2>How we got here, step by step</h2>"
        + _render_story(narration)
        + "</section>"
    )

    # --- top rows -----------------------------------------------------------
    parts.append(
        f"<section class='card'><h2>Top {min(len(rows), MAX_HTML_ROWS)} scored rows</h2>"
        "<p class='small muted'>These are the records the winning model considered most worth your attention.</p>"
        + _render_rows_table(columns, rows)
        + "</section>"
    )

    # --- why these rows -----------------------------------------------------
    explanations = _seq(_get(result, "explanations"))
    if explanations:
        parts.append(
            "<section class='card'><h2>Why these rows?</h2>"
            + _render_explanations(explanations)
            + "</section>"
        )

    parts.append(
        "<footer>"
        f"Generated by {_esc(config.APP_NAME)} v{_esc(config.APP_VERSION)} on {_esc(_fmt_ts(_get(result, 'finished_at')))}. "
        "This report is a single self-contained file &mdash; it works offline with no internet connection."
        "</footer>"
    )
    parts.append("</div></body></html>")

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(parts))
    return path


# ---------------------------------------------------------------------------
# scored-row loading (shared by csv / xlsx / html)
# ---------------------------------------------------------------------------
def _load_scored(src_path: str, limit: int | None = None) -> tuple[list[str], list[tuple]]:
    """Read up to ``limit`` rows from a scored parquet/csv file.

    Tries polars first, then pandas, then the stdlib csv reader.  Raises only if
    every route fails.
    """
    src = Path(str(src_path))
    suffix = src.suffix.lower()
    errors: list[str] = []

    pl = config.get_polars()
    if pl is not None:
        try:
            lf = pl.scan_parquet(src) if suffix == ".parquet" else pl.scan_csv(src, infer_schema_length=5000)
            frame = lf.head(limit).collect() if limit else lf.collect()
            return list(frame.columns), list(frame.rows())
        except Exception as exc:  # pragma: no cover - depends on file contents
            errors.append(f"polars: {exc}")

    pd = config.get_pandas()
    if pd is not None:
        try:
            if suffix == ".parquet":
                frame = pd.read_parquet(src)
                if limit:
                    frame = frame.head(limit)
            else:
                frame = pd.read_csv(src, nrows=limit)
            return [str(c) for c in frame.columns], list(frame.itertuples(index=False, name=None))
        except Exception as exc:  # pragma: no cover
            errors.append(f"pandas: {exc}")

    try:
        with open(src, "r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            columns = next(reader, [])
            rows = [tuple(row) for _, row in zip(range(limit or 10**9), reader)]
        return [str(c) for c in columns], rows
    except Exception as exc:
        errors.append(f"csv: {exc}")

    raise RuntimeError("Could not read scored rows from " + str(src) + " (" + "; ".join(errors) + ")")


def _round_float_columns(columns: list[str], rows: Iterable[Sequence[Any]], digits: int = 6):
    """Round floats to keep CSV/XLSX numbers free of binary noise."""
    for row in rows:
        out: list[Any] = []
        for value in row:
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    out.append(None)
                else:
                    out.append(round(value, digits))
            else:
                out.append(value)
        yield out


def _prepend_bom(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` with a UTF-8 BOM in front."""
    with open(dst, "wb") as out, open(src, "rb") as inp:
        out.write(b"\xef\xbb\xbf")
        shutil.copyfileobj(inp, out, length=1 << 20)


def _write_rows_csv(path: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        if columns:
            writer.writerow(list(columns))
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------
def export_csv(result: Any, out_path: str) -> str:
    """Write every scored row as CSV (UTF-8 with BOM) and return the abs path."""
    path = _abs_path(out_path)
    scored_path = _get(result, "scored_path")

    if scored_path and Path(str(scored_path)).exists():
        src = Path(str(scored_path))
        suffix = src.suffix.lower()
        pl = config.get_polars()

        # fast path: let polars stream the file out, then bolt a BOM on the front
        if pl is not None and suffix in (".parquet", ".csv", ".tsv", ".txt"):
            try:
                frame = pl.read_parquet(src) if suffix == ".parquet" else pl.read_csv(
                    src, infer_schema_length=5000
                )
                float_cols = [
                    name for name, dtype in zip(frame.columns, frame.dtypes)
                    if dtype in (pl.Float32, pl.Float64)
                ]
                if float_cols:
                    frame = frame.with_columns([pl.col(c).round(6) for c in float_cols])
                tmp = Path(path).parent / (Path(path).name + ".mlarena_tmp.csv")
                try:
                    frame.write_csv(tmp)
                    _prepend_bom(tmp, Path(path))
                finally:
                    tmp.unlink(missing_ok=True)
                return path
            except Exception:
                pass

        pd = config.get_pandas()
        if pd is not None:
            try:
                if suffix == ".parquet":
                    frame = pd.read_parquet(src)
                else:
                    frame = pd.read_csv(src)
                float_cols = frame.select_dtypes(include=["float", "float32", "float64"]).columns
                if len(float_cols):
                    frame = frame.copy()
                    frame[float_cols] = frame[float_cols].round(6)
                frame.to_csv(path, index=False, encoding="utf-8-sig")
                return path
            except Exception:
                pass

        try:
            columns, rows = _load_scored(str(src), None)
            _write_rows_csv(path, columns, _round_float_columns(columns, rows))
            return path
        except Exception:
            # last resort: byte-copy with a BOM so Excel still opens it
            _prepend_bom(src, Path(path))
            return path

    # no scored file - fall back to whatever preview rows the run carried
    preview = _as_dict(_get(result, "preview"))
    columns = [str(c) for c in _seq(preview.get("columns"))] or [
        str(c) for c in _seq(_get(result, "scored_columns"))
    ]
    rows = [list(r) for r in _seq(preview.get("rows"))]
    if not columns and not rows:
        # never hand back an empty file - say so instead
        columns = ["note"]
        rows = [["No scored rows were available for this run."]]
    _write_rows_csv(path, columns, _round_float_columns(columns, rows))
    return path


# ---------------------------------------------------------------------------
# export_xlsx
# ---------------------------------------------------------------------------
def _xlsx_value(value: Any) -> Any:
    """Convert anything to something openpyxl can write."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (int, str, bool, _dt.datetime, _dt.date, _dt.time)):
        return value
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(_json_safe(value), ensure_ascii=False)
    try:  # numpy scalars
        item = value.item()  # type: ignore[attr-defined]
        if isinstance(item, (int, float, str, bool)):
            return item
    except Exception:
        pass
    return str(value)


def _style_header(worksheet, ncols: int) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    fill = PatternFill("solid", fgColor="0B0F17")
    font = Font(bold=True, color="FFFFFF", size=11)
    for col in range(1, max(1, ncols) + 1):
        cell = worksheet.cell(row=1, column=col)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(vertical="center")
    worksheet.freeze_panes = "A2"
    worksheet.row_dimensions[1].height = 20


def _autosize(worksheet, columns: Sequence[str], rows: Sequence[Sequence[Any]], cap: int = 46) -> None:
    from openpyxl.utils import get_column_letter

    for idx, name in enumerate(columns, start=1):
        width = len(str(name)) + 2
        for row in rows[:400]:
            if idx - 1 < len(row):
                width = max(width, min(cap, len(str(row[idx - 1] if row[idx - 1] is not None else "")) + 2))
        worksheet.column_dimensions[get_column_letter(idx)].width = max(10, min(cap, width))


def _add_sheet(wb, title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], number_formats: dict | None = None):
    """Add a sheet with a styled header. Never crashes on an empty dataset."""
    ws = wb.create_sheet(title=title)
    columns = [str(c) for c in columns] or ["(no data)"]
    ws.append(columns)
    for row in rows:
        ws.append([_xlsx_value(v) for v in row])
    _style_header(ws, len(columns))
    _autosize(ws, columns, rows)
    if number_formats:
        for idx, name in enumerate(columns, start=1):
            fmt = number_formats.get(name)
            if fmt:
                for r in range(2, ws.max_row + 1):
                    ws.cell(row=r, column=idx).number_format = fmt
    return ws


def export_xlsx(result: Any, out_path: str) -> str:
    """Write the formatted multi-sheet workbook and return its absolute path."""
    path = _abs_path(out_path)

    try:
        from openpyxl import Workbook
        from openpyxl.formatting.rule import ColorScaleRule
    except Exception as exc:  # pragma: no cover - openpyxl is a hard dependency
        raise RuntimeError("openpyxl is required for Excel export") from exc

    dataset = _as_dict(_get(result, "dataset"))
    usecase = _as_dict(_get(result, "usecase"))
    champion = _as_dict(_get(result, "champion"))
    impact = _as_dict(_get(result, "impact"))
    plan = _as_dict(_get(result, "plan"))
    target = _as_dict(_get(result, "target"))
    leaderboard = [_as_dict(x) for x in _seq(_get(result, "leaderboard"))]
    drivers = [_as_dict(x) for x in _seq(_get(result, "drivers"))]
    insights = [_as_dict(x) for x in _seq(_get(result, "insights"))]
    narration = [_as_dict(x) for x in _seq(_get(result, "narration"))]

    # --- scored data --------------------------------------------------------
    scored_columns: list[str] = [str(c) for c in _seq(_get(result, "scored_columns"))]
    scored_rows: list[list] = []
    truncated = False
    scored_path = _get(result, "scored_path")
    if scored_path and Path(str(scored_path)).exists():
        try:
            loaded_cols, loaded_rows = _load_scored(str(scored_path), MAX_XLSX_ROWS + 1)
            if loaded_cols:
                scored_columns = loaded_cols
            if len(loaded_rows) > MAX_XLSX_ROWS:
                loaded_rows = loaded_rows[:MAX_XLSX_ROWS]
                truncated = True
            scored_rows = [list(r) for r in loaded_rows]
        except Exception:
            scored_rows = []
    if not scored_rows:
        preview = _as_dict(_get(result, "preview"))
        preview_cols = [str(c) for c in _seq(preview.get("columns"))]
        if preview_cols:
            scored_columns = preview_cols
        scored_rows = [list(r) for r in _seq(preview.get("rows"))]

    wb = Workbook()
    wb.remove(wb.active)  # type: ignore[arg-type]

    # --- Summary ------------------------------------------------------------
    summary_rows: list[list[Any]] = []
    summary_rows.append(["Field", "Value"])
    summary_rows.append(["Report generated by", f"{config.APP_NAME} v{config.APP_VERSION}"])
    summary_rows.append(["Run ID", _get(result, "run_id")])
    summary_rows.append(["Status", _get(result, "status")])
    summary_rows.append(["Dataset", dataset.get("name") or dataset.get("dataset_id") or _get(result, "dataset_id")])
    summary_rows.append(["Rows analysed", dataset.get("rows")])
    summary_rows.append(["Columns", dataset.get("cols")])
    summary_rows.append(["Rows scored", dataset.get("scored_rows")])
    summary_rows.append(["Engine", dataset.get("engine")])
    summary_rows.append(["Use case", usecase.get("name")])
    summary_rows.append(["Business question", usecase.get("business_question")])
    summary_rows.append(["Task", target.get("task") or usecase.get("task")])
    summary_rows.append(["What we predicted", target.get("target_col")])
    summary_rows.append(["Test strategy", plan.get("test_strategy")])
    summary_rows.append(["Models raced", plan.get("model_count") or len(leaderboard)])
    summary_rows.append(["Champion model", champion.get("display") or champion.get("key")])
    summary_rows.append(["Why it won", champion.get("why_won")])
    summary_rows.append(["Runner-up", champion.get("runner_up")])
    summary_rows.append(["Trade-off", champion.get("tradeoff")])
    summary_rows.append(["Impact label", impact.get("label")])
    summary_rows.append(["Impact value", _fmt_impact(impact.get("value"), impact.get("unit"))])
    summary_rows.append(["Impact formula", impact.get("formula")])
    summary_rows.append(["Impact note", impact.get("note")])
    summary_rows.append(["Executive summary", _executive_summary(result)])
    summary_rows.append(["Started at", _get(result, "started_at")])
    summary_rows.append(["Finished at", _get(result, "finished_at")])
    summary_rows.append(["Duration (ms)", _get(result, "duration_ms")])
    summary_rows.append(["Log events", len(_seq(_get(result, "log")))])
    summary_rows.append(["Scored rows in this workbook", len(scored_rows)])
    if truncated:
        summary_rows.append([
            "Note",
            f"The scored data was truncated to the first {MAX_XLSX_ROWS:,} rows for Excel. "
            "Use the CSV export for the complete set.",
        ])
    if _get(result, "error"):
        summary_rows.append(["Error", _get(result, "error")])

    ws_summary = _add_sheet(wb, "Summary", summary_rows[0], summary_rows[1:])
    ws_summary.column_dimensions["A"].width = 30
    ws_summary.column_dimensions["B"].width = 96
    from openpyxl.styles import Alignment

    for row in ws_summary.iter_rows(min_row=2, min_col=2, max_col=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    # --- Model Leaderboard --------------------------------------------------
    lb_columns = ["Rank", "Model", "Persona", "Family", "Cost", "Status",
                  "Primary metric", "Primary score", "UI score (0-100)",
                  "Train (ms)", "Predict (ms)", "Metrics", "What happened"]
    lb_rows: list[list[Any]] = []
    for i, entry in enumerate(leaderboard):
        metrics = _as_dict(entry.get("metrics"))
        lb_rows.append([
            entry.get("rank") or (i + 1),
            entry.get("display") or entry.get("key"),
            entry.get("persona"),
            entry.get("family"),
            entry.get("cost"),
            entry.get("status") or "ok",
            usecase.get("primary_metric") or entry.get("primary_metric"),
            _num(entry.get("primary")),
            _num(entry.get("score")),
            _num(entry.get("train_ms")),
            _num(entry.get("predict_ms")),
            ", ".join(f"{k}={_fmt_metric_value(k, v)}" for k, v in metrics.items()) if metrics else "",
            entry.get("why") or entry.get("error"),
        ])
    ws_lb = _add_sheet(wb, "Model Leaderboard", lb_columns, lb_rows,
                       number_formats={"Primary score": "0.0000", "UI score (0-100)": "0.0",
                                       "Train (ms)": "#,##0", "Predict (ms)": "#,##0"})
    if lb_rows:
        last = ws_lb.max_row
        ws_lb.conditional_formatting.add(
            f"I2:I{last}",
            ColorScaleRule(start_type="num", start_value=0, start_color="F8696B",
                           mid_type="num", mid_value=50, mid_color="FFEB84",
                           end_type="num", end_value=100, end_color="63BE7B"),
        )

    # --- Key Drivers --------------------------------------------------------
    driver_columns = ["Rank", "Feature", "Importance", "Direction", "In plain words"]
    driver_rows = [
        [i + 1, d.get("feature") or d.get("name"), _num(d.get("importance")),
         d.get("direction"), d.get("plain")]
        for i, d in enumerate(drivers)
    ]
    _add_sheet(wb, "Key Drivers", driver_columns, driver_rows,
               number_formats={"Importance": "0.0000"})

    # --- Insights -----------------------------------------------------------
    insight_rows: list[list[Any]] = []
    for item in insights:
        insight_rows.append(["Insight", item.get("severity") or "info", item.get("title"), item.get("body")])
    for beat in narration:
        insight_rows.append(["Story beat", "", beat.get("title") or beat.get("beat"), beat.get("text")])
    _add_sheet(wb, "Insights", ["Type", "Severity", "Title", "Detail"], insight_rows)

    # --- Scored Data --------------------------------------------------------
    score_col_index = 0
    for idx, name in enumerate(scored_columns):
        low = str(name).lower()
        if "score" in low or "prediction" in low or "probability" in low:
            score_col_index = idx + 1
            break
    ws_scored = _add_sheet(wb, "Scored Data", scored_columns, scored_rows)
    if scored_rows and score_col_index:
        from openpyxl.utils import get_column_letter

        letter = get_column_letter(score_col_index)
        ws_scored.conditional_formatting.add(
            f"{letter}2:{letter}{ws_scored.max_row}",
            ColorScaleRule(start_type="min", start_color="2FBF71",
                           mid_type="percentile", mid_value=50, mid_color="F5C451",
                           end_type="max", end_color="EF6A6A"),
        )

    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# export_summary_json
# ---------------------------------------------------------------------------
def _json_safe(value: Any, depth: int = 0) -> Any:
    """Make any run-result value JSON serialisable."""
    if depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in value]
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe(dataclasses.asdict(value), depth + 1)
        except Exception:
            pass
    try:  # numpy / pandas scalars
        item = value.item()  # type: ignore[attr-defined]
        return _json_safe(item, depth + 1)
    except Exception:
        pass
    return str(value)


def _result_keys(result: Any) -> list[str]:
    if isinstance(result, Mapping):
        return [str(k) for k in result.keys()]
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return [f.name for f in dataclasses.fields(result)]
    try:
        return list(vars(result).keys())
    except Exception:
        return []


def export_summary_json(result: Any, out_path: str) -> str:
    """Write the run result as JSON (log replaced by a count) and return its path."""
    path = _abs_path(out_path)

    payload: dict[str, Any] = {}
    for key in _result_keys(result):
        if key == "log":
            continue
        payload[key] = _json_safe(_get(result, key))

    log = _seq(_get(result, "log"))
    payload["log_event_count"] = len(log)
    payload["log_tail"] = _json_safe(log[-20:]) if log else []

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return path
