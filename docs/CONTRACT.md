# ML Arena — Frozen Interface Contract (v1)

**Read this file completely before writing any code. Do not change any signature in this document.
If you believe a signature must change, implement the frozen one and add an additive extension instead.**

ML Arena is a **100% local** desktop-class web application that democratises ML for non-technical
business users. No APIs, no LLM calls, no cloud. Everything runs on the user's machine.

Root: `C:\ML Arena`
Python venv: `C:\ML Arena\.venv\Scripts\python.exe` (Python 3.11, inherits system site-packages)

---

## 0. Environment bootstrap — MANDATORY

`mlarena/config.py` is written by the lead. **Every module that imports polars must import config first:**

```python
from mlarena import config  # noqa: F401  -- sets POLARS_SKIP_CPU_CHECK before polars loads
import polars as pl
```

Available libraries (already installed, do not add new heavy deps):
`polars 1.44`, `pyarrow 25`, `pandas 2.3`, `numpy 2.4`, `scikit-learn 1.9`, `xgboost 3.2`,
`lightgbm 4.7`, `scipy 1.17`, `openpyxl 3.1`, `fastapi 0.139`, `uvicorn 0.51`, `websockets 15`, `joblib`.

**Never** call the network. **Never** import torch/tensorflow (not installed, and not needed).
"DL-style" capability is delivered via gradient-boosted trees + MLP (`sklearn.neural_network.MLPClassifier/Regressor`)
which are the correct choice for tabular business data on CPU-only machines.

---

## 1. Directory layout (ownership)

```
mlarena/
  config.py            <- LEAD
  dataio/
    loader.py          <- AGENT C
    profiler.py        <- AGENT C
  columns/
    derived.py         <- AGENT C
  usecases/
    registry.py        <- AGENT B
    definitions.py     <- AGENT B
  engine/
    models.py          <- AGENT B
    planner.py         <- AGENT B
    runner.py          <- AGENT B
    evaluate.py        <- AGENT B
    narration.py       <- AGENT B
  export/
    exporters.py       <- AGENT D
  api/
    server.py          <- LEAD
    jobs.py            <- LEAD
web/                   <- AGENT A
samples/               <- AGENT D
tests/                 <- ALL (write tests for your own module)
run.py                 <- LEAD
```

---

## 2. `mlarena/config.py` (LEAD — already written, read it)

```python
ROOT: Path            # C:\ML Arena
WEB_DIR: Path
DATA_DIR: Path        # uploaded files
RUNS_DIR: Path        # runs/<run_id>/
SAMPLES_DIR: Path
def ensure_dirs() -> None
def get_polars():     # -> module or None
def have(module_name: str) -> bool
```

---

## 3. `mlarena/dataio/loader.py` — AGENT C

```python
from dataclasses import dataclass, field
import polars as pl

@dataclass
class LoadResult:
    dataset_id: str
    path: str                 # absolute path on disk
    fmt: str                  # "csv" | "xlsx" | "parquet" | "json" | "tsv"
    rows: int
    cols: int
    size_bytes: int
    load_ms: float
    engine: str               # "polars" | "pandas"
    columns: list[str]
    dtypes: dict[str, str]    # col -> "Int64" | "Float64" | "String" | "Date" | "Datetime" | "Boolean"
    derived_from: str | None = None   # set when this dataset was created by the column engine
    notes: list[str] = field(default_factory=list)   # plain-language notes for the UI log

def load_dataset(src_path: str, dataset_id: str) -> LoadResult
    """Read a file from disk as fast as possible and register it in the store.
    - csv/tsv: try pl.read_csv (or scan_csv + collect for big files). Fall back to
      pandas.read_csv if polars raises. Sniff delimiter and encoding if needed.
    - xlsx: pandas.read_excel(engine="openpyxl") then pl.from_pandas.
    - parquet: pl.read_parquet.
    - json: pl.read_json, fall back to pandas.read_json.
    Must be resilient: if the file is huge (> ~1.5 GB) use lazy scan + sampling to
    report shape, and record a note explaining the sampling.
    """

def register_frame(dataset_id: str, df: pl.DataFrame, *, path: str = "", fmt: str = "memory",
                   derived_from: str | None = None, notes: list[str] | None = None) -> LoadResult
    """Register an already-built frame (used by the derived-column engine)."""

class DatasetStore:
    def get(self, dataset_id: str) -> pl.DataFrame
    def meta(self, dataset_id: str) -> LoadResult
    def has(self, dataset_id: str) -> bool
    def drop(self, dataset_id: str) -> None
    def list(self) -> list[LoadResult]
    def to_pandas(self, dataset_id: str, columns: list[str] | None = None) -> "pandas.DataFrame"

store: DatasetStore = DatasetStore()   # module-level singleton

def frame_pandas(dataset_id: str, columns: list[str] | None = None) -> "pandas.DataFrame"
def dataset_ids() -> list[str]
```

---

## 4. `mlarena/dataio/profiler.py` — AGENT C

```python
@dataclass
class ColumnProfile:
    name: str
    kind: str            # "measure" | "dimension" | "datetime" | "identifier" | "text"
    dtype: str
    semantic: str        # "currency" | "percent" | "count" | "quantity" | "date" | "category"
                         # | "flag" | "free_text" | "id" | "code" | "numeric" | "unknown"
    null_pct: float
    distinct: int
    unique_ratio: float
    samples: list        # up to 5 representative values (JSON-safe)
    stats: dict          # measures: min,max,mean,std,median,p25,p75,sum,skew,negatives_pct,zeros_pct
                         # dimensions: {"top": [[value, count], ...], "coverage": float}
                         # datetime: min,max,span_days
    role_hint: str       # short human sentence, e.g. "Looks like a money amount you can analyse"
    confidence: float    # 0..1

@dataclass
class DatasetProfile:
    dataset_id: str
    rows: int
    cols: int
    profiled_ms: float
    columns: list[ColumnProfile]
    measures: list[str]
    dimensions: list[str]
    datetimes: list[str]
    identifiers: list[str]
    quality: dict        # {"duplicate_rows": int, "null_cells_pct": float, "constant_cols": [str],
                         #  "high_cardinality": [str], "empty_cols": [str], "score": 0..100,
                         #  "headline": "plain-language one-liner"}
    suggestions: list[dict]   # see §5 suggest_derived

def profile(dataset_id: str, *, sample_rows: int = 200_000) -> DatasetProfile
def profile_to_dict(p: DatasetProfile) -> dict   # JSON-safe
```

**Classification rules (be smart — this is a headline feature):**
- `identifier`: integer/string, `unique_ratio > 0.95`, or name matches `id$|_id|^id|code|key|sku|uuid|invoice|order_no`.
- `datetime`: polars Date/Datetime dtype, or name matches `date|dt$|_at$|time|month|year|period`, or string parseable as date > 80%.
- `measure`: numeric dtype **and** `distinct > 25` **and** `unique_ratio > 0.02`. Name hints
  (`amount|price|cost|revenue|sales|margin|discount|qty|quantity|value|total|net|gross|profit|balance|salary|units`)
  raise confidence.
- `dimension`: string/bool/categorical, or numeric with `distinct <= 25` (e.g. a 1-5 rating or flag).
- `text`: string with `unique_ratio > 0.6` and average length > 25.
- Numeric columns that are integers with very few distinct values and name hints like `flag|is_|has_` → `dimension` + semantic `flag`.
- **Never** classify a money column as an identifier just because it is numeric.

`quality.headline` must be a plain-English sentence a non-technical user understands,
e.g. `"42,180 orders, 3 columns look like money, 2 columns have missing values (under 1%)."`

---

## 5. `mlarena/columns/derived.py` — AGENT C

Safe, AST-whitelisted expression engine. **Never use bare `eval`/`exec`.**

```python
@dataclass
class DerivedSpec:
    name: str
    expression: str

def validate_expression(expression: str, available: list[str]) -> tuple[bool, str]
    """(ok, message). Reject unknown column names, unknown functions, attribute access,
    subscripts, comprehensions, lambdas, imports, and any AST node outside the whitelist."""

def preview(dataset_id: str, name: str, expression: str, *, limit: int = 200) -> dict
    """{"ok": bool, "error": str|None, "values": [JSON-safe], "dtype": str,
        "stats": {"min":..,"max":..,"mean":..,"nulls":..}, "row_count": int}"""

def materialize(dataset_id: str, specs: list[DerivedSpec]) -> str
    """Return a NEW dataset_id containing original columns + derived columns.
    Must not mutate the source dataset. Register via loader.register_frame(derived_from=dataset_id)."""

def suggest_derived(profile) -> list[dict]
    """Auto-suggest useful derived columns from column names. Return
    [{"name": str, "expression": str, "reason": str, "confidence": 0..1}].
    Must cover at least these patterns (case/spacing-insensitive name matching):
      list_price & net_price        -> discount_pct = (list_price - net_price) / list_price
      list_price & net_price        -> discount_amount = list_price - net_price
      revenue & cost                -> gross_margin_pct = (revenue - cost) / revenue
      revenue & cost                -> gross_profit = revenue - cost
      quantity & unit_price         -> extended_value = quantity * unit_price
      discount_pct & revenue        -> discount_value = discount_pct * revenue
      start/end date pair           -> duration_days
      any two measures m1, m2       -> <m1>_per_<m2>  (ratio) at lower confidence
    Only suggest when every referenced column exists in the profile.
    """

def function_catalog() -> list[dict]
    """For the UI's formula helper. [{"name","signature","description","example"}].
    Supported functions: abs, round, min, max, sqrt, log, log10, exp, clip, floor, ceil,
    ifelse(cond, a, b), coalesce(a, b, ...), year, month, day, days_between(a, b), isnull(x).
    Supported operators: + - * / ** %, comparisons (== != > >= < <=), and/or/not.
    """
```

---

## 6. `mlarena/usecases/registry.py` — AGENT B

```python
@dataclass
class Role:
    key: str
    label: str          # UI label
    help: str           # plain-language helper text
    accepts: list[str]  # kinds the picker may show, e.g. ["measure"], ["dimension","identifier"]
    required: bool
    multi: bool
    min_count: int = 1

@dataclass
class UseCase:
    key: str
    name: str
    emoji: str
    tagline: str          # <= 8 words
    story: str            # 2-3 plain-language sentences: what this finds and why a business cares
    business_question: str
    task: str             # "classification" | "regression" | "anomaly" | "clustering"
    roles: list[Role]
    target: str           # strategy key implemented in definitions.py
    primary_metric: str   # "roc_auc" | "r2" | "silhouette" | "consensus"
    primary_metric_label: str   # e.g. "Separation power"
    impact_unit: str      # "currency" | "records" | "customers" | "none"
    impact_label: str     # e.g. "Estimated margin at risk"
    sector: str           # "Finance" | "Sales" | "Risk" | "Operations" | "Customer"
    examples: list[str]   # sample file names that fit

def all_usecases() -> list[UseCase]
def get_usecase(key: str) -> UseCase     # raise KeyError with a helpful message
def serialize(uc: UseCase) -> dict       # JSON-safe, includes roles as list of dicts
def catalog() -> list[dict]              # all use cases serialized
```

**The 9 required use cases** (keys are fixed):

| key | task | what it does |
|---|---|---|
| `discount_leakage` | classification | Learn the *expected* discount for each transaction from its context, then flag rows where the actual discount far exceeds expectation (leakage). |
| `margin_overflow` | classification | Flag orders whose realised margin % falls below what the pricing policy implies for that segment. |
| `revenue_predict` | regression | Predict a money/number column from the other columns; rank what drives it. |
| `demand_forecast` | regression | Forecast a measure over time (uses a datetime role); includes a seasonal-naive baseline model. |
| `churn_risk` | classification | Score customers likely to stop buying; rank risk and size the revenue at stake. |
| `credit_default` | classification | Predict probability of default / late payment; translate to expected loss. |
| `transaction_anomaly` | anomaly | Unsupervised: find the records that simply do not look like the others. |
| `segmentation` | clustering | Group customers/products into behavioural segments and describe each one. |
| `price_elasticity` | regression | Estimate how demand responds to price/discount; find the revenue-maximising discount. |

---

## 7. `mlarena/usecases/definitions.py` — AGENT B

```python
@dataclass
class TargetPlan:
    task: str
    frame: pl.DataFrame          # features + target, cleaned, ready for the engine
    target_col: str | None       # None for anomaly/clustering
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    label_names: dict | None     # {0: "Normal", 1: "Leakage"}
    positive_rate: float | None
    impact_col: str | None       # column to SUM for the money-impact number
    impact_label: str
    score_direction: str         # "higher_is_risk" | "higher_is_good"
    key_cols: list[str]          # identifier/dimension columns to carry into the output rows
    notes: list[str]             # plain-language notes emitted to the live log
    extra: dict                  # strategy-specific payload (e.g. {"series_col": "region"})
    baseline: str | None         # e.g. "seasonal_naive"

def build_target(dataset_id: str, uc: UseCase, mapping: dict[str, list[str]],
                 options: dict) -> TargetPlan
```

`mapping` is `{role_key: [column names]}` exactly as the user chose in the UI.

**Target construction guidance (must be defensible statistics, not hand-waving):**
- `discount_leakage`: build features from the *context* roles (segment, product, customer, channel,
  quantity, time). Train a regressor to predict the discount measure from context; `residual =
  actual - expected`. Leakage label = `residual > max(2 * residual_std, 0.15 * expected_scale)`.
  Emit notes explaining this in plain language. `impact_col` = the money column, so impact =
  sum of money at risk on flagged rows.
- `margin_overflow`: same residual framing on margin %, floor = max(0, policy_floor) where
  policy_floor is derived from the 25th percentile of margin in the segment.
- `revenue_predict` / `price_elasticity`: plain supervised regression on the chosen measure.
- `demand_forecast`: sort by the datetime role, build lag/rolling/seasonal features per series
  (`series_col` from a dimension role if given, else global). Hold out the last 20% of time as test.
- `churn_risk`: needs a customer id + date + money measure. Compute per-customer last purchase date;
  label = "no purchase in the last X% of the observed window" (X from `options["churn_window_pct"]`,
  default 25). Aggregate features per customer (recency, frequency, monetary, trend).
- `credit_default`: if a binary/label column is chosen as the outcome role use it directly; otherwise
  derive "late/default" from a status dimension or a days-past-due measure (> 90).
- `transaction_anomaly` / `segmentation`: no target; feature matrix only. For segmentation, prefer
  customer-level aggregation when a customer id role is supplied.

Always: coerce numeric-looking strings to numbers, one-hot / ordinal-encode categoricals later in the
engine (leave them as columns here), cap categorical cardinality (top-N + "other"), drop constant and
all-null columns, and append a plain-language `notes` entry for every transformation you perform.

---

## 8. `mlarena/engine/models.py` — AGENT B

```python
@dataclass
class ModelSpec:
    key: str
    display: str          # "Gradient Boosting"
    persona: str          # "The Step-by-Step Improver"
    plain: str            # 1-2 layman sentences on how it thinks
    strength: str
    weakness: str
    family: str           # "linear" | "tree" | "ensemble" | "boosting" | "neighbour" | "probabilistic" | "neural" | "distance" | "density"
    cost: str             # "fast" | "medium" | "heavy"
    needs_scaling: bool
    build: Callable[[int], object]   # (seed) -> unfitted estimator

def candidates(task: str, *, n_rows: int, n_features: int, budget_s: float) -> list[ModelSpec]
def to_dict(spec: ModelSpec) -> dict     # excludes `build`
```

Minimum model zoo per task (only include libraries that are installed — check with `config.have`):

- **classification**: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`,
  `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, `GradientBoostingClassifier`,
  `XGBClassifier`, `LGBMClassifier`, `KNeighborsClassifier`, `GaussianNB`, `MLPClassifier`
- **regression**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `DecisionTreeRegressor`,
  `RandomForestRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor`,
  `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`, `KNeighborsRegressor`,
  `MLPRegressor`, `seasonal_naive` (custom baseline)
- **anomaly**: `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope`,
  plus a custom `RobustZScoreDetector` (multivariate robust Mahalanobis-style distance)
- **clustering**: `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `AgglomerativeClustering`,
  `GaussianMixture`, `DBSCAN`

`candidates()` must **budget-aware prune**: on large `n_rows` / short `budget_s`, drop `heavy`
models and keep the diverse core so the run always finishes fast. Always keep at least 5 for
supervised tasks. Deterministic given the same inputs (stable ordering).

---

## 9. `mlarena/engine/planner.py` — AGENT B

```python
@dataclass
class Plan:
    task: str
    primary_metric: str
    test_strategy: str        # "stratified_split" | "time_split" | "unsupervised"
    test_size: float
    sample_rows: int | None   # train on a sample when the data is huge
    sample_reason: str
    score_all_rows: bool
    budget_s: float
    models: list[ModelSpec]
    notes: list[str]

def make_plan(uc: UseCase, target: TargetPlan, *, n_rows: int, time_budget_s: float,
              options: dict) -> Plan
```

Sampling rule: if `n_rows > 400_000` train on a stratified sample of 400k; **always score every row**
with the champion afterwards. Record the reason in plain language.

---

## 10. `mlarena/engine/runner.py` — AGENT B (the orchestrator)

```python
Event = dict   # see §11
EmitFn = Callable[[Event], None]

@dataclass
class RunResult:
    run_id: str
    status: str                 # "succeeded" | "failed"
    usecase: dict
    dataset: dict               # {"dataset_id","rows","cols","engine","load_ms","scored_rows"}
    target: dict                # {"task","target_col","label_names","positive_rate","notes"}
    plan: dict                  # {"test_strategy","test_size","sample_rows","sample_reason","budget_s","model_count"}
    leaderboard: list[dict]
    champion: dict
    metrics: dict
    impact: dict                # {"label","value","unit","formula","note"}
    drivers: list[dict]         # [{"feature","importance","direction","plain"}]
    insights: list[dict]        # [{"title","body","severity","icon"}]
    segments: list[dict]        # clustering/anomaly breakdowns, else []
    distributions: dict         # chart data, see §12
    preview: dict               # {"columns":[str], "rows":[[...]]}
    scored_path: str            # parquet/csv of ALL scored rows (with score column)
    scored_columns: list[str]
    exports: dict               # filled by the API layer: {"html","csv","xlsx"}
    narration: list[dict]       # [{"beat","title","text"}] the story, in order
    log: list[Event]            # every event, in order
    error: str | None
    started_at: str             # ISO8601
    finished_at: str
    duration_ms: float

def run(dataset_id: str, usecase_key: str, mapping: dict[str, list[str]],
        options: dict, emit: EmitFn, run_id: str | None = None) -> RunResult
```

`run()` must emit events at every meaningful step so the UI feels alive:
load → profile → prepare → plan (explain which models and **why**) → for each model
(announce persona, plain-language rationale, train, report metric, verdict) → select champion
(explain why it won, referencing the runners-up) → score all rows → drivers → insights → done.

**Every model must be reported even if it fails** — catch per-model exceptions, emit a `warn`
event, and put the model in the leaderboard with `status="failed"` and its `error`. Never let one
bad model kill the run.

`leaderboard` entry:
```python
{"rank": int, "key": str, "display": str, "persona": str, "family": str, "cost": str,
 "status": "ok" | "failed", "error": str | None,
 "metrics": {"roc_auc":..,"accuracy":..,"precision":..,"recall":..,"f1":..,"pr_auc":..,"log_loss":..}
            # or {"r2":..,"mape":..,"rmse":..,"mae":..,"wape":..}
            # or {"silhouette":..,"davies_bouldin":..,"n_clusters":..}
            # or {"contamination":..,"agreement":..}
 "primary": float, "score": float,          # score = normalised 0..100 for the UI bars
 "train_ms": float, "predict_ms": float,
 "why": str}                                # 1-2 plain sentences on how it performed
```

`champion`: the winning leaderboard entry plus
`{"why_won": str, "runner_up": str, "tradeoff": str, "plain_summary": str}`.

`impact`: translate the metric into money/records in plain language. For classification include the
confusion matrix in `metrics` and an explicit sentence like
`"If you review the top 500 flagged orders you would catch ~68% of the leakage."`

`drivers`: from the champion's `feature_importances_` or `coef_` (use permutation importance as a
fallback). Always attach a `plain` sentence naming the feature in business words and its direction.

---

## 11. WebSocket event schema (shared by AGENT A and AGENT B)

```json
{
  "seq": 12,
  "ts": 1758170000123,
  "phase": "load|profile|prepare|plan|train|evaluate|select|score|explain|export|done|error",
  "level": "info|success|warn|error|champion|story",
  "title": "Testing model 3 of 8 — Random Forest",
  "detail": "This model grows hundreds of decision trees and lets them vote...",
  "progress": 0.42,
  "model": "RandomForestClassifier",
  "metrics": {"roc_auc": 0.91},
  "elapsed_ms": 812,
  "payload": {}
}
```
`title` is the bold line, `detail` is the plain-language explanation shown beneath it.
Keep `title` under 80 chars and `detail` to 1-3 short sentences. **Never** use unexplained jargon;
if a technical term is unavoidable, define it in the same sentence.

---

## 12. `distributions` chart payload (AGENT A renders these with offline SVG)

```python
{
 "score_histogram": {"bins": [float], "counts": [int]},          # scores of all scored rows
 "class_balance": {"labels": [str], "values": [int]},            # classification only
 "leaderboard": {"labels": [str], "values": [float]},            # primary metric per model
 "metric_radar": {"axes": [str], "series": [{"name": str, "values": [float]}]},
 "scatter": {"x": [float], "y": [float], "color": [0|1], "x_label": str, "y_label": str},
 "timeseries": {"x": [str], "actual": [float], "predicted": [float]},   # regression/forecast
 "segments": {"labels": [str], "values": [int]},                 # clustering/anomaly
 "feature_impact": {"labels": [str], "values": [float]}
}
```
Only include keys that apply to the task. All values must be JSON-safe.

---

## 13. `mlarena/export/exporters.py` — AGENT D

```python
def export_html(result, out_path: str) -> str
    """Self-contained single-file HTML report: inline CSS, inline SVG charts, no CDN,
    no external fonts, works offline. Dark theme. Includes: header + dataset summary,
    the business question, the story beats, the full model leaderboard with plain-language
    notes, champion card, metric cards, top drivers table, insights, and the top 100
    scored rows. Must be readable by a non-technical executive."""

def export_csv(result, out_path: str) -> str
    """All scored rows as CSV (UTF-8 with BOM so Excel opens it correctly)."""

def export_xlsx(result, out_path: str) -> str
    """Multi-sheet workbook via openpyxl: 'Summary', 'Model Leaderboard', 'Key Drivers',
    'Insights', 'Scored Data' (first 100k rows). Format headers, freeze panes, column widths."""

def export_summary_json(result, out_path: str) -> str
```
`result` is the `RunResult` from §10 (a dataclass; access with `getattr` so a dict also works).

---

## 14. HTTP API (LEAD) — AGENT A consumes this

```
POST /api/datasets                 multipart file field `file` -> LoadResult-ish JSON + profile summary
GET  /api/datasets/{id}            -> {"dataset": {...}, "profile": {...}}
GET  /api/datasets                 -> {"datasets": [...]}
POST /api/datasets/{id}/columns/preview   {name, expression} -> preview() dict
POST /api/datasets/{id}/columns/apply     {columns:[{name,expression}]} -> new LoadResult
GET  /api/datasets/{id}/suggestions       -> {"suggestions": [...]}
GET  /api/usecases                 -> {"usecases": [...], "functions": [...], "operators": [...]}
GET  /api/usecases/{key}           -> {"usecase": {...}}
POST /api/runs                     {dataset_id, usecase, mapping, options} -> {"run_id": "..."}
GET  /api/runs/{run_id}            -> {"run": {...RunResult as dict...}, "events": [...]}
GET  /api/runs/{run_id}/export?fmt=html|csv|xlsx|json -> file download
GET  /api/samples                  -> {"samples": [{"name","path","size_bytes","usecase"}]}
POST /api/samples/{name}/load      -> loads a bundled sample as a dataset -> LoadResult
GET  /api/health                   -> {"ok": true, "version": "..."}
WS   /ws/runs/{run_id}             -> streams §11 events, then {"phase":"done"} and closes
```

---

## 15. Frontend (AGENT A) — `web/`

Vanilla HTML/CSS/JS. **No build step, no CDN, no external fonts** (must work offline).
Dark theme, professional, generous whitespace, subtle depth. Files: `index.html`, `styles.css`,
`app.js`, `charts.js`, `favicon.svg`.

Six steps, always visible as a progress rail:
1. **Start** — hero, product name *ML Arena*, tagline, "Upload your data" drop zone + bundled samples.
2. **Your data** — shape, read time, engine used, column table (kind badge: money / category / date /
   id / text), data-quality headline, and a **"Create a column"** builder with live preview and
   one-click suggestions.
3. **What do you want to find?** — use-case cards grouped by sector, with emoji, tagline and story.
4. **Point us at the right columns** — one picker per role; the picker only offers columns whose
   `kind` is in `role.accepts`; multi-select where `role.multi`; inline "＋ create a column" shortcut.
5. **Watch it work** — live narration console (auto-scroll, colour-coded by level), overall progress
   bar, live model scoreboard that fills in as results arrive, and a friendly "what's happening now"
   banner. Never show a bare spinner.
6. **Your answer** — champion card ("We picked X because…"), plain-language summary, metric cards,
   impact callout, charts, drivers table, insights, top flagged rows, and Export buttons
   (HTML report / CSV / Excel).

Requirements: keyboard accessible, responsive down to 1100px, `prefers-reduced-motion` respected,
graceful error toasts, and a "Start over" action. All copy must be layman-friendly — no
"hyperparameter", "ROC-AUC" or "one-hot encoding" without an immediate plain-English gloss.

---

## 16. Testing

Every agent writes `tests/test_<module>.py` and runs it with
`C:\ML Arena\.venv\Scripts\python.exe -m pytest tests -q` (pytest may be missing — if so, write
plain `assert`-based scripts runnable with `python tests/test_x.py` and say so in your report).

Do not modify files owned by another agent. If you need a change there, note it in your final report.
