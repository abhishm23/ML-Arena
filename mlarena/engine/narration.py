"""ML Arena — the storytelling layer.

**There is no language model in this product.** Every sentence here is a
hand-written template filled with real numbers. That is deliberate: ML Arena runs
entirely on the user's own laptop, offline, and the explanations have to be
identical and reviewable every single time. If a sentence in this file reads
badly, we fix the sentence — we do not ask a model to improvise one.

The rules these templates follow:

1. Never use a technical term without explaining it in the same breath.
   "Separation power (0.5 means the model is guessing, 1.0 means it sorts every
   case perfectly)" — never a bare "ROC-AUC".
2. Always say what the number means for the business, not just what it is.
3. Never oversell. If a model only just beat the alternative, say so.
"""

from __future__ import annotations

from mlarena import config  # noqa: F401  -- env bootstrap


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def beat(key: str, title: str, text: str, **extra) -> dict:
    """One story beat for the narration list."""
    out = {"beat": key, "title": title, "text": text}
    out.update(extra)
    return out


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except Exception:
        return "n/a"


def _num(v, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "n/a"


def metric_phrase(task: str, metric: str, value) -> str:
    """A number, plus what it means, in one breath."""
    if value is None:
        return "could not be measured"
    v = float(value)
    if metric == "roc_auc":
        return (f"separation power {v:.2f} (0.50 means it is guessing, 1.00 means it sorts "
                f"every case perfectly)")
    if metric == "pr_auc":
        return (f"precision-recall score {v:.2f} — how well it finds the rare cases without "
                f"raising false alarms")
    if metric == "r2":
        return (f"explained variation {_pct(v)} — the model accounts for that share of the "
                f"ups and downs in your numbers")
    if metric == "silhouette":
        return (f"group tightness {v:.2f} (above 0.50 is a clear, well-separated set of "
                f"groups; near 0 means the groups overlap)")
    if metric == "consensus":
        return (f"agreement with the crowd {v:.2f} (1.00 means it flags exactly the records "
                f"the other models flag)")
    if metric == "f1":
        return f"balanced score {v:.2f} — a blend of how many it caught and how many it got wrong"
    if metric == "accuracy":
        return f"overall hit rate {_pct(v)}"
    return f"{metric} {_num(v)}"


def _short_metric(task: str, metric: str, value) -> str:
    if value is None:
        return "no measurable score"
    v = float(value)
    if metric == "roc_auc":
        return f"separation power {v:.2f}"
    if metric == "r2":
        return f"explained variation {_pct(v)}"
    if metric == "silhouette":
        return f"group tightness {v:.2f}"
    if metric == "consensus":
        return f"agreement {v:.2f}"
    return f"{metric} {_num(v)}"


def _gap_word(delta: float) -> str:
    a = abs(delta)
    if a < 0.005:
        return "a whisker"
    if a < 0.02:
        return "a little"
    if a < 0.06:
        return "clearly"
    return "by a wide margin"


# --------------------------------------------------------------------------- #
# Per-model narration
# --------------------------------------------------------------------------- #


def model_intro(spec, index: int, total: int, task: str) -> dict:
    """Why this model is in the race and how it thinks."""
    return {
        "title": f"Model {index} of {total}: {spec.display} — {spec.persona}",
        "text": f"{spec.plain} Its strength: {spec.strength.lower()} Its weakness: "
                f"{spec.weakness.lower()}",
    }


def model_verdict(spec, metrics: dict, primary, best_so_far, task: str,
                  primary_metric: str) -> dict:
    """How it did, and whether it beat whoever was leading."""
    if primary is None:
        return {
            "title": f"{spec.display} finished, but there was nothing to measure",
            "text": ("This model ran, but we could not score it fairly — usually because your "
                     "data only contains one of the two outcomes, so there is nothing to "
                     "separate. It will not be considered for the crown."),
        }
    phrase = metric_phrase(task, primary_metric, primary)
    if best_so_far is None:
        text = (f"It scored {phrase}. That is the first result on the board, so it sets the "
                f"bar everyone else has to beat.")
    elif primary > best_so_far:
        text = (f"It scored {phrase}, beating the previous leader "
                f"({_short_metric(task, primary_metric, best_so_far)}) by {_gap_word(primary - best_so_far)}. "
                f"It takes the lead.")
    elif abs(primary - best_so_far) < 1e-9:
        text = (f"It scored {phrase} — dead level with the current leader. When two models "
                f"tie, we prefer the simpler and faster one, because it is easier to trust.")
    else:
        text = (f"It scored {phrase}, behind the current leader "
                f"({_short_metric(task, primary_metric, best_so_far)}) by "
                f"{_gap_word(primary - best_so_far)}. Not the winner this time, but a useful "
                f"second opinion — when very different models agree on a row, that row is "
                f"worth believing.")
    extra = ""
    if task == "classification" and metrics:
        # `all_rows` is the full-file view; the top-level counts are the held-back
        # test split. The full-file numbers are the ones worth quoting here.
        cm = metrics.get("all_rows") or {}
        if cm.get("flagged") is not None and cm.get("actual_positives"):
            extra = (f" At its chosen cut-off it flags {cm['flagged']:,} rows and catches "
                     f"{cm.get('tp', 0):,} of the {cm['actual_positives']:,} real cases.")
    return {"title": f"{spec.display}: {_short_metric(task, primary_metric, primary)}",
            "text": text + extra}


def model_failed(spec, error: str) -> dict:
    return {
        "title": f"{spec.display} could not finish",
        "text": (f"This model hit a problem and was removed from the race: {_plain_error(error)} "
                 f"The other models are unaffected — one bad runner never stops the race."),
    }


def _plain_error(error: str) -> str:
    e = (error or "").lower()
    if "memory" in e:
        return "it ran out of memory on a dataset this size."
    if "single class" in e or "one class" in e or "at least 2 classes" in e:
        return "your data only contains one outcome, so there was nothing for it to separate."
    if "nan" in e or "infinit" in e:
        return "some values in the data could not be used as numbers."
    if "converge" in e:
        return "it could not settle on a stable answer within its time limit."
    if "not enough" in e or "too few" in e:
        return "there were not enough rows for it to learn from."
    return f"the technical reason was: {error}."


# --------------------------------------------------------------------------- #
# The champion
# --------------------------------------------------------------------------- #


def champion_story(champion: dict, leaderboard: list[dict], task: str, impact: dict) -> dict:
    """Why the winner won, what the runner-up offered, and the trade-off."""
    name = champion.get("display", "the winner")
    persona = champion.get("persona", "")
    primary_metric = champion.get("primary_metric") or "roc_auc"
    primary = champion.get("primary")
    others = [e for e in leaderboard
              if e.get("key") != champion.get("key") and e.get("status") == "ok"]
    runner = others[0] if others else None

    if primary is None:
        why = (f"Every model struggled to produce a fair score on this data, so we chose "
               f"{name}{f' ({persona})' if persona else ''} because it finished cleanly and "
               f"gave the steadiest answer.")
    else:
        why = (f"{name} scored {metric_phrase(task, primary_metric, primary)}. "
               f"It came out on top of {len(leaderboard)} models.")
        if runner and runner.get("primary") is not None:
            delta = float(primary) - float(runner["primary"])
            if abs(delta) < 0.005:
                why += (f" It beat {runner.get('display')} by the narrowest of margins, so "
                        f"either would serve you well — we went with {name}"
                        f"{f' ({persona})' if persona else ''} as the steadier of the two.")
            else:
                why += (f" The next best was {runner.get('display')}, "
                        f"{_gap_word(delta)} behind.")

    tradeoff = ""
    if runner:
        c_ms = float(champion.get("train_ms") or 0)
        r_ms = float(runner.get("train_ms") or 0)
        if r_ms > 0 and c_ms > 0 and c_ms > r_ms * 1.6:
            tradeoff = (f"{name} took about {c_ms / 1000:.1f} seconds to train against "
                        f"{r_ms / 1000:.1f} seconds for {runner.get('display')}. You paid for "
                        f"the extra accuracy in waiting time.")
        elif champion.get("family") in ("neural", "ensemble", "boosting"):
            tradeoff = (f"{name} is a black box: it gives you the answer and a ranked list of "
                        f"what mattered, but not a single sentence of the form 'if this, then "
                        f"that'. If you need rules you can hand to a colleague, "
                        f"{runner.get('display')} is the more explainable choice.")
        else:
            tradeoff = (f"{name} is the simpler of the top two, which is exactly what you want "
                        f"when two models are this close.")
    elif champion.get("family") in ("neural", "ensemble", "boosting"):
        tradeoff = ("It is a black box — it will not hand you a simple rule of thumb, only a "
                    "ranked list of what mattered.")

    summary = champion.get("why") or ""
    if impact and impact.get("note"):
        summary = (summary + " " + impact["note"]).strip()

    return {
        "title": f"{name} wins — {persona}" if persona else f"{name} wins",
        "text": why,
        "why_won": why,
        "runner_up": (f"{runner.get('display')} ({_short_metric(task, primary_metric, runner.get('primary'))})"
                      if runner else ""),
        "tradeoff": tradeoff,
        "plain_summary": summary,
    }


def champion_beat(champion: dict, story: dict) -> dict:
    return beat(
        "champion",
        story["title"],
        story["text"] + (f" Trade-off: {story['tradeoff']}" if story.get("tradeoff") else ""),
    )


# --------------------------------------------------------------------------- #
# Drivers and insights
# --------------------------------------------------------------------------- #


def driver_sentence(driver: dict) -> str:
    """One plain sentence naming a driver and which way it pushes."""
    name = driver.get("feature", "this column")
    importance = driver.get("importance")
    direction = driver.get("direction", "neutral")
    kind = driver.get("kind", "number")
    share = f" ({importance:.0f}% of the model's decision-making)" if importance else ""

    if direction == "up":
        tail = (f"Higher values of “{name}” push the answer up." if kind == "number"
                else f"Rows in the “{name}” group tend to sit higher.")
    elif direction == "down":
        tail = (f"Higher values of “{name}” pull the answer down." if kind == "number"
                else f"Rows in the “{name}” group tend to sit lower.")
    elif isinstance(direction, str) and direction.startswith("category:"):
        value = direction.split(":", 1)[1]
        tail = f"“{name}” matters mainly because of the value “{value}”."
    else:
        tail = (f"“{name}” changes the answer, but not consistently in one direction on its "
                f"own.")
    return f"{tail}{share}"


def driver_beats(drivers: list[dict], limit: int = 4) -> list[dict]:
    out = []
    for i, d in enumerate(drivers[:limit], start=1):
        out.append(beat("driver", f"Driver {i}: {d.get('feature')}", driver_sentence(d)))
    return out


# --------------------------------------------------------------------------- #
# One row at a time — "why was THIS one flagged?"
# --------------------------------------------------------------------------- #


def human_value(value, as_percent: bool = False) -> str:
    """Render a value the way a person would read it out loud."""
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"\u201c{value}\u201d"
    if not number == number:  # NaN
        return "missing"
    if as_percent:
        return f"{number * 100:,.1f}%"
    if abs(number - round(number)) < 1e-9 and abs(number) < 1e15:
        return f"{int(round(number)):,}"
    return f"{number:,.4g}"


def local_explanation(feature: str, value, reference, contribution: float, *,
                      is_category: bool, as_percent: bool = False) -> str:
    """One plain sentence: how a single column moved a single row's score.

    `contribution` is how much the score would fall if this column held the value a
    typical record has. Positive means the column pushed this row *up* the list.
    """
    moved = abs(float(contribution or 0.0))
    here = human_value(value, as_percent)
    typical = human_value(reference, as_percent)

    if moved < 5e-4:
        return (f"Its \u201c{feature}\u201d was {here}, about what a typical record "
                f"looks like, so it did not change this row's score.")

    if is_category:
        lead = (f"This record sits in \u201c{feature}\u201d = {here}, where a typical "
                f"record is {typical}.")
    else:
        lead = (f"Its \u201c{feature}\u201d was {here}, where a typical record is "
                f"{typical}.")

    if contribution > 0:
        tail = f"On its own that pushed this row's score up by {moved:.2f}."
    else:
        tail = f"On its own that pulled this row's score down by {moved:.2f}."
    return f"{lead} {tail}"


def row_summary(score, top: list[dict], label: str = "") -> str:
    """The one-line answer to 'why this row?'"""
    if not top:
        return (f"This record scored {float(score):.2f}, but no single column stands out "
                f"as the reason — it is the combination that does it.")
    names = [f"\u201c{t.get('feature')}\u201d" for t in top[:2]]
    if len(names) == 1:
        subject = names[0]
    else:
        subject = f"{names[0]} and {names[1]}"
    return (f"Scored {float(score):.2f}. What pushed it up most was {subject}.")


# --------------------------------------------------------------------------- #
# Metric glossary (used by the UI for tooltips)
# --------------------------------------------------------------------------- #


def metric_glossary() -> dict:
    """Every metric we report, explained the way a shop owner would explain it."""
    return {
        "roc_auc": ("Separation power. How well the model sorts the risky rows above the safe "
                    "ones. 0.5 means it is guessing; 1.0 means every risky row is ranked above "
                    "every safe one."),
        "pr_auc": ("How well the model finds the rare cases without dragging in false alarms. "
                   "More useful than the overall hit rate when the thing you are hunting is "
                   "uncommon."),
        "accuracy": ("The share of rows the model got right. Misleading when one outcome is "
                     "rare — predicting 'nothing ever goes wrong' is 99% accurate and useless."),
        "precision": ("Of the rows we flagged, the share that really were a problem."),
        "recall": ("Of the rows that really were a problem, the share we managed to flag."),
        "f1": "A single balanced score combining precision and recall.",
        "log_loss": ("How over-confident the model's probabilities are. Lower is better; it "
                     "punishes confident wrong answers."),
        "brier": "Average squared gap between the predicted risk and what actually happened.",
        "r2": ("Explained variation. The share of the ups and downs in your number that the "
               "model reproduces. 100% is perfect; 0% means it does no better than always "
               "quoting the average."),
        "rmse": ("Typical size of the miss, in the same units as your column, with big misses "
                 "counted more heavily."),
        "mae": "Typical size of the miss, in the same units as your column.",
        "mape": "Average miss as a percentage of the actual value. Not used when values can be zero.",
        "wape": ("Total error as a share of the total actual — the safe version of a percentage "
                 "error when your column contains zeros."),
        "smape": "A symmetric percentage error that stays sensible when values are small.",
        "silhouette": ("Group tightness. How well-separated the groups are. Above 0.5 is clear, "
                       "around 0 means the groups overlap."),
        "davies_bouldin": "How much the groups overlap. Lower is better.",
        "consensus": ("Agreement with the crowd. Since there is no answer key for unusual "
                      "records, we measure how closely a model agrees with the other models."),
        "agreement": "How closely this model's ranking matches the average of all the models.",
        "contamination": "The share of records this model calls unusual.",
        "lift": ("How much more often a flagged row is a real problem than a randomly chosen "
                 "row would be. A lift of 5 means five times better than picking at random."),
        "topk_capture": ("If a human only reviews the top K flagged rows, this is the share of "
                         "all the real problems they would catch."),
        "feature_importance": ("How much the model relied on each column. Percentages add up to "
                               "100% across the columns shown."),
        "test_size": ("The share of your rows kept back and never shown to the model, used to "
                      "check whether it works on data it has not seen."),
        "sample_rows": ("When a file is very large we train on a fair sample and then score "
                        "every row, so the answers still cover your whole file."),
    }


# --------------------------------------------------------------------------- #
# Run-level beats
# --------------------------------------------------------------------------- #


def opening_beat(uc_name: str, question: str, rows: int, cols: int) -> dict:
    return beat(
        "start",
        f"{uc_name} — let us find out",
        f"You asked: “{question}” We are looking at {rows:,} rows across {cols} columns. "
        f"Nothing leaves this computer.",
    )


def data_beat(notes: list[str]) -> dict:
    text = notes[0] if notes else "Your data is loaded and cleaned."
    extra = ""
    if len(notes) > 1:
        extra = f" {len(notes) - 1} more preparation steps are listed in the log below."
    return beat("data", "Getting your data ready", text + extra)


def plan_beat(plan) -> dict:
    names = ", ".join(m.display for m in plan.models)
    return beat(
        "plan",
        f"{len(plan.models)} models are about to race",
        f"{names}. Each one attacks the problem differently — a straight line, a committee of "
        f"trees, a step-by-step improver — so you can see whether the answer depends on the "
        f"method or is a real pattern in your data.",
    )


def scoring_beat(rows: int) -> dict:
    return beat(
        "score",
        f"Scoring all {rows:,} rows",
        "The winner is now going back over every row in your file — including the rows we held "
        "back for testing — and giving each one a score. That is what you will see in the "
        "results table and the downloadable file.",
    )


def budget_beat(elapsed: float, budget: float, ran: int, skipped: int) -> dict:
    return beat(
        "budget",
        "Time is up — wrapping up",
        f"We used {elapsed:.0f} of the {budget:.0f} seconds allowed, so {skipped} model(s) did "
        f"not get a turn. The {ran} that finished are more than enough to pick a winner from. "
        f"Give the run a bigger time budget if you want everyone to race.",
    )


def impact_beat(impact: dict) -> dict:
    return beat("impact", impact.get("label", "What this is worth"), impact.get("note", ""))


def closing_beat(champion_name: str) -> dict:
    return beat(
        "done",
        "That is the answer",
        f"{champion_name} won the race. Scroll down for what it found, why it thinks so, and "
        f"the rows you should look at first. Everything here was computed on this machine.",
    )


__all__ = [
    "beat",
    "metric_phrase",
    "metric_glossary",
    "model_intro",
    "model_verdict",
    "model_failed",
    "champion_story",
    "champion_beat",
    "driver_sentence",
    "driver_beats",
    "opening_beat",
    "data_beat",
    "plan_beat",
    "scoring_beat",
    "budget_beat",
    "impact_beat",
    "closing_beat",
]
