"""ML Arena — the run plan (contract §9).

Before a single model trains, we decide: how do we test honestly, how many rows
do we train on, how long are we allowed to take, and which models deserve a place
on the starting line.

The one judgement call worth spelling out is sampling. Training on 4 million rows
and testing on the same 4 million rows would take minutes and tell us nothing —
the model would simply be graded on data it had already memorised. So we train on
a fair sample and always score every row afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlarena import config  # noqa: F401  -- env bootstrap

from mlarena.engine import models as model_zoo
from mlarena.engine.models import ModelSpec
from mlarena.usecases.definitions import TargetPlan
from mlarena.usecases.registry import UseCase

TRAIN_SAMPLE_CAP = config.TRAIN_SAMPLE_THRESHOLD      # 400_000 rows
MIN_ROWS_PER_MODEL = 40


@dataclass
class Plan:
    task: str
    primary_metric: str
    test_strategy: str        # "stratified_split" | "time_split" | "unsupervised"
    test_size: float
    sample_rows: int | None
    sample_reason: str
    score_all_rows: bool
    budget_s: float
    models: list[ModelSpec]
    notes: list[str] = field(default_factory=list)


def _human(n: int) -> str:
    return f"{n:,}"


def make_plan(uc: UseCase, target: TargetPlan, *, n_rows: int, time_budget_s: float,
              options: dict | None = None) -> Plan:
    """Decide how to run this particular race."""
    options = options or {}
    notes: list[str] = []
    task = target.task

    budget = float(time_budget_s or options.get("time_budget_s") or config.DEFAULT_TIME_BUDGET_S)
    budget = max(8.0, min(budget, 900.0))

    # ---- how do we test? --------------------------------------------------- #
    if task == "classification":
        test_strategy, test_size = "stratified_split", float(options.get("test_size", 0.2))
        if target.positive_rate is not None and 0 < target.positive_rate < 0.5:
            notes.append(
                f"1 row in 5 is held back for testing, chosen so that the rare outcome "
                f"({target.positive_rate * 100:.1f}% of your rows) appears in the same "
                f"proportion on both sides. Otherwise the test would be unfair."
            )
        else:
            notes.append("1 row in 5 is held back for testing, picked at random.")
    elif task == "regression" and target.extra.get("time_col"):
        test_strategy, test_size = "time_split", float(options.get("test_size", 0.2))
        notes.append(
            "The most recent 1 row in 5 is held back as the test. We never test a forecast "
            "on the past — that would be marking your own homework."
        )
    elif task == "regression":
        test_strategy, test_size = "stratified_split", float(options.get("test_size", 0.2))
        notes.append("1 row in 5 is held back for testing, picked at random.")
    else:
        test_strategy, test_size = "unsupervised", 0.0
        notes.append(
            "There is no right answer to mark against here, so every model sees every row. "
            "We compare the models with each other instead."
        )

    test_size = max(0.1, min(0.4, test_size))
    if n_rows < MIN_ROWS_PER_MODEL * 3 and test_strategy != "unsupervised":
        test_size = 0.25
        notes.append(
            f"With only {_human(n_rows)} rows we kept the test at a quarter of the data, "
            f"which is the smallest honest test we can run."
        )

    # ---- how much data do we train on? ------------------------------------ #
    sample_rows: int | None = None
    if n_rows > TRAIN_SAMPLE_CAP and task in ("classification", "regression"):
        sample_rows = TRAIN_SAMPLE_CAP
        sample_reason = (
            f"Your file has {_human(n_rows)} rows, which is more than the {_human(TRAIN_SAMPLE_CAP)} "
            f"we can train on quickly. We trained on a fair sample of {_human(TRAIN_SAMPLE_CAP)} "
            f"rows and then scored every single row of your file, so nothing was left out of "
            f"the answer."
        )
    elif n_rows > TRAIN_SAMPLE_CAP:
        sample_reason = (
            f"Your file has {_human(n_rows)} rows. Every model still sees every row, because "
            f"these models are fast, but we cap the work internally to keep the run snappy."
        )
    else:
        sample_reason = (
            f"Your file has {_human(n_rows)} rows, which fits comfortably in memory, so every "
            f"model trained on the whole thing."
        )
    notes.append(sample_reason)

    # ---- who is racing? ---------------------------------------------------- #
    n_features = len(target.numeric_cols) + len(target.categorical_cols)
    zoo = model_zoo.candidates(task, n_rows=sample_rows or n_rows, n_features=n_features,
                               budget_s=budget)

    # the "same time last period" rule is a benchmark, not a competitor
    if target.baseline == "seasonal_naive":
        zoo = [m for m in zoo if m.family != "baseline"] + [
            m for m in zoo if m.family == "baseline"]
    else:
        zoo = [m for m in zoo if m.family != "baseline"]

    if not zoo:
        raise ValueError(
            "We could not find any model we are able to run on this data. Check that the "
            "columns you picked contain real variation."
        )

    notes.append(
        f"{len(zoo)} different models will race: "
        + ", ".join(m.display for m in zoo[:6])
        + (f" and {len(zoo) - 6} more" if len(zoo) > 6 else "")
        + ". Each one thinks about the problem differently, which is what makes the "
          "comparison worth doing."
    )
    if budget < 45:
        notes.append(
            f"We have a {budget:.0f} second budget, so the slowest models sat this one out."
        )

    return Plan(
        task=task,
        primary_metric=uc.primary_metric,
        test_strategy=test_strategy,
        test_size=test_size,
        sample_rows=sample_rows,
        sample_reason=sample_reason,
        score_all_rows=True,
        budget_s=budget,
        models=zoo,
        notes=notes,
    )


__all__ = ["Plan", "make_plan", "TRAIN_SAMPLE_CAP"]
