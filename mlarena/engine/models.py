"""ML Arena — the model zoo (contract §8).

Every model in the race gets a *persona* ("The Committee of Experts") and a plain
description of how it thinks. Those strings are shown to the user while the model
trains, so they matter as much as the estimator itself.

Two things in here are not off-the-shelf scikit-learn estimators, and both are
deliberate:

* ``RobustZScoreDetector`` — a multivariate robust-distance outlier detector. It
  uses the median and the median absolute deviation instead of the mean and the
  standard deviation, so a handful of extreme rows cannot drag the "normal" centre
  towards themselves the way they do with a plain z-score.
* ``SeasonalNaiveRegressor`` — the "same time last period" rule. It is the honest
  benchmark a forecast has to beat: if a gradient-boosted model cannot beat
  "whatever happened last week", it has not earned the user's trust.

Both expose the usual ``fit`` / ``predict`` surface so the runner can treat them
exactly like every other competitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mlarena import config  # noqa: F401  -- env bootstrap before heavy imports

_SK = config.get_module("sklearn")
_XGB = config.get_module("xgboost")
_LGBM = config.get_module("lightgbm")

# --------------------------------------------------------------------------- #
# Estimator imports (guarded: the zoo adapts to whatever is installed)
# --------------------------------------------------------------------------- #

if _SK is not None:
    from sklearn.base import BaseEstimator, OutlierMixin, RegressorMixin
    from sklearn.cluster import (
        AgglomerativeClustering,
        BisectingKMeans,
        DBSCAN,
        KMeans,
        MiniBatchKMeans,
    )
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

if _XGB is not None:
    from xgboost import XGBClassifier, XGBRegressor

if _LGBM is not None:
    from lightgbm import LGBMClassifier, LGBMRegressor

# Our two hand-written estimators have to slot into a scikit-learn Pipeline, which
# asks every step for its "tags". Inheriting from scikit-learn's base classes is the
# supported way to answer those questions; without them the Pipeline refuses to run.
if _SK is not None:
    _OutlierBase = (OutlierMixin, BaseEstimator)
    _RegressorBase = (RegressorMixin, BaseEstimator)
else:  # pragma: no cover - environment guarantee
    _OutlierBase = (object,)          # type: ignore[assignment]
    _RegressorBase = (object,)        # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# ModelSpec
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    key: str
    display: str
    persona: str
    plain: str
    strength: str
    weakness: str
    family: str          # linear|tree|ensemble|boosting|neighbour|probabilistic|neural|distance|density|baseline
    cost: str            # fast|medium|heavy
    needs_scaling: bool
    build: Callable[[int], object]


def to_dict(spec: ModelSpec) -> dict:
    """JSON-safe view (the callable is not serialisable)."""
    return {
        "key": spec.key,
        "display": spec.display,
        "persona": spec.persona,
        "plain": spec.plain,
        "strength": spec.strength,
        "weakness": spec.weakness,
        "family": spec.family,
        "cost": spec.cost,
        "needs_scaling": bool(spec.needs_scaling),
    }


# --------------------------------------------------------------------------- #
# Custom estimators
# --------------------------------------------------------------------------- #


class RobustZScoreDetector(*_OutlierBase):
    """Multivariate outlier detector built on median/MAD instead of mean/std.

    A plain z-score uses the average and the spread, both of which are themselves
    dragged around by the very outliers we are hunting. Using the median and the
    median absolute deviation makes the centre of "normal" almost immune to a few
    extreme rows. Each row then gets one number: how far it sits from that centre
    once we account for the fact that the columns move together.
    """

    _estimator_type = "outlier_detector"

    def __init__(self, contamination: float = 0.05, clip: float = 12.0, random_state: int = 42):
        self.contamination = contamination
        self.clip = clip
        self.random_state = random_state

    # -- helpers ----------------------------------------------------------- #
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        Z = (X - self.median_) / self.mad_
        return np.clip(Z, -self.clip, self.clip)

    # -- sklearn-ish API --------------------------------------------------- #
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.n_features_in_ = X.shape[1]
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0) * 1.4826
        mad = np.where(mad <= 1e-12, 1.0, mad)          # constant column -> no vote
        self.mad_ = mad
        Z = self._standardise(X)
        if Z.shape[1] == 1:
            cov = np.array([[float(np.var(Z[:, 0]) or 1.0)]])
        else:
            cov = np.cov(Z, rowvar=False)
            cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0))
            cov = cov + np.eye(cov.shape[0]) * 1e-6     # tiny ridge for invertibility
        self.cov_ = cov
        try:
            self.prec_ = np.linalg.pinv(cov)
        except Exception:
            self.prec_ = np.eye(cov.shape[0])
        d = self._distance(X)
        self.threshold_ = float(np.quantile(d, 1.0 - self.contamination))
        return self

    def _distance(self, X) -> np.ndarray:
        Z = self._standardise(X)
        # squared Mahalanobis distance, computed without materialising n x n
        d2 = np.einsum("ij,jk,ik->i", Z, self.prec_, Z, optimize=True)
        return np.sqrt(np.maximum(d2, 0.0))

    def decision_function(self, X) -> np.ndarray:
        """Higher = more normal (matches the scikit-learn convention)."""
        return self.threshold_ - self._distance(X)

    def score_samples(self, X) -> np.ndarray:
        return -0.5 * (self._distance(X) ** 2)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) < 0.0, -1, 1)

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).predict(X)

    def get_params(self, deep: bool = True) -> dict:
        return {"contamination": self.contamination, "clip": self.clip,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class SeasonalNaiveRegressor(*_RegressorBase):
    """The "same time last period" rule, as a model.

    To forecast next week it simply repeats what happened last week. It is the
    benchmark every other forecaster must beat; if the fancy models cannot beat it,
    the honest answer is to keep using this rule.
    """

    _estimator_type = "regressor"

    def __init__(self, season_length: int = 7, lag_col: str | None = None, random_state: int = 42):
        self.season_length = int(season_length)
        self.lag_col = lag_col
        self.random_state = random_state

    def _resolve_col(self, X) -> str:
        if not hasattr(X, "columns"):
            raise ValueError(
                "The 'same time last period' rule needs to see the named history columns "
                "(it reads the value from one period ago), so it cannot be wrapped in a "
                "feature pipeline. The runner gives it the raw table instead."
            )
        if self.lag_col and self.lag_col in X.columns:
            return self.lag_col
        wanted = f"lag_{self.season_length}"
        if wanted in X.columns:
            return wanted
        lag_cols = [c for c in X.columns if str(c).startswith("lag_")]
        if lag_cols:
            # prefer the lag closest to the seasonal period
            def _dist(name: str) -> int:
                try:
                    return abs(int(str(name).split("_")[-1]) - self.season_length)
                except Exception:
                    return 10 ** 6

            return sorted(lag_cols, key=_dist)[0]
        raise ValueError(
            "The 'same time last period' rule needs a previous-period column, but none "
            "was available. This usually means there is not enough history."
        )

    def fit(self, X, y=None):
        self.lag_col_ = self._resolve_col(X)
        y = np.asarray(y, dtype=float)
        self.fallback_ = float(np.nanmedian(y)) if y.size else 0.0
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        col = np.asarray(X[self.lag_col_], dtype=float)
        col = np.where(np.isfinite(col), col, self.fallback_)
        return col

    def get_params(self, deep: bool = True) -> dict:
        return {"season_length": self.season_length, "lag_col": self.lag_col,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# --------------------------------------------------------------------------- #
# Feature pipeline
# --------------------------------------------------------------------------- #


def make_pipeline(estimator, numeric_cols: list[str], categorical_cols: list[str],
                  needs_scaling: bool):
    """Wrap an estimator with imputation + one-hot encoding (+ scaling when needed).

    The runner hands this a plain pandas frame with named columns; everything that
    has to happen between "raw business data" and "numbers a model can chew" happens
    here. Trees do not care about scale, so we skip the scaler for them — that is
    pure speed, not a modelling choice.
    """
    if _SK is None:  # pragma: no cover - environment guarantee
        return estimator

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_cols = list(numeric_cols or [])
    categorical_cols = list(categorical_cols or [])
    if not numeric_cols and not categorical_cols:
        return estimator

    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if needs_scaling:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    transformers = [("num", num_pipe, numeric_cols)]
    if categorical_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=0.01, max_categories=30,
                                     sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    return Pipeline([("prepare", ColumnTransformer(transformers, remainder="drop")),
                     ("model", estimator)])


# --------------------------------------------------------------------------- #
# Model catalogue builders
# --------------------------------------------------------------------------- #


def _clf(key, display, persona, plain, strength, weakness, family, cost, scaling, fn) -> ModelSpec:
    return ModelSpec(key, display, persona, plain, strength, weakness, family, cost, scaling, fn)


def _classification_zoo() -> list[ModelSpec]:
    zoo: list[ModelSpec] = []
    if _SK is None:
        return zoo

    zoo.append(_clf(
        "logistic", "Logistic Regression", "The Straight-Line Thinker",
        "It draws one straight boundary through your data and says which side each "
        "record falls on. Simple, quick, and easy to argue with — which is exactly "
        "why we always put it in the race.",
        "Very fast, gives clean and explainable weightings for every column.",
        "Only sees straight-line patterns, so it misses rules of the form 'only when "
        "both of these are true'.",
        "linear", "fast", True,
        lambda seed: LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                        random_state=seed),
    ))
    zoo.append(_clf(
        "decision_tree", "Decision Tree", "The Twenty-Questions Player",
        "It asks a chain of yes/no questions — is the discount above 30%? is this a "
        "wholesale account? — until it can make a call. You can read the questions "
        "afterwards and disagree with them.",
        "Reads like a checklist a human can follow and audit.",
        "On its own it tends to memorise your data, so it can look better than it is.",
        "tree", "fast", False,
        lambda seed: DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                            random_state=seed),
    ))
    zoo.append(_clf(
        "random_forest", "Random Forest", "The Committee of Experts",
        "It grows hundreds of different decision trees on random slices of your data "
        "and lets them vote. A single tree can be wrong; a few hundred disagreeing "
        "with each other usually are not.",
        "Reliable and hard to fool, works well straight out of the box.",
        "You cannot read a single rule off it, and it is slower than one tree.",
        "ensemble", "medium", False,
        lambda seed: RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                            min_samples_leaf=2, max_features="sqrt",
                                            random_state=seed),
    ))
    zoo.append(_clf(
        "extra_trees", "Extra Trees", "The Impulsive Committee",
        "The same idea as the committee of experts, except each tree is allowed to "
        "guess its split points at random instead of hunting for the best one. It "
        "trades a little accuracy for a lot of speed.",
        "Faster than a random forest and often just as good on messy business data.",
        "The randomness can blur fine detail in small datasets.",
        "ensemble", "medium", False,
        lambda seed: ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                                          min_samples_leaf=2, max_features="sqrt",
                                          random_state=seed),
    ))
    zoo.append(_clf(
        "hist_gb", "Gradient Boosting", "The Step-by-Step Improver",
        "It builds a small tree, looks at the mistakes it made, and builds the next "
        "tree specifically to fix them. Repeat a couple of hundred times and you get "
        "something remarkably accurate.",
        "Usually the most accurate model on table-shaped business data, and still fast.",
        "It can over-fit if you let it run too long — which is why we cap it and stop early.",
        "boosting", "medium", False,
        lambda seed: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                    early_stopping=True,
                                                    validation_fraction=0.12,
                                                    n_iter_no_change=15,
                                                    random_state=seed),
    ))
    zoo.append(_clf(
        "gradient_boosting", "Classic Gradient Boosting", "The Careful Craftsman",
        "The same step-by-step improvement idea, but built one careful tree at a "
        "time instead of in fast histogram batches. Slower, and occasionally a "
        "little sharper on small datasets.",
        "Very strong on small and medium datasets.",
        "Slow — it is the first thing we drop when your file is large.",
        "boosting", "heavy", False,
        lambda seed: GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                max_depth=3, random_state=seed),
    ))
    if _XGB is not None:
        zoo.append(_clf(
            "xgboost", "XGBoost", "The Championship Booster",
            "The competition-grade version of step-by-step improvement, with extra "
            "polish on how it handles missing values and uneven data. It is the "
            "model that keeps winning public competitions.",
            "Excellent accuracy, handles missing values natively.",
            "Many knobs, and it uses more memory than the simpler boosters.",
            "boosting", "heavy", False,
            lambda seed: XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6,
                                       subsample=0.9, colsample_bytree=0.9,
                                       tree_method="hist", n_jobs=-1,
                                       random_state=seed, eval_metric="logloss",
                                       verbosity=0),
        ))
    if _LGBM is not None:
        zoo.append(_clf(
            "lightgbm", "LightGBM", "The Speedy Booster",
            "Another step-by-step improver, engineered for speed. It grows trees in "
            "a lopsided way that suits wide business tables, so it finishes fast even "
            "on big files.",
            "The fastest strong model on large data.",
            "On very small datasets it can be jumpier than the alternatives.",
            "boosting", "medium", False,
            lambda seed: LGBMClassifier(n_estimators=300, learning_rate=0.08,
                                        num_leaves=31, subsample=0.9,
                                        colsample_bytree=0.9, n_jobs=-1,
                                        random_state=seed, verbose=-1),
        ))
    zoo.append(_clf(
        "knn", "K-Nearest Neighbours", "The Look-Alike Finder",
        "To judge a record it finds the most similar records it has seen before and "
        "goes with the majority answer. No rules at all — just resemblance.",
        "Captures odd local patterns that rule-based models miss.",
        "Gets slow as your file grows, and needs the columns put on the same scale.",
        "neighbour", "heavy", True,
        lambda seed: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
    ))
    zoo.append(_clf(
        "naive_bayes", "Naive Bayes", "The Quick Guesser",
        "It treats every column as independent evidence and multiplies the "
        "probabilities together. The assumption is usually wrong, yet the answer is "
        "often surprisingly good — and it is nearly instant.",
        "Blazing fast, a good sanity check on everything else.",
        "Ignores the fact that columns are usually related to each other.",
        "probabilistic", "fast", True,
        lambda seed: GaussianNB(),
    ))
    zoo.append(_clf(
        "mlp", "Neural Network", "The Mini Brain",
        "A small network of layered 'neurons' that learns curvy, tangled "
        "relationships the other models cannot express. Ours is deliberately small, "
        "because a big one would be slow and would memorise your data.",
        "Can pick up complex interactions between columns.",
        "Needs the columns scaled, and is the slowest model in the race.",
        "neural", "heavy", True,
        lambda seed: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120,
                                   early_stopping=True, n_iter_no_change=8,
                                   validation_fraction=0.12, alpha=1e-3,
                                   random_state=seed),
    ))
    return zoo


def _regression_zoo() -> list[ModelSpec]:
    zoo: list[ModelSpec] = []
    if _SK is None:
        return zoo

    zoo.append(_clf(
        "linear", "Linear Regression", "The Straight-Line Thinker",
        "It fits one straight relationship between your inputs and the number you "
        "are predicting. If the answer were simply 'each extra unit adds 4 rupees', "
        "this is the model that would say so.",
        "Instant, and every coefficient is a sentence you can read out loud.",
        "Misses anything curved, and one extreme row can tilt the whole line.",
        "linear", "fast", True,
        lambda seed: LinearRegression(),
    ))
    zoo.append(_clf(
        "ridge", "Ridge Regression", "The Cautious Line Fitter",
        "A straight-line fit with the volume turned down: it deliberately shrinks "
        "every effect a little so that no single column can dominate. That makes it "
        "much steadier when your columns overlap.",
        "Stable when columns are correlated or numerous.",
        "Shrinks real effects slightly, so it is rarely the single best number.",
        "linear", "fast", True,
        lambda seed: Ridge(alpha=1.0, random_state=seed),
    ))
    zoo.append(_clf(
        "lasso", "Lasso Regression", "The Ruthless Editor",
        "Also a straight line, but it deletes the columns that are not pulling their "
        "weight. You end up with a short list of inputs that actually matter.",
        "Produces a short, quotable list of drivers.",
        "Can throw away a genuinely useful column if it overlaps with another.",
        "linear", "fast", True,
        lambda seed: Lasso(alpha=0.001, max_iter=5000, random_state=seed),
    ))
    zoo.append(_clf(
        "elasticnet", "Elastic Net", "The Balanced Editor",
        "A middle path between the cautious fitter and the ruthless editor: it "
        "shrinks weak effects and drops the useless ones without being brutal about "
        "overlapping columns.",
        "Good when you have many overlapping inputs.",
        "Two dials to balance, so it is rarely the outright winner.",
        "linear", "fast", True,
        lambda seed: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000,
                                random_state=seed),
    ))
    zoo.append(_clf(
        "decision_tree", "Decision Tree", "The Twenty-Questions Player",
        "It splits your data with a chain of yes/no questions until each group is "
        "consistent, then predicts the average of the group a record lands in.",
        "Follows bends and thresholds that a straight line cannot.",
        "Left alone it memorises noise, so we cap its depth.",
        "tree", "fast", False,
        lambda seed: DecisionTreeRegressor(max_depth=12, min_samples_leaf=5,
                                           random_state=seed),
    ))
    zoo.append(_clf(
        "random_forest", "Random Forest", "The Committee of Experts",
        "Hundreds of different trees each make a prediction and the forest averages "
        "them. Averaging is what turns an unreliable single tree into a dependable "
        "answer.",
        "Strong, stable, and forgiving of messy columns.",
        "Cannot extrapolate beyond the range it has seen, and takes longer to train.",
        "ensemble", "medium", False,
        lambda seed: RandomForestRegressor(n_estimators=150, n_jobs=-1,
                                           min_samples_leaf=2, max_features=1.0 / 3.0,
                                           random_state=seed),
    ))
    zoo.append(_clf(
        "extra_trees", "Extra Trees", "The Impulsive Committee",
        "The same voting idea, but each tree picks its split points at random rather "
        "than searching for the best one — quicker to build, nearly as accurate.",
        "Fast for how accurate it is.",
        "Slightly noisier predictions than a tuned forest.",
        "ensemble", "medium", False,
        lambda seed: ExtraTreesRegressor(n_estimators=150, n_jobs=-1,
                                         min_samples_leaf=2, max_features=1.0 / 3.0,
                                         random_state=seed),
    ))
    zoo.append(_clf(
        "hist_gb", "Gradient Boosting", "The Step-by-Step Improver",
        "It builds a small tree, studies where it was wrong, and builds the next tree "
        "to correct exactly those rows. Two hundred rounds of that gets very close to "
        "the truth.",
        "Usually the most accurate option on business tables.",
        "Needs guarding against over-fitting, which we do by stopping it early.",
        "boosting", "medium", False,
        lambda seed: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                                   early_stopping=True,
                                                   validation_fraction=0.12,
                                                   n_iter_no_change=15,
                                                   random_state=seed),
    ))
    zoo.append(_clf(
        "gradient_boosting", "Classic Gradient Boosting", "The Careful Craftsman",
        "The original step-by-step improver: one small tree at a time, each fixing "
        "the last one's mistakes.",
        "Excellent on small datasets.",
        "Slow — we drop it first when your file is big.",
        "boosting", "heavy", False,
        lambda seed: GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                               max_depth=3, random_state=seed),
    ))
    if _XGB is not None:
        zoo.append(_clf(
            "xgboost", "XGBoost", "The Championship Booster",
            "The competition-grade step-by-step improver, with careful handling of "
            "missing values built in.",
            "Top accuracy and native missing-value handling.",
            "Heavier on memory than the simpler boosters.",
            "boosting", "heavy", False,
            lambda seed: XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=6,
                                      subsample=0.9, colsample_bytree=0.9,
                                      tree_method="hist", n_jobs=-1,
                                      random_state=seed, verbosity=0),
        ))
    if _LGBM is not None:
        zoo.append(_clf(
            "lightgbm", "LightGBM", "The Speedy Booster",
            "A step-by-step improver tuned for speed on wide tables, so it stays "
            "quick even when your file is large.",
            "The fastest of the strong models.",
            "Can be jumpy on very small datasets.",
            "boosting", "medium", False,
            lambda seed: LGBMRegressor(n_estimators=300, learning_rate=0.08,
                                       num_leaves=31, subsample=0.9,
                                       colsample_bytree=0.9, n_jobs=-1,
                                       random_state=seed, verbose=-1),
        ))
    zoo.append(_clf(
        "knn", "K-Nearest Neighbours", "The Look-Alike Finder",
        "It predicts by finding the most similar records it has already seen and "
        "averaging their answers. No formula — just resemblance.",
        "Picks up local patterns that formulas miss.",
        "Slow on big files, and it needs the columns on a common scale.",
        "neighbour", "heavy", True,
        lambda seed: KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
    ))
    zoo.append(_clf(
        "mlp", "Neural Network", "The Mini Brain",
        "A small layered network that learns curved, tangled relationships between "
        "your inputs and the number you are predicting.",
        "Captures complex interactions between columns.",
        "Needs scaling and is the slowest model here.",
        "neural", "heavy", True,
        lambda seed: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100,
                                  early_stopping=True, n_iter_no_change=8,
                                  validation_fraction=0.12, alpha=1e-3,
                                  random_state=seed),
    ))
    zoo.append(_clf(
        "seasonal_naive", "Same Time Last Period", "The 'Same Time Last Year' Rule",
        "It ignores everything except what happened at this point in the previous "
        "cycle, and repeats it. It is the benchmark: if a clever model cannot beat "
        "this, the clever model is not worth using.",
        "Almost free to compute, and hard to beat when your data is strongly seasonal.",
        "Completely blind to trends, promotions, price changes and anything new.",
        "baseline", "fast", False,
        lambda seed: SeasonalNaiveRegressor(season_length=7),
    ))
    return zoo


def _anomaly_zoo() -> list[ModelSpec]:
    zoo: list[ModelSpec] = []
    if _SK is None:
        return zoo

    zoo.append(_clf(
        "isolation_forest", "Isolation Forest", "The Odd-One-Out Spotter",
        "It repeatedly slices the data at random. Ordinary records survive many "
        "slices before being separated out; unusual ones get isolated almost "
        "immediately, which is how it finds them.",
        "Works with no examples of wrongdoing, and scales to large files.",
        "On very high-dimensional data the random slices lose their edge.",
        "ensemble", "medium", False,
        lambda seed: IsolationForest(n_estimators=200, max_samples="auto",
                                     contamination="auto", n_jobs=-1,
                                     random_state=seed),
    ))
    zoo.append(_clf(
        "robust_zscore", "Robust Distance", "The Yardstick Keeper",
        "It measures how far each record sits from the middle of your data, using "
        "the median rather than the average so that a few extreme rows cannot move "
        "the middle towards themselves.",
        "Fast, transparent and easy to explain to a sceptical colleague.",
        "Assumes the normal rows form one blob, so it can over-flag genuinely "
        "separate groups.",
        "distance", "fast", True,
        lambda seed: RobustZScoreDetector(contamination=0.05, random_state=seed),
    ))
    zoo.append(_clf(
        "lof", "Local Outlier Factor", "The Neighbourhood Watch",
        "Instead of asking 'is this record odd overall?', it asks 'is this record odd "
        "compared with its immediate neighbours?'. A quiet street and a busy street "
        "each get judged against their own normal.",
        "Catches records that are strange only relative to their local area.",
        "Slower on large files, and needs a sensible idea of 'neighbourhood'.",
        "density", "heavy", True,
        lambda seed: LocalOutlierFactor(n_neighbors=20, novelty=True,
                                        contamination="auto", n_jobs=-1),
    ))
    zoo.append(_clf(
        "one_class_svm", "One-Class SVM", "The Boundary Drawer",
        "It draws a smooth fence around the bulk of your data and calls anything "
        "outside the fence unusual. The fence can bend to follow complicated shapes.",
        "Flexible boundary that copes with curvy data.",
        "Slow, and sensitive to how the columns are scaled.",
        "distance", "heavy", True,
        lambda seed: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
    ))
    zoo.append(_clf(
        "elliptic_envelope", "Elliptic Envelope", "The Shape Fitter",
        "It fits one big ellipse around the centre of your data and flags whatever "
        "falls outside it. Very quick to picture and explain.",
        "Simple, fast, and easy to draw on a slide.",
        "Assumes your normal data is one neat oval, which is often not true.",
        "distance", "heavy", True,
        lambda seed: EllipticEnvelope(support_fraction=None, contamination=0.05,
                                      random_state=seed),
    ))
    return zoo


def _clustering_zoo() -> list[ModelSpec]:
    zoo: list[ModelSpec] = []
    if _SK is None:
        return zoo

    zoo.append(_clf(
        "kmeans", "K-Means", "The Grouping Sorter",
        "It decides how many groups to make, then repeatedly shuffles every record "
        "into whichever group centre it is closest to until things settle down.",
        "Fast, predictable, and easy to describe to a colleague.",
        "You have to tell it how many groups to look for, and it prefers round groups.",
        "distance", "medium", True,
        lambda seed: KMeans(n_clusters=3, n_init=5, random_state=seed),
    ))
    zoo.append(_clf(
        "minibatch_kmeans", "Mini-Batch K-Means", "The Fast Sorter",
        "The same grouping idea, but it sorts small handfuls of records at a time "
        "instead of all of them. Nearly the same answer, a fraction of the time.",
        "The quickest way to group a very large file.",
        "Slightly less tidy groups than the full version.",
        "distance", "fast", True,
        lambda seed: MiniBatchKMeans(n_clusters=3, n_init=5, batch_size=2048,
                                     random_state=seed),
    ))
    zoo.append(_clf(
        "bisecting_kmeans", "Bisecting K-Means", "The Divide-and-Conquer Sorter",
        "It starts with everyone in one group and keeps splitting the messiest group "
        "in two until it has enough. That suits data with a few big groups and some "
        "small ones.",
        "Handles uneven group sizes better than plain K-Means.",
        "The splits are decided greedily, so it can miss the ideal arrangement.",
        "distance", "medium", True,
        lambda seed: BisectingKMeans(n_clusters=3, random_state=seed),
    ))
    zoo.append(_clf(
        "gaussian_mixture", "Gaussian Mixture", "The Soft Grouper",
        "It assumes your customers come from a handful of overlapping bell curves "
        "and works out which mixture of them best explains everyone. Records can "
        "belong partly to two groups.",
        "Gives a probability of belonging, not just a hard label.",
        "Assumes each group is a neat bell shape.",
        "probabilistic", "medium", True,
        lambda seed: GaussianMixture(n_components=3, covariance_type="diag",
                                     random_state=seed, max_iter=200),
    ))
    zoo.append(_clf(
        "agglomerative", "Agglomerative Clustering", "The Family-Tree Builder",
        "It starts with every record as its own group and repeatedly merges the two "
        "closest groups until only the number you asked for remain.",
        "Builds a natural hierarchy you can cut at any level.",
        "Slow on large files.",
        "distance", "heavy", True,
        lambda seed: AgglomerativeClustering(n_clusters=3, linkage="ward"),
    ))
    zoo.append(_clf(
        "dbscan", "DBSCAN", "The Crowd Finder",
        "It looks for dense crowds of records and calls everything left over noise. "
        "Crucially, you do not have to tell it how many groups to expect.",
        "Finds groups of any shape and openly admits some records belong to none.",
        "Struggles when your groups have very different densities.",
        "density", "heavy", True,
        lambda seed: DBSCAN(eps=0.5, min_samples=10, n_jobs=-1),
    ))
    return zoo


# --------------------------------------------------------------------------- #
# Budget-aware candidate selection
# --------------------------------------------------------------------------- #

_ZOOS: dict[str, Callable[[], list[ModelSpec]]] = {
    "classification": _classification_zoo,
    "regression": _regression_zoo,
    "anomaly": _anomaly_zoo,
    "clustering": _clustering_zoo,
}

_SUPERVISED = {"classification", "regression"}

# Models we never want to lose even under heavy pruning: they cover different ways
# of thinking (a straight line, a committee, a step-by-step improver, a fast guess).
_CORE = {
    "classification": ["logistic", "random_forest", "hist_gb", "naive_bayes", "decision_tree"],
    "regression": ["linear", "random_forest", "hist_gb", "extra_trees", "decision_tree"],
    "anomaly": ["isolation_forest", "robust_zscore", "lof"],
    "clustering": ["kmeans", "minibatch_kmeans", "gaussian_mixture"],
}

# Clustering models whose number of groups we choose at run time.
K_MODELS = {"kmeans", "minibatch_kmeans", "bisecting_kmeans", "gaussian_mixture",
            "agglomerative"}
K_SEARCH_RANGE = (2, 3, 4, 5, 6)


def full_zoo(task: str) -> list[ModelSpec]:
    """Every model we could run for a task, before any pruning."""
    builder = _ZOOS.get(task)
    return builder() if builder else []


def candidates(task: str, *, n_rows: int, n_features: int, budget_s: float) -> list[ModelSpec]:
    """The models we will actually run, given the size of the job and the clock.

    Deterministic: the same inputs always produce the same field, in the same order.
    """
    zoo = full_zoo(task)
    if not zoo:
        return []

    floor = 5 if task in _SUPERVISED else 3

    # --- how tight are we? -------------------------------------------------- #
    tight = budget_s < 60.0 or n_rows > 120_000 or n_features > 250
    very_tight = budget_s < 30.0 or n_rows > 400_000 or n_features > 600

    def _allowed(spec: ModelSpec) -> bool:
        if spec.cost == "heavy" and tight:
            return False
        if spec.cost == "medium" and very_tight and spec.key not in _CORE.get(task, []):
            return False
        if spec.key == "knn" and n_rows > 150_000:
            return False
        if spec.key == "gradient_boosting" and n_rows > 60_000:
            return False
        if spec.key == "mlp" and n_rows > 200_000:
            return False
        if spec.key in {"one_class_svm", "elliptic_envelope"} and n_rows > 60_000:
            return False
        if spec.key == "lof" and n_rows > 120_000:
            return False
        if spec.key == "agglomerative" and n_rows > 60_000:
            return False
        if spec.key == "dbscan" and n_rows > 150_000:
            return False
        if spec.key == "xgboost" and n_rows > 800_000:
            return False
        return True

    kept = [s for s in zoo if _allowed(s)]

    # --- guarantee a diverse floor ----------------------------------------- #
    if len(kept) < floor:
        have = {s.key for s in kept}
        for key in _CORE.get(task, []):
            if len(kept) >= floor:
                break
            if key in have:
                continue
            for s in zoo:
                if s.key == key:
                    kept.append(s)
                    have.add(key)
                    break
        if len(kept) < floor:  # last resort: add back in catalogue order
            have = {s.key for s in kept}
            for s in zoo:
                if len(kept) >= floor:
                    break
                if s.key not in have:
                    kept.append(s)
                    have.add(s.key)

    # stable, catalogue order
    order = {s.key: i for i, s in enumerate(zoo)}
    kept.sort(key=lambda s: order[s.key])

    # the seasonal baseline is a benchmark, not a competitor: always last
    kept.sort(key=lambda s: 1 if s.family == "baseline" else 0)
    return kept


__all__ = [
    "ModelSpec",
    "to_dict",
    "candidates",
    "full_zoo",
    "make_pipeline",
    "RobustZScoreDetector",
    "SeasonalNaiveRegressor",
    "K_MODELS",
    "K_SEARCH_RANGE",
]
