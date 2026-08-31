"""Optuna multivariable search with chronological validation and holdout proof."""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Optional

import optuna
import pandas as pd

from backtesting.oscillation import run_backtest
from optimization.random_search import SEARCH_SPACE, fitness


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", message=r"Argument .* is an experimental feature.*")


def _fold_data(df: pd.DataFrame, folds: int, warmup: int = 150):
    data = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)
    folds = max(2, min(int(folds), 8))
    usable_start = min(warmup, max(0, len(data) // 4))
    boundaries = [usable_start + round((len(data) - usable_start) * index / folds) for index in range(folds + 1)]
    result = []
    for index in range(folds):
        start, end = boundaries[index], boundaries[index + 1]
        if end - start < 10:
            continue
        chunk = data.iloc[max(0, start - warmup):end].copy()
        evaluation_start = str(data.iloc[start].get("timestamp", "")) or None
        result.append((chunk, evaluation_start))
    return result


def evaluate(df: pd.DataFrame, params: Dict[str, Any], folds: int,
             initial_equity: float, benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    results = []
    for fold_number, (chunk, evaluation_start) in enumerate(_fold_data(df, folds), 1):
        evaluation_end = str(chunk.iloc[-1].get("timestamp", "")) if len(chunk) else None
        benchmark = benchmark_df
        if benchmark is not None and evaluation_start and "timestamp" in benchmark.columns:
            benchmark = benchmark[pd.to_datetime(benchmark["timestamp"]).dt.date >= pd.Timestamp(evaluation_start).date()]
        metrics = run_backtest(chunk, params, initial_equity=initial_equity, start=evaluation_start, end=evaluation_end, benchmark_df=benchmark)["metrics"]
        results.append({"fold": fold_number, "score": fitness(metrics), "metrics": metrics})
    scores = [item["score"] for item in results]
    returns = [item["metrics"]["return_pct"] for item in results]
    aggregate = median(scores) - (pstdev(scores) * 0.5 if len(scores) > 1 else 0)
    return {
        "score": round(aggregate, 4),
        "median_score": round(median(scores), 4),
        "score_stdev": round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "mean_return_pct": round(mean(returns), 4),
        "positive_periods": sum(value > 0 for value in returns),
        "folds": results,
    }


def _suggest_parameters(trial: optuna.Trial, base_params: Dict[str, Any], locked_keys: set[str]) -> Dict[str, Any]:
    params = dict(base_params)
    for key, (low, high, kind) in SEARCH_SPACE.items():
        if key in locked_keys:
            continue
        if kind == "int":
            params[key] = trial.suggest_int(key, int(low), int(high))
        else:
            params[key] = trial.suggest_float(key, float(low), float(high))
    return params


def _searchable_base(base_params: Dict[str, Any], locked_keys: set[str] | None = None) -> Dict[str, Any]:
    locked_keys = locked_keys or set()
    values = {}
    for key, (low, high, kind) in SEARCH_SPACE.items():
        fallback = (low + high) / 2
        value = min(high, max(low, base_params.get(key, fallback)))
        if key not in locked_keys:
            values[key] = int(round(value)) if kind == "int" else float(value)
    return values


def run_adaptive_search(df: pd.DataFrame, base_params: Optional[Dict[str, Any]] = None,
                        max_iterations: int = 250, patience: int = 25,
                        folds: int = 4, seed: int = 42,
                        initial_equity: float = 100_000,
                        store: bool = True, benchmark_df: Optional[pd.DataFrame] = None,
                        locked_keys: Optional[list[str]] = None) -> Dict[str, Any]:
    max_iterations = max(10, min(int(max_iterations), 2000))
    patience = max(5, min(int(patience), 200))
    base = dict(base_params or {})
    locked = set(locked_keys or []) & set(SEARCH_SPACE)
    ordered = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)
    holdout_index = max(1, int(len(ordered) * 0.8))
    training = ordered.iloc[:holdout_index].copy()
    holdout = ordered.iloc[max(0, holdout_index - 150):].copy()
    holdout_start = str(ordered.iloc[holdout_index].get("timestamp", "")) if holdout_index < len(ordered) else None
    training_folds = max(2, int(folds) - 1)
    baseline = evaluate(training, base, training_folds, initial_equity, benchmark_df=benchmark_df)

    startup_trials = min(20, max(5, max_iterations // 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=startup_trials,
        )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(_searchable_base(base, locked))

    def objective(trial: optuna.Trial):
        params = _suggest_parameters(trial, base, locked)
        evaluation = evaluate(training, params, training_folds, initial_equity, benchmark_df=benchmark_df)
        trial.set_user_attr("evaluation", evaluation)
        return evaluation["score"]

    history = []
    best_seen = float("-inf")
    stale = 0
    status = "max_iterations"
    base_searchable = _searchable_base(base, locked)
    for iteration in range(1, max_iterations + 1):
        study.optimize(objective, n_trials=1, catch=(ValueError,))
        trial = study.trials[-1]
        improved = trial.value is not None and trial.value > best_seen + 0.01
        if improved:
            best_seen = trial.value
            stale = 0
        else:
            stale += 1
        changed = {
            key: {"from": base_searchable[key], "to": round(value, 6) if isinstance(value, float) else value}
            for key, value in trial.params.items()
            if value != base_searchable[key]
        }
        history.append({
            "iteration": iteration,
            "parameters": trial.params,
            "changed_parameters": changed,
            "changed_count": len(changed),
            "candidate_score": round(trial.value, 4) if trial.value is not None else None,
            "best_score": round(study.best_value, 4),
            "accepted": improved,
        })
        if stale >= patience and iteration >= startup_trials:
            status = "converged"
            break

    best_params = {**base, **study.best_params}
    best_evaluation = study.best_trial.user_attrs["evaluation"]
    baseline_full = run_backtest(df, base, initial_equity=initial_equity, benchmark_df=benchmark_df)
    full_result = run_backtest(df, best_params, initial_equity=initial_equity, benchmark_df=benchmark_df)
    baseline_holdout = run_backtest(holdout, base, initial_equity=initial_equity, start=holdout_start, benchmark_df=benchmark_df)["metrics"]
    best_holdout = run_backtest(holdout, best_params, initial_equity=initial_equity, start=holdout_start, benchmark_df=benchmark_df)["metrics"]
    baseline_scores = [fold["score"] for fold in baseline["folds"]] + [fitness(baseline_holdout)]
    best_scores = [fold["score"] for fold in best_evaluation["folds"]] + [fitness(best_holdout)]
    baseline_returns = [fold["metrics"]["return_pct"] for fold in baseline["folds"]] + [baseline_holdout["return_pct"]]
    best_returns = [fold["metrics"]["return_pct"] for fold in best_evaluation["folds"]] + [best_holdout["return_pct"]]
    periods_better = sum(best > baseline_score for best, baseline_score in zip(best_scores, baseline_scores))
    baseline_full_score = fitness(baseline_full["metrics"])
    best_full_score = fitness(full_result["metrics"])
    comparison = {
        "periods_better": periods_better,
        "periods_tested": len(best_scores),
        "score_improvement": round(best_evaluation["score"] - baseline["score"], 4),
        "baseline_mean_return_pct": round(mean(baseline_returns), 4),
        "best_mean_return_pct": round(mean(best_returns), 4),
        "holdout_better": fitness(best_holdout) > fitness(baseline_holdout),
        "overall_exceeds_baseline": best_full_score > baseline_full_score,
        "full_score_improvement": round(best_full_score - baseline_full_score, 4),
        "full_return_improvement_pct": round(full_result["metrics"]["return_pct"] - baseline_full["metrics"]["return_pct"], 4),
        "stable_improvement": bool(best_scores) and periods_better >= max(2, len(best_scores) - 1),
    }
    try:
        importance = {key: round(value, 4) for key, value in optuna.importance.get_param_importances(study).items()}
    except (ImportError, RuntimeError, ValueError, ZeroDivisionError):
        importance = {}
    ready = status == "converged" and comparison["stable_improvement"] and comparison["holdout_better"] and comparison["overall_exceeds_baseline"]
    payload = {
        "engine": "Optuna TPESampler (multivariate)",
        "status": status,
        "ready": ready,
        "seed": seed,
        "iterations": len(history),
        "baseline": baseline,
        "best_validation": best_evaluation,
        "best_params": best_params,
        "locked_keys": sorted(locked),
        "baseline_full_backtest": baseline_full,
        "best_full_backtest": full_result,
        "holdout": {"baseline": baseline_holdout, "learned": best_holdout},
        "comparison": comparison,
        "parameter_importance": importance,
        "history": history,
    }
    if store:
        _store(payload)
    return payload


def _store(payload: Dict[str, Any]):
    directory = Path(__file__).resolve().parent.parent / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (directory / f"optuna-search-{timestamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
