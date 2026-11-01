import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from . import backtest, championship, charts, data
from . import model as m
from .features import add_upcoming, build_features

RESULTS = Path("results")


def _parse_grid(text: str | None) -> dict[str, int]:
    """Convierte "VER=5,ANT=4" en parrilla final tras sanciones."""
    if not text:
        return {}
    pairs = (item.split("=") for item in text.split(","))
    return {code.strip().upper(): int(slot) for code, slot in pairs}


def _apply_grid(entrants: pd.DataFrame, grid: dict[str, int]) -> pd.DataFrame:
    """La API da el orden de qualy; las sanciones de parrilla se indican a mano."""
    unknown = set(grid) - set(entrants["code"])
    if unknown:
        raise SystemExit(f"--grid: pilotos desconocidos: {sorted(unknown)}")
    return entrants.assign(grid=entrants["code"].map(grid).fillna(entrants["grid"]).astype(int))


def _frame_for(history: pd.DataFrame, race: pd.Series, grid: dict[str, int] | None = None) -> pd.DataFrame:
    """Variables de una carrera; si es futura, toma los inscritos de la qualy ya disputada."""
    year, round_ = int(race["year"]), int(race["round"])
    if ((history["year"] == year) & (history["round"] == round_)).any():
        return build_features(data.enrich(history))
    entrants = data.fetch_qualifying(year, round_)
    if not entrants.empty and grid:
        entrants = _apply_grid(entrants, grid)
    if entrants.empty:
        entrants = history.loc[history["date"] == history["date"].max(), ["driver", "code", "team"]]
    fields = {k: race[k] for k in ("year", "round", "date", "race", "circuit", "lat", "lon")}
    return build_features(data.enrich(add_upcoming(history, fields, entrants)))


def _next_race(schedule: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    done = history[history["year"] == schedule["year"].iloc[0]]["round"].max()
    return schedule[schedule["round"] > done].iloc[0]


def cmd_predict(args) -> None:
    history = data.load_history(last=args.year)
    schedule = data.fetch_schedule(args.year)
    race = schedule[schedule["round"] == args.round].iloc[0] if args.round else _next_race(schedule, history)
    frame = _frame_for(history, race, _parse_grid(args.grid))
    table_race, ranks, _, features, win = m.forecast(frame, args.year * 100 + int(race["round"]), runs=args.runs)
    table = m.summarize(table_race, ranks, win)
    mode = "después de la qualy" if "grid_pct" in features else "antes de la qualy"
    print(f"{race['race']} {args.year} ({mode})")
    view = table.copy()
    floor = 100 / args.runs
    for col in ("win", "podium", "points"):
        view[col] = [f"<{floor:.5f}%" if v == 0 else f"{v * 100:.5f}%" for v in table[col]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"{args.runs:,} simulaciones; '<' = no ocurrió en ninguna, cota de resolución.")
    RESULTS.mkdir(exist_ok=True)
    stem = f"{args.year}_{int(race['round']):02d}_{race['circuit']}"
    table.to_csv(RESULTS / f"{stem}.csv", index=False, float_format="%.8g")
    charts.win_chart(table, f"{race['race']} {args.year}: probabilidades ({mode})", str(RESULTS / f"{stem}.png"), args.runs)


def cmd_championship(args) -> None:
    history = data.load_history(last=args.year)
    schedule = data.fetch_schedule(args.year)
    done = history[history["year"] == args.year]["round"].max()
    left = schedule[schedule["round"] > done]
    grid = _parse_grid(args.grid)
    frames = [(_frame_for(history, r, grid if r["round"] == done + 1 else None), args.year * 100 + int(r["round"]), bool(r["sprint"])) for _, r in left.iterrows()]
    season = history[(history["year"] == args.year) & (history["position"] == 1)]
    table = championship.simulate_title(frames, data.fetch_standings(args.year), season["code"].value_counts().to_dict())
    view = table[["code", "points_now", "title", "title_lo", "title_hi", "exp_points"]].head(8).copy()
    print("Probabilidad de título (entre paréntesis, error Monte Carlo al 90%, no incertidumbre del modelo):")

    def pct(x: float) -> str:
        return f"{x*100:.2f}%" if x >= 0.005 else f"{x*100:.3f}%"

    for _, row in view.iterrows():
        print(f"  {row['code']:<3} {row['points_now']:>5.0f} pts  "
              f"título {pct(row['title']):>8}  ({pct(row['title_lo'])}–{pct(row['title_hi'])})  "
              f"pts esp. {row['exp_points']:.1f}")
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{args.year}_championship.csv", index=False, float_format="%.4f")


def cmd_backtest(args) -> None:
    frame = build_features(data.enrich(data.load_history(last=args.year)))
    sections = []
    post_holdout = None
    for mode, sigma in (("post", m.SIGMA_POST_QUALI), ("pre", m.SIGMA_PRE_QUALI)):
        results = backtest.run(frame, sigmas=(sigma,), mode=mode, blend_alpha=m.BLEND_ALPHA)
        for label, part in (("2025-26 holdout", results[results["year"] >= 2025]), ("2026 only", results[results["year"] == 2026])):
            table = backtest.summary(part).assign(mode=mode, sample=label, races=part["race_id"].nunique())
            sections.append(table.reset_index())
        if mode == "post" and post_holdout is None:
            post_holdout = results[results["year"] >= 2025]
    out = pd.concat(sections)
    print(out.to_string(index=False))

    print("\nCalibración de victoria (holdout 2025-26), ECE menor es mejor:")
    for method in ("blend", f"model_s{m.SIGMA_POST_QUALI}", "grid"):
        ece = backtest.expected_calibration_error(post_holdout, method)
        print(f"  {method:<12} ECE={ece:.3f}")

    cis = backtest.bootstrap_ci(post_holdout, n=1000)
    pivot = cis[cis["method"].isin(["blend", "grid"]) & cis["metric"].isin(["winner_hit", "logloss", "brier", "top3_prob"])]
    print("\nBootstrap 90% sobre carreras (holdout 2025-26):")
    print(pivot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    paired = pd.concat([backtest.paired_bootstrap(post_holdout, "blend", other)
                        for other in (f"model_s{m.SIGMA_POST_QUALI}", "grid")])
    print("\nBootstrap pareado (blend - referencia); si cruza 0, no hay diferencia clara:")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "backtest.csv", index=False)
    cis.to_csv(RESULTS / "backtest_bootstrap.csv", index=False)
    paired.to_csv(RESULTS / "backtest_paired.csv", index=False)
    backtest.calibration(post_holdout, "blend").to_csv(RESULTS / "backtest_calibration.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="f1pred")
    sub = parser.add_subparsers(required=True)
    for name, handler in (("predict", cmd_predict), ("championship", cmd_championship), ("backtest", cmd_backtest)):
        command = sub.add_parser(name)
        command.add_argument("--year", type=int, default=date.today().year)
        command.set_defaults(handler=handler)
        if name == "predict":
            command.add_argument("--round", type=int)
        if name in ("predict", "championship"):
            command.add_argument("--runs", type=int, default=1_000_000)
            command.add_argument("--grid", help='parrilla tras sanciones, p. ej. "VER=5,ANT=4"')
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
