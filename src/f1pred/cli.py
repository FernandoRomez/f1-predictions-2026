import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from . import backtest, championship, charts, data
from . import model as m
from .features import add_upcoming, build_features

RESULTS = Path("results")


def _frame_for(history: pd.DataFrame, race: pd.Series) -> pd.DataFrame:
    """Variables de una carrera; si es futura, toma los inscritos de la qualy ya disputada."""
    year, round_ = int(race["year"]), int(race["round"])
    if ((history["year"] == year) & (history["round"] == round_)).any():
        return build_features(data.enrich(history))
    entrants = data.fetch_qualifying(year, round_)
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
    frame = _frame_for(history, race)
    table_race, ranks, _, features = m.forecast(frame, args.year * 100 + int(race["round"]))
    table = m.summarize(table_race, ranks)
    mode = "después de la qualy" if "grid_pct" in features else "antes de la qualy"
    print(f"{race['race']} {args.year} ({mode})")
    print(table.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RESULTS.mkdir(exist_ok=True)
    stem = f"{args.year}_{int(race['round']):02d}_{race['circuit']}"
    table.to_csv(RESULTS / f"{stem}.csv", index=False, float_format="%.4f")
    charts.win_chart(table, f"{race['race']} {args.year}: probabilidad de victoria ({mode})", str(RESULTS / f"{stem}.png"))


def cmd_championship(args) -> None:
    history = data.load_history(last=args.year)
    schedule = data.fetch_schedule(args.year)
    done = history[history["year"] == args.year]["round"].max()
    left = schedule[schedule["round"] > done]
    frames = [(_frame_for(history, r), args.year * 100 + int(r["round"]), bool(r["sprint"])) for _, r in left.iterrows()]
    season = history[(history["year"] == args.year) & (history["position"] == 1)]
    table = championship.simulate_title(frames, data.fetch_standings(args.year), season["code"].value_counts().to_dict())
    print(table.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{args.year}_championship.csv", index=False, float_format="%.4f")


def cmd_backtest(args) -> None:
    frame = build_features(data.enrich(data.load_history(last=args.year)))
    sections = []
    for mode, sigma in (("post", m.SIGMA_POST_QUALI), ("pre", m.SIGMA_PRE_QUALI)):
        results = backtest.run(frame, sigmas=(sigma,), mode=mode)
        for label, part in (("2025-26 holdout", results[results["year"] >= 2025]), ("2026 only", results[results["year"] == 2026])):
            table = backtest.summary(part).assign(mode=mode, sample=label, races=part["race_id"].nunique())
            sections.append(table.reset_index())
    out = pd.concat(sections)
    print(out.to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "backtest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="f1pred")
    sub = parser.add_subparsers(required=True)
    for name, handler in (("predict", cmd_predict), ("championship", cmd_championship), ("backtest", cmd_backtest)):
        command = sub.add_parser(name)
        command.add_argument("--year", type=int, default=date.today().year)
        command.set_defaults(handler=handler)
        if name == "predict":
            command.add_argument("--round", type=int)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
