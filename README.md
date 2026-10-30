# F1 Predictions 2026

Probabilistic Formula 1 forecasts: win, podium and points odds for every driver, plus a title simulation. Every number is validated walk-forward, so each race is predicted using only the races before it.

Inspired by [mar-antaya/2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions). This is a from-scratch rewrite with no shared code.

![Mexico City 2026 forecast](results/2026_19_rodriguez.png)

## Latest forecast: Mexico City GP, before qualifying

| Driver | Team | Win | Podium |
|---|---|---:|---:|
| VER | Red Bull | 31.9% | 67.4% |
| ANT | Mercedes | 23.5% | 60.2% |
| LEC | Ferrari | 18.1% | 53.9% |
| NOR | McLaren | 13.1% | 45.9% |
| HAM | Ferrari | 6.5% | 30.5% |

Treat this as a rough prior: the pre-qualifying model is much weaker than the post-qualifying one (see below). Rerun `f1pred predict` after qualifying. The simulation gives Antonelli the title in effectively every run from a 123-point lead with 125 points left, though Russell is not yet mathematically out.

## What is different from the 2025 project

| 2025 scripts | This project |
|---|---|
| About 13 rows, one race | 3,854 driver-race rows from 2018 to 2026 |
| Hand-typed values per race | Features computed automatically |
| Random train/test split | Walk-forward backtest over 88 races |
| One predicted time per driver | Monte Carlo probabilities |
| Fuzzy metric | Compared against naive baselines |

Key ideas:

- **No leakage:** a test changes a race's results and checks that its own pre-race features stay identical. It caught a real leak during development.
- **Regulation resets:** team form restarts in 2022 and 2026, and races under current rules get extra weight.
- **Team lineage:** renamed teams are joined, for example Sauber to Audi.
- **Two models:** the pre-qualifying model uses form only; the post-qualifying model adds grid and qualifying gap.
- **Reliability:** retirements are simulated from smoothed driver and team rates.

## Backtest

Simulation noise was chosen on 2023-24 only. These 42 races from 2025-26 were not used for that choice, although they did inform the design.

| Method | Win log-loss | Win Brier | Winner picked | Top-3 overlap | Spearman |
|---|---:|---:|---:|---:|---:|
| Post-qualifying model | **0.968** | **0.478** | 64.3% | 69.0% | **0.669** |
| Grid baseline | 1.150 | 0.522 | **69.0%** | **69.8%** | 0.654 |
| Pre-qualifying model | 1.678 | 0.792 | 33.3% | 57.1% | 0.588 |
| Driver form baseline | 1.951 | 0.841 | 23.8% | 54.0% | 0.566 |

Lower is better for log-loss and Brier. The grid baseline needs qualifying, so the fair comparison for the pre-qualifying model is driver form.

The honest reading:

- The post-qualifying model gives better probabilities than the grid overall, but simply picking pole still finds the winner more often.
- In the 18 races of 2026 alone, the grid beats it on win log-loss (1.198 vs 1.226) and Brier (0.489 vs 0.532). The sample is small, and the new regulations leave little data.
- Before qualifying, the model picks the winner only about one race in three.

## Usage

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
.venv/bin/f1pred predict                # next race, uses qualifying once available
.venv/bin/f1pred predict --round 20
.venv/bin/f1pred championship
.venv/bin/f1pred backtest
.venv/bin/python -m pytest
```

No API key is needed. Data comes from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) and is cached in `data/`.

## Limitations

- No weather, tyre strategy, safety cars, penalties or lap telemetry.
- Drivers' noise is independent, so shared events such as rain are not modelled.
- Future races in the title simulation reuse today's form, so long-range odds are overconfident.
- The title simulation stops if sprint weekends remain, and it ignores countback.
- Entrants for an upcoming race are copied from the latest race.

## License

MIT
