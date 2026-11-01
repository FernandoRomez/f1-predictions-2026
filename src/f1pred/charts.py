import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _label(value: float) -> str:
    return f"{value:.1f}%" if value >= 0.1 else f"{value:.5f}".rstrip("0") + "%"


def win_chart(table: pd.DataFrame, title: str, path: str, runs: int) -> None:
    """Victoria, podio y puntos en escala log; sin eventos se marca la resolución."""
    rows = table.iloc[::-1]
    floor = 100 / runs
    panels = [("win", "Victoria"), ("podium", "Podio"), ("points", "Puntos (top 10)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4.4, 0.34 * len(rows))), sharey=True)
    for ax, (col, name) in zip(axes, panels):
        values = rows[col] * 100
        colors = ["#c8553d" if code == "PER" else "#2a6f97" for code in rows["code"]]
        ax.barh(rows["code"], values.clip(lower=floor), left=floor / 10, color=colors, height=0.6)
        for y, value in enumerate(values):
            text = f"<{_label(floor)}" if value == 0 else _label(value)
            ax.text(max(value, floor) * 1.3, y, text, va="center", fontsize=8, color="#333333")
        ax.set_xscale("log")
        ax.set_xlim(floor / 10, 3000)
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("% (escala log)", color="#555555")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x", color="#e6e6e6")
        ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
