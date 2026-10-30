import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def win_chart(table: pd.DataFrame, title: str, path: str, top: int = 8) -> None:
    rows = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(rows["code"], rows["win"] * 100, color="#2a6f97", height=0.6)
    for y, value in enumerate(rows["win"] * 100):
        ax.text(value + 0.6, y, f"{value:.1f}%", va="center", fontsize=10, color="#333333")
    ax.set_title(title, loc="left", fontsize=13)
    ax.set_xlabel("Win probability (%)", color="#555555")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
