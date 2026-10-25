from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_rolling(
    path: Path,
    values: list[float],
    windows: list[int],
    title: str = "GRPO average reward",
    xlabel: str = "step",
    ylabel: str = "average reward",
) -> None:
    if not values:
        return
    if any(w <= 0 for w in windows):
        raise ValueError("Rolling windows must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Prefix sums let us calculate each rolling average in O(1).
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    plt.figure(figsize=(12, 5))

    for window in windows:
        rolling = [
            (prefix[i + 1] - prefix[max(0, i + 1 - window)])
            / min(i + 1, window)
            for i in range(len(values))
        ]

        plt.plot(
            rolling,
            label=f"{window}-step average",
            marker="o" if window == 1 and len(windows) == 1 else None,
            markersize=2,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()