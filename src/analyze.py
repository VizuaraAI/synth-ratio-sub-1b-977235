"""Turn a directory of finished runs into alpha*(N) and the paper's tables.

    python -m src.analyze --runs runs --out results

Produces:
    results/all_runs.csv        one row per run, the raw material
    results/table1_main.tex     the LaTeX body of the paper's Table 1
    results/alpha_star.json     alpha*(N) with confidence intervals
    results/alpha_star.png      the figure 2 panels, with YOUR numbers

The curve fit is a parabola in log(alpha + eps). A parabola is the simplest
form that has a single interior minimum, which is exactly the hypothesis
under test; if your measured curve is clearly not parabolic, say so in the
paper rather than forcing this fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 0.01  # so that alpha = 0 is representable on a log axis
SIZE_PARAMS = {"70m": 7.0e7, "160m": 1.6e8, "410m": 4.1e8, "830m": 8.3e8}
PUBLISHED_ANCHORS = {3.0e9: 12.5, 6.0e9: 25.0}  # Kang et al., EMNLP 2025


def collect(runs_dir: Path) -> pd.DataFrame:
    """Read every summary.json, preferring eval.json where it exists."""
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        row = json.loads(summary_path.read_text())
        eval_path = summary_path.parent / "eval.json"
        if eval_path.exists():
            ev = json.loads(eval_path.read_text())
            row["val_loss"] = ev.get("val_loss", row.get("final_val_loss"))
            row["val_loss_stderr"] = ev.get("val_loss_stderr")
            if "downstream" in ev:
                row["downstream_mean"] = ev["downstream"].get("mean")
        else:
            row["val_loss"] = row.get("final_val_loss")
        rows.append(row)

    if not rows:
        raise SystemExit(f"no runs with a summary.json under {runs_dir}")
    df = pd.DataFrame(rows)
    df["alpha_pct"] = df["alpha"] * 100
    df["params"] = df["size"].map(SIZE_PARAMS)
    return df.sort_values(["params", "alpha"])


def fit_optimum(alphas: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Return (alpha* as a fraction, fitted minimum loss).

    Fits loss = a*u^2 + b*u + c with u = log(alpha + EPS). If the fit is
    concave (a <= 0) there is no interior minimum, so we fall back to the
    best measured point and let the caller notice via the flat curve.
    """
    if len(alphas) < 3:
        best = int(np.argmin(losses))
        return float(alphas[best]), float(losses[best])

    u = np.log(alphas + EPS)
    a, b, c = np.polyfit(u, losses, 2)
    if a <= 0:
        best = int(np.argmin(losses))
        return float(alphas[best]), float(losses[best])

    u_star = -b / (2 * a)
    alpha_star = float(np.clip(np.exp(u_star) - EPS, 0.0, float(alphas.max())))
    return alpha_star, float(np.polyval([a, b, c], np.log(alpha_star + EPS)))


def bootstrap_ci(
    alphas: np.ndarray, losses: np.ndarray, noise: float, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Resample the loss curve under the measured seed noise to bound alpha*."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        perturbed = losses + rng.normal(0.0, noise, size=losses.shape)
        draws.append(fit_optimum(alphas, perturbed)[0])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze(df: pd.DataFrame, style: str = "balanced") -> dict:
    subset = df[df["style"] == style] if "style" in df else df
    results = {}

    for size, group in subset.groupby("size"):
        # Average seed replicates at each alpha, and use their spread as the
        # noise floor. Without replicates we cannot say an optimum is real.
        per_alpha = group.groupby("alpha")["val_loss"].agg(["mean", "std", "count"])
        alphas = per_alpha.index.to_numpy(dtype=float)
        losses = per_alpha["mean"].to_numpy(dtype=float)

        noise = float(np.nanmean(per_alpha["std"])) if per_alpha["count"].max() > 1 else 0.0
        alpha_star, loss_star = fit_optimum(alphas, losses)
        baseline = float(losses[np.argmin(np.abs(alphas))])

        entry = {
            "size": size,
            "params": SIZE_PARAMS.get(size),
            "alpha_star": alpha_star,
            "alpha_star_pct": 100 * alpha_star,
            "loss_at_alpha_star": loss_star,
            "loss_at_zero": baseline,
            "delta_best": baseline - loss_star,
            "seed_noise": noise,
            "n_alphas": int(len(alphas)),
            "has_replicates": bool(per_alpha["count"].max() > 1),
        }
        if noise > 0:
            lo, hi = bootstrap_ci(alphas, losses, noise)
            entry["alpha_star_ci95"] = [100 * lo, 100 * hi]
            # The headline test: is the optimum distinguishable from zero?
            entry["distinguishable_from_zero"] = bool(lo > 0.005)
            entry["beats_real_only"] = bool(entry["delta_best"] > noise)
        results[size] = entry

    ordered = [results[s] for s in sorted(results, key=lambda s: SIZE_PARAMS.get(s, 0))]
    stars = [e["alpha_star_pct"] for e in ordered]
    return {
        "per_size": ordered,
        "monotone_increasing": bool(all(b >= a - 1e-9 for a, b in zip(stars, stars[1:]))),
        "published_anchors": {f"{int(k / 1e9)}B": v for k, v in PUBLISHED_ANCHORS.items()},
    }


def write_table(summary: dict, df: pd.DataFrame, out_path: Path) -> None:
    """Emit the LaTeX rows for Table 1 so you can paste over the placeholders."""
    grid = sorted(df["alpha"].unique())
    lines = []
    for entry in summary["per_size"]:
        cells = []
        for a in grid:
            match = df[(df["size"] == entry["size"]) & (np.isclose(df["alpha"], a))]
            cells.append(f"{match['val_loss'].mean():.3f}" if len(match) else "--")
        lines.append(
            f"    {entry['size'].upper():<5}& "
            + " & ".join(cells)
            + f" & {entry['alpha_star_pct']:.1f}\\% & {entry['delta_best']:+.3f} \\\\"
        )
    header = " & ".join(f"${a * 100:g}\\%$" for a in grid)
    out_path.write_text(
        f"% columns: model size & {header} & $\\alpha^\\star$ & $\\Delta_{{best}}$\n"
        + "\n".join(lines)
        + "\n"
    )
    print(f"wrote {out_path}")


def plot(summary: dict, df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)

    for entry in summary["per_size"]:
        group = df[df["size"] == entry["size"]].groupby("alpha")["val_loss"].mean()
        ax1.plot(100 * group.index, group.values, marker="o", markersize=4, label=entry["size"])
        ax1.axvline(entry["alpha_star_pct"], linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("synthetic share alpha (%)")
    ax1.set_ylabel("held-out real-web validation loss")
    ax1.set_title("(a) Loss versus synthetic share", loc="left", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    xs = [e["params"] for e in summary["per_size"]]
    ys = [e["alpha_star_pct"] for e in summary["per_size"]]
    errs = [
        [max(0.0, e["alpha_star_pct"] - e.get("alpha_star_ci95", [e["alpha_star_pct"]] * 2)[0])
         for e in summary["per_size"]],
        [max(0.0, e.get("alpha_star_ci95", [0, e["alpha_star_pct"]])[1] - e["alpha_star_pct"])
         for e in summary["per_size"]],
    ]
    ax2.errorbar(xs, ys, yerr=errs, fmt="o", capsize=3, label="this work (measured)")
    ax2.scatter(
        list(PUBLISHED_ANCHORS), list(PUBLISHED_ANCHORS.values()),
        facecolors="none", edgecolors="crimson", s=50, label="published anchors",
    )
    ax2.set_xscale("log")
    ax2.axhline(0, linestyle=":", linewidth=0.8, color="grey")
    ax2.set_xlabel("model parameters N (log scale)")
    ax2.set_ylabel("optimal synthetic share alpha* (%)")
    ax2.set_title("(b) Optimal share versus scale", loc="left", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results")
    parser.add_argument("--style", default="balanced")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(Path(args.runs))
    df.to_csv(out_dir / "all_runs.csv", index=False)

    subset = df[df["style"] == args.style]
    if subset.empty:
        raise SystemExit(f"no runs with style={args.style!r}; found {sorted(df['style'].unique())}")

    summary = analyze(df, args.style)
    (out_dir / "alpha_star.json").write_text(json.dumps(summary, indent=2))
    write_table(summary, subset, out_dir / "table1_main.tex")
    plot(summary, subset, out_dir / "alpha_star.png")

    print("\nalpha*(N):")
    for entry in summary["per_size"]:
        ci = entry.get("alpha_star_ci95")
        ci_text = f"  95% CI [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "  (no seed replicates)"
        print(f"  {entry['size']:>5}: {entry['alpha_star_pct']:5.1f}%{ci_text}")
    print(f"\nmonotone increasing in N: {summary['monotone_increasing']}")


if __name__ == "__main__":
    main()
