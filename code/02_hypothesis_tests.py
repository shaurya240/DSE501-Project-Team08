"""
DSE501 Term Project - Team 8
Analysis of U.S. Domestic Flight Delay Causes

Script 02: Inferential Statistics - Hypothesis Tests H1-H4
----------------------------------------------------------
Performs the four hypothesis tests outlined in the proposal:
  H1 - Carrier effect on delay rate            (ANOVA / Kruskal-Wallis)
  H2 - Seasonal variation in delay rate        (ANOVA + Tukey HSD)
  H3 - Weather vs Carrier as delay drivers     (Two-proportion z-test)
  H4 - Volume-vs-delay-rate correlation        (Pearson, Spearman, OLS)

All results saved to ../output/hypothesis_tests.json and printed to stdout.
"""

import json
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")
FIG_DIR = os.path.join(HERE, "..", "figures")
DATA_PATH = os.path.join(OUT_DIR, "clean_data.csv")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
})

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helper: format p-values consistently
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if p < 1e-10:
        return "p < 1e-10"
    if p < 0.001:
        return f"p = {p:.2e}"
    return f"p = {p:.4f}"


# ---------------------------------------------------------------------------
# H1 - Carrier effect on delay rate
# ---------------------------------------------------------------------------
def test_h1_carrier(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("H1: Carrier effect on delay rate")
    print("=" * 70)

    groups = [g["delay_rate"].values
              for _, g in df.groupby("carrier") if len(g) >= 5]
    print(f"  k groups (carriers): {len(groups)}")
    print(f"  N total            : {sum(len(g) for g in groups)}")

    # Assumption checks
    sample = df["delay_rate"].sample(n=min(5000, len(df)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    print(f"  Shapiro-Wilk normality (sample): "
          f"W = {sw_stat:.4f}, {fmt_p(sw_p)}  "
          f"-> {'normal' if sw_p > ALPHA else 'NOT normal'}")

    lev_stat, lev_p = stats.levene(*groups, center="median")
    print(f"  Levene's test (equal variances): "
          f"W = {lev_stat:.4f}, {fmt_p(lev_p)}  "
          f"-> {'equal' if lev_p > ALPHA else 'UNEQUAL'} variances")

    # Parametric and non-parametric tests
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    print(f"\n  One-way ANOVA       : F = {f_stat:.3f}, {fmt_p(f_p)}")
    print(f"  Kruskal-Wallis      : H = {h_stat:.3f}, {fmt_p(h_p)}")

    # Effect size (eta-squared)
    grand_mean = df["delay_rate"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((df["delay_rate"] - grand_mean) ** 2))
    eta_sq = ss_between / ss_total
    print(f"  Effect size (eta^2) : {eta_sq:.4f}")

    # Post-hoc Tukey HSD on the 8 carriers with the largest sample size
    top_carriers = (df.groupby("carrier").size()
                    .sort_values(ascending=False).head(8).index.tolist())
    sub = df[df["carrier"].isin(top_carriers)]
    sub_groups = [sub[sub["carrier"] == c]["delay_rate"].values for c in top_carriers]
    tukey = stats.tukey_hsd(*sub_groups)

    posthoc_rows = []
    for i, j in combinations(range(len(top_carriers)), 2):
        posthoc_rows.append({
            "carrier_1": top_carriers[i],
            "carrier_2": top_carriers[j],
            "mean_diff": float(tukey.statistic[i, j]),
            "p_value": float(tukey.pvalue[i, j]),
            "significant": bool(tukey.pvalue[i, j] < ALPHA),
        })

    n_sig = sum(r["significant"] for r in posthoc_rows)
    print(f"  Tukey HSD: {n_sig}/{len(posthoc_rows)} top-8 pairs differ significantly")

    decision = "REJECT H0" if h_p < ALPHA else "FAIL TO REJECT H0"
    print(f"\n  DECISION: {decision} (using Kruskal-Wallis)")

    return {
        "n_groups": len(groups),
        "shapiro_W": float(sw_stat), "shapiro_p": float(sw_p),
        "levene_W": float(lev_stat), "levene_p": float(lev_p),
        "anova_F": float(f_stat), "anova_p": float(f_p),
        "kruskal_H": float(h_stat), "kruskal_p": float(h_p),
        "eta_squared": float(eta_sq),
        "decision": decision,
        "posthoc_top8": posthoc_rows,
    }


# ---------------------------------------------------------------------------
# H2 - Seasonal variation
# ---------------------------------------------------------------------------
def test_h2_seasonal(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("H2: Seasonal variation in delay rate (months Jan-Aug)")
    print("=" * 70)

    months = sorted(df["month"].unique())
    groups = [df[df["month"] == m]["delay_rate"].values for m in months]
    print(f"  Months : {months}")
    print(f"  N      : {sum(len(g) for g in groups)}")

    lev_stat, lev_p = stats.levene(*groups, center="median")
    print(f"  Levene W = {lev_stat:.4f}, {fmt_p(lev_p)}")

    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    print(f"  ANOVA       : F = {f_stat:.3f}, {fmt_p(f_p)}")
    print(f"  Kruskal-W   : H = {h_stat:.3f}, {fmt_p(h_p)}")

    grand_mean = df["delay_rate"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((df["delay_rate"] - grand_mean) ** 2))
    eta_sq = ss_between / ss_total
    print(f"  eta^2       : {eta_sq:.4f}")

    # Tukey HSD across all 8 months
    tukey = stats.tukey_hsd(*groups)
    posthoc_rows = []
    for i, j in combinations(range(len(months)), 2):
        posthoc_rows.append({
            "month_1": int(months[i]),
            "month_2": int(months[j]),
            "mean_diff": float(tukey.statistic[i, j]),
            "p_value": float(tukey.pvalue[i, j]),
            "significant": bool(tukey.pvalue[i, j] < ALPHA),
        })
    n_sig = sum(r["significant"] for r in posthoc_rows)
    print(f"  Tukey HSD : {n_sig}/{len(posthoc_rows)} month pairs differ significantly")

    decision = "REJECT H0" if f_p < ALPHA else "FAIL TO REJECT H0"
    print(f"\n  DECISION: {decision}")

    # Significance heatmap of Tukey HSD pairwise p-values
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                   5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
    pmat = np.ones((len(months), len(months)))
    for i in range(len(months)):
        for j in range(len(months)):
            pmat[i, j] = tukey.pvalue[i, j]
    plot_df = pd.DataFrame(pmat,
                           index=[month_names[m] for m in months],
                           columns=[month_names[m] for m in months])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sig_mask = plot_df < ALPHA
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="RdYlGn",
                cbar_kws={"label": "Tukey HSD p-value"}, ax=ax,
                vmin=0, vmax=0.1, linewidths=0.4, linecolor="white")
    ax.set_title("Tukey HSD pairwise p-values across months\n"
                 "(red = significant difference at α=0.05, green = not significant)")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_h2_tukey_heatmap.png")
    plt.savefig(out)
    plt.close()
    print(f"  [fig] saved {out}")

    return {
        "anova_F": float(f_stat), "anova_p": float(f_p),
        "kruskal_H": float(h_stat), "kruskal_p": float(h_p),
        "levene_W": float(lev_stat), "levene_p": float(lev_p),
        "eta_squared": float(eta_sq),
        "decision": decision,
        "tukey_pairs": posthoc_rows,
    }


# ---------------------------------------------------------------------------
# H3 - Weather vs Carrier delay shares
# ---------------------------------------------------------------------------
def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> tuple:
    """Pooled two-proportion z-test."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_two = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_lo = (p1 - p2) - 1.96 * se_diff
    ci_hi = (p1 - p2) + 1.96 * se_diff
    return z, p_two, p1, p2, ci_lo, ci_hi


def test_h3_weather_vs_carrier(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("H3: Weather vs Carrier as drivers of delay MINUTES")
    print("=" * 70)

    total_minutes = (df["carrier_delay"].sum() + df["weather_delay"].sum()
                     + df["nas_delay"].sum() + df["security_delay"].sum()
                     + df["late_aircraft_delay"].sum())
    weather_min = int(df["weather_delay"].sum())
    carrier_min = int(df["carrier_delay"].sum())
    total_minutes = int(total_minutes)

    print(f"  Total delay minutes : {total_minutes:,}")
    print(f"  Weather minutes     : {weather_min:,} ({100*weather_min/total_minutes:.2f}%)")
    print(f"  Carrier minutes     : {carrier_min:,} ({100*carrier_min/total_minutes:.2f}%)")

    z, p, p1, p2, lo, hi = two_prop_z(weather_min, total_minutes,
                                      carrier_min, total_minutes)
    print(f"\n  Two-proportion z-test:")
    print(f"    p_weather  = {p1:.4f}")
    print(f"    p_carrier  = {p2:.4f}")
    print(f"    difference = {p1-p2:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])")
    print(f"    z = {z:.4f},  {fmt_p(p)}")

    # Same test on counts of delayed flights
    weather_ct = int(df["weather_ct"].sum())
    carrier_ct = int(df["carrier_ct"].sum())
    total_ct = int(df["arr_del15"].sum())
    z2, p2_, p1c, p2c, lo2, hi2 = two_prop_z(weather_ct, total_ct,
                                             carrier_ct, total_ct)
    print(f"\n  Sanity check on flight COUNTS:")
    print(f"    p_weather  = {p1c:.4f}")
    print(f"    p_carrier  = {p2c:.4f}")
    print(f"    z = {z2:.4f},  {fmt_p(p2_)}")

    decision = "REJECT H0" if p < ALPHA else "FAIL TO REJECT H0"
    print(f"\n  DECISION: {decision}")

    # Visualization
    causes = ["Late Aircraft", "Carrier", "NAS", "Weather", "Security"]
    minutes = [int(df["late_aircraft_delay"].sum()), carrier_min,
               int(df["nas_delay"].sum()), weather_min,
               int(df["security_delay"].sum())]
    pct = [100 * m / total_minutes for m in minutes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    bars = ax.barh(causes[::-1], pct[::-1], color=colors[::-1],
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, pct[::-1]):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=10)
    ax.axvline((carrier_min / total_minutes) * 100, color="#1f77b4",
               linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline((weather_min / total_minutes) * 100, color="#ff7f0e",
               linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Share of total delay minutes (%)")
    ax.set_title(f"Carrier ({100*carrier_min/total_minutes:.1f}%) vs "
                 f"Weather ({100*weather_min/total_minutes:.1f}%)\n"
                 f"Two-proportion z-test: z = {z:.2f}, {fmt_p(p)}")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_h3_weather_vs_carrier.png")
    plt.savefig(out)
    plt.close()
    print(f"  [fig] saved {out}")

    return {
        "weather_minutes": weather_min,
        "carrier_minutes": carrier_min,
        "total_minutes": total_minutes,
        "p_weather": float(p1),
        "p_carrier": float(p2),
        "difference": float(p1 - p2),
        "ci_low": float(lo), "ci_high": float(hi),
        "z_stat": float(z), "p_value": float(p),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# H4 - Flight Volume vs Delay Rate
# ---------------------------------------------------------------------------
def test_h4_volume(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("H4: Airport flight volume vs delay rate")
    print("=" * 70)

    apt = df.groupby("airport").agg(
        total_flights=("arr_flights", "sum"),
        total_delays=("arr_del15", "sum"),
    )
    apt["delay_rate"] = apt["total_delays"] / apt["total_flights"]
    apt = apt[apt["total_flights"] >= 100]
    print(f"  Airports analyzed : {len(apt)} (min 100 flights)")

    x = apt["total_flights"].values
    y = apt["delay_rate"].values
    log_x = np.log10(x)

    pearson_r, pearson_p = stats.pearsonr(x, y)
    pearson_r_log, pearson_p_log = stats.pearsonr(log_x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print(f"  Pearson  (raw)   : r = {pearson_r:.4f}, {fmt_p(pearson_p)}")
    print(f"  Pearson  (log10) : r = {pearson_r_log:.4f}, {fmt_p(pearson_p_log)}")
    print(f"  Spearman (rank)  : rho = {spearman_r:.4f}, {fmt_p(spearman_p)}")

    # OLS regression on log(volume)
    X = log_x.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_hat = reg.predict(X)
    r2 = reg.score(X, y)
    n = len(y)

    # Manual t-test on slope
    residuals = y - y_hat
    ss_res = np.sum(residuals ** 2)
    ss_xx = np.sum((log_x - log_x.mean()) ** 2)
    se_slope = np.sqrt(ss_res / (n - 2) / ss_xx)
    t_stat = reg.coef_[0] / se_slope
    p_slope = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

    print(f"\n  OLS regression: delay_rate ~ a + b*log10(volume)")
    print(f"    intercept = {reg.intercept_:.5f}")
    print(f"    slope     = {reg.coef_[0]:.5f}  (per decade of volume)")
    print(f"    R^2       = {r2:.4f}")
    print(f"    t = {t_stat:.4f}, {fmt_p(p_slope)}")

    decision_pearson = "REJECT H0" if pearson_p_log < ALPHA else "FAIL TO REJECT H0"
    print(f"\n  DECISION (Pearson on log-volume): {decision_pearson}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    ax.scatter(x, y, alpha=0.55, s=22, color="#1f77b4",
               edgecolor="black", linewidth=0.3)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ys = reg.intercept_ + reg.coef_[0] * np.log10(xs)
    ax.plot(xs, ys, color="red", linewidth=2,
            label=f"OLS fit (R²={r2:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Total arriving flights (log scale)")
    ax.set_ylabel("Airport delay rate")
    ax.set_title(f"(a) OLS fit on log10(volume)\n"
                 f"r = {pearson_r_log:.3f}, {fmt_p(pearson_p_log)}")
    ax.legend()

    ax = axes[1]
    apt["volume_quintile"] = pd.qcut(
        apt["total_flights"], 5,
        labels=["Q1 (smallest)", "Q2", "Q3", "Q4", "Q5 (largest)"]
    )
    sns.boxplot(data=apt.reset_index(), x="volume_quintile", y="delay_rate",
                ax=ax, palette="viridis", linewidth=0.7, fliersize=3)
    ax.set_xlabel("Airport volume quintile")
    ax.set_ylabel("Airport delay rate")
    ax.set_title("(b) Delay rate by volume quintile")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_h4_volume_regression.png")
    plt.savefig(out)
    plt.close()
    print(f"  [fig] saved {out}")

    return {
        "n_airports": int(len(apt)),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "pearson_r_log": float(pearson_r_log), "pearson_p_log": float(pearson_p_log),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "ols_intercept": float(reg.intercept_),
        "ols_slope": float(reg.coef_[0]),
        "ols_r2": float(r2),
        "ols_t": float(t_stat),
        "ols_p_slope": float(p_slope),
        "decision": decision_pearson,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    print(f"[main] loaded {len(df):,} records, "
          f"{df['carrier'].nunique()} carriers, "
          f"{df['airport'].nunique()} airports")

    results = {
        "alpha": ALPHA,
        "n_records": int(len(df)),
        "h1_carrier": test_h1_carrier(df),
        "h2_seasonal": test_h2_seasonal(df),
        "h3_weather_vs_carrier": test_h3_weather_vs_carrier(df),
        "h4_volume_correlation": test_h4_volume(df),
    }

    out = os.path.join(OUT_DIR, "hypothesis_tests.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main] wrote {out}")
    print("\n[main] Script 02 complete.")


if __name__ == "__main__":
    main()
