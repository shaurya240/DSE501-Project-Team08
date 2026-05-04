"""
DSE501 Term Project - Team 8
Analysis of U.S. Domestic Flight Delay Causes

Script 01: Data Loading and Exploratory Data Analysis
-----------------------------------------------------
Loads the Bureau of Transportation Statistics (BTS) Airline Delay Cause
dataset (Jan-Aug 2024) and performs initial exploration, cleaning, and
feature engineering.

Outputs cleaned data + descriptive figures to ../output and ../figures.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and global plotting style
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "Airline_Delay_Cause.xlsx")
FIG_DIR = os.path.join(HERE, "..", "figures")
OUT_DIR = os.path.join(HERE, "..", "output")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
})

CAUSE_COLORS = {
    "Late Aircraft": "#d62728",
    "Carrier":       "#1f77b4",
    "NAS":           "#2ca02c",
    "Weather":       "#ff7f0e",
    "Security":      "#9467bd",
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the BTS Airline Delay Cause dataset."""
    df = pd.read_excel(path)
    print(f"[load_data] shape = {df.shape}")
    print(f"[load_data] columns = {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records with missing values (negligible per the proposal)."""
    n_before = len(df)
    df = df.dropna().copy()
    n_after = len(df)
    print(f"[clean_data] dropped {n_before - n_after} records with NaNs "
          f"({100*(n_before-n_after)/n_before:.3f}%)")
    df = df[df["arr_flights"] > 0].copy()
    print(f"[clean_data] final shape = {df.shape}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived metrics required by the study."""
    df["delay_rate"] = df["arr_del15"] / df["arr_flights"]
    df["avg_delay_min"] = np.where(
        df["arr_del15"] > 0, df["arr_delay"] / df["arr_del15"], 0.0
    )
    df["cancel_rate"] = df["arr_cancelled"] / df["arr_flights"]
    df["divert_rate"] = df["arr_diverted"] / df["arr_flights"]

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    }
    df["month_name"] = df["month"].map(month_names)

    season_map = {
        1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
    }
    df["season"] = df["month"].map(season_map)
    return df


def descriptive_summary(df: pd.DataFrame) -> dict:
    """Compute and print a high-level summary matching the proposal."""
    total_flights = df["arr_flights"].sum()
    total_delays = df["arr_del15"].sum()
    overall_delay_rate = total_delays / total_flights

    causes = ["carrier", "weather", "nas", "security", "late_aircraft"]
    cause_counts = {c: df[f"{c}_ct"].sum() for c in causes}
    cause_minutes = {c: df[f"{c}_delay"].sum() for c in causes}

    summary = {
        "n_records": len(df),
        "n_carriers": df["carrier"].nunique(),
        "n_airports": df["airport"].nunique(),
        "total_flights": int(total_flights),
        "total_delays": int(total_delays),
        "overall_delay_rate": float(overall_delay_rate),
        "cause_counts": {k: int(v) for k, v in cause_counts.items()},
        "cause_minutes": {k: int(v) for k, v in cause_minutes.items()},
    }

    print("\n" + "=" * 60)
    print("DESCRIPTIVE SUMMARY")
    print("=" * 60)
    print(f"Records              : {summary['n_records']:,}")
    print(f"Carriers             : {summary['n_carriers']}")
    print(f"Airports             : {summary['n_airports']}")
    print(f"Total flights        : {summary['total_flights']:,}")
    print(f"Total delayed (15+m) : {summary['total_delays']:,}")
    print(f"Overall delay rate   : {summary['overall_delay_rate']:.4f} "
          f"({100*summary['overall_delay_rate']:.2f}%)")
    print("\nDelay cause breakdown (count of delayed flights):")
    for c in causes:
        ct = cause_counts[c]
        share = ct / total_delays
        avg_min = cause_minutes[c] / total_flights
        print(f"  {c:15s} : {int(ct):>9,}  "
              f"({100*share:5.2f}%)  avg {avg_min:.2f} min/flight")
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_cause_breakdown(df: pd.DataFrame) -> None:
    """Bar chart of delay-cause shares (counts and minutes)."""
    causes = ["late_aircraft", "carrier", "nas", "weather", "security"]
    pretty = {"late_aircraft": "Late Aircraft", "carrier": "Carrier",
              "nas": "NAS", "weather": "Weather", "security": "Security"}
    counts = [df[f"{c}_ct"].sum() for c in causes]
    minutes = [df[f"{c}_delay"].sum() for c in causes]
    labels = [pretty[c] for c in causes]
    colors = [CAUSE_COLORS[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title("(a) Delayed flights by cause", fontsize=11)
    ax.set_ylabel("Number of delayed flights")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(labels, [m / 1e6 for m in minutes], color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.set_title("(b) Total delay minutes by cause (millions)", fontsize=11)
    ax.set_ylabel("Delay minutes (millions)")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, minutes):
        ax.text(b.get_x() + b.get_width() / 2, v / 1e6,
                f"{v/1e6:.2f}M", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cause_breakdown.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def fig_carrier_delay_rate(df: pd.DataFrame) -> None:
    """Boxplot of delay rate per carrier (sorted by median)."""
    code_to_name = (df.drop_duplicates("carrier")
                    .set_index("carrier")["carrier_name"].to_dict())
    medians = df.groupby("carrier")["delay_rate"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="carrier", y="delay_rate", order=order,
                ax=ax, palette="coolwarm", fliersize=2, linewidth=0.7)
    ax.set_xlabel("Carrier (IATA code)")
    ax.set_ylabel("Delay rate (delayed / total arrivals)")
    ax.set_title("Delay rate distribution by carrier (Jan–Aug 2024)")
    ax.set_ylim(0, df["delay_rate"].quantile(0.99))

    legend_text = ", ".join(
        [f"{c}={code_to_name.get(c, c).split(' Air')[0][:14]}" for c in order[:8]]
    )
    ax.text(0.01, -0.30, "Top-8 carriers (lowest median): " + legend_text,
            transform=ax.transAxes, fontsize=7, color="dimgray")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_carrier_delay_rate.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def fig_monthly_trend(df: pd.DataFrame) -> None:
    """Line plot of mean delay rate per month with 95% CI band, plus stacked bar of cause shares."""
    monthly = df.groupby("month")["delay_rate"].agg(["mean", "std", "count"]).reset_index()
    monthly["se"] = monthly["std"] / np.sqrt(monthly["count"])
    monthly["ci"] = 1.96 * monthly["se"]
    monthly["month_name"] = monthly["month"].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(monthly["month_name"], monthly["mean"], "o-", color="#1f77b4",
            linewidth=2, markersize=7)
    ax.fill_between(monthly["month_name"],
                    monthly["mean"] - monthly["ci"],
                    monthly["mean"] + monthly["ci"],
                    color="#1f77b4", alpha=0.2, label="95% CI")
    ax.set_xlabel("Month (2024)")
    ax.set_ylabel("Mean delay rate")
    ax.set_title("(a) Monthly mean delay rate with 95% CI")
    ax.legend()

    ax = axes[1]
    causes = ["late_aircraft", "carrier", "nas", "weather", "security"]
    pretty = {"late_aircraft": "Late Aircraft", "carrier": "Carrier",
              "nas": "NAS", "weather": "Weather", "security": "Security"}
    by_month = df.groupby("month")[[f"{c}_delay" for c in causes]].sum()
    by_month_pct = by_month.div(by_month.sum(axis=1), axis=0) * 100
    by_month_pct.columns = [pretty[c] for c in causes]
    by_month_pct.index = [monthly["month_name"].iloc[i-1] for i in by_month_pct.index]

    bottom = np.zeros(len(by_month_pct))
    for col in by_month_pct.columns:
        ax.bar(by_month_pct.index, by_month_pct[col], bottom=bottom,
               label=col, color=CAUSE_COLORS[col], edgecolor="white", linewidth=0.4)
        bottom += by_month_pct[col].values
    ax.set_ylabel("Share of total delay minutes (%)")
    ax.set_xlabel("Month (2024)")
    ax.set_title("(b) Composition of delay minutes by month")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_monthly_trend.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def fig_volume_vs_rate(df: pd.DataFrame) -> None:
    """Scatter plot of airport flight volume vs. delay rate."""
    airport_agg = df.groupby("airport").agg(
        total_flights=("arr_flights", "sum"),
        total_delays=("arr_del15", "sum"),
    )
    airport_agg["delay_rate"] = airport_agg["total_delays"] / airport_agg["total_flights"]
    airport_agg = airport_agg[airport_agg["total_flights"] >= 100]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(airport_agg["total_flights"], airport_agg["delay_rate"],
               alpha=0.55, s=22, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Total arriving flights (log scale, Jan–Aug 2024)")
    ax.set_ylabel("Airport delay rate")
    ax.set_title("Airport flight volume vs. delay rate")

    top = airport_agg.nlargest(8, "total_flights")
    for code, row in top.iterrows():
        ax.annotate(code, (row["total_flights"], row["delay_rate"]),
                    fontsize=8, color="darkred", weight="bold",
                    xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_volume_vs_rate.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def fig_heatmap_carrier_month(df: pd.DataFrame) -> None:
    """Heatmap of mean delay rate by carrier x month."""
    pivot = df.pivot_table(values="delay_rate", index="carrier",
                           columns="month", aggfunc="mean")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                   5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
    pivot.columns = [month_names[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(pivot, cmap="RdYlGn_r", annot=True, fmt=".2f",
                cbar_kws={"label": "Mean delay rate"}, ax=ax,
                linewidths=0.4, linecolor="white", annot_kws={"size": 8})
    ax.set_xlabel("Month (2024)")
    ax.set_ylabel("Carrier (IATA code, sorted by overall mean)")
    ax.set_title("Mean delay rate by carrier × month")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_heatmap_carrier_month.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def main():
    df = load_data()
    df = clean_data(df)
    df = add_features(df)
    summary = descriptive_summary(df)

    df.to_csv(os.path.join(OUT_DIR, "clean_data.csv"), index=False)

    pd.Series(summary).to_json(os.path.join(OUT_DIR, "summary.json"), indent=2)

    fig_cause_breakdown(df)
    fig_carrier_delay_rate(df)
    fig_monthly_trend(df)
    fig_volume_vs_rate(df)
    fig_heatmap_carrier_month(df)
    print("\n[main] Script 01 complete.")


if __name__ == "__main__":
    main()
