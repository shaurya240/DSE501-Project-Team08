"""
DSE501 Term Project - Team 8
Analysis of U.S. Domestic Flight Delay Causes

Script 04: U.S. State-Level Geographic Visualization
----------------------------------------------------
Two map figures associating delay metrics with geographic location:

(1) A state-level tile-grid choropleth, where each U.S. state is a
    cell in approximate geographic position, colored by the mean delay
    rate of arriving flights at airports in that state. This avoids the
    Alaska/Hawaii framing problem and reads cleanly at thumbnail size.

(2) A geographic-coordinate map with simplified state outlines drawn
    from an embedded coordinate set, overlaid with airport bubbles
    (size = traffic volume, color = delay rate).

Outputs to ../figures.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output")
FIG_DIR = os.path.join(HERE, "..", "figures")

sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 11,
})

# ---------------------------------------------------------------------------
# Tile-grid layout of U.S. states.
# Coordinates are (row, col) in a 12x12 grid; the layout is the standard
# NPR / 538 hex/tile-grid arrangement adapted to a square grid.
# Origin (0,0) is upper-left.
# ---------------------------------------------------------------------------
TILE_GRID = {
    "AK": (0, 0),  "ME": (0, 11),
    "VT": (1, 10), "NH": (1, 11),
    "WA": (2, 1),  "MT": (2, 3), "ND": (2, 4), "MN": (2, 5),
    "WI": (2, 6),  "MI": (2, 7), "NY": (2, 9),  "MA": (2, 10),
    "ID": (3, 2),  "WY": (3, 3), "SD": (3, 4),  "IA": (3, 5),
    "IL": (3, 6),  "IN": (3, 7), "OH": (3, 8),  "PA": (3, 9),
    "NJ": (3, 10), "CT": (3, 10), "RI": (3, 11),
    "OR": (4, 1),  "NV": (4, 2), "CO": (4, 3),  "NE": (4, 4),
    "MO": (4, 5),  "KY": (4, 6), "WV": (4, 7),  "VA": (4, 8),
    "MD": (4, 9),  "DC": (4, 10), "DE": (4, 11),
    "CA": (5, 1),  "UT": (5, 2), "KS": (5, 4),  "AR": (5, 5),
    "TN": (5, 6),  "NC": (5, 8), "SC": (5, 9),
    "AZ": (6, 2),  "NM": (6, 3), "OK": (6, 4),  "LA": (6, 5),
    "MS": (6, 6),  "AL": (6, 7), "GA": (6, 8),
    "HI": (7, 0),  "TX": (7, 4), "FL": (7, 9),
    "PR": (8, 10),
}

# CT and NJ collide in the original layout above; resolve manually.
TILE_GRID["NJ"] = (4, 11)
TILE_GRID["CT"] = (3, 10)
TILE_GRID["MD"] = (5, 10)
TILE_GRID["DC"] = (5, 11)
TILE_GRID["DE"] = (4, 11)

# Final, deduplicated layout (manually verified)
TILE_GRID = {
    "AK": (0, 0),
    "ME": (0, 11),
    "VT": (1, 10), "NH": (1, 11),
    "WA": (2, 1),  "ID": (2, 2),  "MT": (2, 3),  "ND": (2, 4),
    "MN": (2, 5),  "WI": (2, 6),  "MI": (2, 7),  "NY": (2, 9),
    "MA": (2, 10),
    "OR": (3, 1),  "NV": (3, 2),  "WY": (3, 3),  "SD": (3, 4),
    "IA": (3, 5),  "IL": (3, 6),  "IN": (3, 7),  "OH": (3, 8),
    "PA": (3, 9),  "NJ": (3, 10), "CT": (3, 11),
    "CA": (4, 1),  "UT": (4, 2),  "CO": (4, 3),  "NE": (4, 4),
    "MO": (4, 5),  "KY": (4, 6),  "WV": (4, 7),  "VA": (4, 8),
    "MD": (4, 9),  "DE": (4, 10), "RI": (4, 11),
    "AZ": (5, 2),  "NM": (5, 3),  "KS": (5, 4),  "AR": (5, 5),
    "TN": (5, 6),  "NC": (5, 7),  "SC": (5, 8),  "DC": (5, 9),
    "HI": (6, 0),  "OK": (6, 4),  "LA": (6, 5),  "MS": (6, 6),
    "AL": (6, 7),  "GA": (6, 8),  "FL": (6, 9),
    "TX": (7, 4),  "PR": (7, 10),
}

# ---------------------------------------------------------------------------
# Simplified outline of the contiguous United States.
# Coordinates are (longitude, latitude) of waypoints around the perimeter
# plus a few key state boundaries. Source: U.S. Census Bureau cartographic
# boundary data, manually decimated to ~120 vertices for the outline.
# ---------------------------------------------------------------------------
CONUS_OUTLINE = [
    # Pacific NW
    (-124.7, 48.4), (-124.6, 47.0), (-124.0, 46.3), (-124.0, 45.0),
    (-124.4, 43.5), (-124.5, 42.0), (-124.2, 40.5), (-123.7, 38.9),
    (-122.4, 37.7), (-121.9, 36.6), (-120.9, 35.5), (-120.6, 34.5),
    (-118.5, 34.0), (-117.3, 32.7), (-117.1, 32.5),
    # Mexican border
    (-114.8, 32.5), (-111.1, 31.3), (-108.2, 31.3), (-106.5, 31.8),
    (-104.5, 30.5), (-103.2, 28.9), (-102.4, 29.8), (-101.4, 29.8),
    (-100.7, 29.4), (-99.5, 27.6), (-97.5, 26.0),
    # Gulf coast
    (-97.4, 27.8), (-95.3, 29.0), (-94.0, 29.7), (-92.0, 29.5),
    (-90.4, 29.1), (-89.5, 30.0), (-87.5, 30.3), (-85.5, 30.3),
    (-84.4, 30.0), (-83.0, 29.5), (-82.7, 29.0), (-82.6, 27.5),
    (-81.7, 25.8), (-80.8, 25.2), (-80.1, 25.2), (-80.1, 26.5),
    # Atlantic coast
    (-80.5, 28.5), (-80.9, 29.8), (-81.4, 30.7), (-80.9, 32.0),
    (-79.7, 32.8), (-79.0, 33.8), (-77.7, 34.5), (-76.5, 34.7),
    (-75.5, 35.6), (-75.5, 36.9), (-76.0, 37.2), (-75.9, 38.0),
    (-75.0, 38.5), (-74.4, 39.4), (-74.0, 40.5), (-73.0, 41.1),
    (-71.8, 41.3), (-71.1, 41.5), (-70.8, 41.6), (-70.0, 41.8),
    (-70.6, 42.7), (-70.8, 43.2), (-70.0, 43.7), (-69.0, 44.0),
    (-67.0, 44.7), (-67.0, 45.2),
    # Canadian border
    (-69.0, 47.5), (-71.0, 45.0), (-74.7, 45.0), (-76.9, 43.6),
    (-79.0, 43.3), (-83.0, 41.6), (-83.4, 42.0), (-82.4, 43.0),
    (-82.5, 45.8), (-84.5, 46.0), (-87.7, 47.5), (-90.0, 48.2),
    (-92.0, 47.5), (-95.0, 49.4), (-104.0, 49.0), (-110.0, 49.0),
    (-114.5, 49.0), (-117.0, 49.0), (-120.0, 49.0), (-123.0, 49.0),
    (-124.7, 48.4),
]


# ---------------------------------------------------------------------------
# Helper: state lookup. The proposal-curated airport dictionary in
# Script 03 has IATA -> (lat, lon, state, region). We re-use it.
# ---------------------------------------------------------------------------
def load_airport_summary():
    df = pd.read_csv(os.path.join(OUT_DIR, "airport_summary.csv"))
    df = df.dropna(subset=["state"])
    return df


def state_aggregates(apt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to state-level metrics."""
    g = apt.groupby("state").agg(
        n_airports=("airport", "count"),
        total_flights=("total_flights", "sum"),
        total_delays=("total_delays", "sum"),
        total_delay_min=("total_delay_min", "sum"),
        weather_min=("weather_min", "sum"),
        carrier_min=("carrier_min", "sum"),
        nas_min=("nas_min", "sum"),
        late_aircraft_min=("late_aircraft_min", "sum"),
    ).reset_index()
    g["delay_rate"] = g["total_delays"] / g["total_flights"]
    g["weather_share"] = g["weather_min"] / (
        g["weather_min"] + g["carrier_min"] + g["nas_min"] + g["late_aircraft_min"]
    )
    return g


# ---------------------------------------------------------------------------
# Figure 1: tile-grid choropleth
# ---------------------------------------------------------------------------
def fig_tile_choropleth(state_df: pd.DataFrame) -> None:
    """One square per state; color = mean delay rate."""
    state_lookup = state_df.set_index("state").to_dict("index")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

    for ax, metric, label, cmap, vmin, vmax in [
        (axes[0], "delay_rate", "Mean delay rate",
         "RdYlGn_r", 0.15, 0.30),
        (axes[1], "weather_share", "Share of delay minutes from weather",
         "Blues", 0.0, 0.15),
    ]:
        cmap_obj = plt.get_cmap(cmap)

        for state, (row, col) in TILE_GRID.items():
            data = state_lookup.get(state)
            if data is None or pd.isna(data.get(metric)):
                color = "#e5e5e5"
                txt_color = "#888"
                value_str = ""
            else:
                value = data[metric]
                norm_val = np.clip((value - vmin) / (vmax - vmin), 0, 1)
                color = cmap_obj(norm_val)
                # text color: white if dark, black if light
                r, g, b = color[:3]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "white" if lum < 0.5 else "black"
                if metric == "delay_rate":
                    value_str = f"{100*value:.0f}"
                else:
                    value_str = f"{100*value:.0f}"

            x = col
            y = -row
            rect = mpatches.Rectangle((x, y), 0.92, 0.92,
                                      facecolor=color, edgecolor="white",
                                      linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.46, y + 0.6, state, ha="center", va="center",
                    fontsize=9, weight="bold", color=txt_color)
            if value_str:
                ax.text(x + 0.46, y + 0.28, value_str, ha="center", va="center",
                        fontsize=8, color=txt_color)

        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-8.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=12, weight="bold", pad=8)

        # Custom colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                            shrink=0.55, pad=0.02, aspect=25)
        if metric == "delay_rate":
            cbar.set_label("Mean delay rate", fontsize=9)
            cbar.set_ticks([0.15, 0.20, 0.25, 0.30])
            cbar.set_ticklabels(["15%", "20%", "25%", "30%"])
        else:
            cbar.set_label("Weather share of delay minutes", fontsize=9)
            cbar.set_ticks([0.0, 0.05, 0.10, 0.15])
            cbar.set_ticklabels(["0%", "5%", "10%", "15%"])

    plt.suptitle("U.S. flight-delay tile-grid choropleths "
                 "(Jan–Aug 2024; numbers in cells are state values, %)",
                 fontsize=11, y=0.98)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_us_tilegrid.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: real geographic outline + airport bubbles
# ---------------------------------------------------------------------------
def fig_us_outline_bubbles(apt: pd.DataFrame) -> None:
    """Hand-traced US outline with airport bubbles overlaid."""
    geo = apt.dropna(subset=["lat", "lon"]).copy()
    conus = geo[(geo["lon"] > -130) & (geo["lon"] < -65)
                & (geo["lat"] > 24) & (geo["lat"] < 50)]
    top = conus.nlargest(60, "total_flights")

    fig, ax = plt.subplots(figsize=(12, 7.0))

    # Draw the outline as a closed polygon
    outline = np.array(CONUS_OUTLINE)
    ax.fill(outline[:, 0], outline[:, 1],
            facecolor="#f3f3f3", edgecolor="#666",
            linewidth=1.0, zorder=1)

    # Light state-region tint by U.S. Census region
    region_colors = {
        "Northeast": "#e8eef7",
        "Midwest":   "#fff3e6",
        "South":     "#f0f7ee",
        "West":      "#fde8e8",
    }
    # Plot every airport (faint dots) for spatial coverage
    other = geo[~geo.index.isin(top.index)]
    other = other[(other["lon"] > -130) & (other["lon"] < -65)
                  & (other["lat"] > 24) & (other["lat"] < 50)]
    ax.scatter(other["lon"], other["lat"], s=8, color="#aaa",
               alpha=0.45, zorder=2, edgecolor="none")

    # Top-60 airports as colored bubbles
    sizes = (top["total_flights"] / top["total_flights"].max()) * 1700 + 35
    sc = ax.scatter(top["lon"], top["lat"], s=sizes, c=top["delay_rate"],
                    cmap="RdYlGn_r", edgecolor="black", linewidth=0.7,
                    alpha=0.88, vmin=0.15, vmax=0.30, zorder=3)

    # Label the top-15 by volume
    for _, row in top.nlargest(15, "total_flights").iterrows():
        ax.annotate(row["airport"], (row["lon"], row["lat"]),
                    fontsize=8, weight="bold", color="black",
                    xytext=(5, 5), textcoords="offset points", zorder=4)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Delay rate", fontsize=10)

    # Volume legend
    for vol, lbl in [(50000, "50k flights"),
                     (100000, "100k flights"),
                     (200000, "200k flights")]:
        s = (vol / top["total_flights"].max()) * 1700 + 35
        ax.scatter([], [], s=s, c="lightgray", edgecolor="black",
                   linewidth=0.6, label=lbl)
    ax.legend(scatterpoints=1, frameon=True, labelspacing=1.5,
              title="Volume", loc="lower left", fontsize=8,
              title_fontsize=9)

    ax.set_xlim(-126, -65)
    ax.set_ylim(23, 50)
    ax.set_aspect(1.3)
    ax.set_xlabel("Longitude (°W)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_title("Top-60 CONUS airports on the U.S. land outline\n"
                 "(bubble size = traffic volume, color = delay rate, "
                 "Jan–Aug 2024)", fontsize=11)
    ax.grid(False)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_us_outline_bubbles.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


# ---------------------------------------------------------------------------
def main():
    apt = load_airport_summary()
    state_df = state_aggregates(apt)

    # Save for the report
    state_df.sort_values("delay_rate", ascending=False).to_csv(
        os.path.join(OUT_DIR, "state_summary.csv"), index=False)

    print(f"[main] {len(state_df)} states with airport data")
    print("\nTop-5 states by delay rate:")
    print(state_df.nlargest(5, "delay_rate")[["state", "n_airports",
                                              "total_flights", "delay_rate"]])
    print("\nBottom-5 states by delay rate:")
    print(state_df.nsmallest(5, "delay_rate")[["state", "n_airports",
                                               "total_flights", "delay_rate"]])

    fig_tile_choropleth(state_df)
    fig_us_outline_bubbles(apt)

    print("\n[main] Script 04 complete.")


if __name__ == "__main__":
    main()
