"""
DSE501 Term Project - Team 8
Analysis of U.S. Domestic Flight Delay Causes

Script 03: Geographic visualization and Extended Modeling
---------------------------------------------------------
1. Geographic map of the top 60 airports, with bubble size = traffic
   volume and color = delay rate.
2. Regional analysis (4 U.S. Census regions).
3. Multiple linear regression: delay rate as a function of carrier,
   month, log(volume) and weather share. We compare three nested models
   (M1 base, M2 with carrier, M3 full) using R^2 and adjusted R^2.

Outputs to ../figures and ../output.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from scipy import stats

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


# ---------------------------------------------------------------------------
# Curated lookup of major U.S. airport coordinates and Census regions.
# 100 airports here cover the great majority of total traffic. The dataset
# uses standard 3-letter IATA codes. State assignments follow the U.S. Census
# Bureau's four-region classification (Northeast / Midwest / South / West).
# Source: openflights.org, FAA airport database, U.S. Census Bureau.
# ---------------------------------------------------------------------------
AIRPORTS = {
    "ATL": (33.6407, -84.4277, "GA", "South"),
    "DFW": (32.8998, -97.0403, "TX", "South"),
    "DEN": (39.8561, -104.6737, "CO", "West"),
    "ORD": (41.9742, -87.9073, "IL", "Midwest"),
    "LAX": (33.9416, -118.4085, "CA", "West"),
    "JFK": (40.6413, -73.7781, "NY", "Northeast"),
    "LAS": (36.0840, -115.1537, "NV", "West"),
    "MCO": (28.4312, -81.3081, "FL", "South"),
    "CLT": (35.2140, -80.9431, "NC", "South"),
    "MIA": (25.7959, -80.2870, "FL", "South"),
    "SEA": (47.4502, -122.3088, "WA", "West"),
    "PHX": (33.4342, -112.0116, "AZ", "West"),
    "EWR": (40.6895, -74.1745, "NJ", "Northeast"),
    "SFO": (37.6213, -122.3790, "CA", "West"),
    "IAH": (29.9902, -95.3368, "TX", "South"),
    "BOS": (42.3656, -71.0096, "MA", "Northeast"),
    "FLL": (26.0742, -80.1506, "FL", "South"),
    "MSP": (44.8848, -93.2223, "MN", "Midwest"),
    "LGA": (40.7769, -73.8740, "NY", "Northeast"),
    "DTW": (42.2124, -83.3534, "MI", "Midwest"),
    "PHL": (39.8744, -75.2424, "PA", "Northeast"),
    "SLC": (40.7899, -111.9791, "UT", "West"),
    "BWI": (39.1774, -76.6684, "MD", "South"),
    "DCA": (38.8512, -77.0402, "VA", "South"),
    "IAD": (38.9531, -77.4565, "VA", "South"),
    "MDW": (41.7868, -87.7522, "IL", "Midwest"),
    "TPA": (27.9755, -82.5332, "FL", "South"),
    "SAN": (32.7338, -117.1933, "CA", "West"),
    "BNA": (36.1245, -86.6782, "TN", "South"),
    "AUS": (30.1975, -97.6664, "TX", "South"),
    "HOU": (29.6454, -95.2789, "TX", "South"),
    "DAL": (32.8471, -96.8517, "TX", "South"),
    "PDX": (45.5887, -122.5975, "OR", "West"),
    "STL": (38.7487, -90.3700, "MO", "Midwest"),
    "RDU": (35.8776, -78.7875, "NC", "South"),
    "SJC": (37.3626, -121.9290, "CA", "West"),
    "SMF": (38.6954, -121.5908, "CA", "West"),
    "MSY": (29.9934, -90.2580, "LA", "South"),
    "OAK": (37.7126, -122.2197, "CA", "West"),
    "PIT": (40.4915, -80.2329, "PA", "Northeast"),
    "MCI": (39.2976, -94.7139, "MO", "Midwest"),
    "CLE": (41.4117, -81.8498, "OH", "Midwest"),
    "IND": (39.7173, -86.2944, "IN", "Midwest"),
    "CMH": (39.9980, -82.8919, "OH", "Midwest"),
    "JAX": (30.4941, -81.6879, "FL", "South"),
    "ABQ": (35.0402, -106.6094, "NM", "West"),
    "ANC": (61.1742, -149.9961, "AK", "West"),
    "BUF": (42.9405, -78.7322, "NY", "Northeast"),
    "BUR": (34.2007, -118.3590, "CA", "West"),
    "CVG": (39.0489, -84.6678, "KY", "South"),
    "ELP": (31.8072, -106.3779, "TX", "South"),
    "GEG": (47.6199, -117.5339, "WA", "West"),
    "HNL": (21.3187, -157.9224, "HI", "West"),
    "OGG": (20.8986, -156.4305, "HI", "West"),
    "OMA": (41.3032, -95.8941, "NE", "Midwest"),
    "ONT": (34.0560, -117.6012, "CA", "West"),
    "ORF": (36.8946, -76.2012, "VA", "South"),
    "PBI": (26.6832, -80.0956, "FL", "South"),
    "PVD": (41.7240, -71.4283, "RI", "Northeast"),
    "RIC": (37.5052, -77.3197, "VA", "South"),
    "RNO": (39.4990, -119.7681, "NV", "West"),
    "ROC": (43.1186, -77.6724, "NY", "Northeast"),
    "SAT": (29.5337, -98.4698, "TX", "South"),
    "SDF": (38.1740, -85.7361, "KY", "South"),
    "SJU": (18.4394, -66.0018, "PR", "South"),
    "SNA": (33.6757, -117.8682, "CA", "West"),
    "SYR": (43.1112, -76.1063, "NY", "Northeast"),
    "TUL": (36.1984, -95.8881, "OK", "South"),
    "TUS": (32.1161, -110.9410, "AZ", "West"),
    "ABE": (40.6521, -75.4408, "PA", "Northeast"),
    "ALB": (42.7483, -73.8017, "NY", "Northeast"),
    "BDL": (41.9389, -72.6832, "CT", "Northeast"),
    "BHM": (33.5629, -86.7535, "AL", "South"),
    "BIL": (45.8077, -108.5429, "MT", "West"),
    "BOI": (43.5644, -116.2228, "ID", "West"),
    "BTV": (44.4719, -73.1533, "VT", "Northeast"),
    "BZN": (45.7776, -111.1530, "MT", "West"),
    "CAE": (33.9388, -81.1195, "SC", "South"),
    "CHS": (32.8986, -80.0405, "SC", "South"),
    "DAY": (39.9024, -84.2194, "OH", "Midwest"),
    "DSM": (41.5340, -93.6631, "IA", "Midwest"),
    "EUG": (44.1246, -123.2110, "OR", "West"),
    "FAR": (46.9207, -96.8158, "ND", "Midwest"),
    "FAT": (36.7762, -119.7181, "CA", "West"),
    "FSD": (43.5820, -96.7419, "SD", "Midwest"),
    "GPT": (30.4073, -89.0700, "MS", "South"),
    "GRR": (42.8808, -85.5228, "MI", "Midwest"),
    "GSO": (36.0978, -79.9374, "NC", "South"),
    "GSP": (34.8957, -82.2189, "SC", "South"),
    "HSV": (34.6372, -86.7751, "AL", "South"),
    "ICT": (37.6499, -97.4331, "KS", "Midwest"),
    "ILM": (34.2706, -77.9026, "NC", "South"),
    "JAN": (32.3112, -90.0759, "MS", "South"),
    "LBB": (33.6636, -101.8228, "TX", "South"),
    "LEX": (38.0365, -84.6059, "KY", "South"),
    "LGB": (33.8177, -118.1516, "CA", "West"),
    "LIT": (34.7294, -92.2243, "AR", "South"),
    "MEM": (35.0424, -89.9767, "TN", "South"),
    "MKE": (42.9472, -87.8966, "WI", "Midwest"),
    "MSN": (43.1399, -89.3375, "WI", "Midwest"),
    "MYR": (33.6797, -78.9283, "SC", "South"),
    "OKC": (35.3931, -97.6007, "OK", "South"),
    "PNS": (30.4734, -87.1866, "FL", "South"),
    "PSP": (33.8297, -116.5067, "CA", "West"),
    "PWM": (43.6462, -70.3093, "ME", "Northeast"),
    "RSW": (26.5362, -81.7552, "FL", "South"),
    "SAV": (32.1276, -81.2021, "GA", "South"),
    "SBA": (34.4262, -119.8403, "CA", "West"),
    "SHV": (32.4466, -93.8256, "LA", "South"),
    "SRQ": (27.3954, -82.5544, "FL", "South"),
    "TYS": (35.8110, -83.9941, "TN", "South"),
    "XNA": (36.2818, -94.3068, "AR", "South"),
    "ATW": (44.2581, -88.5191, "WI", "Midwest"),
    "GRB": (44.4851, -88.1296, "WI", "Midwest"),
    "FNT": (42.9655, -83.7436, "MI", "Midwest"),
    "AVL": (35.4362, -82.5418, "NC", "South"),
    "BGR": (44.8074, -68.8281, "ME", "Northeast"),
    "BTR": (30.5332, -91.1496, "LA", "South"),
    "CRP": (27.7704, -97.5012, "TX", "South"),
    "MAF": (31.9425, -102.2019, "TX", "South"),
    "MFE": (26.1758, -98.2386, "TX", "South"),
    "AMA": (35.2194, -101.7060, "TX", "South"),
    "ABI": (32.4113, -99.6819, "TX", "South"),
    "TLH": (30.3965, -84.3503, "FL", "South"),
    "VPS": (30.4832, -86.5254, "FL", "South"),
    "ECP": (30.3417, -85.7975, "FL", "South"),
    "MOB": (30.6912, -88.2428, "AL", "South"),
    "MGM": (32.3006, -86.3939, "AL", "South"),
    "FAY": (34.9912, -78.8803, "NC", "South"),
    "OAJ": (34.8292, -77.6125, "NC", "South"),
    "EWN": (35.0730, -77.0429, "NC", "South"),
}


def airport_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate airport-level metrics and attach lat/lon and region."""
    apt = df.groupby(["airport", "airport_name"]).agg(
        total_flights=("arr_flights", "sum"),
        total_delays=("arr_del15", "sum"),
        total_delay_min=("arr_delay", "sum"),
        weather_min=("weather_delay", "sum"),
        carrier_min=("carrier_delay", "sum"),
        nas_min=("nas_delay", "sum"),
        late_aircraft_min=("late_aircraft_delay", "sum"),
        n_carriers=("carrier", "nunique"),
    ).reset_index()
    apt["delay_rate"] = apt["total_delays"] / apt["total_flights"]
    apt["avg_delay_min"] = apt["total_delay_min"] / apt["total_delays"]

    apt["lat"] = apt["airport"].map(lambda c: AIRPORTS.get(c, (None,) * 4)[0])
    apt["lon"] = apt["airport"].map(lambda c: AIRPORTS.get(c, (None,) * 4)[1])
    apt["state"] = apt["airport"].map(lambda c: AIRPORTS.get(c, (None,) * 4)[2])
    apt["region"] = apt["airport"].map(lambda c: AIRPORTS.get(c, (None,) * 4)[3])
    return apt


# ---------------------------------------------------------------------------
# Geographic visualization (matplotlib only - no external map libs)
# ---------------------------------------------------------------------------
def fig_geo_map(apt: pd.DataFrame) -> None:
    """Bubble map of CONUS airports with size = volume, color = delay rate."""
    geo = apt.dropna(subset=["lat", "lon"]).copy()
    # Restrict to contiguous U.S. for the main map (drop AK, HI, PR)
    conus = geo[(geo["lon"] > -130) & (geo["lon"] < -65)
                & (geo["lat"] > 24) & (geo["lat"] < 50)]
    top = conus.nlargest(60, "total_flights")

    fig, ax = plt.subplots(figsize=(11.5, 6.7))

    # Approximate U.S. continental outline using state plot
    # We'll just draw a simple bounding rectangle as backdrop since we
    # cannot rely on cartopy / basemap in this environment.
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_facecolor("#f7f7f7")

    # Light grid for geographic context
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")

    sizes = (top["total_flights"] / top["total_flights"].max()) * 1500 + 30
    sc = ax.scatter(top["lon"], top["lat"], s=sizes, c=top["delay_rate"],
                    cmap="RdYlGn_r", edgecolor="black", linewidth=0.6,
                    alpha=0.85, vmin=0.15, vmax=0.30)

    # Label the top 12 by volume
    for _, row in top.nlargest(12, "total_flights").iterrows():
        ax.annotate(row["airport"], (row["lon"], row["lat"]),
                    fontsize=8, weight="bold", color="black",
                    xytext=(5, 5), textcoords="offset points")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delay rate")
    ax.set_title("Top-60 CONUS airports: bubble size = traffic volume, "
                 "color = delay rate (Jan–Aug 2024)")

    # Volume legend
    for vol, lbl in [(50000, "50k"), (100000, "100k"), (200000, "200k")]:
        s = (vol / top["total_flights"].max()) * 1500 + 30
        ax.scatter([], [], s=s, c="lightgray", edgecolor="black",
                   linewidth=0.6, label=f"{lbl} flights")
    ax.legend(scatterpoints=1, frameon=True, labelspacing=1.4,
              title="Volume", loc="lower left", fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_geo_map.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")


def fig_regional_breakdown(apt: pd.DataFrame) -> dict:
    """Boxplot + cause composition by Census region."""
    geo = apt.dropna(subset=["region"]).copy()
    print(f"\n[regions] mapped {len(geo)}/{len(apt)} airports to a region "
          f"({100*len(geo)/len(apt):.1f}%)")

    region_summary = geo.groupby("region").agg(
        n_airports=("airport", "count"),
        total_flights=("total_flights", "sum"),
        mean_delay_rate=("delay_rate", "mean"),
        median_delay_rate=("delay_rate", "median"),
    ).round(4)
    print("\nRegional summary:")
    print(region_summary)

    # ANOVA across regions
    groups = [g["delay_rate"].values for _, g in geo.groupby("region")]
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    print(f"\nRegional ANOVA  : F = {f_stat:.3f}, p = {f_p:.4e}")
    print(f"Regional Kruskal: H = {h_stat:.3f}, p = {h_p:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    region_order = ["Northeast", "Midwest", "South", "West"]
    sns.boxplot(data=geo, x="region", y="delay_rate", order=region_order,
                ax=ax, palette="Set2", linewidth=0.8, fliersize=3)
    ax.set_title(f"(a) Delay rate by U.S. Census region\n"
                 f"ANOVA: F = {f_stat:.2f}, p = {f_p:.2e}")
    ax.set_xlabel("Region")
    ax.set_ylabel("Airport delay rate")

    ax = axes[1]
    cause_cols = ["late_aircraft_min", "carrier_min", "nas_min", "weather_min"]
    cause_labels = ["Late Aircraft", "Carrier", "NAS", "Weather"]
    cause_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    by_region = geo.groupby("region")[cause_cols].sum()
    by_region_pct = by_region.div(by_region.sum(axis=1), axis=0) * 100
    by_region_pct = by_region_pct.reindex(region_order)
    by_region_pct.columns = cause_labels

    bottom = np.zeros(len(by_region_pct))
    for col, color in zip(cause_labels, cause_colors):
        ax.bar(by_region_pct.index, by_region_pct[col], bottom=bottom,
               label=col, color=color, edgecolor="white", linewidth=0.5)
        bottom += by_region_pct[col].values
    ax.set_ylabel("Share of delay minutes (%)")
    ax.set_xlabel("Region")
    ax.set_title("(b) Delay-cause composition by region")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_regional.png")
    plt.savefig(out)
    plt.close()
    print(f"[fig] saved {out}")

    return {
        "anova_F": float(f_stat), "anova_p": float(f_p),
        "kruskal_H": float(h_stat), "kruskal_p": float(h_p),
        "region_summary": region_summary.reset_index().to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Multivariate regression model comparison
# ---------------------------------------------------------------------------
def fit_multivariate_models(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("EXTENDED MODEL: multivariate linear regression")
    print("=" * 70)

    work = df.copy()
    work["log_volume"] = np.log10(work["arr_flights"].clip(lower=1))
    work["weather_share"] = (
        work["weather_delay"] /
        (work["arr_delay"].replace(0, np.nan))
    ).fillna(0)
    work["nas_share"] = (
        work["nas_delay"] /
        (work["arr_delay"].replace(0, np.nan))
    ).fillna(0)

    y = work["delay_rate"].values
    n = len(y)

    # M1: log_volume + month
    X1 = pd.get_dummies(work["month"], prefix="m", drop_first=True).astype(float)
    X1["log_volume"] = work["log_volume"].values

    # M2: + carrier
    X2 = X1.copy()
    carrier_dum = pd.get_dummies(work["carrier"], prefix="c",
                                 drop_first=True).astype(float)
    X2 = pd.concat([X2.reset_index(drop=True),
                    carrier_dum.reset_index(drop=True)], axis=1)

    # M3: + weather_share + nas_share
    X3 = X2.copy()
    X3["weather_share"] = work["weather_share"].values
    X3["nas_share"] = work["nas_share"].values

    def fit_and_score(X, name):
        reg = LinearRegression().fit(X, y)
        y_hat = reg.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(ss_res / n)

        cv_r2 = cross_val_score(LinearRegression(), X, y,
                                cv=KFold(5, shuffle=True, random_state=42),
                                scoring="r2")
        print(f"  {name:30s}  p={p:>3d}  R^2={r2:.4f}  "
              f"AdjR^2={adj_r2:.4f}  RMSE={rmse:.4f}  "
              f"CV-R^2={cv_r2.mean():.4f}±{cv_r2.std():.4f}")
        return {
            "name": name, "p": int(p), "r2": float(r2),
            "adj_r2": float(adj_r2), "rmse": float(rmse),
            "cv_r2_mean": float(cv_r2.mean()),
            "cv_r2_std": float(cv_r2.std()),
        }

    m1 = fit_and_score(X1, "M1: month + log_volume")
    m2 = fit_and_score(X2, "M2: + carrier")
    m3 = fit_and_score(X3, "M3: + cause shares")

    # Standardised top coefficients from M3 for interpretability
    sc = StandardScaler()
    Xz = sc.fit_transform(X3)
    reg_z = LinearRegression().fit(Xz, (y - y.mean()) / y.std())
    coefs = pd.Series(reg_z.coef_, index=X3.columns).sort_values(key=np.abs,
                                                                ascending=False)
    print("\n  Top-12 standardized predictors (M3):")
    for name, val in coefs.head(12).items():
        print(f"    {name:25s}  beta = {val:+.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    names = ["M1", "M2", "M3"]
    r2_vals = [m1["r2"], m2["r2"], m3["r2"]]
    adj_vals = [m1["adj_r2"], m2["adj_r2"], m3["adj_r2"]]
    cv_vals = [m1["cv_r2_mean"], m2["cv_r2_mean"], m3["cv_r2_mean"]]
    x_pos = np.arange(len(names))
    w = 0.27
    ax.bar(x_pos - w, r2_vals, w, label="R²", color="#1f77b4")
    ax.bar(x_pos, adj_vals, w, label="Adj. R²", color="#ff7f0e")
    ax.bar(x_pos + w, cv_vals, w, label="5-fold CV R²", color="#2ca02c")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel("R² score")
    ax.set_title("(a) Nested model comparison")
    ax.legend()
    for xp, v in zip(x_pos - w, r2_vals):
        ax.text(xp, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xp, v in zip(x_pos, adj_vals):
        ax.text(xp, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xp, v in zip(x_pos + w, cv_vals):
        ax.text(xp, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax = axes[1]
    top12 = coefs.head(12)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top12.values]
    ax.barh(range(len(top12)), top12.values[::-1],
            color=colors[::-1], edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels(top12.index[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Standardized coefficient (β)")
    ax.set_title("(b) Top-12 standardized predictors (M3)")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_model_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"  [fig] saved {out}")

    return {
        "M1": m1, "M2": m2, "M3": m3,
        "top_coefficients": [
            {"feature": str(n), "beta": float(v)} for n, v in coefs.head(12).items()
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    apt = airport_lookup(df)
    apt.to_csv(os.path.join(OUT_DIR, "airport_summary.csv"), index=False)

    fig_geo_map(apt)
    region_results = fig_regional_breakdown(apt)
    model_results = fit_multivariate_models(df)

    out = os.path.join(OUT_DIR, "extended_analysis.json")
    with open(out, "w") as f:
        json.dump({"regional": region_results,
                   "models": model_results}, f, indent=2)
    print(f"\n[main] wrote {out}")
    print("\n[main] Script 03 complete.")


if __name__ == "__main__":
    main()
