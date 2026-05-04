# Analysis of U.S. Domestic Flight Delay Causes

**DSE501 – Statistics for Data Analysts (Spring C 2026)**
**Team 8** &nbsp;·&nbsp; James Czeranko, Ryan Garcia, Shaurya, Yaksh Maulesh Pancholi

An inferential study of carrier, seasonal, and geographic drivers of arrival
delays at 356 U.S. domestic airports between January and August 2024, using
the U.S. Department of Transportation Bureau of Transportation Statistics
(BTS) *Airline Delay Cause* dataset.

---

## Headline Results

| Hypothesis | Test | Statistic | p-value | Decision |
|---|---|---:|---:|---|
| **H1** Carrier effect on delay rate | One-way ANOVA / Kruskal–Wallis | *F* = 124.48 / *H* = 2,663.83 | < 10⁻¹⁰ | Reject *H₀* |
| **H2** Seasonal variation (Jan–Aug) | One-way ANOVA + Tukey HSD | *F* = 276.76 | < 10⁻¹⁰ | Reject *H₀* |
| **H3** Weather vs. carrier delay minutes | Two-proportion *z*-test | *z* = −4,388.84 | < 10⁻¹⁰ | Reject *H₀* |
| **H4** Volume vs. delay rate | Pearson on log₁₀ volume | *r* = 0.399 | < 10⁻¹⁰ | Reject *H₀* |

A nested multivariate model comparison shows that adding **carrier identity**
to a baseline of seasonality and volume more than doubles the explained
variance (R² 0.120 → 0.256), establishing carrier as the single most
important driver in this setting.

The full written report (18 pages, LaTeX) is in `latex/main.pdf`.

---

## Repository Structure

```
flight_delay_project/
├── README.md                       # this file
├── requirements.txt                # Python dependencies
├── data/
│   └── Airline_Delay_Cause.xlsx    # raw BTS dataset (Jan–Aug 2024)
├── code/
│   ├── 01_load_and_explore.py      # cleaning, EDA, descriptive figures
│   ├── 02_hypothesis_tests.py      # H1–H4 with assumption checks
│   └── 03_geo_and_models.py        # geographic viz + multivariate models
├── output/
│   ├── clean_data.csv              # cleaned, feature-engineered dataset
│   ├── airport_summary.csv         # airport-level aggregates with lat/lon
│   ├── summary.json                # descriptive statistics
│   ├── hypothesis_tests.json       # full test results for H1–H4
│   └── extended_analysis.json      # regional ANOVA + nested-model results
├── figures/                        # all PNG figures (200 dpi)
│   ├── fig_cause_breakdown.png
│   ├── fig_carrier_delay_rate.png
│   ├── fig_monthly_trend.png
│   ├── fig_heatmap_carrier_month.png
│   ├── fig_volume_vs_rate.png
│   ├── fig_h2_tukey_heatmap.png
│   ├── fig_h3_weather_vs_carrier.png
│   ├── fig_h4_volume_regression.png
│   ├── fig_geo_map.png
│   ├── fig_regional.png
│   └── fig_model_comparison.png
└── latex/
    ├── main.tex                    # source of the final report
    ├── main.pdf                    # compiled report (18 pages)
    ├── references.bib              # BibTeX bibliography (16 entries)
    └── figures/                    # mirror of /figures/ for LaTeX
```

---

## Reproducing the Analysis

### 1. Set up the environment

```bash
git clone <this-repo>
cd flight_delay_project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is included under `data/`. If you want the latest BTS release,
download it from
<https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp>
and replace `data/Airline_Delay_Cause.xlsx`.

### 2. Run the analysis pipeline

The three scripts are designed to be run sequentially:

```bash
cd code
python 01_load_and_explore.py    # ~10 s — cleans data, writes CSV + 5 figures
python 02_hypothesis_tests.py    # ~10 s — H1–H4 + Tukey HSD heatmap
python 03_geo_and_models.py      # ~15 s — geographic map + nested models
```

Each script prints its results to `stdout` and writes JSON summaries to
`output/`. All figures land in `figures/` at 200 dpi.

### 3. Build the report

```bash
cd ../latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `latex/main.pdf` (18 pages, ~1.4 MB).

---

## Methods Summary

| Hypothesis | Method | Software |
|---|---|---|
| H1 | One-way ANOVA, Kruskal–Wallis, Tukey HSD, η² effect size | `scipy.stats` |
| H2 | One-way ANOVA, Kruskal–Wallis, full Tukey HSD pairwise grid | `scipy.stats` |
| H3 | Pooled two-proportion z-test on delay-minute shares | NumPy + `scipy.stats.norm` |
| H4 | Pearson, Spearman, OLS regression on log₁₀ volume | `scipy.stats`, `sklearn.linear_model` |
| Extension | Nested OLS models M1/M2/M3 with 5-fold CV; standardized coefficients | `scikit-learn` |

Assumption checks (Shapiro–Wilk for normality, Levene for equal variance)
are reported alongside every parametric test. Where assumptions fail, the
non-parametric Kruskal–Wallis result is treated as the primary test.

---

## Data

- **Source.** U.S. Department of Transportation, Bureau of Transportation
  Statistics — *Airline On-Time Performance and Causes of Flight Delays*.
- **Coverage.** Jan–Aug 2024, 21 reporting carriers, 356 airports.
- **Records.** 15,060 raw → 15,046 after dropping 14 records (0.09%) with
  missing values.
- **Aggregate volume.** 5,022,488 flights, of which 1,129,387 (22.49%)
  were delayed by 15 or more minutes.
- **License.** BTS data is public-domain U.S. government data.

The 21 columns include arrival counts (`arr_flights`, `arr_del15`),
total delay minutes (`arr_delay`), and per-cause counts and minutes for
the five DOT cause categories (carrier, weather, NAS, security, late
aircraft).

---

## Key Figures

| File | What it shows |
|---|---|
| `fig_cause_breakdown.png` | Marginal split of delays by DOT cause |
| `fig_carrier_delay_rate.png` | Boxplots of delay rate per carrier |
| `fig_monthly_trend.png` | Monthly mean delay rate + cause composition |
| `fig_heatmap_carrier_month.png` | Carrier × month delay-rate heatmap |
| `fig_volume_vs_rate.png` | Log-scale scatter of volume vs delay rate |
| `fig_h2_tukey_heatmap.png` | Pairwise Tukey HSD p-values across months |
| `fig_h3_weather_vs_carrier.png` | Weather vs carrier delay-minute shares |
| `fig_h4_volume_regression.png` | OLS fit + quintile boxplot for H4 |
| `fig_geo_map.png` | Bubble map of top-60 CONUS airports |
| `fig_regional.png` | U.S. Census-region delay-rate comparison |
| `fig_model_comparison.png` | Nested-model R² and top standardized coefficients |

---

## License & Attribution

Code released under the MIT License. Report text © Team 8, 2026.
The BTS dataset is in the public domain.

If you build on this work, please cite the underlying BTS release and the
methodological references listed in `latex/references.bib`.
