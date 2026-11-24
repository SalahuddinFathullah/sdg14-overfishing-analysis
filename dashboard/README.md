# Dashboard — interactive results page

`index.html` is a single offline page with every result of the analysis:

- **Verdict summary**: how many hypothesis tests reject H0, are mixed or weak, or fail to reject, plus the world 2030 forecast.
- **Analysis map** (figure O.01).
- **Verdict table**: every hypothesis (H1a–H5d, the CPUE explanation and the stage rules). Click a row to read its question, evidence and note; the figure gallery then shows only that hypothesis's figures.
- **Forecast to 2030 (H5d)**: choose any of the 8 series to see the observed values, the chosen model's forecast and its 95% interval (hover for values), plus the back-test table.
- **Figure gallery**: all 71 figures, filterable by report section, pipeline step and hypothesis, or limited to the 20 main-body figures. Click a figure to enlarge it.

## Open it

Download or clone the repo and double-click `dashboard/index.html`. It needs no server, internet connection or installed libraries. The images are linked from their folders in the repo (`../1-visualisation/figures/...` etc.), so keep the page inside the repo folder.

GitHub shows `.html` files as source code, not as a page. To view it online, enable GitHub Pages for the repo (Settings → Pages → deploy from `main`, folder `/root`); it is then at `https://salahuddinfathullah.github.io/sdg14-overfishing-analysis/dashboard/`.

## Rebuild it

The data are embedded in the page, so rebuild it whenever results change:

```bash
python 3-hypothesis-tests/hypothesis_tests.py
python 4-overview/overview.py          # writes 4-overview/figure_index.csv
python dashboard/build_dashboard.py    # writes dashboard/index.html
```

| File | Role |
|---|---|
| `build_dashboard.py` | Reads `3-hypothesis-tests/results/decisions.csv`, the H5d forecast tables and `4-overview/figure_index.csv`, rebuilds the observed series with the same loaders H5d uses, and writes the data into the template |
| `template.html` | Layout, styles and drawing code (plain JavaScript, no libraries); light and dark mode follow the system setting |
| `index.html` | Generated output; do not edit by hand |

**Verdict groups.** Each decision is grouped by its first words: "Reject H0", "Mixed", "Weak", "Fail to reject H0", "Rule not met", otherwise "Descriptive". The summary counts only hypothesis tests (codes starting with H) and combines Mixed and Weak.
