# Rainfall Prediction using Machine Learning

Internship project structure — Zephyr Technologies and Solutions Pvt Ltd

```
Rainfall_Prediction_Project/
│── dataset/
│   ├── make_dataset.py        <- generates rainfall_dataset.csv
│   └── rainfall_dataset.csv   <- 10,950 rows, 20 weather columns
│── notebooks/
│   ├── Rainfall_Prediction.ipynb   <- MAIN NOTEBOOK (already run, with outputs)
│   └── rainfall_prediction.py      <- same pipeline as a plain script
│── models/
│   ├── logistic_regression_model.pkl   <- best-performing model (no district dependency)
│   ├── scaler.pkl
│   ├── label_encoder_psc.pkl
│   └── label_encoder_wsc.pkl
│── graphs/
│   ├── Rainfall_Class_Count.png
│   ├── Rainfall_Distribution.png
│   ├── Correlation_Heatmap.png
│   ├── Confusion_Matrix.png
│   └── Model_Accuracy_Comparison.png
└── README.md
```

## How to run (Google Colab or Jupyter Notebook)

1. Upload the whole `Rainfall_Prediction_Project` folder to Google Drive, or
   open Google Colab and upload just `notebooks/Rainfall_Prediction.ipynb`
   plus `dataset/rainfall_dataset.csv`.
2. If running in Colab, adjust the path in the first data-loading cell from
   `../dataset/rainfall_dataset.csv` to wherever you placed the CSV
   (e.g. `/content/rainfall_dataset.csv`).
3. Run all cells: **Runtime → Run all**.
4. Take your own screenshots of each output cell (dataset preview, EDA
   graphs, accuracy printouts, classification report, confusion matrix)
   for your report appendix — this is what "screenshots" means in your
   report structure.

## About the dataset

Your original dataset file was missing, so this project uses a newly
generated dataset (`make_dataset.py`) with the **same 20 columns** your
report describes (YEAR, MO, DY, RH2M, T2MDEW, QV2M, PS, WS50M,
PRECTOTCORR, T2MWET, WD50M, T2M_MAX, T2M_MIN, ALLSKY_SFC_UV_INDEX, TS,
PSC, WSC, DISTRICT, LATITUDE, LONGITUDE) across 5 Karnataka districts
and 6 years, with realistic seasonal rainfall patterns (monsoon months
wetter, summer drier).

**Important — accuracy will differ from your report:** with this
generated data, Logistic Regression is the best model at **75.98%**
accuracy, not the 91.51% your report states. That number cannot be
reproduced without your original dataset, since it depends on the
exact data. If
you still have the original `.csv`/`.ipynb` anywhere (email attachments,
Colab's "Recent" list, browser downloads folder, Google Drive trash),
recovering that is the only way to get back the exact 91.51% figure.
Otherwise, use the numbers this run actually produces, and be ready to
explain in the viva that they come from your own re-run.

## Pipeline summary (matches report Section 05/06)

1. **Data Collection** — load the CSV.
2. **Preprocessing** — check missing values; Label Encode PSC and WSC;
   create `Rainfall_Class` from `PRECTOTCORR`; train/test split 80/20
   (before scaling, to avoid leakage); StandardScaler fit on train only.
   (DISTRICT as a named category was later removed in favour of
   LATITUDE/LONGITUDE — see "Connecting to a live weather API" below —
   so the model generalizes to any location, not just the 5 original
   districts.)
3. **EDA** — Rainfall Class Count, Rainfall Distribution, Correlation
   Heatmap.
4. **Feature Selection** — all weather/time/location columns except the
   rainfall amount itself.
5. **Model Selection & Training** — Logistic Regression, Decision Tree
   (max_depth=8), Random Forest (n_estimators=100).
6. **Evaluation** — Accuracy Score, Classification Report, Confusion
   Matrix; best model saved as a `.pkl` file.

## Connecting to a live weather API (NASA POWER)

Your dataset's column names (RH2M, PS, WS50M, PRECTOTCORR, etc.) match
the parameter names used by **NASA POWER**
(https://power.larc.nasa.gov/) — a free, no-signup weather/climate API
run by NASA. This is almost certainly where the original project's data
came from, so it's a natural fit.

New files in `api/`:

- **`common_features.py`** — shared logic (pressure/wind categories,
  feature column order) so training and live prediction stay consistent.
- **`fetch_power_data.py`** — downloads real historical daily weather
  for a given latitude/longitude and date range, and saves it in this
  project's exact dataset format. Use this to replace the synthetic
  dataset with real data, then rerun `notebooks/Rainfall_Prediction.ipynb`
  to retrain on it.

  ```bash
  cd api
  pip install -r requirements.txt
  python fetch_power_data.py --lat 12.9716 --lon 77.5946 \
      --district "Bengaluru Urban" --start 20200101 --end 20251231 \
      --out ../dataset/rainfall_dataset_real.csv
  ```
  (`--district` here is just a label stored in the CSV for your own
  reference — any name works, it isn't used to restrict anything.)

- **`predict_live.py`** — fetches the most recent available weather for
  **any** latitude/longitude and runs it through your already-trained
  model to predict rain / no rain right now.

  ```bash
  cd api
  python predict_live.py --lat 13.0827 --lon 80.2707 --place "Chennai"
  ```

  Earlier versions of this script required the location to be one of
  5 pre-set districts, because DISTRICT was trained as a fixed named
  category — a label encoder like that only ever recognizes the exact
  names it was shown. That's since been fixed: the model is retrained
  on LATITUDE/LONGITUDE instead of a district name, so any coordinates
  work, anywhere in the world. `--place` is optional and only affects
  the printed label.

**Note:** these scripts need internet access to reach the NASA POWER
API — run them on your own machine, not inside a sandboxed notebook
environment without network access. I wrote and compile-checked the
code here, but could not make a live request to verify the actual API
response, so test it once yourself and watch for any small
error-handling fixes needed on your end.

## Web app — live rainfall predictor (`app/index.html`)

A self-contained, single-file web page where you type **any place
name** (not just the 5 original districts), enter today's weather
(humidity, dew point, specific humidity, pressure, both temperatures,
surface temperature, wind speed/direction, UV index, month, day), and
get a live rain / no-rain prediction — no server, no Python runtime
needed once the page is open.

**Finding a location:** type a place name (city, district, town —
anywhere) and click **Find**; it looks up latitude/longitude using the
free [Open-Meteo Geocoding API](https://open-meteo.com) (no sign-up
needed). This needs internet access and works in a normal browser; if
it can't reach the API (offline, or a sandboxed preview blocking
external requests), just type latitude/longitude into the two fields
below it by hand instead — the prediction works exactly the same way
either way.

**How it works:** a Decision Tree (max depth 6, trained on
`dataset/rainfall_dataset.csv`, test accuracy 76.1%) was exported
directly from scikit-learn into a JavaScript function
(`predictTree()` inside `app/index.html`), reusing the exact split
thresholds the trained tree learned. This is a separate, simplified
tree trained on **unscaled** features (trees don't need scaling) so
the thresholds are in plain, human-readable units like "humidity
<= 70%". Like the notebook's saved model, it uses LATITUDE/LONGITUDE
rather than a fixed district category, so it works for any location.

Open it by double-clicking `app/index.html` in any browser, or host
it for free on **GitHub Pages**:

```bash
# from the repo root, after pushing to GitHub:
# Settings → Pages → Deploy from branch → main → /app folder
```

This matches the "Web / Mobile Application" item in the project's
future-scope slide — useful to show live during a viva.
