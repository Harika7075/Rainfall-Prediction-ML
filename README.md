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
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── label_encoder_district.pkl
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
generated data, Random Forest is the best model at **76.03%** accuracy,
not the 91.51% your report states. That number cannot be reproduced
without your original dataset, since it depends on the exact data. If
you still have the original `.csv`/`.ipynb` anywhere (email attachments,
Colab's "Recent" list, browser downloads folder, Google Drive trash),
recovering that is the only way to get back the exact 91.51% figure.
Otherwise, use the numbers this run actually produces, and be ready to
explain in the viva that they come from your own re-run.

## Pipeline summary (matches report Section 05/06)

1. **Data Collection** — load the CSV.
2. **Preprocessing** — check missing values; Label Encode DISTRICT, PSC,
   WSC; create `Rainfall_Class` from `PRECTOTCORR`; train/test split
   80/20 (before scaling, to avoid leakage); StandardScaler fit on train
   only.
3. **EDA** — Rainfall Class Count, Rainfall Distribution, Correlation
   Heatmap.
4. **Feature Selection** — all weather/time/location columns except the
   rainfall amount itself.
5. **Model Selection & Training** — Logistic Regression, Decision Tree
   (max_depth=8), Random Forest (n_estimators=100).
6. **Evaluation** — Accuracy Score, Classification Report, Confusion
   Matrix; best model saved as a `.pkl` file.
