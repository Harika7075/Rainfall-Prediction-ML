"""
Fetches the most recent available weather data for a location from NASA
POWER, then uses your already-trained model (models/*.pkl) to predict
whether it will rain.

Usage:
    python predict_live.py --lat 12.9716 --lon 77.5946 --district "Bengaluru Urban"

IMPORTANT LIMITATION:
The saved DISTRICT label encoder only knows the 5 districts used when
training on the synthetic dataset (Bengaluru Urban, Chikkaballapur,
Kolar, Tumakuru, Ramanagara). If you pass a different district name,
this will raise an error. Retrain on real data (see fetch_power_data.py
+ rerun the notebook) if you need other locations.
"""
import argparse
import pickle
import datetime as dt
import pandas as pd
import requests

from common_features import pressure_category, wind_category, FEATURE_ORDER

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "T2M_MAX", "T2M_MIN", "RH2M", "T2MDEW", "T2MWET", "QV2M",
    "PS", "WS50M", "WD50M", "ALLSKY_SFC_UV_INDEX", "TS",
]

MODEL_DIR = "../models"


def fetch_latest(lat, lon, lookback_days=45):
    """POWER's near-real-time data can lag by more than two weeks
    depending on the parameter, so pull a wider window and use the most
    recent row available. Falls back to filling any still-missing minor
    fields with dataset defaults rather than failing outright."""
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    param_data = resp.json()["properties"]["parameter"]

    dates = sorted(param_data[PARAMETERS[0]].keys())

    # Sane physical ranges per parameter. POWER sometimes returns odd
    # sentinel values (not just -999) for a field that isn't ready yet,
    # e.g. a wind direction of -73710. Anything outside these bounds is
    # treated as missing rather than trusted.
    VALID_RANGE = {
        "T2M_MAX": (-50, 60), "T2M_MIN": (-50, 60), "RH2M": (0, 100),
        "T2MDEW": (-60, 50), "T2MWET": (-60, 50), "QV2M": (0, 40),
        "PS": (50, 110), "WS50M": (0, 100), "WD50M": (0, 360),
        "ALLSKY_SFC_UV_INDEX": (0, 20), "TS": (-50, 70),
    }

    def is_valid(param, v):
        if v is None or v == -999:
            return False
        lo, hi = VALID_RANGE.get(param, (-1e9, 1e9))
        return lo <= v <= hi

    # Core features the model relies on most; minor ones can fall back
    # to dataset-mean defaults if POWER hasn't published them yet.
    core = ["RH2M", "T2M_MAX", "T2M_MIN", "WS50M"]

    # First pass: a day where every parameter is present and sane
    for d in reversed(dates):
        values = {p: param_data[p].get(d) for p in PARAMETERS}
        if all(is_valid(p, v) for p, v in values.items()):
            return d, values, []

    # Second pass: a day where at least the core parameters are sane;
    # anything else gets reported as "missing" so the caller fills it
    # with a dataset-mean default instead of trusting a bad value.
    for d in reversed(dates):
        values = {p: param_data[p].get(d) for p in PARAMETERS}
        if all(is_valid(p, values[p]) for p in core):
            missing = [p for p in PARAMETERS if not is_valid(p, values[p])]
            return d, values, missing

    raise RuntimeError(
        f"No usable data in the last {lookback_days} days from POWER for "
        f"this location. Try increasing --lookback or a different lat/lon."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--district", type=str, required=True,
                     help="Must be one of the districts the model was trained on")
    ap.add_argument("--lookback", type=int, default=45,
                     help="How many past days to search for usable data (default 45)")
    args = ap.parse_args()

    # Dataset-mean fallbacks for any field POWER hasn't published yet
    PARAM_DEFAULTS = {
        "T2M_MAX": 29.66, "T2M_MIN": 21.67, "RH2M": 61.51, "T2MDEW": 21.95,
        "T2MWET": 24.85, "QV2M": 9.85, "PS": 96.53, "WS50M": 4.68,
        "WD50M": 179.09, "ALLSKY_SFC_UV_INDEX": 5.97, "TS": 30.66,
    }

    print(f"Fetching latest available weather for ({args.lat}, {args.lon}) ...")
    date_str, values, missing = fetch_latest(args.lat, args.lon, args.lookback)
    if missing:
        print(f"Note: POWER hasn't published {missing} for {date_str} yet; "
              f"using typical dataset values for those.")
        for p in missing:
            values[p] = PARAM_DEFAULTS[p]
    print(f"Using data from {date_str}: {values}")

    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder_district.pkl", "rb") as f:
        le_district = pickle.load(f)
    # pick whichever trained model file exists
    import glob
    model_files = glob.glob(f"{MODEL_DIR}/*_model.pkl")
    if not model_files:
        raise RuntimeError("No trained model found in ../models/")
    with open(model_files[0], "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {model_files[0]}")

    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])

    try:
        district_encoded = le_district.transform([args.district])[0]
    except ValueError:
        raise SystemExit(
            f"'{args.district}' was not one of the districts used in training "
            f"({list(le_district.classes_)}). Retrain on real data to add new "
            f"locations — see fetch_power_data.py."
        )

    psc = pressure_category(values["PS"])
    wsc = wind_category(values["WS50M"])
    # PSC/WSC were label-encoded alphabetically during training:
    # High=0, Low=1, Medium=2 ; Calm=0, Moderate=1, Strong=2
    psc_map = {"High": 0, "Low": 1, "Medium": 2}
    wsc_map = {"Calm": 0, "Moderate": 1, "Strong": 2}

    row = {
        "YEAR": year, "MO": month, "DY": day,
        "RH2M": values["RH2M"], "T2MDEW": values["T2MDEW"], "QV2M": values["QV2M"],
        "PS": values["PS"], "WS50M": values["WS50M"], "T2MWET": values["T2MWET"],
        "WD50M": values["WD50M"], "T2M_MAX": values["T2M_MAX"], "T2M_MIN": values["T2M_MIN"],
        "ALLSKY_SFC_UV_INDEX": values["ALLSKY_SFC_UV_INDEX"], "TS": values["TS"],
        "PSC": psc_map[psc], "WSC": wsc_map[wsc],
        "DISTRICT": district_encoded, "LATITUDE": args.lat, "LONGITUDE": args.lon,
    }
    X_live = pd.DataFrame([row])[FEATURE_ORDER]
    X_live_scaled = scaler.transform(X_live)

    pred = model.predict(X_live_scaled)[0]
    label = "RAIN" if pred == 1 else "NO RAIN"
    print(f"\nPrediction for {args.district} on {date_str}: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_live_scaled)[0]
        print(f"Confidence — No Rain: {proba[0]*100:.1f}%  |  Rain: {proba[1]*100:.1f}%")


if __name__ == "__main__":
    main()
