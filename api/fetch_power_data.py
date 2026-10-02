"""
Fetches REAL historical weather data from NASA POWER (free, no API key
needed) and saves it in the exact column format used by this project's
dataset, so you can retrain the model on real data instead of the
synthetic dataset.

NASA POWER docs: https://power.larc.nasa.gov/docs/services/api/

Usage:
    python fetch_power_data.py --lat 12.9716 --lon 77.5946 \
        --district "Bengaluru Urban" --start 20200101 --end 20251231 \
        --out ../dataset/rainfall_dataset_real.csv

Notes:
- POWER data usually has a ~2-3 day reporting delay, so don't request
  dates from the last few days.
- One request per district keeps the URL simple; call this script once
  per district you want and concatenate the CSVs, or extend it to loop
  over a list of districts.
"""
import argparse
import sys
import pandas as pd
import requests

from common_features import pressure_category, wind_category

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameter names exactly as POWER expects them. These map directly onto
# this project's dataset columns.
PARAMETERS = [
    "T2M_MAX", "T2M_MIN", "RH2M", "T2MDEW", "T2MWET", "QV2M",
    "PS", "WS50M", "WD50M", "PRECTOTCORR", "ALLSKY_SFC_UV_INDEX", "TS",
]


def fetch(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        param_data = data["properties"]["parameter"]
    except KeyError:
        raise RuntimeError(f"Unexpected response from POWER API: {data}")

    # Each parameter is a dict of {yyyymmdd: value}
    dates = sorted(param_data[PARAMETERS[0]].keys())
    rows = []
    for d in dates:
        year, month, day = int(d[:4]), int(d[4:6]), int(d[6:8])
        row = {"YEAR": year, "MO": month, "DY": day}
        for p in PARAMETERS:
            row[p] = param_data[p].get(d)
        rows.append(row)

    df = pd.DataFrame(rows)
    # POWER uses -999 as a missing-value sentinel
    df = df.replace(-999, pd.NA)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--district", type=str, required=True)
    ap.add_argument("--start", type=str, required=True, help="YYYYMMDD")
    ap.add_argument("--end", type=str, required=True, help="YYYYMMDD")
    ap.add_argument("--out", type=str, default="../dataset/rainfall_dataset_real.csv")
    args = ap.parse_args()

    print(f"Fetching NASA POWER data for {args.district} "
          f"({args.lat}, {args.lon}) from {args.start} to {args.end} ...")
    df = fetch(args.lat, args.lon, args.start, args.end)

    df["PSC"] = df["PS"].apply(pressure_category)
    df["WSC"] = df["WS50M"].apply(wind_category)
    df["DISTRICT"] = args.district
    df["LATITUDE"] = args.lat
    df["LONGITUDE"] = args.lon

    # match the training column order (PRECTOTCORR kept here; it's the
    # source used to build Rainfall_Class later in the pipeline)
    cols = ["YEAR", "MO", "DY", "RH2M", "T2MDEW", "QV2M", "PS", "WS50M",
            "PRECTOTCORR", "T2MWET", "WD50M", "T2M_MAX", "T2M_MIN",
            "ALLSKY_SFC_UV_INDEX", "TS", "PSC", "WSC", "DISTRICT",
            "LATITUDE", "LONGITUDE"]
    df = df[cols]

    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing values from POWER.")

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
