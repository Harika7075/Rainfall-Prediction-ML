"""
Generates a realistic historical weather dataset for the
Rainfall Prediction project, with the exact columns described
in the internship report:
YEAR, MO, DY, RH2M, T2MDEW, QV2M, PS, WS50M, PRECTOTCORR,
T2MWET, WD50M, T2M_MAX, T2M_MIN, ALLSKY_SFC_UV_INDEX, TS,
PSC, WSC, DISTRICT, LATITUDE, LONGITUDE
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

DISTRICTS = {
    "Bengaluru Urban": (12.9716, 77.5946),
    "Chikkaballapur": (13.4355, 77.7315),
    "Kolar":          (13.1372, 78.1298),
    "Tumakuru":       (13.3392, 77.1010),
    "Ramanagara":     (12.7217, 77.2812),
}

N_YEARS = 6
START_YEAR = 2020
rows = []

for district, (lat, lon) in DISTRICTS.items():
    for year in range(START_YEAR, START_YEAR + N_YEARS):
        for month in range(1, 13):
            days_in_month = 28 if month == 2 else (30 if month in [4,6,9,11] else 31)
            if month in [6,7,8,9]:
                season = "monsoon"
            elif month in [3,4,5]:
                season = "summer"
            else:
                season = "winter"

            for day in range(1, days_in_month + 1):
                if season == "monsoon":
                    rh2m = rng.normal(82, 6)
                    rain_prob = 0.62
                    rain_amt_scale = 14
                    t2m_max = rng.normal(29, 2)
                    ps = rng.normal(96.5, 0.6)
                elif season == "summer":
                    rh2m = rng.normal(45, 8)
                    rain_prob = 0.12
                    rain_amt_scale = 4
                    t2m_max = rng.normal(35, 2.5)
                    ps = rng.normal(95.8, 0.6)
                else:
                    rh2m = rng.normal(55, 7)
                    rain_prob = 0.18
                    rain_amt_scale = 5
                    t2m_max = rng.normal(27, 2)
                    ps = rng.normal(97.0, 0.6)

                rh2m = float(np.clip(rh2m, 15, 100))
                rains = rng.random() < rain_prob
                prectotcorr = float(max(0, rng.exponential(rain_amt_scale))) if rains else 0.0

                t2mdew = t2m_max - (100 - rh2m) / 5 + rng.normal(0, 1)
                qv2m = rh2m / 100 * rng.normal(16, 2)
                ws50m = float(np.clip(rng.normal(6 if season=="monsoon" else 4, 1.5), 0.5, 20))
                t2mwet = t2m_max - (100 - rh2m) / 8 + rng.normal(0, 0.8)
                wd50m = float(rng.uniform(0, 360))
                t2m_min = t2m_max - rng.normal(8, 1.5)
                uv_index = float(np.clip(rng.normal(7 if season!="monsoon" else 4, 1.5), 0, 12))
                ts = t2m_max + rng.normal(1, 0.5)

                psc = "Low" if ps < 96.2 else ("Medium" if ps < 97.0 else "High")
                wsc = "Calm" if ws50m < 3 else ("Moderate" if ws50m < 7 else "Strong")

                rows.append([
                    year, month, day, round(rh2m,2), round(t2mdew,2), round(qv2m,2),
                    round(ps,2), round(ws50m,2), round(prectotcorr,2), round(t2mwet,2),
                    round(wd50m,2), round(t2m_max,2), round(t2m_min,2), round(uv_index,2),
                    round(ts,2), psc, wsc, district, lat, lon
                ])

cols = ["YEAR","MO","DY","RH2M","T2MDEW","QV2M","PS","WS50M","PRECTOTCORR",
        "T2MWET","WD50M","T2M_MAX","T2M_MIN","ALLSKY_SFC_UV_INDEX","TS",
        "PSC","WSC","DISTRICT","LATITUDE","LONGITUDE"]

df = pd.DataFrame(rows, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/rainproj/dataset/rainfall_dataset.csv", index=False)
print("Saved:", df.shape)
print(df.head())
