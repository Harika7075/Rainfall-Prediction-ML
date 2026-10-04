"""
Shared feature-engineering helpers.
These MUST match exactly what was used in notebooks/rainfall_prediction.py
and dataset/make_dataset.py, or the saved model/scaler will misbehave.
"""

def pressure_category(ps_kpa: float) -> str:
    """Same thresholds used when the training dataset was generated."""
    if ps_kpa < 96.2:
        return "Low"
    elif ps_kpa < 97.0:
        return "Medium"
    else:
        return "High"


def wind_category(ws50m: float) -> str:
    """Same thresholds used when the training dataset was generated."""
    if ws50m < 3:
        return "Calm"
    elif ws50m < 7:
        return "Moderate"
    else:
        return "Strong"


# Column order the model was trained on (X columns, i.e. everything except
# PRECTOTCORR and Rainfall_Class). Must match notebooks/rainfall_prediction.py
#
# NOTE: DISTRICT (as a fixed category) was removed from this list. A label
# encoder for district names only ever knows the handful of places it saw
# during training, so it breaks on anything new. LATITUDE/LONGITUDE carry
# the location signal instead, which generalizes to any place without
# retraining.
FEATURE_ORDER = [
    "YEAR", "MO", "DY", "RH2M", "T2MDEW", "QV2M", "PS", "WS50M",
    "T2MWET", "WD50M", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_UV_INDEX", "TS",
    "PSC_ENC", "WSC_ENC", "LATITUDE", "LONGITUDE",
]
