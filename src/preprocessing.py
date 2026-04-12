"""
preprocessing.py
-----------------
Full-featured, defensive preprocessing pipeline matching the Kaggle training
notebook exactly. Handles partial inputs gracefully so the Streamlit app can
pass only the features the user filled in – everything else defaults to a
sensible zero / median / 'None' value before the pipeline runs.
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Load feature columns saved during training
# ─────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

try:
    feature_columns = joblib.load(FEATURE_PATH)
except FileNotFoundError:
    feature_columns = []
    print(f"[WARNING] feature_columns.pkl not found at {FEATURE_PATH}")


# ─────────────────────────────────────────────
# Ordinal encoding maps
# ─────────────────────────────────────────────
QUAL_MAP            = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
BSMT_EXPOSURE_MAP   = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
BSMT_FIN_MAP        = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
GARAGE_FINISH_MAP   = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
FUNCTIONAL_MAP      = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
FENCE_MAP           = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
LOTSHAPE_MAP        = {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0}
LANDSLOPE_MAP       = {"Gtl": 2, "Mod": 1, "Sev": 0}
ELECTRICAL_MAP      = {"SBrkr": 4, "FuseA": 3, "FuseF": 2, "FuseP": 1, "Mix": 0}
PAVEDDRIVE_MAP      = {"Y": 1, "P": 0.5, "N": 0}


# ─────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────
ORDINAL_NA_COLS = [
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "PoolQC", "Fence",
]

QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

ONEHOT_COLS = [
    "MSSubClass", "MSZoning", "Neighborhood", "Street", "Alley", "LandContour",
    "LotConfig", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "Foundation", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Heating", "GarageType", "SaleType", "SaleCondition", "MiscFeature",
]

NUM_ZERO_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

LOG_COLS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

# Sensible defaults for every raw input column
# (used to back-fill anything the user left blank)
COLUMN_DEFAULTS = {
    # ── Identifiers / always dropped ──────────────────────────────────────────
    "Id": 0,
    "Utilities": "AllPub",

    # ── Lot / Location ─────────────────────────────────────────────────────────
    "MSSubClass": "20",
    "MSZoning": "RL",
    "LotFrontage": None,        # filled by median later
    "LotArea": 8000,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",

    # ── Building ───────────────────────────────────────────────────────────────
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 5,
    "OverallCond": 5,
    "YearBuilt": 1990,
    "YearRemodAdd": 1990,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",

    # ── Basement ───────────────────────────────────────────────────────────────
    "BsmtQual": "None",
    "BsmtCond": "None",
    "BsmtExposure": "None",
    "BsmtFinType1": "None",
    "BsmtFinSF1": 0,
    "BsmtFinType2": "None",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": 0,
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,

    # ── HVAC ───────────────────────────────────────────────────────────────────
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",

    # ── Main floor ────────────────────────────────────────────────────────────
    "1stFlrSF": 1000,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 1500,
    "FullBath": 2,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": "None",

    # ── Garage ─────────────────────────────────────────────────────────────────
    "GarageType": "None",
    "GarageYrBlt": None,        # filled by GarageAge=0 logic
    "GarageFinish": "None",
    "GarageCars": 2,
    "GarageArea": 400,
    "GarageQual": "None",
    "GarageCond": "None",
    "PavedDrive": "Y",

    # ── Outdoor ────────────────────────────────────────────────────────────────
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "None",
    "MiscVal": 0,

    # ── Sale ───────────────────────────────────────────────────────────────────
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Insert sensible defaults for every column that is absent from df."""
    for col, val in COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw input DataFrame into the feature matrix expected by the
    trained stacking model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw house-feature rows. Missing columns are filled with defaults
        so partial inputs (e.g. from the Streamlit UI) work correctly.

    Returns
    -------
    pd.DataFrame  –  Aligned, fully-encoded feature matrix.
    """
    df = df.copy()

    # ── 0. Fill absent columns with sensible defaults ─────────────────────────
    df = _fill_defaults(df)

    # ── 1. Drop unused columns ────────────────────────────────────────────────
    df.drop(columns=["Id", "Utilities"], errors="ignore", inplace=True)

    # ── 2. MSSubClass → string ────────────────────────────────────────────────
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    # ── 3. Fill ordinal NA cols ───────────────────────────────────────────────
    for col in ORDINAL_NA_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # ── 4. Ordinal encodings ──────────────────────────────────────────────────
    for col in QUAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP).fillna(0)

    _safe_map(df, "BsmtExposure",  BSMT_EXPOSURE_MAP,  0)
    _safe_map(df, "BsmtFinType1",  BSMT_FIN_MAP,       0)
    _safe_map(df, "BsmtFinType2",  BSMT_FIN_MAP,       0)
    _safe_map(df, "GarageFinish",  GARAGE_FINISH_MAP,  0)
    _safe_map(df, "Functional",    FUNCTIONAL_MAP,     7)   # default = Typ
    _safe_map(df, "Fence",         FENCE_MAP,          0)
    _safe_map(df, "LotShape",      LOTSHAPE_MAP,       3)   # default = Reg
    _safe_map(df, "LandSlope",     LANDSLOPE_MAP,      2)   # default = Gtl
    _safe_map(df, "Electrical",    ELECTRICAL_MAP,     4)   # default = SBrkr
    _safe_map(df, "PavedDrive",    PAVEDDRIVE_MAP,     1)   # default = Y

    # ── 5. LotFrontage median imputation ──────────────────────────────────────
    if "LotFrontage" in df.columns:
        median_lf = df["LotFrontage"].median()
        if pd.isna(median_lf):
            median_lf = 70.0
        df["LotFrontage"] = df["LotFrontage"].fillna(median_lf)

    # ── 6. Categorical NA fills ───────────────────────────────────────────────
    for col in ["Alley", "MasVnrType", "GarageType", "MiscFeature"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    # ── 7. Binary: CentralAir ────────────────────────────────────────────────
    if "CentralAir" in df.columns:
        df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0}).fillna(1)

    # ── 8. Feature engineering: age features ──────────────────────────────────
    if "YrSold" in df.columns:
        yr_sold = df["YrSold"]
        df["HouseAge"]   = (yr_sold - df.get("YearBuilt",    pd.Series([1990] * len(df), index=df.index))).clip(lower=0)
        df["RemodAge"]   = (yr_sold - df.get("YearRemodAdd", pd.Series([1990] * len(df), index=df.index))).clip(lower=0)
        df["GarageAge"]  = (yr_sold - df.get("GarageYrBlt",  pd.Series([np.nan] * len(df), index=df.index)))

    df.drop(columns=["YearBuilt", "YearRemodAdd", "GarageYrBlt"], errors="ignore", inplace=True)

    if "GarageAge" in df.columns:
        df["GarageAge"] = df["GarageAge"].fillna(0).clip(lower=0)

    # ── 9. Zero imputation for area / count features ──────────────────────────
    for col in NUM_ZERO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # ── 10. Log transform skewed numeric features ─────────────────────────────
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))

    # ── 11. Cyclical month encoding ───────────────────────────────────────────
    if "MoSold" in df.columns:
        mo = pd.to_numeric(df["MoSold"], errors="coerce").fillna(6)
        df["MoSold_sin"] = np.sin(2 * np.pi * mo / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * mo / 12)
        df.drop(columns=["MoSold"], inplace=True)

    # ── 12. Remaining NA fixes ────────────────────────────────────────────────
    for col in ["BsmtFullBath", "BsmtHalfBath", "GarageCars"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "KitchenQual" in df.columns:
        kq_median = df["KitchenQual"].median()
        df["KitchenQual"] = df["KitchenQual"].fillna(kq_median if not pd.isna(kq_median) else 3)

    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna(FUNCTIONAL_MAP["Typ"])

    if "Electrical" in df.columns:
        elec_median = df["Electrical"].median()
        df["Electrical"] = df["Electrical"].fillna(elec_median if not pd.isna(elec_median) else 4)

    # ── 13. One-hot encoding ──────────────────────────────────────────────────
    ohe_present = [c for c in ONEHOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=False)

    # ── 14. Align with training feature columns ───────────────────────────────
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    else:
        # Fallback: cast all remaining object cols to 0
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = 0

    # ── 15. Final numeric cast & NaN cleanup ──────────────────────────────────
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _safe_map(df: pd.DataFrame, col: str, mapping: dict, default) -> None:
    """Apply an ordinal mapping in-place; unknown values → default."""
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(default)