import pandas as pd
import numpy as np
import sys
import os
from data_viz import render_data_viz
from prometheus_client import start_http_server,Counter, Summary,REGISTRY,Histogram

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

def get_metric(name, cls, *args):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return cls(name, *args)

def get_histogram(name, documentation, buckets, labelnames=None):
    if name in REGISTRY._names_to_collectors:
        old_metric = REGISTRY._names_to_collectors[name]

        # REMOVE old broken metric
        try:
            REGISTRY.unregister(old_metric)
        except:
            pass

    return Histogram(
        name,
        documentation,
        buckets=buckets,
        labelnames=labelnames or []
    )

REQUEST_COUNT = get_metric(
    "app_requests_total",
    Counter,
    "Total number of prediction requests"
)

PREDICTION_LATENCY = get_metric(
    "prediction_latency_seconds",
    Summary,
    "Time spent on prediction"
)

ERROR_COUNT = get_metric(
    "app_errors_total",
    Counter,
    "Total number of prediction errors"
)

REQUESTS_BY_STATUS = get_metric(
    "requests_by_status",
    Counter,
    "Requests by status",
    ["status"]
)

PREDICTION_VALUES = get_histogram(
    "prediction_values",
    "Distribution of predicted prices",
    buckets=[50000,100000,150000,200000,300000,500000,1000000]
)

INPUT_FEATURE_DISTRIBUTION = get_histogram(
    "input_feature_distribution",
    "Distribution of important input features",
    buckets=[0,500,1000,2000,5000,10000,20000,50000],
    labelnames=["feature"]
)

import threading

def start_metrics_server():
    start_http_server(8000)

if "metrics_thread" not in st.session_state:
    thread = threading.Thread(target=start_metrics_server, daemon=True)
    thread.start()
    st.session_state["metrics_thread"] = True

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0e1117;
    --surface:  #161b27;
    --card:     #1c2333;
    --border:   #2a3450;
    --gold:     #c9a84c;
    --gold-lt:  #e8c97a;
    --text:     #e8eaf2;
    --muted:    #8892a4;
    --green:    #3ecf8e;
    --blue:     #4a9eff;
    --red:      #f06060;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Sidebar text fix */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: var(--text);
}

/* DO NOT hide header */
#MainMenu, footer { visibility: hidden; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0;
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    accent-color: var(--gold);
}

[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stTable"] table {
    background: var(--card);
    border-radius: 8px;
}

thead tr th {
    background: var(--border) !important;
}

/* Radio fix */
[data-testid="stRadio"] label {
    font-size: 0.95rem;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR IMPORT  (lazy so page renders even if models missing)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_predictor():
    from src.predictor import predict_with_breakdown
    return predict_with_breakdown


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card(content_fn, title=""):
    with st.container():
        if title:
            st.markdown(f"<p style='color:var(--muted);font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.25rem'>{title}</p>", unsafe_allow_html=True)
        content_fn()


def metric_card(label, value, delta=None, color="var(--gold)"):
    delta_html = ""
    if delta:
        delta_html = f"<span style='font-size:.85rem;color:var(--muted)'>{delta}</span>"
    st.markdown(f"""
    <div style='background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.4rem;'>
        <p style='margin:0;font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)'>{label}</p>
        <p style='margin:.25rem 0 0;font-size:1.7rem;font-weight:700;font-family:"Playfair Display",serif;color:{color}'>{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(icon, title, subtitle=""):
    sub_html = f"<p style='color:var(--muted);font-size:.95rem;margin-top:.3rem'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
        <h2 style='margin-bottom:0;color:var(--gold)'>{icon} {title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.4rem'>🏠</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.3rem;color:var(--gold);font-weight:700'>EstimaAI</div>
        <div style='font-size:.78rem;color:var(--muted);letter-spacing:.1em'>HOUSE PRICE PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Predict Price","Data Visulization", "Model Insights", "Feature Guide", "About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.78rem;color:var(--muted);line-height:1.8'>
        <b style='color:var(--text)'>Model Stack</b><br>
        ElasticNet · XGBoost<br>
        CatBoost · Ridge Meta<br><br>
        <b style='color:var(--text)'>Dataset</b><br>
        Ames Housing (Kaggle)<br>
        1,460 training samples<br>
        79 original features
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – PREDICT PRICE
# ═════════════════════════════════════════════════════════════════════════════
if page == "Predict Price":

    st.markdown("""
    <div style='margin-bottom:2rem'>
        <h1 style='margin-bottom:.25rem'>Property Valuation</h1>
        <p style='color:var(--muted)'>Fill in as many details as you know. Fields you skip use sensible defaults.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── TIER 1 – ESSENTIAL ───────────────────────────────────────────────────
    section_header("🏗️", "Essential Features", "These 8 fields drive ~70 % of prediction accuracy.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        OverallQual = st.select_slider(
            "Overall Quality",
            options=list(range(1, 11)),
            value=5,
            help="Rates overall material and finish (1=Very Poor → 10=Excellent)",
        )
        st.caption("Material & finish quality")

    with c2:
        GrLivArea = st.number_input(
            "Above-Grade Living Area (sq ft)",
            min_value=300, max_value=10000, value=1500, step=50,
            help="Above-grade (ground) living area in square feet",
        )
        st.caption("Above-ground living space")

    with c3:
        TotalBsmtSF = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0, max_value=6000, value=800, step=50,
            help="Total square feet of basement; 0 = no basement",
        )
        st.caption("0 = no basement")

    with c4:
        GarageCars = st.number_input(
            "Garage Capacity (cars)",
            min_value=0, max_value=5, value=2,
            help="Number of cars that fit in garage; 0 = no garage",
        )
        st.caption("0 = no garage")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        YearBuilt = st.number_input(
            "Year Built",
            min_value=1800, max_value=2024, value=2000,
            help="Original construction year",
        )
    with c6:
        YearRemodAdd = st.number_input(
            "Year Remodelled",
            min_value=1800, max_value=2024, value=2000,
            help="Year remodeled; same as Year Built if never remodelled",
        )
    with c7:
        Neighborhood = st.selectbox("Neighborhood", [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
            "NridgHt", "Gilbert", "Sawyer", "BrkSide", "Crawfor",
            "Mitchel", "NoRidge", "Timber",  "IDOTRR",  "ClearCr",
            "StoneBr", "SWISU",   "Blmngtn", "MeadowV", "Veenker",
            "BrDale",  "NPkVill", "Blueste",
        ], help="Physical location within Ames city limits")
    with c8:
        YrSold = st.number_input("Year Sold", min_value=2006, max_value=2024, value=2010,
                                 help="Year sold (2006–2010 typical, extended for demo)")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── TIER 2 – IMPORTANT ───────────────────────────────────────────────────
    section_header("🔑", "Important Features", "These significantly refine the estimate.")

    with st.expander("Building & Rooms", expanded=True):
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        with b1:
            HouseStyle = st.selectbox("House Style", ["1Story","2Story","1.5Fin","1.5Unf","2.5Fin","2.5Unf","SFoyer","SLvl"],
                                      help="Style of dwelling (e.g., 1Story, 2Story)")
        with b2:
            BldgType = st.selectbox("Building Type", ["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"],
                                    help="Type of dwelling (single-family, duplex, townhouse, etc.)")
        with b3:
            OverallCond = st.select_slider("Overall Condition", options=list(range(1,11)), value=5,
                                           help="Rates overall condition (1=Very Poor → 10=Excellent)")
        with b4:
            TotRmsAbvGrd = st.number_input("Total Rooms (above grade)", 2, 20, 6,
                                            help="Total rooms above grade (excluding bathrooms)")
        with b5:
            BedroomAbvGr = st.number_input("Bedrooms", 0, 10, 3,
                                            help="Bedrooms above grade (does not include basement)")
        with b6:
            FullBath = st.number_input("Full Bathrooms", 0, 5, 2,
                                        help="Full bathrooms above grade")

        b7, b8, b9, b10, b11, b12 = st.columns(6)
        with b7:
            HalfBath = st.number_input("Half Bathrooms", 0, 3, 0,
                                        help="Half bathrooms above grade")
        with b8:
            KitchenAbvGr = st.number_input("Kitchens", 0, 3, 1,
                                            help="Number of kitchens above grade")
        with b9:
            Fireplaces = st.number_input("Fireplaces", 0, 5, 0,
                                          help="Number of fireplaces")
        with b10:
            st_1stFlrSF = st.number_input("1st Floor SF", 300, 5000, 1000, step=50,
                                           help="First floor square feet")
        with b11:
            st_2ndFlrSF = st.number_input("2nd Floor SF", 0, 3000, 0, step=50,
                                           help="Second floor square feet (0 if none)")
        with b12:
            LotArea = st.number_input("Lot Area (sq ft)", 1000, 200000, 8000, step=500,
                                       help="Lot size in square feet")

    with st.expander("Quality Ratings", expanded=True):
        q1, q2, q3, q4, q5 = st.columns(5)
        QUAL_OPTIONS = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
        with q1:
            KitchenQual  = st.selectbox("Kitchen Quality",  QUAL_OPTIONS, index=3,
                                         help="Kitchen quality (Ex=Excellent, Gd=Good, TA=Typical, Fa=Fair, Po=Poor, None)")
        with q2:
            ExterQual    = st.selectbox("Exterior Quality", QUAL_OPTIONS, index=3,
                                         help="Exterior material quality")
        with q3:
            ExterCond    = st.selectbox("Exterior Condition", QUAL_OPTIONS, index=3,
                                         help="Exterior material condition")
        with q4:
            HeatingQC    = st.selectbox("Heating Quality",  QUAL_OPTIONS, index=3,
                                         help="Heating quality and condition")
        with q5:
            CentralAir   = st.selectbox("Central Air", ["Y","N"],
                                         help="Central air conditioning (Y=Yes, N=No)")

    with st.expander("Garage", expanded=False):
        g1, g2, g3, g4 = st.columns(4)
        GARAGE_QUAL_OPTIONS = ["None","Po","Fa","TA","Gd","Ex"]
        GARAGE_FINISH_OPTIONS = ["None","Unf","RFn","Fin"]
        with g1:
            GarageType   = st.selectbox("Garage Type", ["None","Attchd","Detchd","BuiltIn","CarPort","BasmtGrg","2Types"],
                                         help="Garage location (Attached, Detached, Built-In, etc.)")
        with g2:
            GarageFinish = st.selectbox("Garage Finish", GARAGE_FINISH_OPTIONS, index=0,
                                         help="Interior finish of garage (Fin=Finished, RFn=Rough finished, Unf=Unfinished, None)")
        with g3:
            GarageYrBlt  = st.number_input("Garage Year Built", 1900, 2024, 2000,
                                            help="Year garage was built")
        with g4:
            GarageArea   = st.number_input("Garage Area (sq ft)", 0, 2000, 400, step=50,
                                            help="Garage size in square feet")

    with st.expander("Basement", expanded=False):
        bq1, bq2, bq3, bq4 = st.columns(4)
        BSMT_QUAL_OPTIONS = ["None","Po","Fa","TA","Gd","Ex"]
        BSMT_EXP_OPTIONS  = ["None","No","Mn","Av","Gd"]
        BSMT_FIN_OPTIONS  = ["None","Unf","LwQ","Rec","BLQ","ALQ","GLQ"]
        with bq1:
            BsmtQual      = st.selectbox("Basement Quality",   BSMT_QUAL_OPTIONS, index=0,
                                          help="Basement height/quality (Ex=Excellent, Gd=Good, TA=Typical, Fa=Fair, Po=Poor, None)")
        with bq2:
            BsmtCond      = st.selectbox("Basement Condition", BSMT_QUAL_OPTIONS, index=0,
                                          help="Basement general condition")
        with bq3:
            BsmtExposure  = st.selectbox("Basement Exposure",  BSMT_EXP_OPTIONS,  index=0,
                                          help="Walkout or garden level walls (Gd=Good, Av=Average, Mn=Minimum, No=No exposure, None)")
        with bq4:
            BsmtFinType1  = st.selectbox("Basement Finish Type", BSMT_FIN_OPTIONS, index=0,
                                          help="Rating of finished basement area (GLQ=Good Living Quarters, ALQ=Average, etc.)")
        bsf1, bsf2, bsf3, bsf4 = st.columns(4)
        with bsf1:
            BsmtFinSF1    = st.number_input("Finished Basement SF", 0, 3000, 0, step=50,
                                             help="Finished basement area (sq ft)")
        with bsf2:
            BsmtUnfSF     = st.number_input("Unfinished Basement SF", 0, 3000, 0, step=50,
                                             help="Unfinished basement area (sq ft)")
        with bsf3:
            BsmtFullBath  = st.number_input("Basement Full Baths", 0, 3, 0,
                                              help="Full bathrooms in basement")
        with bsf4:
            BsmtHalfBath  = st.number_input("Basement Half Baths", 0, 2, 0,
                                              help="Half bathrooms in basement")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── TIER 3 – OPTIONAL ────────────────────────────────────────────────────
    section_header("⚙️", "Optional / Advanced Features", "Expand for granular control.")

    with st.expander("Lot & Location"):
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            MSSubClass  = st.selectbox("Building Class (MSSubClass)", ["20","30","40","45","50","60","70","75","80","85","90","120","150","160","180","190"],
                                        help="Identifies type of dwelling (e.g., 20=1-Story 1946+, 60=2-Story 1946+)")
        with l2:
            MSZoning    = st.selectbox("Zoning", ["RL","RM","FV","RH","C (all)"],
                                        help="General zoning classification (RL=Residential Low, RM=Medium, etc.)")
        with l3:
            LotFrontage = st.number_input("Lot Frontage (ft)", 0, 300, 70,
                                           help="Linear feet of street connected to property")
        with l4:
            LotShape    = st.selectbox("Lot Shape",  ["Reg","IR1","IR2","IR3"],
                                        help="General shape of property (Reg=Regular, IR1=Slightly irregular, etc.)")
        l5, l6, l7, l8 = st.columns(4)
        with l5:
            LandContour = st.selectbox("Land Contour", ["Lvl","Bnk","HLS","Low"],
                                        help="Flatness of property (Lvl=Level, Bnk=Banked, HLS=Hillside, Low=Depression)")
        with l6:
            LotConfig   = st.selectbox("Lot Config",   ["Inside","Corner","CulDSac","FR2","FR3"],
                                        help="Lot configuration (Inside, Corner, Cul-de-sac, etc.)")
        with l7:
            LandSlope   = st.selectbox("Land Slope",   ["Gtl","Mod","Sev"],
                                        help="Slope of property (Gtl=Gentle, Mod=Moderate, Sev=Severe)")
        with l8:
            Condition1  = st.selectbox("Proximity Condition", ["Norm","Feedr","PosN","PosA","Artery","RRAn","RRAe","RRNn","RRNe"],
                                        help="Proximity to various conditions (Norm=Normal, Artery=Adjacent to arterial street, etc.)")

    with st.expander("Construction & Exterior"):
        cx1, cx2, cx3, cx4 = st.columns(4)
        with cx1:
            Foundation  = st.selectbox("Foundation",  ["PConc","CBlock","BrkTil","Wood","Slab","Stone"],
                                        help="Type of foundation (PConc=Poured Concrete, CBlock=Cinder Block, etc.)")
        with cx2:
            RoofStyle   = st.selectbox("Roof Style",  ["Gable","Hip","Gambrel","Mansard","Flat","Shed"],
                                        help="Type of roof (Gable, Hip, etc.)")
        with cx3:
            Exterior1st = st.selectbox("Exterior 1st", ["VinylSd","HdBoard","MetalSd","Wd Sdng","Plywood","CemntBd","BrkFace","WdShing","Stucco","AsbShng"],
                                        help="Exterior covering on house")
        with cx4:
            MasVnrType  = st.selectbox("Masonry Veneer", ["None","BrkFace","Stone","BrkCmn"],
                                        help="Masonry veneer type (None, Brick Face, Stone, etc.)")
        cx5, cx6 = st.columns([1,3])
        with cx5:
            MasVnrArea  = st.number_input("Masonry Veneer Area (sq ft)", 0, 2000, 0,
                                           help="Masonry veneer area in square feet")

    with st.expander("Outdoor & Extras"):
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            WoodDeckSF    = st.number_input("Wood Deck SF",     0, 1500, 0, step=25,
                                             help="Wood deck area in square feet")
        with o2:
            OpenPorchSF   = st.number_input("Open Porch SF",    0, 600,  0, step=10,
                                             help="Open porch area in square feet")
        with o3:
            EnclosedPorch = st.number_input("Enclosed Porch SF",0, 600,  0, step=10,
                                             help="Enclosed porch area in square feet")
        with o4:
            ScreenPorch   = st.number_input("Screen Porch SF",  0, 600,  0, step=10,
                                             help="Screen porch area in square feet")
        o5, o6, o7, o8 = st.columns(4)
        with o5:
            PoolArea      = st.number_input("Pool Area SF",     0, 800,  0, step=10,
                                             help="Pool area in square feet")
        with o6:
            Fence         = st.selectbox("Fence", ["None","GdPrv","MnPrv","GdWo","MnWw"],
                                          help="Fence quality (GdPrv=Good Privacy, MnPrv=Minimum Privacy, etc.)")
        with o7:
            PavedDrive    = st.selectbox("Paved Driveway", ["Y","P","N"],
                                          help="Paved driveway (Y=Yes, P=Partial, N=No)")
        with o8:
            Functional    = st.selectbox("Functionality", ["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"],
                                          help="Home functionality rating (Typ=Typical, Min1=Minor deductions, etc.)")

    with st.expander("Sale Information"):
        s1, s2, s3 = st.columns(3)
        with s1:
            MoSold       = st.number_input("Month Sold", 1, 12, 6,
                                            help="Month sold (1=January, 12=December)")
        with s2:
            SaleType     = st.selectbox("Sale Type", ["WD","CWD","VWD","COD","Con","ConLw","ConLI","ConLD","Oth"],
                                         help="Type of sale (WD=Warranty Deed, CWD=Cash, etc.)")
        with s3:
            SaleCondition = st.selectbox("Sale Condition", ["Normal","Abnorml","AdjLand","Alloca","Family","Partial"],
                                          help="Condition of sale (Normal, Abnormal, etc.)")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── VALIDATION ───────────────────────────────────────────────────────────
    def validate() -> list[str]:
        errs = []
        if LotArea <= 0:
            errs.append("Lot Area must be > 0.")
        if GrLivArea < 300:
            errs.append("Above-grade living area seems too small (< 300 sq ft).")
        if YearBuilt < 1800:
            errs.append("Year Built looks unrealistic (< 1800).")
        if YearRemodAdd < YearBuilt:
            errs.append("Year Remodelled cannot be before Year Built.")
        if TotalBsmtSF > GrLivArea * 1.5:
            errs.append("Basement area is unusually large compared to living area – please double-check.")
        if OverallQual < 1 or OverallQual > 10:
            errs.append("Overall Quality must be between 1 and 10.")
        return errs

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("Get Estimate", use_container_width=True, type="primary")

    if predict_clicked:
        errors = validate()
        if errors:
            for e in errors:
                st.error(f"⚠️ {e}")
        else:
            with st.spinner("Running ensemble models…"):
                try:
                    predict_fn = load_predictor()
                    input_df = pd.DataFrame([{
                        # Tier 1
                        "OverallQual":   OverallQual,
                        "GrLivArea":     GrLivArea,
                        "TotalBsmtSF":   TotalBsmtSF,
                        "GarageCars":    GarageCars,
                        "YearBuilt":     YearBuilt,
                        "YearRemodAdd":  YearRemodAdd,
                        "Neighborhood":  Neighborhood,
                        "YrSold":        YrSold,
                        # Tier 2 – building
                        "HouseStyle":    HouseStyle,
                        "BldgType":      BldgType,
                        "OverallCond":   OverallCond,
                        "TotRmsAbvGrd":  TotRmsAbvGrd,
                        "BedroomAbvGr":  BedroomAbvGr,
                        "FullBath":      FullBath,
                        "HalfBath":      HalfBath,
                        "KitchenAbvGr":  KitchenAbvGr,
                        "Fireplaces":    Fireplaces,
                        "1stFlrSF":      st_1stFlrSF,
                        "2ndFlrSF":      st_2ndFlrSF,
                        "LotArea":       LotArea,
                        # Tier 2 – quality
                        "KitchenQual":   KitchenQual,
                        "ExterQual":     ExterQual,
                        "ExterCond":     ExterCond,
                        "HeatingQC":     HeatingQC,
                        "CentralAir":    CentralAir,
                        # Tier 2 – garage
                        "GarageType":    GarageType,
                        "GarageFinish":  GarageFinish,
                        "GarageYrBlt":   GarageYrBlt,
                        "GarageArea":    GarageArea,
                        # Tier 2 – basement
                        "BsmtQual":      BsmtQual,
                        "BsmtCond":      BsmtCond,
                        "BsmtExposure":  BsmtExposure,
                        "BsmtFinType1":  BsmtFinType1,
                        "BsmtFinSF1":    BsmtFinSF1,
                        "BsmtUnfSF":     BsmtUnfSF,
                        "BsmtFullBath":  BsmtFullBath,
                        "BsmtHalfBath":  BsmtHalfBath,
                        # Tier 3 – lot
                        "MSSubClass":    MSSubClass,
                        "MSZoning":      MSZoning,
                        "LotFrontage":   LotFrontage,
                        "LotShape":      LotShape,
                        "LandContour":   LandContour,
                        "LotConfig":     LotConfig,
                        "LandSlope":     LandSlope,
                        "Condition1":    Condition1,
                        # Tier 3 – construction
                        "Foundation":    Foundation,
                        "RoofStyle":     RoofStyle,
                        "Exterior1st":   Exterior1st,
                        "MasVnrType":    MasVnrType,
                        "MasVnrArea":    MasVnrArea,
                        # Tier 3 – outdoor
                        "WoodDeckSF":    WoodDeckSF,
                        "OpenPorchSF":   OpenPorchSF,
                        "EnclosedPorch": EnclosedPorch,
                        "ScreenPorch":   ScreenPorch,
                        "PoolArea":      PoolArea,
                        "Fence":         Fence,
                        "PavedDrive":    PavedDrive,
                        "Functional":    Functional,
                        # Tier 3 – sale
                        "MoSold":        MoSold,
                        "SaleType":      SaleType,
                        "SaleCondition": SaleCondition,
                    }])
                    import time
                    start = time.time()
                    st.write("METRIC TRIGGERED")

                    # ─── METRICS START ───
                    REQUEST_COUNT.inc()
                    
                    # track input distribution (MULTI FEATURE)
                    INPUT_FEATURE_DISTRIBUTION.labels("GrLivArea").observe(GrLivArea)
                    INPUT_FEATURE_DISTRIBUTION.labels("TotalBsmtSF").observe(TotalBsmtSF)
                    INPUT_FEATURE_DISTRIBUTION.labels("GarageCars").observe(GarageCars)
                    INPUT_FEATURE_DISTRIBUTION.labels("TotRmsAbvGrd").observe(TotRmsAbvGrd)
                    INPUT_FEATURE_DISTRIBUTION.labels("LotArea").observe(LotArea)
                    INPUT_FEATURE_DISTRIBUTION.labels("GarageArea").observe(GarageArea)
                    INPUT_FEATURE_DISTRIBUTION.labels("LotFrontage").observe(LotFrontage)


                    result = predict_fn(input_df)

                    latency = time.time() - start
                    PREDICTION_LATENCY.observe(latency)
                    st.write("Counter:", REQUEST_COUNT._value.get())
                    price  = result["final"]

                    # ─── OUTPUT METRIC ───
                    PREDICTION_VALUES.observe(price)
                    REQUESTS_BY_STATUS.labels("success").inc()
                    

                    # ── RESULT DISPLAY ────────────────────────────────────────
                    st.markdown("<br/>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#1c2333 60%,#23304a);border:1.5px solid var(--gold);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem'>
                        <p style='color:var(--muted);margin:0;font-size:.85rem;letter-spacing:.12em;text-transform:uppercase'>Ensemble Estimate</p>
                        <h1 style='color:var(--gold-lt);font-family:"Playfair Display",serif;font-size:3rem;margin:.4rem 0'>${price:,.0f}</h1>
                        <p style='color:var(--muted);margin:0;font-size:.9rem'>±  confidence range: <b style='color:var(--text)'>${price*0.88:,.0f} – ${price*1.12:,.0f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Per-model breakdown ───────────────────────────────────
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.markdown(f"""
                    <div style='background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;text-align:center'>
                        <div style='font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em'>ElasticNet</div>
                        <div style='font-size:1.45rem;font-weight:700;color:var(--blue)'>${result["elasticnet"]:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                    mc2.markdown(f"""
                    <div style='background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;text-align:center'>
                        <div style='font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em'>XGBoost</div>
                        <div style='font-size:1.45rem;font-weight:700;color:var(--green)'>${result["xgboost"]:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                    mc3.markdown(f"""
                    <div style='background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem 1.25rem;text-align:center'>
                        <div style='font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em'>CatBoost</div>
                        <div style='font-size:1.45rem;font-weight:700;color:var(--red)'>${result["catboost"]:,.0f}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br/>", unsafe_allow_html=True)

                    # ── Input summary ─────────────────────────────────────────
                    with st.expander("View Input Summary"):
                        summary_data = {
                            "Feature": [
                                "Overall Quality","Living Area","Basement Area","Garage Capacity",
                                "Year Built","Year Remodelled","Neighborhood","Year Sold",
                                "House Style","Bedrooms","Full Baths","Fireplaces",
                                "Kitchen Quality","Exterior Quality","Central Air",
                            ],
                            "Value": [
                                f"{OverallQual}/10", f"{GrLivArea:,} sq ft", f"{TotalBsmtSF:,} sq ft", f"{GarageCars} cars",
                                YearBuilt, YearRemodAdd, Neighborhood, YrSold,
                                HouseStyle, BedroomAbvGr, FullBath, Fireplaces,
                                KitchenQual, ExterQual, CentralAir,
                            ],
                        }
                        st.dataframe(pd.DataFrame(summary_data))
                    with st.expander("Show Model Feature Vector"):
                        from src.preprocessing import preprocess

                        processed_df = preprocess(input_df)

                        st.write("Total features used by model:", processed_df.shape[1])

                        st.dataframe(processed_df)

                except FileNotFoundError as e:
                    ERROR_COUNT.inc()
                    REQUESTS_BY_STATUS.labels("error").inc()
                    st.error(f"**Model files not found.**\n\n{e}")
                except Exception as e:
                    ERROR_COUNT.inc()
                    REQUESTS_BY_STATUS.labels("error").inc()
                    st.error(f"**Prediction failed:** {e}")
                    st.exception(e)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":

    st.markdown("""
<div style="margin-bottom:2rem">
<h1 style="margin-bottom:.25rem">Model Insights</h1>
<p style="color:var(--muted)">
Understand how the ensemble predicts house prices and how each model contributes.
</p>
</div>
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # ARCHITECTURE
    # ─────────────────────────────────────────────────────────
    section_header("", "Ensemble Architecture")

    st.markdown("""
<div style="background:var(--card);
            border:1px solid var(--border);
            border-radius:16px;
            padding:2rem">

<div style="display:flex;
            justify-content:center;
            align-items:center;
            gap:2rem;
            flex-wrap:wrap">

<div style="background:#182236;border:1px solid var(--blue);border-radius:12px;padding:1rem 1.2rem;text-align:center;min-width:180px">
<div style="font-size:.75rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase">Base Models</div>
<div style="margin-top:.8rem">
<div style="background:#243757;border-radius:6px;padding:.3rem .6rem;margin:.2rem">ElasticNet</div>
<div style="background:#1e3d2f;border-radius:6px;padding:.3rem .6rem;margin:.2rem">XGBoost</div>
<div style="background:#3d2020;border-radius:6px;padding:.3rem .6rem;margin:.2rem">CatBoost</div>
</div>
</div>

<div style="font-size:1.8rem;color:var(--gold)">→</div>

<div style="background:#2a2a1e;border:1px solid var(--gold);border-radius:12px;padding:1rem 1.2rem;text-align:center;min-width:160px">
<div style="font-size:.75rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase">Meta Model</div>
<div style="margin-top:.8rem;font-weight:600">Ridge Regressor</div>
</div>

<div style="font-size:1.8rem;color:var(--gold)">→</div>

<div style="background:#1a2a1a;border:1px solid var(--green);border-radius:12px;padding:1rem 1.2rem;text-align:center;min-width:180px">
<div style="font-size:.75rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase">Output</div>
<div style="margin-top:.8rem;color:var(--gold-lt);font-weight:600">log(SalePrice) → expm1 → USD</div>
</div>

</div>

<p style="color:var(--muted);font-size:.9rem;margin-top:1.5rem">
ElasticNet, XGBoost and CatBoost generate out-of-fold predictions via 10-fold cross-validation.
These predictions are stacked and fed into a Ridge regression meta-model. The target is
log-transformed during training and exponentiated back to USD for final predictions.
</p>

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # PERFORMANCE METRICS
    # ─────────────────────────────────────────────────────────
    section_header("", "Cross-Validation Performance")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("ElasticNet R²", "0.7759")

    with m2:
        metric_card("XGBoost R²", "0.8928", color="var(--green)")

    with m3:
        metric_card("CatBoost R²", "0.9020", color="var(--red)")

    with m4:
        metric_card("Stacked Ensemble R²", "0.9104", color="var(--gold)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # MODEL BREAKDOWN
    # ─────────────────────────────────────────────────────────
    section_header("", "Model Breakdown")

    CARD_HEIGHT = "260px"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--blue);border-radius:14px;
padding:1.5rem;height:{CARD_HEIGHT};display:flex;flex-direction:column;justify-content:space-between">

<div>
<h3 style="color:var(--blue);margin-top:0">ElasticNet</h3>
<p style="color:var(--muted);font-size:.9rem">
Linear regression with L1 and L2 penalties. Performs feature selection
while stabilizing correlated variables.
</p>
</div>

<ul style="color:var(--muted);font-size:.85rem;margin:0">
<li>α = 0.001</li>
<li>l1_ratio = 0.1</li>
<li>max_iter = 500</li>
</ul>

</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--green);border-radius:14px;
padding:1.5rem;height:{CARD_HEIGHT};display:flex;flex-direction:column;justify-content:space-between">

<div>
<h3 style="color:var(--green);margin-top:0">XGBoost</h3>
<p style="color:var(--muted);font-size:.9rem">
Gradient boosted decision trees capturing nonlinear interactions
between housing features.
</p>
</div>

<ul style="color:var(--muted);font-size:.85rem;margin:0">
<li>n_estimators = 950</li>
<li>depth = 5</li>
<li>learning_rate = 0.01</li>
</ul>

</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--red);border-radius:14px;
padding:1.5rem;height:{CARD_HEIGHT};display:flex;flex-direction:column;justify-content:space-between">

<div>
<h3 style="color:var(--red);margin-top:0">CatBoost</h3>
<p style="color:var(--muted);font-size:.9rem">
Ordered boosting algorithm optimized for tabular data
with strong performance on categorical features.
</p>
</div>

<ul style="color:var(--muted);font-size:.85rem;margin:0">
<li>iterations = 1500</li>
<li>depth = 6</li>
<li>learning_rate = 0.01</li>
</ul>

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # PREPROCESSING PIPELINE
    # ─────────────────────────────────────────────────────────
    section_header("", "Feature Engineering Pipeline")

    steps = [
        ("1","Default Fill","Missing values replaced using domain defaults."),
        ("2","Ordinal Encoding","Quality features mapped to numeric scale."),
        ("3","Age Features","HouseAge, RemodAge and GarageAge derived."),
        ("4","Log Transform","Right-skewed numeric features transformed using log1p."),
        ("5","Cyclical Month","MoSold encoded with sine/cosine representation."),
        ("6","One-Hot Encoding","Nominal categorical features expanded."),
        ("7","Feature Alignment","Columns aligned to training schema."),
        ("8","RobustScaler","Applied only to ElasticNet input path."),
    ]

    for num, name, desc in steps:

        html = f"""
<div style="background:var(--card);
border:1px solid var(--border);
border-radius:10px;
padding:.8rem 1rem;
margin-bottom:.6rem;
display:flex;
gap:1rem;
align-items:center">

<div style="background:var(--gold);
color:black;
border-radius:50%;
width:26px;
height:26px;
display:flex;
align-items:center;
justify-content:center;
font-weight:700;
font-size:.85rem">
{num}
</div>

<div>
<div style="font-weight:600">{name}</div>
<div style="color:var(--muted);font-size:.9rem">{desc}</div>
</div>

</div>
"""
        st.markdown(html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – FEATURE GUIDE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Feature Guide":

    st.markdown("""
    <div style='margin-bottom:2rem'>
        <h1 style='margin-bottom:.25rem'>Feature Guide</h1>
        <p style='color:var(--muted)'>
        Explanation of important model features and how they affect price predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)


    # ───────────── Quality Scale ─────────────
    section_header("", "Quality & Condition Scale")

    q1,q2,q3,q4,q5,q6 = st.columns(6)

    with q1:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>Ex</div>
        <div style='color:var(--muted);font-size:.85rem'>Excellent</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 5</div>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>Gd</div>
        <div style='color:var(--muted);font-size:.85rem'>Good</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 4</div>
        </div>
        """, unsafe_allow_html=True)

    with q3:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>TA</div>
        <div style='color:var(--muted);font-size:.85rem'>Typical</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 3</div>
        </div>
        """, unsafe_allow_html=True)

    with q4:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>Fa</div>
        <div style='color:var(--muted);font-size:.85rem'>Fair</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 2</div>
        </div>
        """, unsafe_allow_html=True)

    with q5:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>Po</div>
        <div style='color:var(--muted);font-size:.85rem'>Poor</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 1</div>
        </div>
        """, unsafe_allow_html=True)

    with q6:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);
        border-radius:12px;padding:1rem;text-align:center'>
        <div style='font-size:1.3rem;font-weight:700;color:var(--gold)'>None</div>
        <div style='color:var(--muted);font-size:.85rem'>Not Applicable</div>
        <div style='font-size:.8rem;color:var(--muted);margin-top:.4rem'>numeric = 0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <p style='color:var(--muted);font-size:.9rem;margin-top:1rem'>
    Used for: Exterior quality, basement quality, heating quality,
    kitchen quality, garage quality, fireplace quality and pool quality.
    </p>
    """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # ───────────── Feature Tiers ─────────────
    section_header("", "Feature Impact Tiers")

    t1,t2,t3 = st.columns(3)

    with t1:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--red);
        border-radius:14px;padding:1.4rem;height:100%'>
        <h3 style='color:var(--red)'>Tier 1 · Core Drivers</h3>
        <b>OverallQual</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Construction quality</span><br><br>

        <b>GrLivArea</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Above-grade living area</span><br><br>

        <b>TotalBsmtSF</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Total basement area</span><br><br>

        <b>GarageCars</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Garage capacity</span><br><br>

        <b>YearBuilt</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Construction year</span><br><br>

        <b>Neighborhood</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Property location</span>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--gold);
        border-radius:14px;padding:1.4rem;height:100%'>
        <h3 style='color:var(--gold)'>Tier 2 · Strong Signals</h3>
        <b>KitchenQual</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Kitchen finish quality</span><br><br>

        <b>ExterQual</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Exterior material quality</span><br><br>

        <b>BsmtQual</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Basement quality</span><br><br>

        <b>Fireplaces</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Number of fireplaces</span><br><br>

        <b>GarageArea</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Garage size</span><br><br>

        <b>YearRemodAdd</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Remodel year</span>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--green);
        border-radius:14px;padding:1.4rem;height:100%'>
        <h3 style='color:var(--green)'>Tier 3 · Contextual</h3>
        <b>LotArea</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Lot size</span><br><br>

        <b>MSSubClass</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Dwelling class</span><br><br>

        <b>MoSold</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Month sold</span><br><br>

        <b>Condition1</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Nearby conditions</span><br><br>

        <b>Foundation</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Foundation type</span><br><br>

        <b>WoodDeckSF</b><br>
        <span style='color:var(--muted);font-size:.85rem'>Deck size</span>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    # ───────────── Neighborhood Table ─────────────
    section_header("", "Neighbourhood Reference")

    nbhd_df = pd.DataFrame({
        "Neighbourhood":["NridgHt","NoRidge","StoneBr","Timber","Veenker",
        "Somerst","CollgCr","ClearCr","Crawfor","Gilbert",
        "Blmngtn","NAmes","Edwards","OldTown","BrkSide",
        "IDOTRR","MeadowV","BrDale","NPkVill","SWISU",
        "Sawyer","Mitchel"],

        "Typical Price":["$315k","$307k","$310k","$243k","$238k",
        "$229k","$197k","$213k","$210k","$193k",
        "$195k","$145k","$128k","$122k","$125k",
        "$100k","$98k","$104k","$141k","$142k",
        "$136k","$153k"],

        "Market Tier":["Premium","Premium","Premium","High","High",
        "High","Mid-High","Mid","Mid","Mid",
        "Mid","Average","Below Avg","Below Avg","Below Avg",
        "Low","Low","Low","Mid","Mid","Mid","Mid"]
    })

    st.dataframe(nbhd_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div style='margin-bottom:2rem'>
        <h1 style='margin-bottom:.25rem'>About EstimaAI</h1>
        <p style='color:var(--muted)'>Project overview, dataset details, and technical notes.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.75rem;line-height:1.9'>
        <h3 style='color:var(--gold);margin-top:0'>Project Summary</h3>
        <p>EstimaAI is a production-ready house price prediction system built on the
        <b>Ames Housing Dataset</b> — a widely-used Kaggle competition dataset with
        <b>1,460 training samples</b> and <b>79 original features</b> describing
        residential properties in Ames, Iowa (USA), sold 2006–2010.</p>

        <p>The system uses a <b>3-model stacking ensemble</b>:
        ElasticNet provides a linear regularised baseline;
        XGBoost and CatBoost capture non-linear interactions.
        Their out-of-fold predictions are blended by a Ridge meta-learner.</p>

        <p>All predictions are made on <b>log-transformed prices</b> (log1p), then
        exponentiated back to USD for display. This stabilises variance and improves
        RMSE on a log scale — the official Kaggle evaluation metric.</p>

        <h3 style='color:var(--gold)'>Dataset Details</h3>
        <ul style='color:var(--muted)'>
            <li>Source: <b style='color:var(--text)'>Kaggle · House Prices: Advanced Regression Techniques</b></li>
            <li>Rows: <b style='color:var(--text)'>1,460 train / 1,459 test</b></li>
            <li>Features: <b style='color:var(--text)'>79 raw → ~250 after encoding</b></li>
            <li>Target: <b style='color:var(--text)'>SalePrice (USD)</b> · range $34,900 – $755,000</li>
            <li>Median sale price: <b style='color:var(--text)'>~$163,000</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.75rem'>
        <h3 style='color:var(--gold);margin-top:0'>Tech Stack</h3>
        <div style='display:flex;flex-direction:column;gap:.6rem;font-size:.88rem'>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🐍 Python 3.11</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🔢 NumPy / Pandas</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🤖 Scikit-Learn</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🌲 XGBoost 2.x</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🐱 CatBoost</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>🎈 Streamlit 1.x</div>
            <div style='background:var(--surface);border-radius:8px;padding:.5rem .8rem'>💾 joblib (model I/O)</div>
        </div>
        </div>

        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;text-align:center;color:var(--muted);font-size:.85rem'>
        Built with ❤️ using Python · Scikit-Learn · XGBoost · CatBoost · Streamlit
        &nbsp;|&nbsp; Dataset: Ames Housing (De Cock, 2011)
        &nbsp;|&nbsp; For educational purposes
    </div>
    """, unsafe_allow_html=True)

elif page=="Data Visulization":
    render_data_viz()