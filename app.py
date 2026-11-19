from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


st.set_page_config(page_title="DineScope • Restaurant Intelligence", page_icon="🍽️", layout="wide")
DATA_FILE = Path(__file__).parent / "Zomato Restaurant Dataset.csv"
NUM = ["Price range", "Votes", "Average Cost for two"]
CAT = ["City", "Cuisines", "Currency", "Has Table booking", "Has Online delivery"]
FEATURES = NUM + CAT
RATING_FEATURES = FEATURES
PRICE_FEATURES = [c for c in FEATURES if c != "Price range"]

st.markdown("""
<style>
:root { color-scheme:light; --ink:#202322; --muted:#59625c; --paper:#f5f6f2; --card:#fff; --line:#dfe3dc; --accent:#c74430; --green:#315f4b; }
html, body, [data-testid="stAppViewContainer"], .stApp { background:#f5f6f2 !important; color:#202322 !important; font-family:Arial, sans-serif !important; }
[data-testid="stHeader"] { background:rgba(245,246,242,.96) !important; }
[data-testid="stMain"] { background:#f5f6f2 !important; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
[data-testid="stWidgetLabel"] p, label, .stCaption, small { color:#303832 !important; }
[data-testid="stSidebar"] { display:block !important; background:#182b27 !important; border-right:1px solid #29433c; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#e9f0eb !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#e9f0eb !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#b7c9bf !important; }
[data-testid="stSidebar"] [role="radiogroup"] label { background:#203831; border:1px solid #345348; padding:10px 12px; border-radius:10px; margin:4px 0; color:#f1f5f2 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#2b493e; }
[data-testid="stSidebar"] [role="radiogroup"] label p { color:#f1f5f2 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#d96a4d; border-color:#e78a70; }
h1,h2,h3,h4 { color:#202322 !important; letter-spacing:-.03em; }
.hero { background:#243a32; border-radius:22px; padding:40px 48px; color:white; margin:4px 0 23px; position:relative; overflow:hidden; min-height:320px; display:flex; flex-direction:column; justify-content:center; }
.hero:after { content:'✳'; position:absolute; right:36px; top:4px; font-size:112px; color:#ffffff16; }
.hero h1, [data-testid="stMarkdownContainer"] .hero h1 { color:#ffffff !important; font-size:58px !important; line-height:1.12 !important; margin:14px 0 18px !important; max-width:1000px; }
.hero p, [data-testid="stMarkdownContainer"] .hero p { color:#e1ece5 !important; margin:0; font-size:21px; }
.eyebrow { color:#9f3626; text-transform:uppercase; letter-spacing:.13em; font-size:14px; font-weight:800; }
.hero .eyebrow, [data-testid="stMarkdownContainer"] .hero .eyebrow { color:#fff1ec !important; font-size:42px !important; font-weight:900 !important; letter-spacing:.09em; line-height:1.2; }
.section-title { margin:22px 0 3px; font-size:22px; font-weight:800; }
.section-sub { color:var(--muted); margin-bottom:16px; font-size:13px; }
.metric-card { background:var(--card); border:1px solid var(--line); padding:18px 20px; border-radius:16px; min-height:98px; box-shadow:0 4px 16px #18231a08; }
.metric-label { color:var(--muted); font-size:12px; font-weight:600; }
.metric-value { color:var(--ink); font:800 27px 'Manrope',sans-serif; margin-top:4px; }
.result-card { background:#fff; border:1px solid var(--line); border-left:5px solid var(--accent); padding:22px 24px; border-radius:15px; margin:14px 0; }
.result-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.11em; font-weight:700; }
.result-value { font:800 32px 'Manrope',sans-serif; margin:5px 0; }
.step { color:var(--green); font-weight:700; font-size:12px; }
div[data-testid="stMetric"] { background:white !important; border:1px solid var(--line); border-radius:15px; padding:14px 16px; }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"], div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color:#202322 !important; }
div[data-testid="stTabs"] button { color:#303832 !important; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--accent) !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] { background:#fff !important; border-color:#aeb8ae !important; }
div[data-baseweb="select"] *, div[data-baseweb="input"] *, div[data-baseweb="base-input"] *, input, textarea { color:#202322 !important; -webkit-text-fill-color:#202322 !important; }
div[role="listbox"], ul[role="listbox"], li[role="option"] { background:#fff !important; color:#202322 !important; }
div.stButton > button { border-radius:10px; border:0; background:var(--accent); color:white !important; font-weight:700; padding:.55rem 1rem; }
div.stButton > button:hover { color:white !important; background:#a93424; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def prepare_data():
    raw = pd.read_csv(DATA_FILE)
    df = raw.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()
    df["Cuisines"] = df["Cuisines"].fillna("Not specified").astype(str).str.strip()
    for column in CAT:
        if column in df:
            df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    for column in ["Aggregate rating", "Average Cost for two", "Price range", "Votes"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Aggregate rating", "Average Cost for two", "Price range"])
    return raw, df


def make_preprocessor(features, scale=False):
    numeric = [c for c in NUM + ["Aggregate rating"] if c in features]
    categorical = [c for c in CAT if c in features]
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical),
    ])


@st.cache_resource
def train_and_compare():
    _, df = prepare_data()
    rated = df[df["Aggregate rating"] > 0]
    x_rating = rated[RATING_FEATURES]
    x_price = df[PRICE_FEATURES]
    y_rating = rated["Aggregate rating"]
    y_price = df["Price range"].astype(int)
    xr_train, xr_test, yr_train, yr_test = train_test_split(x_rating, y_rating, test_size=.2, random_state=42)
    xp_train, xp_test, yp_train, yp_test = train_test_split(x_price, y_price, test_size=.2, random_state=42, stratify=y_price)

    regression_estimators = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0), True),
        "Decision Tree Regressor": (DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42), False),
        "Random Forest Regressor": (RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1), False),
    }
    classifiers = {
        "Logistic Regression": (LogisticRegression(max_iter=1200, class_weight="balanced"), True),
        "Decision Tree (Gini)": (DecisionTreeClassifier(criterion="gini", max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42), False),
        "Random Forest Classifier": (RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1), False),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=9, weights="distance"), True),
    }
    rating_models, price_models, reg_scores, class_scores = {}, {}, [], []
    for name, (estimator, scale) in regression_estimators.items():
        pipe = Pipeline([("prep", make_preprocessor(RATING_FEATURES, scale)), ("model", estimator)])
        pipe.fit(xr_train, yr_train)
        pred = pipe.predict(xr_test)
        rating_models[name] = pipe
        reg_scores.append({"Algorithm": name, "MAE ↓": mean_absolute_error(yr_test, pred),
                           "RMSE ↓": root_mean_squared_error(yr_test, pred), "R² ↑": r2_score(yr_test, pred)})
    for name, (estimator, scale) in classifiers.items():
        pipe = Pipeline([("prep", make_preprocessor(PRICE_FEATURES, scale)), ("model", estimator)])
        pipe.fit(xp_train, yp_train)
        pred = pipe.predict(xp_test)
        price_models[name] = pipe
        class_scores.append({"Algorithm": name, "Accuracy ↑": accuracy_score(yp_test, pred),
                             "Precision ↑": precision_score(yp_test, pred, average="weighted", zero_division=0),
                             "Recall ↑": recall_score(yp_test, pred, average="weighted", zero_division=0),
                             "F1 ↑": f1_score(yp_test, pred, average="weighted", zero_division=0)})
    return rating_models, price_models, pd.DataFrame(reg_scores), pd.DataFrame(class_scores)


raw, data = prepare_data()
rating_models, price_models, regression_scores, classification_scores = train_and_compare()

with st.sidebar:
    st.markdown("<div style='font-size:25px;font-weight:800;color:#ffffff;margin:8px 0 0'>🍽️ DineScope</div><div style='color:#b7c9bf;font-size:11px;letter-spacing:.15em;margin:2px 0 18px'>RESTAURANT INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#a9c1b5;font-size:11px;font-weight:700;letter-spacing:.13em;margin:8px 0'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Dashboard", "Visual Analytics", "Rating Prediction", "Price Tier Prediction", "Model Comparison"], label_visibility="collapsed")
    st.divider()
    st.caption("ML project · Zomato restaurant data")

st.markdown("<div class='hero'><div class='eyebrow'>RESTAURANT MARKET ANALYSIS</div><h1>DineScope</h1><p>Discover restaurant patterns. Explore the market. Make informed predictions.</p></div>", unsafe_allow_html=True)
st.markdown(f"<div class='step'>PROJECT WORKFLOW &nbsp; 01 Clean data &nbsp; → &nbsp; 02 Explore &nbsp; → &nbsp; 03 Predict rating &nbsp; → &nbsp; 04 Predict price tier &nbsp; → &nbsp; 05 Compare</div>", unsafe_allow_html=True)
with st.expander("Restaurant inputs · used by the prediction tabs", expanded=True):
    cities = sorted(data.City.unique().tolist())
    cuisines = sorted(data.Cuisines.unique().tolist())
    currencies = sorted(data.Currency.unique().tolist())
    row1 = st.columns(3)
    city = row1[0].selectbox("City", cities, index=cities.index("New Delhi") if "New Delhi" in cities else 0)
    cuisine = row1[1].selectbox("Cuisine(s)", cuisines, index=cuisines.index("North Indian") if "North Indian" in cuisines else 0)
    currency = row1[2].selectbox("Currency", currencies, index=currencies.index("Indian Rupees(Rs.)") if "Indian Rupees(Rs.)" in currencies else 0)
    row2 = st.columns(3)
    booking = row2[0].selectbox("Table booking", ["Yes", "No"])
    delivery = row2[1].selectbox("Online delivery", ["Yes", "No"])
    price_range = row2[2].slider("Price range · 1 low – 4 high", 1, 4, 2)
    row3 = st.columns(2)
    votes = row3[0].number_input("Votes", 0, 100000, 100, step=10)
    cost = row3[1].number_input("Average cost for two", 0, 1000000, 500, step=100)
    st.caption("Rating and price models use different target specific inputs. Cost values span currencies in the dataset.")

if page == "Dashboard":
    st.markdown("<div class='section-title'>Dataset at a glance</div><div class='section-sub'>A quick overview of the cleaned restaurant records used to train the models.</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Restaurants", "Cities", "Cuisine entries", "Average rating"],
                                 [f"{len(data):,}", f"{data.City.nunique():,}", f"{data.Cuisines.nunique():,}", f"{data['Aggregate rating'].mean():.2f} / 5"]):
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Your model workspace</div><div class='section-sub'>Four regression algorithms estimate rating. Four classification algorithms predict the restaurant's 1–4 price tier.</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("#### ★ Rating prediction")
        st.write("Explore four regressors using average rating as the target. Compare MAE, RMSE, and R² on a held-out test set.")
    with b:
        st.markdown("#### ◈ Price category")
        st.write("Classify Zomato's price range (1–4) with four classifiers. Compare accuracy, precision, recall, and F1.")
    st.caption("Dataset cleaning: duplicate rows removed, categorical blanks filled, numeric columns coerced to numbers, and missing target rows dropped. Identifiers and post-rating labels are excluded from model inputs.")

if page == "Visual Analytics":
    st.markdown("<div class='section-title'>Explore the dataset</div><div class='section-sub'>Inspect cleaning outcomes, target distributions, and common restaurant locations.</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows in source", f"{len(raw):,}")
    m2.metric("Rows after cleaning", f"{len(data):,}")
    m3.metric("Duplicate rows removed", f"{len(raw) - raw.drop_duplicates().shape[0]:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Rating distribution")
        rating_hist = data["Aggregate rating"].value_counts(bins=10, sort=False).sort_index()
        st.bar_chart(pd.DataFrame({"Restaurant count": rating_hist.values}, index=[str(x) for x in rating_hist.index]))
    with right:
        st.markdown("#### Price range distribution")
        st.bar_chart(data["Price range"].value_counts().sort_index().rename("Restaurants"))
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top cities")
        st.bar_chart(data["City"].value_counts().head(12).sort_values().rename("Restaurants"))
    with right:
        st.markdown("#### Missing values · before cleaning")
        missing = raw.isna().sum().sort_values(ascending=False).head(12)
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True, height=325)
    with st.expander("Preview cleaned data"):
        st.dataframe(data.head(20), use_container_width=True)

def input_row(with_rating=False):
    values = {"Price range": price_range, "Votes": votes, "Average Cost for two": cost,
              "City": city, "Cuisines": cuisine, "Currency": currency,
              "Has Table booking": booking, "Has Online delivery": delivery}
    return pd.DataFrame([values], columns=RATING_FEATURES if not with_rating else PRICE_FEATURES)

if page == "Rating Prediction":
    st.markdown("<div class='section-title'>Predict aggregate rating</div><div class='section-sub'>Estimate a restaurant score from 0 to 5 using the restaurant inputs in the panel above.</div>", unsafe_allow_html=True)
    chosen_reg = st.selectbox("Choose regression algorithm", list(rating_models), key="reg_choice")
    rating_prediction = float(rating_models[chosen_reg].predict(input_row())[0])
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div class='result-card'><div class='result-label'>Estimated aggregate rating</div><div class='result-value'>{max(0,min(5,rating_prediction)):.2f}<span style='font-size:16px;color:#777'> / 5</span></div><div>Using {chosen_reg}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("#### Regression model snapshot")
        score = regression_scores.set_index("Algorithm").loc[chosen_reg]
        st.write(f"On the held-out test set, this model has MAE **{score['MAE ↓']:.3f}**, RMSE **{score['RMSE ↓']:.3f}**, and R² **{score['R² ↑']:.3f}**.")
        st.caption("Inputs: city, cuisine, currency, service options, price tier, votes, and average cost. Aggregate rating is not included as an input.")

if page == "Price Tier Prediction":
    st.markdown("<div class='section-title'>Predict price category</div><div class='section-sub'>Classify the Zomato price range from 1 (lower) to 4 (higher).</div>", unsafe_allow_html=True)
    chosen_clf = st.selectbox("Choose classification algorithm", list(price_models), key="clf_choice")
    x = input_row(with_rating=True)
    model = price_models[chosen_clf]
    category_prediction = int(model.predict(x)[0])
    confidence = float(model.predict_proba(x).max()) if hasattr(model, "predict_proba") else None
    c1, c2 = st.columns([1, 2])
    c1.markdown(f"<div class='result-card'><div class='result-label'>Predicted price tier</div><div class='result-value'>{category_prediction}<span style='font-size:16px;color:#777'> / 4</span></div><div>Using {chosen_clf}</div></div>", unsafe_allow_html=True)
    with c2:
        score = classification_scores.set_index("Algorithm").loc[chosen_clf]
        st.markdown("#### Classification model snapshot")
        st.write(f"Held-out accuracy **{score['Accuracy ↑']:.1%}** · weighted F1 **{score['F1 ↑']:.1%}**.")
        if confidence is not None:
            st.caption(f"Highest class probability: {confidence:.0%}. This indicates model confidence, not a guarantee.")
        st.caption("Inputs exclude Price range, the target being predicted.")

if page == "Model Comparison":
    st.markdown("<div class='section-title'>Compare model performance</div><div class='section-sub'>All scores use a fixed 80/20 train/test split with random_state=42. Arrows show the preferred direction.</div>", unsafe_allow_html=True)
    st.markdown("#### Rating regression · target: Aggregate rating")
    reg_view = regression_scores.copy()
    for col in ["MAE ↓", "RMSE ↓", "R² ↑"]:
        reg_view[col] = reg_view[col].map(lambda v: f"{v:.3f}")
    st.dataframe(reg_view, hide_index=True, use_container_width=True)
    best_reg = regression_scores.loc[regression_scores["MAE ↓"].idxmin()]
    st.success(f"Lowest rating MAE: {best_reg['Algorithm']} ({best_reg['MAE ↓']:.3f}).")
    st.markdown("#### Price category classification · target: Price range")
    clf_view = classification_scores.copy()
    for col in ["Accuracy ↑", "Precision ↑", "Recall ↑", "F1 ↑"]:
        clf_view[col] = clf_view[col].map(lambda v: f"{v:.1%}")
    st.dataframe(clf_view, hide_index=True, use_container_width=True)
    best_clf = classification_scores.loc[classification_scores["F1 ↑"].idxmax()]
    st.success(f"Highest weighted F1: {best_clf['Algorithm']} ({best_clf['F1 ↑']:.1%}).")
    st.caption("Classifiers use weighted averages so each price tier contributes according to its number of test examples. These scores are estimates from this dataset split.")

st.markdown("<hr><div style='color:#858a84;font-size:12px'>DineScope · Restaurant Intelligence · Student machine learning project</div>", unsafe_allow_html=True)
