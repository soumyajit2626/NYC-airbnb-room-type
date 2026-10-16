import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="NYC Airbnb 3D Predictor", page_icon="🏙️", layout="wide")

NUM_COLS = ["latitude", "longitude", "price", "minimum_nights",
            "number_of_reviews", "reviews_per_month", "availability_365"]
CAT_COLS = ["neighbourhood_group", "neighbourhood"]
COLORS = {"Entire home/apt": [0, 229, 255],
          "Private room": [255, 64, 129],
          "Shared room": [255, 214, 0]}
HEX = {k: "#%02x%02x%02x" % tuple(v) for k, v in COLORS.items()}

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 20% 0%, #16213e 0%, #0b0f1a 55%);}
header[data-testid="stHeader"] {background: transparent;}
.hero {font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
       background: linear-gradient(90deg,#00e5ff,#ff4081); -webkit-background-clip: text;
       -webkit-text-fill-color: transparent;}
.sub {color:#8fa0c8; margin-bottom: 1rem;}
.card {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
       border-radius: 18px; padding: 22px; backdrop-filter: blur(8px);
       box-shadow: 0 0 30px var(--glow);}
.card .label {color:#8fa0c8; font-size:.8rem; text-transform:uppercase; letter-spacing:2px;}
.card .value {font-size: 2rem; font-weight: 800; color: var(--c);}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("Model_1.pkl")
    ohe = joblib.load("ohe.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, ohe, scaler, meta


model, ohe, scaler, meta = load_artifacts()
sample = pd.DataFrame(meta["sample"]) if meta.get("sample") else None
if sample is not None:
    sample["color"] = sample["room_type"].map(COLORS)

# ---------- Sidebar controls (the prediction updates live) ----------
with st.sidebar:
    st.markdown("### 🎛️ Listing controls")
    group = st.selectbox("Borough", sorted(meta["neighbourhoods"]))
    hood = st.selectbox("Neighbourhood", meta["neighbourhoods"][group])
    price = st.slider("Price per night ($)", 0, int(meta["price_cap"]), 120, 5)
    min_nights = st.slider("Minimum nights", 1, int(meta["nights_cap"]), 2)
    availability = st.slider("Availability (days/year)", 0, 365, 180)
    n_reviews = st.slider("Number of reviews", 0, 500, 10)
    rev_month = st.slider("Reviews per month", 0.0, 15.0, 0.5, 0.1)
    st.markdown("**📍 Location**")
    lat = st.slider("Latitude", 40.50, 40.92, 40.73, 0.005)
    lon = st.slider("Longitude", -74.25, -73.70, -73.95, 0.005)

# ---------- Predict ----------
num_df = pd.DataFrame([[lat, lon, price, min_nights, n_reviews, rev_month, availability]],
                      columns=NUM_COLS)
cat_df = pd.DataFrame([[group, hood]], columns=CAT_COLS)
X = pd.concat([pd.DataFrame(scaler.transform(num_df)),
               pd.DataFrame(ohe.transform(cat_df))], axis=1).to_numpy()
pred = model.predict(X)[0]
proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
conf = proba[pred]
c = HEX[pred]

st.markdown('<div class="hero">NYC Airbnb Room-Type Predictor</div>'
            '<div class="sub">Move the sliders — the model, the 3D city and the charts update instantly.</div>',
            unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
k1.markdown(f'<div class="card" style="--c:{c};--glow:{c}33"><div class="label">Predicted type</div>'
            f'<div class="value">{pred}</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="card" style="--c:{c};--glow:{c}33"><div class="label">Confidence</div>'
            f'<div class="value">{conf:.0%}</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="card" style="--c:{c};--glow:{c}33"><div class="label">Location</div>'
            f'<div class="value" style="font-size:1.3rem">{hood}, {group}</div></div>',
            unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3 = st.tabs(["🗺️ 3D City Map", "🧊 3D Price Space", "📊 Confidence"])

with tab1:
    layers = []
    if sample is not None:
        layers.append(pdk.Layer(
            "ColumnLayer", data=sample, get_position=["longitude", "latitude"],
            get_elevation="price", elevation_scale=5, radius=70,
            get_fill_color="color", extruded=True, pickable=True, auto_highlight=True))
    me = pd.DataFrame([{"longitude": lon, "latitude": lat, "price": price,
                        "room_type": pred, "color": COLORS[pred] + [255]}])
    layers.append(pdk.Layer(
        "ColumnLayer", data=me, get_position=["longitude", "latitude"],
        get_elevation="price", elevation_scale=5, radius=180,
        get_fill_color="color", extruded=True, pickable=True))
    st.pydeck_chart(pdk.Deck(
        layers=layers, map_style="dark",
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=10.8,
                                         pitch=55, bearing=20),
        tooltip={"html": "<b>{room_type}</b><br/>${price}/night",
                 "style": {"backgroundColor": "#131a2b", "color": "white"}}),
        height=560)
    st.caption("Column height = nightly price. Your listing is the tall, wide column. "
               "Hold right-click / Ctrl + drag to tilt and rotate.")

with tab2:
    fig = go.Figure()
    if sample is not None:
        for rt, g in sample.groupby("room_type"):
            fig.add_trace(go.Scatter3d(
                x=g["longitude"], y=g["latitude"], z=g["price"], mode="markers", name=rt,
                marker=dict(size=2.5, color=HEX[rt], opacity=0.6)))
    fig.add_trace(go.Scatter3d(
        x=[lon], y=[lat], z=[price], mode="markers", name="Your listing",
        marker=dict(size=10, color="#ffffff", symbol="diamond",
                    line=dict(color=c, width=4))))
    fig.update_layout(template="plotly_dark", height=560, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis_title="Longitude", yaxis_title="Latitude",
                                 zaxis_title="Price ($)", bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Drag to rotate, scroll to zoom, click legend items to hide groups.")

with tab3:
    labels = list(proba)
    fig2 = go.Figure(go.Bar(
        x=[proba[l] for l in labels], y=labels, orientation="h",
        marker=dict(color=[HEX[l] for l in labels]),
        text=[f"{proba[l]:.1%}" for l in labels], textposition="auto"))
    fig2.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(range=[0, 1], tickformat=".0%"),
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)