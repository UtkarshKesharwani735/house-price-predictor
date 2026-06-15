import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(ttl=0)
def load_artifacts():
    model      = joblib.load(os.path.join(BASE, "best_house_model.pkl"))
    scaler     = joblib.load(os.path.join(BASE, "scaler.pkl"))
    encoders   = joblib.load(os.path.join(BASE, "label_encoders.pkl"))
    features   = joblib.load(os.path.join(BASE, "feature_names.pkl"))
    locations  = joblib.load(os.path.join(BASE, "location_list.pkl"))
    area_types = joblib.load(os.path.join(BASE, "area_type_list.pkl"))
    return model, scaler, encoders, features, locations, area_types

model, scaler, label_encoders, feature_names, location_list, area_type_list = load_artifacts()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Base ── */
.stApp { background: #08090d; }
.block-container { padding: 2rem 2rem 3rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography overrides ── */
label { color: #6b7280 !important; font-size: 0.78rem !important;
        font-weight: 500 !important; letter-spacing: 0.06em !important;
        text-transform: uppercase !important; }

/* ── Navbar strip ── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 2rem 0; margin-bottom: 0.5rem;
    border-bottom: 1px solid #13151c;
}
.nav-logo {
    display: flex; align-items: center; gap: 0.55rem;
    color: #f9fafb; font-size: 0.95rem; font-weight: 700;
}
.nav-dot {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
}
.nav-tag {
    background: #13151c; color: #6b7280; border: 1px solid #1f2333;
    border-radius: 20px; padding: 0.25rem 0.75rem;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.05em;
}

/* ── Hero ── */
.hero { padding: 2.5rem 0 3rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    color: #7c3aed; font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 1rem;
}
.hero-h1 {
    color: #f9fafb; font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700; line-height: 1.1; letter-spacing: -0.025em;
    margin-bottom: 0.8rem;
}
.hero-h1 em { font-style: normal; color: #7c3aed; }
.hero-sub {
    color: #6b7280; font-size: 0.95rem; line-height: 1.7;
    max-width: 440px; margin-bottom: 2rem;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
    background: #0d0f16; border: 1px solid #1f2333;
    border-radius: 999px; padding: 0.3rem 0.9rem;
    color: #9ca3af; font-size: 0.75rem; font-weight: 500;
    display: flex; align-items: center; gap: 0.4rem;
}
.pill-dot { width: 6px; height: 6px; border-radius: 50%; background: #7c3aed; }

/* ── Layout cols ── */
.panel {
    background: #0d0f16; border: 1px solid #1a1d28;
    border-radius: 16px; padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    color: #374151; font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 1.4rem; display: flex; align-items: center; gap: 0.5rem;
}
.panel-label::after {
    content: ''; flex: 1; height: 1px; background: #1a1d28;
}

/* ── Inputs: override Streamlit select/number ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #08090d !important;
    border: 1px solid #1f2333 !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}

/* ── CTA button ── */
div[data-testid="stButton"] > button {
    width: 100%; padding: 0.9rem 1.5rem;
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    color: #fff; font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem; font-weight: 600; letter-spacing: 0.01em;
    border: none; border-radius: 10px; cursor: pointer;
    transition: opacity 0.15s ease; margin-top: 0.4rem;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }

/* ── Result card ── */
.result-card {
    position: relative; overflow: hidden;
    background: #0d0f16; border: 1px solid #1a1d28;
    border-radius: 16px; padding: 2.2rem 1.8rem; text-align: center;
}
.result-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
}
.result-card-glow {
    position: absolute; width: 200px; height: 200px;
    border-radius: 50%; background: rgba(124,58,237,0.06);
    top: -60px; right: -60px; pointer-events: none;
}
.result-eyebrow {
    font-family: 'DM Mono', monospace;
    color: #374151; font-size: 0.68rem; letter-spacing: 0.14em;
    text-transform: uppercase; margin-bottom: 0.8rem;
}
.result-price {
    font-size: clamp(2rem, 5vw, 2.8rem); font-weight: 700;
    letter-spacing: -0.03em; line-height: 1;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.4rem;
}
.result-inr { color: #374151; font-size: 0.9rem; margin-bottom: 1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(16,185,129,0.08); color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.18); border-radius: 999px;
    padding: 0.28rem 0.85rem; font-size: 0.72rem; font-weight: 600;
}
.result-meta { color: #374151; font-size: 0.78rem; margin-top: 0.8rem;
               font-family: 'DM Mono', monospace; letter-spacing: 0.04em; }

/* ── Summary list ── */
.s-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid #13151c;
}
.s-row:last-child { border-bottom: none; }
.s-key { color: #4b5563; font-size: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }
.s-val { color: #d1d5db; font-size: 0.82rem; font-weight: 600;
          font-family: 'DM Mono', monospace; }

/* ── Empty state ── */
.empty {
    background: #0d0f16; border: 1px dashed #1f2333;
    border-radius: 16px; padding: 3rem 1.5rem; text-align: center;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-title { color: #374151; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.4rem; }
.empty-sub { color: #1f2333; font-size: 0.8rem; line-height: 1.7; }
.empty-sub strong { color: #7c3aed; }

/* ── Footer ── */
.foot {
    text-align: center; padding: 2rem 0 0.5rem;
    color: #1f2333; font-size: 0.72rem; font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
}
.foot span { color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">
        <div class="nav-dot">🏠</div>
        House Price Predictor
    </div>
    <div class="nav-tag">ML · Random Forest</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">// AI-Powered Estimation</div>
    <h1 class="hero-h1">Know your property's<br><em>true market value.</em></h1>
    <p class="hero-sub">
        Enter a few details about any property and get an
        instant, data-driven price estimate — no agents, no guesswork.
    </p>
    <div class="pill-row">
        <div class="pill"><div class="pill-dot"></div> 13,000+ Properties Trained</div>
        <div class="pill"><div class="pill-dot"></div> 85%+ Accuracy</div>
        <div class="pill"><div class="pill-dot"></div> 100+ Locations</div>
        <div class="pill"><div class="pill-dot"></div> 3 ML Models Tested</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-label">Property Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        area_type  = st.selectbox("Area Type",  area_type_list)
        location   = st.selectbox("Location",   location_list)
        total_sqft = st.number_input("Total Sqft", min_value=300, max_value=30000, value=1200, step=50)
    with c2:
        bhk     = st.number_input("BHK",        min_value=1, max_value=20, value=3, step=1)
        bath    = st.number_input("Bathrooms",   min_value=1, max_value=20, value=2, step=1)
        balcony = st.number_input("Balconies",   min_value=0, max_value=10, value=1, step=1)

    st.markdown('</div>', unsafe_allow_html=True)
    predict_btn = st.button("Estimate Price →")

with right:
    if predict_btn:
        raw = {
            'area_type'  : area_type,
            'location'   : location,
            'total_sqft' : total_sqft,
            'bath'       : bath,
            'balcony'    : balcony,
            'bhk'        : bhk,
        }

        input_df = pd.DataFrame([raw])

        if input_df['location'][0] not in label_encoders['location'].classes_:
            input_df['location'] = 'Other'

        for col in ['area_type', 'location']:
            input_df[col] = label_encoders[col].transform(input_df[col])

        input_df['sqft_per_bhk'] = input_df['total_sqft'] / (input_df['bhk'] + 1)
        input_df['bath_per_bhk'] = input_df['bath'] / (input_df['bhk'] + 1)
        input_df['total_rooms']  = input_df['bhk'] + input_df['bath']

        input_df = input_df[feature_names]
        price    = model.predict(scaler.transform(input_df))[0]
        price    = max(price, 0)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-glow"></div>
            <div class="result-eyebrow">Estimated Market Value</div>
            <div class="result-price">₹ {price:.2f} L</div>
            <div class="result-inr">≈ ₹ {price*100000:,.0f}</div>
            <div class="badge">✓ Confidence: High</div>
            <div class="result-meta">{total_sqft:,} sqft · {bhk} BHK · {location}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:1rem;"><div class="panel-label">Input Summary</div>', unsafe_allow_html=True)

        icons = {
            "area_type": "🏗", "location": "📍", "total_sqft": "📐",
            "bath": "🚿", "balcony": "🌿", "bhk": "🛏"
        }
        rows = ""
        for k, v in raw.items():
            unit = " sqft" if k == "total_sqft" else ""
            dv   = f"{v:,}{unit}" if isinstance(v, (int, float)) else str(v)
            rows += f'<div class="s-row"><span class="s-key">{icons.get(k,"")} {k.replace("_"," ").title()}</span><span class="s-val">{dv}</span></div>'
        st.markdown(rows, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty">
            <div class="empty-icon">🏠</div>
            <div class="empty-title">Awaiting property details</div>
            <div class="empty-sub">
                Fill in the form on the left and click<br>
                <strong>Estimate Price</strong> to see results here.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
    House Price Prediction &nbsp;·&nbsp; <span>Random Forest Model</span> &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)

   