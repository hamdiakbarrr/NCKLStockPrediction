import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG & STYLING ---
st.set_page_config(page_title="NCKL Intelligence Terminal", layout="wide")

# CSS Kustom (SUDAH DIPERBAIKI: Tanpa kurung siku yang salah)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
    }
    
    .main {
        background-color: #0e1117;
    }

    /* Card Style Mewah */
    .metric-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #343a40;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        text-align: center;
        color: white;
    }
    
    /* Tombol Gradasi */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px;
        transition: 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,114,255,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_nckl_rf.pkl")

# --- Images ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "Images", "NCKL_Logo.png")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except:
    st.error("Model .pkl tidak ditemukan di folder models/")
    st.stop()

# --- HEADER SECTION ---
col_logo, col_title = st.columns([1, 6])

with col_logo:
    # Cek apakah file benar-benar ada sebelum mencoba menampilkannya
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    else:
        # Cadangan jika logo hilang agar UI tetap cantik
        st.markdown("<h1 style='margin:0; padding:0;'>🚢</h1>", unsafe_allow_html=True)
        st.caption("Logo Error")

with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>NCKL Intelligence Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#808495; font-size:1.1rem;'>Professional Grade Machine Learning for PT Trimegah Bangun Persada Tbk.</p>", unsafe_allow_html=True)

st.divider()
# --- INPUT SECTION ---
with st.container():
    with st.form("prediction_form"):
        st.markdown("### 📊 Market Parameters")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            price_now = st.number_input("Last Close Price (NCKL)", value=1000)
            open_nckl = st.number_input("Opening Price", value=1000)
            vol_nckl = st.number_input("Volume", value=1000000)
            
        with c2:
            high_nckl = st.number_input("High", value=1010)
            low_nckl = st.number_input("Low", value=990)
            rsi = st.slider("RSI (14)", 0, 100, 50)
            
        with c3:
            ma_5 = st.number_input("MA 5", value=1000)
            ma_10 = st.number_input("MA 10", value=995)
            p_nickel = st.number_input("Nickel Price ($)", value=17000)
            usd_idr = st.number_input("USD/IDR", value=15700)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("ANALYSIS DATA & PREDICT")

# --- RESULTS (PREMIUM REFINEMENT) ---
if submit:
    features = np.array([[open_nckl, high_nckl, low_nckl, vol_nckl, ma_5, ma_10, rsi, p_nickel, usd_idr]])
    delta_pred = model.predict(features)[0]
    estimasi_besok = price_now + delta_pred
    persen_perubahan = (delta_pred / price_now) * 100
    
    # Menentukan Warna Aksen berdasarkan hasil
    accent_color = "#00ffcc" if delta_pred > 0 else "#ff4b4b"
    status_text = "BULLISH" if delta_pred > 0 else "BEARISH"
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 🎯 Intelligence Output: <span style='color:{accent_color};'>{status_text}</span>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    
    # Card 1: Delta
    with m1:
        st.markdown(f'''<div class="metric-card">
            <p style="color:#808495; font-size:0.85rem; margin-bottom:5px; font-weight:700;">PREDICTED DELTA</p>
            <h1 style="color:{accent_color}; margin:0; font-size:2.2rem;">{delta_pred:+.2f} <span style="font-size:1rem;">pts</span></h1>
        </div>''', unsafe_allow_html=True)
        
    # Card 2: Percentage Change
    with m2:
        st.markdown(f'''<div class="metric-card">
            <p style="color:#808495; font-size:0.85rem; margin-bottom:5px; font-weight:700;">PROJECTION (%)</p>
            <h1 style="color:{accent_color}; margin:0; font-size:2.2rem;">{persen_perubahan:+.2f}%</h1>
        </div>''', unsafe_allow_html=True)
        
    # Card 3: Target Price (Highlight Utama)
    with m3:
        st.markdown(f'''<div class="metric-card" style="border: 1px solid {accent_color}55;">
            <p style="color:#808495; font-size:0.85rem; margin-bottom:5px; font-weight:700;">ESTIMATED PRICE</p>
            <h1 style="color:#ffffff; margin:0; font-size:2.2rem;">Rp {estimasi_besok:,.0f}</h1>
        </div>''', unsafe_allow_html=True)

    # Indikator Progress Bar Kecil untuk Visualisasi Bullish/Bearish
    st.markdown("<br>", unsafe_allow_html=True)
    if delta_pred > 0:
        st.success(f"🚀 **Model Optimistic**: NCKL menunjukkan tren akumulasi. Target harga besok berada di area Rp {estimasi_besok:,.0f}.")
    else:
        st.error(f"⚠️ **Model Cautious**: Waspada potensi koreksi. Support level diprediksi bergeser ke area Rp {estimasi_besok:,.0f}.")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("© 2026 NCKL Intelligence Dashboard | Disclaimer: High Risk Investment | Developed by: Stockgain.")