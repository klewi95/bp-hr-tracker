import os
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Streamlit Page Config ---
st.set_page_config(page_title="Blutdruck Analyse", layout="wide")
# Einheitliches Styling mit Fallback
try:
    plt.style.use('seaborn-whitegrid')
except (OSError, IOError):
    plt.style.use('ggplot')

st.title("📊 Blutdruck- und Pulsanalyse")

# --- Funktionen ---
@st.cache_data
def load_json_data(uploaded_file):
    try:
        data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Ungültiges JSON-Format.")
        st.stop()
    if not isinstance(data, list) or not data:
        st.error("Erwartet eine Liste von Datensätzen im JSON.")
        st.stop()
    required_cols = {'datum', 'systolisch', 'diastolisch', 'puls'}
    if not required_cols.issubset(data[0].keys()):
        st.error("JSON fehlt mindestens eine der Spalten: " + ", ".join(required_cols))
        st.stop()
    df = pd.DataFrame(data)
    # Datum mit aktuellem Jahr ergänzen und parsen
    year = datetime.now().year
    df['datum_raw'] = df['datum'].astype(str)
    df['datum'] = pd.to_datetime(
        df['datum_raw'] + f'.{year}',
        format='%d.%m.%Y',
        errors='coerce'
    )
    if df['datum'].isna().any():
        st.warning("Mindestens ein Datum konnte nicht geparst werden.")
    return df.sort_values('datum')

@st.cache_data
def compute_trend(df, column):
    if len(df) < 2:
        return np.full(len(df), np.nan)
    x = np.arange(len(df))
    y = df[column].values
    coef = np.polyfit(x, y, 1)
    return np.poly1d(coef)(x)

@st.cache_data
def compute_summary(df, column):
    return {
        "Ø": round(df[column].mean(), 1),
        "Min": int(df[column].min()),
        "Max": int(df[column].max())
    }


def create_pdf_report(fig, stats, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Blutdruck Analyse", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.image(output_path, x=10, y=30, w=190)
    pdf.set_y(160)
    for label, s in stats.items():
        pdf.cell(0, 10, f"{label} Ø: {s['Ø']} (Min: {s['Min']}, Max: {s['Max']})", ln=True)
    return pdf

# --- JSON-Upload ---
uploaded_file = st.file_uploader("📁 Lade deine Blutdruckdaten (JSON) hoch", type="json")
if not uploaded_file:
    st.info("Bitte lade eine JSON-Datei hoch, um deine Daten anzuzeigen.")
    st.stop()

df = load_json_data(uploaded_file)

# Trendlinien berechnen
for col in ['systolisch', 'diastolisch', 'puls']:
    df[f'{col}Trend'] = compute_trend(df, col)

# Statistik
syst_stats = compute_summary(df, 'systolisch')
diast_stats = compute_summary(df, 'diastolisch')
puls_stats = compute_summary(df, 'puls')

# Optimale Bereiche
opt_ranges = {
    'systolisch': (90, 120),
    'diastolisch': (60, 80),
    'puls': (60, 80)
}

# --- Plot erstellen ---
fig, ax1 = plt.subplots(figsize=(10, 5))
dates = df['datum']

# Blutdruckkurven
ax1.plot(dates, df['systolisch'], marker='o', label="Systolisch")
ax1.plot(dates, df['systolischTrend'], linestyle='--', alpha=0.5)
ax1.plot(dates, df['diastolisch'], marker='o', label="Diastolisch")
ax1.plot(dates, df['diastolischTrend'], linestyle='--', alpha=0.5)

# Puls auf zweiter Achse, aber ohne eigene Achsenbeschriftung
ax2 = ax1.twinx()
ax2.plot(dates, df['puls'], marker='o', label="Puls")
ax2.plot(dates, df['pulsTrend'], linestyle='--', alpha=0.5)
# Achsbeschriftung der rechten Achse entfernen und Ticks ausblenden
ax2.set_ylabel("")
ax2.set_yticklabels([])
ax2.tick_params(left=False, right=False, labelleft=False, labelright=False)

# Highlight optimale Bereiche
ax1.axhspan(*opt_ranges['systolisch'], alpha=0.1)
ax1.axhspan(*opt_ranges['diastolisch'], alpha=0.1)
# Für die Puls-Schattierung zweite Achse nutzen, aber Bereich auf linke Achse projizieren
ax1.axhspan(opt_ranges['puls'][0], opt_ranges['puls'][1], alpha=0.05)

# Beschriftungen
ax1.set_xlabel("Datum")
ax1.set_ylabel("mmHg / bpm")
fig.autofmt_xdate()
fig.legend(loc='upper center', ncol=3)
fig.tight_layout()

st.pyplot(fig)

# --- Zusammenfassung ---
st.subheader("📌 Zusammenfassung")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Systolisch Ø", f"{syst_stats['Ø']} mmHg")
    st.text(f"Min: {syst_stats['Min']} | Max: {syst_stats['Max']}")
    st.text(f"Optimal: {opt_ranges['systolisch'][0]}–{opt_ranges['systolisch'][1]} ")

with col2:
    st.metric("Diastolisch Ø", f"{diast_stats['Ø']} mmHg")
    st.text(f"Min: {diast_stats['Min']} | Max: {diast_stats['Max']}")
    st.text(f"Optimal: {opt_ranges['diastolisch'][0]}–{opt_ranges['diastolisch'][1]}")

with col3:
    st.metric("Puls Ø", f"{puls_stats['Ø']} bpm")
    st.text(f"Min: {puls_stats['Min']} | Max: {puls_stats['Max']}")
    st.text(f"Optimal: {opt_ranges['puls'][0]}–{opt_ranges['puls'][1]}")

# Einzelwerte-Tabelle
st.subheader("📅 Einzelwerte")
st.dataframe(df[['datum', 'systolisch', 'diastolisch', 'puls']], use_container_width=True)

# --- PDF Export ---
st.subheader("📄 Export")
if st.button("Als PDF speichern"):
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        pdf = create_pdf_report(fig, {
            'Systolisch': syst_stats,
            'Diastolisch': diast_stats,
            'Puls': puls_stats
        }, tmp_img.name)

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp_pdf.name)
        tmp_pdf.close()

        with open(tmp_pdf.name, "rb") as f:
            st.download_button("📥 PDF herunterladen", f, file_name="blutdruck_bericht.pdf")
    finally:
        tmp_img.close()
        if os.path.exists(tmp_img.name): os.remove(tmp_img.name)
        if 'tmp_pdf' in locals() and os.path.exists(tmp_pdf.name): os.remove(tmp_pdf.name)
