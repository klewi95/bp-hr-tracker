import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import json
import os

st.set_page_config(page_title="Blutdruck Analyse", layout="wide")

st.title("📊 Blutdruck- und Pulsanalyse")

# JSON Upload
uploaded_file = st.file_uploader("📁 Lade deine Blutdruckdaten (JSON) hoch", type="json")

if uploaded_file is not None:
    data = json.load(uploaded_file)
    df = pd.DataFrame(data)
    
    # Trendlinien berechnen
    def add_trendline(df, column):
        x = np.arange(len(df))
        y = df[column].values
        coef = np.polyfit(x, y, 1)
        return np.poly1d(coef)(x)

    df['systolischTrend'] = add_trendline(df, 'systolisch')
    df['diastolischTrend'] = add_trendline(df, 'diastolisch')
    df['pulsTrend'] = add_trendline(df, 'puls')

    # Statistik
    def summary_stats(column):
        return {
            "Ø": round(df[column].mean(), 1),
            "Min": df[column].min(),
            "Max": df[column].max()
        }

    syst_stats = summary_stats("systolisch")
    diast_stats = summary_stats("diastolisch")
    puls_stats = summary_stats("puls")

    # Optimale Bereiche
    opt_syst = (90, 120)
    opt_diast = (60, 80)
    opt_puls = (60, 80)

    # Diagramm
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    labels = df['datum']

    ax1.plot(x, df['systolisch'], label="Systolisch", color='red', marker='o')
    ax1.plot(x, df['systolischTrend'], '--', color='red', alpha=0.5)

    ax1.plot(x, df['diastolisch'], label="Diastolisch", color='blue', marker='o')
    ax1.plot(x, df['diastolischTrend'], '--', color='blue', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, df['puls'], label="Puls", color='green', marker='o')
    ax2.plot(x, df['pulsTrend'], '--', color='green', alpha=0.5)

    ax1.axhspan(*opt_syst, color='red', alpha=0.1, label="Optimal Systolisch")
    ax1.axhspan(*opt_diast, color='blue', alpha=0.1, label="Optimal Diastolisch")
    ax2.axhspan(*opt_puls, color='green', alpha=0.1, label="Optimal Puls")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("mmHg")
    ax2.set_ylabel("bpm")
    fig.legend(loc='upper left')
    fig.tight_layout()

    st.pyplot(fig)

    # Zusammenfassung
    st.subheader("📌 Zusammenfassung")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Systolisch Ø", f"{syst_stats['Ø']} mmHg")
        st.text(f"Min: {syst_stats['Min']}, Max: {syst_stats['Max']}")
        st.text(f"Optimal: {opt_syst[0]}–{opt_syst[1]}")

    with col2:
        st.metric("Diastolisch Ø", f"{diast_stats['Ø']} mmHg")
        st.text(f"Min: {diast_stats['Min']}, Max: {diast_stats['Max']}")
        st.text(f"Optimal: {opt_diast[0]}–{opt_diast[1]}")

    with col3:
        st.metric("Puls Ø", f"{puls_stats['Ø']} bpm")
        st.text(f"Min: {puls_stats['Min']}, Max: {puls_stats['Max']}")
        st.text(f"Optimal: {opt_puls[0]}–{opt_puls[1]}")

    # Tabelle
    st.subheader("📅 Einzelwerte")
    st.dataframe(df[['datum', 'systolisch', 'diastolisch', 'puls']], use_container_width=True)

    # PDF Export
    st.subheader("📄 Export")
    if st.button("Als PDF speichern"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            image_path = tmpfile.name

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Blutdruck Analyse", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.image(image_path, x=10, y=30, w=190)

        pdf.ln(100)
        pdf.set_y(160)
        pdf.cell(0, 10, f"Systolisch Ø: {syst_stats['Ø']} mmHg (Min: {syst_stats['Min']}, Max: {syst_stats['Max']})", ln=True)
        pdf.cell(0, 10, f"Diastolisch Ø: {diast_stats['Ø']} mmHg (Min: {diast_stats['Min']}, Max: {diast_stats['Max']})", ln=True)
        pdf.cell(0, 10, f"Puls Ø: {puls_stats['Ø']} bpm (Min: {puls_stats['Min']}, Max: {puls_stats['Max']})", ln=True)

        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(pdf_output.name)

        with open(pdf_output.name, "rb") as f:
            st.download_button("📥 PDF herunterladen", f, file_name="blutdruck_bericht.pdf")

        os.remove(image_path)
        os.remove(pdf_output.name)

else:
    st.info("Bitte lade eine JSON-Datei hoch, um deine Daten anzuzeigen.")
