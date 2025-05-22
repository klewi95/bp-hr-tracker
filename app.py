import os
import json
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Konstanten ---
COL_DATE = 'datum'
COL_DATE_RAW = 'datum_raw'
COL_SYSTOLIC = 'systolisch'
COL_DIASTOLIC = 'diastolisch'
COL_PULSE = 'puls'
TREND_SUFFIX = 'Trend'

KEY_SYSTOLIC = 'systolisch'
KEY_DIASTOLIC = 'diastolisch'
KEY_PULSE = 'puls'

# --- Streamlit Page Config ---
st.set_page_config(page_title="Blutdruck Analyse", layout="wide")

# Einheitliches Styling mit Fallback
try:
    plt.style.use('seaborn-v0_8-whitegrid') # Aktualisierter Name für seaborn-Stile
except (OSError, IOError):
    try:
        plt.style.use('seaborn-whitegrid') # Älterer Name als Fallback
    except (OSError, IOError):
        plt.style.use('ggplot') # Generischer Fallback

st.title("📊 Blutdruck- und Pulsanalyse")

# --- Funktionen ---
@st.cache_data
def load_json_data(uploaded_file):
    """
    Lädt Blutdruckdaten aus einer JSON-Datei, validiert sie und bereitet sie auf.
    """
    try:
        data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Ungültiges JSON-Format.")
        st.stop()

    if not isinstance(data, list) or not data:
        st.error("Erwartet eine Liste von Datensätzen im JSON-Format.")
        st.stop()

    required_cols = {COL_DATE, COL_SYSTOLIC, COL_DIASTOLIC, COL_PULSE}
    if not required_cols.issubset(data[0].keys()):
        st.error(f"JSON fehlt mindestens eine der Spalten: {', '.join(required_cols)}")
        st.stop()

    df = pd.DataFrame(data)

    # Datum parsen:
    # Annahme: Das Datum in JSON ist im Format TT.MM und bezieht sich auf das aktuelle Jahr.
    # Für Daten über mehrere Jahre oder andere Formate müsste dies angepasst werden.
    current_year = datetime.now().year
    df[COL_DATE_RAW] = df[COL_DATE].astype(str) # Rohdatum für eventuelle Fehleranalyse behalten
    df[COL_DATE] = pd.to_datetime(
        df[COL_DATE_RAW] + f'.{current_year}',
        format='%d.%m.%Y',
        errors='coerce' # Fehlerhafte Daten werden zu NaT (Not a Time)
    )

    if df[COL_DATE].isna().any():
        st.warning(
            "Mindestens ein Datum konnte nicht korrekt interpretiert werden und wurde ignoriert. "
            "Bitte stelle sicher, dass das Format 'TT.MM' ist."
        )
        df = df.dropna(subset=[COL_DATE]) # Zeilen mit ungültigem Datum entfernen

    if df.empty:
        st.error("Keine validen Datensätze nach der Datumsprüfung vorhanden.")
        st.stop()

    return df.sort_values(COL_DATE)

@st.cache_data
def compute_trend(df, column):
    """
    Berechnet eine lineare Trendlinie für eine gegebene Spalte im DataFrame.
    """
    if len(df) < 2: # Trendberechnung benötigt mindestens zwei Punkte
        return np.full(len(df), np.nan)
    x_values = np.arange(len(df))
    y_values = df[column].values
    # Fehlende Werte in y_values vor polyfit behandeln (z.B. durch Füllen oder Ignorieren)
    # Hier: Einfache Annahme, dass keine NaNs in den relevanten Spalten für Trend sind
    # In einer robusteren Anwendung: df[column].fillna(df[column].mean(), inplace=True) oder ähnliches
    if np.isnan(y_values).any(): # Falls doch NaNs vorhanden sind
        # Ersetze NaNs mit dem Mittelwert für die Trendberechnung, oder gib np.nan zurück
        # Diese Strategie kann je nach Anwendungsfall variieren
        # return np.full(len(df), np.nan) # konservative Variante
        mask = ~np.isnan(y_values)
        if np.sum(mask) < 2: # Nicht genug Datenpunkte ohne NaN
            return np.full(len(df), np.nan)
        x_values_clean = np.arange(np.sum(mask))
        y_values_clean = y_values[mask]
        coef = np.polyfit(x_values_clean, y_values_clean, 1)
        trend_values_clean = np.poly1d(coef)(x_values_clean)
        # Trend auf die ursprüngliche Länge mit NaNs für fehlende Werte mappen
        trend_values = np.full(len(df), np.nan)
        trend_values[mask] = trend_values_clean
        return trend_values

    coef = np.polyfit(x_values, y_values, 1)
    return np.poly1d(coef)(x_values)


@st.cache_data
def compute_summary(df, column):
    """
    Berechnet deskriptive Statistiken (Mittelwert, Min, Max) für eine Spalte.
    """
    if df[column].empty or df[column].isnull().all():
        return {"Ø": "N/A", "Min": "N/A", "Max": "N/A"}
    return {
        "Ø": round(df[column].mean(), 1),
        "Min": int(df[column].min()),
        "Max": int(df[column].max())
    }

def create_pdf_report(figure, stats_data, temp_image_path):
    """
    Erstellt ein FPDF-Objekt mit dem Analysebericht.
    Die Positionierung der Elemente ist statisch und für dieses Layout optimiert.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Blutdruck Analyse Bericht", ln=True, align="C")
    pdf.ln(10) # Zusätzlicher Abstand

    # Bild einfügen
    # Die Bildbreite ist auf 190mm gesetzt, Höhe wird proportional angepasst.
    # x und y Positionen sind für A4 optimiert.
    pdf.image(temp_image_path, x=10, y=30, w=190)

    # Position für Statistiken nach dem Bild (Höhe des Bildes ist variabel, hier ca. 90-100mm)
    # Wir setzen y konservativ, damit es nicht mit dem Bild überlappt.
    # Annahme: Bildhöhe ist ca. 100mm (190mm breit / typisches Seitenverhältnis). 30 (Start-Y) + 100 = 130.
    # Wir setzen den Start für Text auf y=140.
    pdf.set_y(140)
    pdf.set_font("Arial", "", 12)

    for label, s_values in stats_data.items():
        if s_values["Ø"] == "N/A":
            text = f"{label}: Daten nicht verfügbar"
        else:
            text = f"{label} - Durchschnitt: {s_values['Ø']} (Min: {s_values['Min']}, Max: {s_values['Max']})"
        pdf.cell(0, 10, text, ln=True)
    return pdf

# --- JSON-Upload ---
uploaded_file = st.file_uploader("📁 Lade deine Blutdruckdaten (JSON) hoch", type="json")

if not uploaded_file:
    st.info("Bitte lade eine JSON-Datei hoch, um deine Daten anzuzeigen.")
    st.stop()

df_blood_pressure = load_json_data(uploaded_file)

# --- Datenverarbeitung und Analyse ---
# Trendlinien berechnen
for col in [COL_SYSTOLIC, COL_DIASTOLIC, COL_PULSE]:
    df_blood_pressure[f'{col}{TREND_SUFFIX}'] = compute_trend(df_blood_pressure, col)

# Statistiken berechnen
systolic_summary = compute_summary(df_blood_pressure, COL_SYSTOLIC)
diastolic_summary = compute_summary(df_blood_pressure, COL_DIASTOLIC)
pulse_summary = compute_summary(df_blood_pressure, COL_PULSE)

# Optimale Bereiche definieren
optimal_ranges = {
    KEY_SYSTOLIC: (90, 120),
    KEY_DIASTOLIC: (60, 80),
    KEY_PULSE: (60, 80) # Puls optimal Bereich in bpm
}

# --- Plot erstellen ---
fig, ax1 = plt.subplots(figsize=(12, 6)) # Etwas breiter für bessere Lesbarkeit
dates = df_blood_pressure[COL_DATE]

# Blutdruckkurven (Systolisch, Diastolisch)
ax1.plot(dates, df_blood_pressure[COL_SYSTOLIC], marker='o', linestyle='-', label="Systolisch (mmHg)")
ax1.plot(dates, df_blood_pressure[f'{COL_SYSTOLIC}{TREND_SUFFIX}'], linestyle='--', color='blue', alpha=0.7, label="Trend Systolisch")
ax1.plot(dates, df_blood_pressure[COL_DIASTOLIC], marker='o', linestyle='-', label="Diastolisch (mmHg)")
ax1.plot(dates, df_blood_pressure[f'{COL_DIASTOLIC}{TREND_SUFFIX}'], linestyle='--', color='green', alpha=0.7, label="Trend Diastolisch")

# Puls auf einer zweiten Y-Achse darstellen
ax2 = ax1.twinx()
ax2.plot(dates, df_blood_pressure[COL_PULSE], marker='s', linestyle='-', color='red', label="Puls (bpm)")
ax2.plot(dates, df_blood_pressure[f'{COL_PULSE}{TREND_SUFFIX}'], linestyle='--', color='red', alpha=0.5, label="Trend Puls")

# Y-Achsen Beschriftungen
ax1.set_xlabel("Datum")
ax1.set_ylabel("Blutdruck (mmHg)")
ax2.set_ylabel("Puls (bpm)")

# Optimale Bereiche hervorheben
# Systolisch und Diastolisch auf ax1
ax1.axhspan(*optimal_ranges[KEY_SYSTOLIC], color='blue', alpha=0.05, label=f"Optimal Systolisch ({optimal_ranges[KEY_SYSTOLIC][0]}-{optimal_ranges[KEY_SYSTOLIC][1]})")
ax1.axhspan(*optimal_ranges[KEY_DIASTOLIC], color='green', alpha=0.05, label=f"Optimal Diastolisch ({optimal_ranges[KEY_DIASTOLIC][0]}-{optimal_ranges[KEY_DIASTOLIC][1]})")

# Puls-Optimalbereich auf ax2, da er eine eigene Skala hat
# Kommentar: Der Optimalbereich für Puls wird auf ax2 (rechte Y-Achse) gezeichnet.
ax2.axhspan(*optimal_ranges[KEY_PULSE], color='red', alpha=0.05, label=f"Optimal Puls ({optimal_ranges[KEY_PULSE][0]}-{optimal_ranges[KEY_PULSE][1]})")


# Diagramm-Layout und Legende
fig.autofmt_xdate() # Formatiert Datumsanzeige auf X-Achse schön
# Legenden von beiden Achsen kombinieren
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Duplikate in Legenden vermeiden, falls Labels gleich sind (z.B. für axhspan)
unique_labels = {}
for line, label in zip(lines + lines2, labels + labels2):
    if label not in unique_labels:
        unique_labels[label] = line
fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))


fig.tight_layout(rect=[0, 0, 1, 0.95]) # Platz für die Legende über dem Plot lassen
st.pyplot(fig)


# --- Zusammenfassung der Statistiken ---
st.subheader("📌 Zusammenfassung der Werte")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Systolisch Ø", f"{systolic_summary['Ø']} mmHg" if systolic_summary['Ø'] != "N/A" else "N/A")
    if systolic_summary['Min'] != "N/A":
        st.text(f"Min: {systolic_summary['Min']} / Max: {systolic_summary['Max']}")
        st.text(f"Optimal: {optimal_ranges[KEY_SYSTOLIC][0]} – {optimal_ranges[KEY_SYSTOLIC][1]} mmHg")

with col2:
    st.metric("Diastolisch Ø", f"{diastolic_summary['Ø']} mmHg" if diastolic_summary['Ø'] != "N/A" else "N/A")
    if diastolic_summary['Min'] != "N/A":
        st.text(f"Min: {diastolic_summary['Min']} / Max: {diastolic_summary['Max']}")
        st.text(f"Optimal: {optimal_ranges[KEY_DIASTOLIC][0]} – {optimal_ranges[KEY_DIASTOLIC][1]} mmHg")

with col3:
    st.metric("Puls Ø", f"{pulse_summary['Ø']} bpm" if pulse_summary['Ø'] != "N/A" else "N/A")
    if pulse_summary['Min'] != "N/A":
        st.text(f"Min: {pulse_summary['Min']} / Max: {pulse_summary['Max']}")
        st.text(f"Optimal: {optimal_ranges[KEY_PULSE][0]} – {optimal_ranges[KEY_PULSE][1]} bpm")

# --- Einzelwerte-Tabelle ---
st.subheader("📅 Einzelwerte")
display_df = df_blood_pressure[[COL_DATE, COL_SYSTOLIC, COL_DIASTOLIC, COL_PULSE]].copy()
display_df[COL_DATE] = display_df[COL_DATE].dt.strftime('%d.%m.%Y') # Formatierung für Anzeige
st.dataframe(display_df.rename(columns={
    COL_DATE: "Datum",
    COL_SYSTOLIC: "Systolisch",
    COL_DIASTOLIC: "Diastolisch",
    COL_PULSE: "Puls"
}), use_container_width=True)


# --- PDF Export ---
st.subheader("📄 Export als PDF")
if st.button("Als PDF speichern"):
    # Temporäre Datei für das Diagramm-Bild erstellen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
        fig.savefig(tmp_img_file.name, dpi=300, bbox_inches='tight') # Höhere DPI für bessere PDF Qualität
        tmp_img_path = tmp_img_file.name

    report_stats = {
        'Systolische Werte': systolic_summary,
        'Diastolische Werte': diastolic_summary,
        'Pulswerte': pulse_summary
    }

    # PDF Objekt erstellen (Funktion gibt jetzt das Objekt zurück)
    pdf_doc = create_pdf_report(fig, report_stats, tmp_img_path)

    # Temporäre Datei für das PDF erstellen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
        pdf_doc.output(tmp_pdf_file.name, "F")
        tmp_pdf_path = tmp_pdf_file.name

    # Temporäres Bild löschen, nachdem es im PDF verwendet wurde
    if os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

    # PDF zum Download anbieten
    with open(tmp_pdf_path, "rb") as f_pdf:
        st.download_button(
            label="📥 PDF herunterladen",
            data=f_pdf,
            file_name=f"Blutdruck_Analyse_{datetime.now().strftime('%Y-%m-%d')}.pdf",
            mime="application/pdf"
        )

    # Temporäre PDF-Datei löschen, nachdem sie zum Download angeboten wurde
    # Normalerweise würde man sie nach dem Download-Vorgang löschen,
    # aber Streamlit's Download-Button ist da etwas speziell.
    # Man könnte sie hier löschen oder dem OS überlassen. Sicherer ist es, sie zu löschen.
    # Allerdings muss die Datei existieren, solange der Download-Button sichtbar ist
    # und potenziell geklickt werden kann.
    # Für eine einfache Anwendung ist es oft okay, sie nicht sofort zu löschen,
    # oder man implementiert komplexere State-Logik.
    # Hier lassen wir sie erstmal bestehen, da der Button sie braucht.
    # Besser: Löschen in einem `finally` Block, aber das ist hier mit dem with-Statement für tmp_pdf_file
    # nicht direkt so umsetzbar, da der Download-Button außerhalb des `with`-Blocks steht (implizit)
    # Stattdessen kann man sie registrieren und beim nächsten Skriptlauf löschen, oder den User informieren.

    # Für diesen Fall ist es am einfachsten, die Löschung der PDF-Datei dem Betriebssystem zu überlassen
    # oder sie nach einer gewissen Zeit zu entfernen, wenn man eine Hintergrundaufgabe hätte.
    # Da `delete=False` gesetzt wurde, bleibt sie bestehen.
    # Wenn der Button geklickt wurde und der Download startet, könnte man sie löschen.
    # Hier ist eine pragmatische Lösung:
    # st.session_state['pdf_to_delete'] = tmp_pdf_path (um es später zu löschen)
    # Aber für Einfachheit wird es hier weggelassen.

    # Wichtiger Hinweis zur Löschung der tmp_pdf_path:
    # Die Datei tmp_pdf_path muss für den Download-Button zugänglich bleiben.
    # Streamlit hält die App nicht an, während der Download-Button angezeigt wird.
    # Ein `os.remove(tmp_pdf_path)` direkt hier würde den Download fehlschlagen lassen.
    # Temporäre Dateien mit `delete=False` werden normalerweise nicht automatisch gelöscht.
    # Der Nutzer ist dafür verantwortlich, oder sie werden beim Systemneustart etc. gelöscht.
    # In einer produktiven Umgebung würde man hierfür eine robustere Strategie wählen.
    # z.B. Dateien in einem Session-spezifischen Temp-Ordner ablegen und diesen Ordner bei Session-Ende löschen.
