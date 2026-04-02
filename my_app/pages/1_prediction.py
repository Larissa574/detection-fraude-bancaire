from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from history_store import append_history_entry


st.markdown(
    """
### Détection de fraude bancaire

Cette application propose 2 modes:
- Mode 1: saisie manuelle d'une transaction
- Mode 2: upload d'un fichier CSV
"""
)
st.warning("Ce modèle est un outil d'aide et ne remplace pas la vérification humaine.")


MODEL_CACHE_VERSION = "2026-04-02-sklearn-only"


@st.cache_resource
def load_model(cache_version: str = MODEL_CACHE_VERSION):
    _ = cache_version
    base_dir = Path(__file__).resolve().parent.parent.parent
    models_dir = base_dir / "models"

    candidates = ["model_sklearn.joblib", "model.joblib"]
    load_errors = []

    for filename in candidates:
        model_path = models_dir / filename
        if not model_path.exists():
            continue
        try:
            return joblib.load(model_path)
        except ModuleNotFoundError as error:
            load_errors.append(f"{filename}: {error}")
        except Exception as error:
            load_errors.append(f"{filename}: {type(error).__name__}: {error}")

    if load_errors:
        raise RuntimeError(" ; ".join(load_errors))

    raise FileNotFoundError(
        "Aucun modèle exploitable trouvé dans models/. Fichiers attendus: model_sklearn.joblib ou model.joblib"
    )


def analyze_dataframe(df, model, preprocessor, threshold, features):
    missing_columns = [column for column in features if column not in df.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:10])
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing_preview}")

    x_data = df[features]
    x_processed = preprocessor.transform(x_data)
    proba = model.predict_proba(x_processed)[:, 1]
    prediction = (proba >= threshold).astype(int)

    result_df = df.copy()
    result_df["fraude_predite"] = prediction
    result_df["Probabilité fraude (%)"] = (proba * 100).round(2)
    result_df["Résultat"] = result_df["fraude_predite"].map({0: "LÉGITIME", 1: "FRAUDE"})
    return result_df


def generate_pdf_report(lines):
    if not lines:
        lines = ["Rapport vide"]

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    for line in lines:
        content.append(Paragraph(str(line), styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)
    buffer.seek(0)
    return buffer.read()


def build_report_lines(result_df, source_name):
    total_transactions = len(result_df)
    fraude_total = int(result_df["fraude_predite"].sum())
    legit_total = total_transactions - fraude_total
    max_proba = float(result_df["Probabilité fraude (%)"].max()) if total_transactions else 0.0

    lines = [
        "Rapport de détection de fraude bancaire",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {source_name}",
        "",
        f"Transactions analysées: {total_transactions}",
        f"Fraudes détectées: {fraude_total}",
        f"Transactions légitimes: {legit_total}",
        f"Probabilité max observée: {max_proba:.2f}%",
    ]

    return lines


def persist_summary(result_df, mode, source):
    total_transactions = int(len(result_df))
    fraude_total = int(result_df["fraude_predite"].sum())
    blocked_amount = 0.0
    if "Amount" in result_df.columns:
        amount_series = pd.to_numeric(result_df["Amount"], errors="coerce").fillna(0.0)
        blocked_amount = float(amount_series[result_df["fraude_predite"] == 1].sum())

    append_history_entry(
        {
            "mode": mode,
            "source": source,
            "total_transactions": total_transactions,
            "frauds": fraude_total,
            "blocked_amount": blocked_amount,
            "mean_probability": float(result_df["Probabilité fraude (%)"].mean()),
            "max_probability": float(result_df["Probabilité fraude (%)"].max()),
        }
    )


def render_result(analysis_df, report_pdf):
    fraude_total = int(analysis_df["fraude_predite"].sum())
    total_transactions = len(analysis_df)

    if fraude_total > 0:
        st.error(f"FRAUDE - {fraude_total} transaction(s) suspecte(s) détectée(s)")
    else:
        st.success("LÉGITIME - aucune transaction suspecte détectée")

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions analysées", total_transactions)
    col2.metric("Fraudes détectées", fraude_total)
    col3.metric("Probabilité moyenne", f"{analysis_df['Probabilité fraude (%)'].mean():.2f}%")

    st.subheader("Résultats par transaction")
    visible_columns = ["Résultat", "Probabilité fraude (%)"]
    for optional_col in ["Amount", "Time"]:
        if optional_col in analysis_df.columns:
            visible_columns.append(optional_col)
    st.dataframe(analysis_df[visible_columns], use_container_width=True)

    if report_pdf is not None:
        report_name = f"rapport_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            "Télécharger le rapport PDF",
            data=report_pdf,
            file_name=report_name,
            mime="application/pdf",
            use_container_width=True,
        )


if "last_analysis_df" not in st.session_state:
    st.session_state["last_analysis_df"] = None
if "last_report_pdf" not in st.session_state:
    st.session_state["last_report_pdf"] = None


try:
    artifact = load_model()
except Exception as load_err:
    st.error(f"Impossible de charger le modèle: {load_err}")
    st.stop()

modele = artifact["model"]
preprocessor = artifact["preprocessor"]
threshold = float(artifact["threshold"])
features = artifact["features"]

tab_manual, tab_csv = st.tabs(["Mode 1 - Saisie manuelle", "Mode 2 - Upload CSV"])

with tab_manual:
    st.markdown("#### Entrer les valeurs de transaction")
    st.caption("L'analyse démarre uniquement quand vous cliquez sur le bouton 'Analyser la transaction'.")

    with st.form("manual_prediction_form", enter_to_submit=False):
        form_values = {}
        cols = st.columns(3)

        for index, feature_name in enumerate(features):
            default_value = 0.0
            step_value = 0.01
            if feature_name == "Amount":
                default_value = 100.0
            if feature_name == "Time":
                step_value = 1.0

            with cols[index % 3]:
                form_values[feature_name] = st.number_input(
                    feature_name,
                    value=float(default_value),
                    step=float(step_value),
                    format="%.6f",
                )

        submit_manual = st.form_submit_button("Analyser la transaction")

    if submit_manual:
        try:
            input_df = pd.DataFrame([form_values], columns=features)
            result_df = analyze_dataframe(input_df, modele, preprocessor, threshold, features)
            report_lines = build_report_lines(result_df, "saisie_manuelle")
            report_pdf = generate_pdf_report(report_lines)

            st.session_state["last_analysis_df"] = result_df
            st.session_state["last_report_pdf"] = report_pdf
            st.session_state["csv_analysis"] = result_df

            persist_summary(result_df, mode="manuel", source="saisie_manuelle")
            st.success("Analyse manuelle terminée")
        except Exception as manual_err:
            st.error(f"Erreur pendant l'analyse manuelle: {manual_err}")

with tab_csv:
    st.markdown("#### Upload d'un fichier CSV")
    csv_file = st.file_uploader(
        "Glissez et déposez un fichier CSV",
        type="csv",
        help="Le fichier doit contenir les colonnes utilisées par le modèle.",
    )

    if csv_file is not None:
        try:
            source_df = pd.read_csv(csv_file)
            st.subheader("Aperçu des données")
            st.dataframe(source_df.head(), use_container_width=True)
            st.write("Nombre de transactions:", source_df.shape[0])

            if st.button("Lancer l'analyse CSV"):
                result_df = analyze_dataframe(source_df, modele, preprocessor, threshold, features)
                report_lines = build_report_lines(result_df, csv_file.name)
                report_pdf = generate_pdf_report(report_lines)

                st.session_state["last_analysis_df"] = result_df
                st.session_state["last_report_pdf"] = report_pdf
                st.session_state["csv_analysis"] = result_df

                persist_summary(result_df, mode="csv", source=csv_file.name)
                st.success("Analyse CSV terminée")
        except Exception as csv_err:
            st.error(f"Erreur pendant l'analyse CSV: {csv_err}")
    else:
        st.info("Veuillez uploader un fichier CSV pour lancer l'analyse.")


if st.session_state["last_analysis_df"] is not None:
    render_result(st.session_state["last_analysis_df"], st.session_state["last_report_pdf"])