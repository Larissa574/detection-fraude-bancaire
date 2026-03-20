import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# -------------------------
# Interface
# -------------------------

st.markdown("""
### Détection de fraude bancaire

Bienvenue. Cette application aide la banque à détecter les transactions frauduleuses.
Veuillez uploader un fichier CSV contenant les transactions (Mode 2).
""")

st.warning("⚠️ Ce modèle est un outil d'aide et ne remplace pas la vérification humaine.")

st.markdown("#### Mode 2 : Upload fichier CSV")

# -------------------------
# Upload fichier
# -------------------------

csv = st.file_uploader(
    "Glissez et déposez un fichier CSV",
    type="csv",
    help="Seuls les fichiers CSV sont autorisés"
)

# -------------------------
# Chargement du modèle
# -------------------------

@st.cache_resource
def load_model():

    base_dir = Path(__file__).resolve().parent.parent.parent
    models_dir = base_dir / "models"

    preferred_model = models_dir / "model.joblib"
    if preferred_model.exists():
        return joblib.load(preferred_model)

    model_files = list(models_dir.glob("*.joblib"))

    if not model_files:
        st.error("Aucun modèle trouvé dans le dossier models.")
        st.stop()

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

    return joblib.load(latest_model)


artifact = load_model()

modele = artifact["model"]
preprocessor = artifact["preprocessor"]
threshold = artifact["threshold"]
features = artifact["features"]


def analyze_dataframe(df):
    missing_columns = [column for column in features if column not in df.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:10])
        raise ValueError(
            f"Colonnes manquantes dans le CSV: {missing_preview}"
        )

    X = df[features]
    X_processed = preprocessor.transform(X)
    proba = modele.predict_proba(X_processed)[:, 1]
    prediction = (proba >= threshold).astype(int)

    result_df = df.copy()
    result_df["fraude_predite"] = prediction
    result_df["Probabilité fraude (%)"] = (proba * 100).round(2)
    result_df["Résultat"] = result_df["fraude_predite"].map(
        {0: "✅ LÉGITIME", 1: "🚨 FRAUDE"}
    )
    return result_df

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO


def generate_pdf_report(lines):
    if not lines:
        lines = ["Rapport vide"]

    buffer = BytesIO()  # fichier en mémoire (important pour Streamlit)

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    content = []

    for line in lines:
        content.append(Paragraph(str(line), styles["Normal"]))
        content.append(Spacer(1, 8))  # petit espace entre lignes

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
        f"Fichier source: {source_name}",
        "",
        f"Transactions analysées: {total_transactions}",
        f"Fraudes détectées: {fraude_total}",
        f"Transactions légitimes: {legit_total}",
        f"Probabilité max observée: {max_proba:.2f}%",
        "",
        "Top transactions suspectes:",
    ]

    suspect_rows = result_df[result_df["fraude_predite"] == 1].copy()
    suspect_rows = suspect_rows.sort_values("Probabilité fraude (%)", ascending=False).head(20)

    if suspect_rows.empty:
        lines.append("Aucune transaction frauduleuse détectée.")
    else:
        for row_index, row in suspect_rows.iterrows():
            lines.append(
                f"- Ligne {row_index + 1}: probabilité {row['Probabilité fraude (%)']:.2f}%"
            )

    return lines


if "csv_analysis" not in st.session_state:
    st.session_state["csv_analysis"] = None
if "csv_file_signature" not in st.session_state:
    st.session_state["csv_file_signature"] = None
if "csv_report_pdf" not in st.session_state:
    st.session_state["csv_report_pdf"] = None

# -------------------------
# Analyse des données
# -------------------------

if csv is not None:

    file_signature = f"{csv.name}-{csv.size}"
    if st.session_state["csv_file_signature"] != file_signature:
        st.session_state["csv_file_signature"] = file_signature
        st.session_state["csv_analysis"] = None
        st.session_state["csv_report_pdf"] = None

    df = pd.read_csv(csv)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    st.write("Nombre de transactions :", df.shape[0])

    if st.button("Lancer l'analyse de fraude"):

        try:
            analysis_df = analyze_dataframe(df)
            st.session_state["csv_analysis"] = analysis_df
            report_lines = build_report_lines(analysis_df, csv.name)
            st.session_state["csv_report_pdf"] = generate_pdf_report(report_lines)
            st.success("Analyse terminée")

        except Exception as e:
            st.error(f"Erreur pendant l'analyse : {e}")

    analysis_df = st.session_state["csv_analysis"]
    if analysis_df is not None:
        fraude_total = int(analysis_df["fraude_predite"].sum())
        total_transactions = len(analysis_df)

        if fraude_total > 0:
            st.error(f"🚨 FRAUDE — {fraude_total} transaction(s) suspecte(s) détectée(s)")
        else:
            st.success("✅ LÉGITIME — aucune transaction suspecte détectée")

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions analysées", total_transactions)
        col2.metric("Fraudes détectées", fraude_total)
        col3.metric(
            "Probabilité moyenne",
            f"{analysis_df['Probabilité fraude (%)'].mean():.2f}%",
        )

        st.subheader("Résultats par transaction")
        visible_columns = ["Résultat", "Probabilité fraude (%)"]
        for optional_col in ["Amount", "Time"]:
            if optional_col in analysis_df.columns:
                visible_columns.append(optional_col)

        st.dataframe(analysis_df[visible_columns])

        if st.session_state["csv_report_pdf"] is not None:
            report_name = f"rapport_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                "Télécharger le rapport PDF",
                data=st.session_state["csv_report_pdf"],
                file_name=report_name,
                mime="application/pdf",
                use_container_width=True,
            )

else:
    st.info("Veuillez uploader un fichier CSV pour lancer l'analyse.")