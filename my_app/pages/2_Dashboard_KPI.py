from pathlib import Path

import pandas as pd
import streamlit as st


st.title("Dashboard KPI - Détection de fraude")
st.caption("KPI calculés sur un CSV uploadé ou sur le fichier local `creditcard.csv`.")


def _detect_amount_column(df: pd.DataFrame) -> str:
    for col in ["Amount", "amount", "Montant", "montant"]:
        if col in df.columns:
            return col
    raise ValueError("Colonne montant introuvable (ex: Amount).")


def _detect_fraud_flag(df: pd.DataFrame) -> pd.Series:
    candidate = None
    for col in ["fraude_predite", "Class", "class", "fraud", "is_fraud"]:
        if col in df.columns:
            candidate = col
            break

    if candidate is None:
        raise ValueError("Colonne fraude introuvable (ex: Class ou fraude_predite).")

    series = df[candidate]

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "true": "1",
                "false": "0",
                "fraude": "1",
                "fraud": "1",
                "légitime": "0",
                "legitime": "0",
                "legit": "0",
            }
        )
    )
    return normalized.isin(["1", "yes", "y", "oui", "o", "true", "fraude", "fraud"]).astype(int)


def _build_hour_column(df: pd.DataFrame) -> pd.Series:
    if "Time" in df.columns:
        time_num = pd.to_numeric(df["Time"], errors="coerce").fillna(0)
        return ((time_num // 3600) % 24).astype(int)
    return pd.Series(df.index % 24, index=df.index, dtype="int64")


def _load_dashboard_df(uploaded_csv):
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        return df, f"CSV uploadé: {uploaded_csv.name}"

    default_csv = Path(__file__).resolve().parent.parent.parent / "creditcard.csv"
    if not default_csv.exists():
        raise FileNotFoundError("Fichier `creditcard.csv` introuvable.")

    df = pd.read_csv(default_csv)
    return df, "Source locale: creditcard.csv"


uploaded_csv = st.file_uploader("Uploader un CSV pour le dashboard (optionnel)", type="csv")

prediction_available = st.session_state.get("csv_analysis") is not None
use_prediction = False

if prediction_available:
    source_choice = st.radio(
        "Source des données",
        ["Résultats de la dernière prédiction", "creditcard.csv / CSV uploadé"],
        horizontal=True,
    )
    use_prediction = source_choice == "Résultats de la dernière prédiction"

try:
    if use_prediction:
        data = st.session_state["csv_analysis"]
        source_label = "Source: résultats de la dernière prédiction"
    else:
        data, source_label = _load_dashboard_df(uploaded_csv)
    fraud_flag = _detect_fraud_flag(data)
    amount_col = _detect_amount_column(data)
    amount = pd.to_numeric(data[amount_col], errors="coerce").fillna(0.0)
    hours = _build_hour_column(data)

    total_transactions = int(len(data))
    total_frauds = int(fraud_flag.sum())
    montant_total_bloque_xof = float(amount[fraud_flag == 1].sum())
    taux_fraude = (total_frauds / total_transactions * 100) if total_transactions else 0.0

    st.info(source_label)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre total transactions analysées", f"{total_transactions:,}".replace(",", " "))
    c2.metric("Nombre fraudes détectées", f"{total_frauds:,}".replace(",", " "))
    c3.metric("Montant total bloqué (XOF)", f"{montant_total_bloque_xof:,.0f}".replace(",", " "))
    c4.metric("Taux de fraude (%)", f"{taux_fraude:.2f}%")

    st.subheader("Graphique fraudes par heure")
    fraude_par_heure = (
        pd.DataFrame({"heure": hours, "fraude": fraud_flag})
        .query("fraude == 1")
        .groupby("heure")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("Nombre de fraudes")
        .to_frame()
    )
    st.bar_chart(fraude_par_heure)

    st.subheader("Graphique montant moyen fraude vs légitime")
    montant_moyen = (
        pd.DataFrame({"flag": fraud_flag, "amount": amount})
        .groupby("flag", as_index=False)["amount"]
        .mean()
    )
    montant_moyen["Type"] = montant_moyen["flag"].map({0: "LÉGITIME", 1: "FRAUDE"})
    montant_moyen = montant_moyen.set_index("Type")[["amount"]].rename(columns={"amount": "Montant moyen (XOF)"})
    st.bar_chart(montant_moyen)

except Exception as err:
    st.error(f"Impossible de calculer le dashboard : {err}")
