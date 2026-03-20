import csv
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="AfriBank - Détection de fraude", layout="wide")


def apply_navy_white_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #FFFFFF;
                color: #001F3F;
            }

            h1, h2, h3, h4, h5, h6, p, label, span {
                color: #001F3F;
            }

            [data-testid="stSidebar"] {
                background-color: #001F3F;
            }

            [data-testid="stSidebar"] * {
                color: #FFFFFF !important;
            }

            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid rgba(0, 31, 63, 0.2);
                border-radius: 12px;
                padding: 10px;
            }

            div.stAlert {
                border-left: 4px solid #001F3F;
            }

            div.stButton > button {
                background-color: #001F3F;
                color: #FFFFFF;
                border: 1px solid #001F3F;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_kpis():
    csv_path = Path(__file__).resolve().parent.parent / "creditcard.csv"

    if not csv_path.exists():
        return None

    transactions = 0
    fraudes = 0
    montant_bloque = 0.0

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            transactions += 1
            is_fraud = row.get("Class", "0").strip() == "1"
            if is_fraud:
                fraudes += 1
                try:
                    montant_bloque += float(row.get("Amount", 0.0) or 0.0)
                except ValueError:
                    pass

    return {
        "transactions": transactions,
        "fraudes": fraudes,
        "montant_bloque": montant_bloque,
    }


def main():
    apply_navy_white_theme()

    logo_path = Path(__file__).resolve().parent / "image.jpg"

    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=110)
    with col_title:
        st.title("AfriBank")
        st.caption("Plateforme de détection de fraude bancaire")

    kpis = _load_kpis()

    if kpis is None:
        st.error("Fichier `creditcard.csv` introuvable. Les KPI ne peuvent pas être calculés.")
        return

    transactions = kpis["transactions"]
    fraudes = kpis["fraudes"]
    montant_bloque = kpis["montant_bloque"]
    taux_fraude = (fraudes / transactions * 100) if transactions else 0.0
    ticket_moyen = (montant_bloque / fraudes) if fraudes else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analysées", f"{transactions:,}".replace(",", " "))
    c2.metric("Fraudes détectées", f"{fraudes:,}".replace(",", " "))
    c3.metric("Montant total bloqué (XOF)", f"{montant_bloque:,.0f}".replace(",", " "))

    s1, s2 = st.columns(2)
    s1.metric("Taux de fraude", f"{taux_fraude:.3f}%")
    s2.metric("Montant moyen bloqué / fraude (XOF)", f"{ticket_moyen:,.0f}".replace(",", " "))

    st.markdown(
        """
        ### Description
        AfriBank est un outil d'aide à la détection de transactions frauduleuses.
        Les indicateurs ci-dessus sont calculés directement sur le jeu de données `creditcard.csv`.
        """
    )

    st.success("👈 Utilisez le menu de gauche pour accéder aux modules de prédiction, performance et dashboard KPI.")


if __name__ == "__main__":
    main()
