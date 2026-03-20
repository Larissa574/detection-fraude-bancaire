from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve


st.title("Performance du modèle")
st.caption("Évaluation du modèle sur le jeu de données labellisé avec seuil de décision ajustable en temps réel.")


@st.cache_resource
def load_model_artifact():
	model_path = Path(__file__).resolve().parent.parent.parent / "models" / "model.joblib"
	if not model_path.exists():
		st.error("Fichier modèle introuvable: `models/model.joblib`.")
		st.stop()
	return joblib.load(model_path)


@st.cache_data(show_spinner=True)
def load_evaluation_scores():
	artifact = load_model_artifact()
	csv_path = Path(__file__).resolve().parent.parent.parent / "creditcard.csv"

	if not csv_path.exists():
		raise FileNotFoundError("Fichier `creditcard.csv` introuvable.")

	df = pd.read_csv(csv_path)

	features = artifact["features"]
	missing_columns = [column for column in features + ["Class"] if column not in df.columns]
	if missing_columns:
		raise ValueError(f"Colonnes manquantes dans `creditcard.csv`: {', '.join(missing_columns)}")

	X = df[features]
	y_true = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)

	X_processed = artifact["preprocessor"].transform(X)
	y_score = artifact["model"].predict_proba(X_processed)[:, 1]

	fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)

	return {
		"y_true": y_true,
		"y_score": y_score,
		"fpr": fpr,
		"tpr": tpr,
		"roc_thresholds": roc_thresholds,
		"roc_auc": roc_auc,
		"default_threshold": float(artifact["threshold"]),
		"total_rows": int(len(df)),
	}


def compute_threshold_metrics(y_true, y_score, threshold):
	y_pred = (y_score >= threshold).astype(int)
	matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
	precision = precision_score(y_true, y_pred, zero_division=0)
	recall = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred, zero_division=0)
	return {
		"y_pred": y_pred,
		"matrix": matrix,
		"precision": precision,
		"recall": recall,
		"f1": f1,
	}


def build_confusion_matrix_figure(matrix):
	labels_x = ["Prédit légitime", "Prédit fraude"]
	labels_y = ["Réel légitime", "Réel fraude"]

	figure = go.Figure(
		data=go.Heatmap(
			z=matrix,
			x=labels_x,
			y=labels_y,
			text=matrix,
			texttemplate="%{text}",
			colorscale=[[0.0, "#E8EEF7"], [1.0, "#C62828"]],
			hovertemplate="%{y}<br>%{x}<br>Valeur: %{z}<extra></extra>",
			showscale=False,
		)
	)
	figure.update_layout(
		margin=dict(l=20, r=20, t=30, b=20),
		height=420,
		xaxis_title="Prédiction",
		yaxis_title="Vérité terrain",
		template="plotly_white",
	)
	return figure


def build_roc_figure(fpr, tpr, roc_auc, roc_thresholds, selected_threshold):
	operating_index = min(
		range(len(roc_thresholds)),
		key=lambda index: abs(float(roc_thresholds[index]) - selected_threshold),
	)

	figure = go.Figure()
	figure.add_trace(
		go.Scatter(
			x=fpr,
			y=tpr,
			mode="lines",
			name=f"ROC (AUC = {roc_auc:.4f})",
			line=dict(color="#0B5ED7", width=3),
		)
	)
	figure.add_trace(
		go.Scatter(
			x=[0, 1],
			y=[0, 1],
			mode="lines",
			name="Référence aléatoire",
			line=dict(color="#8A8F98", width=2, dash="dash"),
		)
	)
	figure.add_trace(
		go.Scatter(
			x=[fpr[operating_index]],
			y=[tpr[operating_index]],
			mode="markers",
			name=f"Seuil {selected_threshold:.2f}",
			marker=dict(size=12, color="#C62828"),
			hovertemplate=(
				f"Seuil: {selected_threshold:.2f}<br>FPR: {fpr[operating_index]:.4f}"
				f"<br>TPR: {tpr[operating_index]:.4f}<extra></extra>"
			),
		)
	)
	figure.update_layout(
		margin=dict(l=20, r=20, t=30, b=20),
		height=420,
		xaxis_title="False Positive Rate",
		yaxis_title="True Positive Rate",
		template="plotly_white",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return figure


try:
	evaluation = load_evaluation_scores()
except Exception as error:
	st.error(f"Impossible de charger les données de performance : {error}")
	st.stop()


selected_threshold = st.slider(
	"Seuil de décision",
	min_value=0.0,
	max_value=1.0,
	value=float(evaluation["default_threshold"]),
	step=0.01,
	help="Une transaction est classée comme fraude si sa probabilité est supérieure ou égale à ce seuil.",
)

metrics = compute_threshold_metrics(evaluation["y_true"], evaluation["y_score"], selected_threshold)
tn, fp, fn, tp = metrics["matrix"].ravel()

top1, top2, top3, top4 = st.columns(4)
top1.metric("Precision", f"{metrics['precision']:.3f}")
top2.metric("Recall", f"{metrics['recall']:.3f}")
top3.metric("F1-score", f"{metrics['f1']:.3f}")
top4.metric("AUC ROC", f"{evaluation['roc_auc']:.3f}")

st.caption(
	(
		f"Transactions évaluées: {evaluation['total_rows']:,}. "
		f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}"
	).replace(",", " ")
)

left_col, right_col = st.columns(2)

with left_col:
	st.subheader("Matrice de confusion interactive")
	st.plotly_chart(build_confusion_matrix_figure(metrics["matrix"]), use_container_width=True)

with right_col:
	st.subheader("Courbe ROC")
	st.plotly_chart(
		build_roc_figure(
			evaluation["fpr"],
			evaluation["tpr"],
			evaluation["roc_auc"],
			evaluation["roc_thresholds"],
			selected_threshold,
		),
		use_container_width=True,
	)
