import json
from pathlib import Path

import joblib
import plotly.graph_objects as go
import streamlit as st


st.title("Performance du modèle")
st.caption("Évaluation du modèle sur le jeu de données labellisé avec seuil de décision ajustable en temps réel.")


MODEL_CACHE_VERSION = "2026-04-02-sklearn-only"


@st.cache_resource
def load_model_artifact(cache_version: str = MODEL_CACHE_VERSION):
	_ = cache_version
	models_dir = Path(__file__).resolve().parent.parent.parent / "models"
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


@st.cache_data(show_spinner=True)
def load_evaluation_scores():
	artifact = load_model_artifact()
	summary_path = Path(__file__).resolve().parent.parent.parent / "models" / "performance_summary.json"

	if summary_path.exists():
		summary = json.loads(summary_path.read_text(encoding="utf-8"))
		threshold_metrics = summary.get("threshold_metrics", [])
		roc_points = summary.get("roc_points", [])
		if not threshold_metrics or not roc_points:
			raise ValueError("Le fichier `models/performance_summary.json` est incomplet.")
		return {
			"threshold_metrics": threshold_metrics,
			"roc_points": roc_points,
			"roc_auc": float(summary.get("roc_auc", 0.0)),
			"default_threshold": float(summary.get("default_threshold", artifact["threshold"])),
			"total_rows": int(summary.get("total_rows", 0)),
			"positive_rows": int(summary.get("positive_rows", 0)),
			"negative_rows": int(summary.get("negative_rows", 0)),
		}

	raise FileNotFoundError("Fichier `models/performance_summary.json` introuvable.")


def compute_threshold_metrics(summary, threshold):
	selected_threshold = round(float(threshold), 2)
	metric_entry = next(
		(entry for entry in summary["threshold_metrics"] if round(float(entry["threshold"]), 2) == selected_threshold),
		None,
	)
	if metric_entry is None:
		metric_entry = min(
			summary["threshold_metrics"],
			key=lambda entry: abs(float(entry["threshold"]) - selected_threshold),
		)

	matrix = [[int(metric_entry["tn"]), int(metric_entry["fp"])], [int(metric_entry["fn"]), int(metric_entry["tp"])] ]
	precision = float(metric_entry["precision"])
	recall = float(metric_entry["recall"])
	f1 = float(metric_entry["f1"])
	return {
		"matrix": matrix,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"threshold": selected_threshold,
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


def build_roc_figure(roc_points, roc_auc, selected_threshold):
	fpr = [float(point["fpr"]) for point in roc_points]
	tpr = [float(point["tpr"]) for point in roc_points]
	thresholds = [float(point["threshold"]) for point in roc_points]
	operating_index = min(range(len(thresholds)), key=lambda index: abs(thresholds[index] - selected_threshold))

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

metrics = compute_threshold_metrics(evaluation, selected_threshold)
tn, fp = metrics["matrix"][0]
fn, tp = metrics["matrix"][1]

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
		build_roc_figure(evaluation["roc_points"], evaluation["roc_auc"], selected_threshold),
		use_container_width=True,
	)
