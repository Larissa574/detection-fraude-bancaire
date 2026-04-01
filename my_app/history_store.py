import json
from datetime import datetime
from pathlib import Path
import re
import uuid
from typing import Dict, List

import streamlit as st

_HISTORY_DIR = Path(__file__).resolve().parent / "analysis_history"
_HISTORY_SCOPE_KEY = "history_scope"
_HISTORY_SCOPE_QUERY_PARAM = "history_scope"


def _normalize_legacy_entry(day_key: str, payload: Dict) -> Dict:
    return {
        "timestamp": payload.get("updated_at", day_key),
        "date": day_key,
        "mode": payload.get("mode", "legacy"),
        "source": payload.get("source", "historique_legacy"),
        "total_transactions": int(payload.get("transactions", 0) or 0),
        "frauds": int(payload.get("fraudes", 0) or 0),
        "blocked_amount": float(payload.get("montant_bloque", 0.0) or 0.0),
        "mean_probability": float(payload.get("mean_probability", 0.0) or 0.0),
        "max_probability": float(payload.get("max_probability", 0.0) or 0.0),
    }


def _sanitize_scope(raw_scope: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", raw_scope).strip("_")
    return cleaned[:64]


def _get_history_scope() -> str:
    query_scope = None

    try:
        query_scope = st.query_params.get(_HISTORY_SCOPE_QUERY_PARAM)
    except Exception:
        query_scope = None

    if isinstance(query_scope, list):
        query_scope = query_scope[0] if query_scope else None

    if query_scope:
        sanitized_query_scope = _sanitize_scope(str(query_scope))
        if sanitized_query_scope:
            try:
                st.session_state[_HISTORY_SCOPE_KEY] = sanitized_query_scope
            except Exception:
                pass
            return sanitized_query_scope

    session_scope = None
    try:
        session_scope = st.session_state.get(_HISTORY_SCOPE_KEY)
    except Exception:
        session_scope = None

    if not session_scope:
        session_scope = uuid.uuid4().hex
        try:
            st.session_state[_HISTORY_SCOPE_KEY] = session_scope
        except Exception:
            pass

    sanitized_session_scope = _sanitize_scope(str(session_scope)) or uuid.uuid4().hex

    try:
        st.query_params[_HISTORY_SCOPE_QUERY_PARAM] = sanitized_session_scope
    except Exception:
        pass

    return sanitized_session_scope


def _history_file() -> Path:
    scope = _get_history_scope()
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR / f"history_{scope}.json"


def load_history() -> List[Dict]:
    history_file = _history_file()
    if not history_file.exists():
        return []

    try:
        raw = json.loads(history_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(raw, list):
        entries = [entry for entry in raw if isinstance(entry, dict)]
    elif isinstance(raw, dict):
        entries = [_normalize_legacy_entry(key, value) for key, value in raw.items() if isinstance(value, dict)]
    else:
        entries = []

    return sorted(entries, key=lambda entry: str(entry.get("timestamp", "")))


def save_history(entries: List[Dict]) -> None:
    history_file = _history_file()
    history_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history_entry(entry: Dict, keep_last: int = 1000) -> None:
    entries = load_history()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = {
        "timestamp": timestamp,
        "date": timestamp[:10],
        "mode": entry.get("mode", "inconnu"),
        "source": entry.get("source", "inconnu"),
        "total_transactions": int(entry.get("total_transactions", 0) or 0),
        "frauds": int(entry.get("frauds", 0) or 0),
        "blocked_amount": float(entry.get("blocked_amount", 0.0) or 0.0),
        "mean_probability": float(entry.get("mean_probability", 0.0) or 0.0),
        "max_probability": float(entry.get("max_probability", 0.0) or 0.0),
    }

    entries.append(normalized)
    if keep_last > 0:
        entries = entries[-keep_last:]

    save_history(entries)


def compute_kpis(entries: List[Dict]) -> Dict:
    total_analyses = len(entries)
    total_transactions = sum(int(entry.get("total_transactions", 0) or 0) for entry in entries)
    total_frauds = sum(int(entry.get("frauds", 0) or 0) for entry in entries)
    total_blocked_amount = sum(float(entry.get("blocked_amount", 0.0) or 0.0) for entry in entries)

    fraud_rate = (total_frauds / total_transactions * 100) if total_transactions else 0.0
    avg_blocked_amount = (total_blocked_amount / total_frauds) if total_frauds else 0.0

    return {
        "total_analyses": total_analyses,
        "total_transactions": total_transactions,
        "total_frauds": total_frauds,
        "total_blocked_amount": total_blocked_amount,
        "fraud_rate": fraud_rate,
        "avg_blocked_amount": avg_blocked_amount,
    }
