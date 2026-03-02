from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from model import LogAutoencoder, export_onnx, reconstruction_errors, train_autoencoder
from transformer_model import (
    Transformer,
    export_transformer_onnx,
    train_transformer_autoencoder,
    transformer_reconstruction_errors,
)
from device_utils import resolve_torch_device
from preprocessor import (
    build_vectorizer,
    infer_endpoint,
    infer_severity,
    infer_weak_label,
    merge_log_lines,
    normalize_log_line,
    prepare_lines_for_model,
    read_log_file,
    remove_placeholder_tokens,
    summarize_preprocessing,
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
ARTIFACT_DIR = BASE_DIR / "artifacts"
for target_dir in (UPLOAD_DIR, ARTIFACT_DIR):
    target_dir.mkdir(parents=True, exist_ok=True)


class TrainRequest(BaseModel):
    model_name: str = Field(min_length=2, max_length=60)
    model_type: Literal["autoencoder", "transformer"] = Field(default="autoencoder")
    selected_logs: list[str] = Field(default_factory=list)
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=32, ge=4, le=1024)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)
    hidden_dim: int = Field(default=64, ge=16, le=2048)
    transformer_d_model: int = Field(default=64, ge=16, le=1024)
    transformer_num_heads: int = Field(default=8, ge=1, le=16)
    transformer_num_layers: int = Field(default=2, ge=1, le=12)
    transformer_d_ff: int = Field(default=256, ge=32, le=4096)
    transformer_dropout: float = Field(default=0.1, ge=0.0, le=0.8)
    max_features: int = Field(default=256, ge=32, le=10000)
    threshold_quantile: float = Field(default=0.95, ge=0.5, le=0.999)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=42, ge=0, le=2_147_483_647)
    training_rounds: int = Field(default=5, ge=1, le=10)


class AnalyzeRequest(BaseModel):
    model_name: str
    selected_logs: list[str] = Field(default_factory=list)


class PreprocessPreviewRequest(BaseModel):
    selected_logs: list[str] = Field(default_factory=list)
    sample_limit: int = Field(default=25, ge=1, le=100)


app = FastAPI(title="Server Log Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']} (type: {error['type']})")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validierungsfehler bei der Anfrage",
            "errors": errors,
            "body": str(exc.body) if hasattr(exc, 'body') else None
        },
    )


def _model_paths(model_name: str) -> dict[str, Path]:
    model_dir = ARTIFACT_DIR / model_name
    return {
        "dir": model_dir,
        "weights": model_dir / "model.pt",
        "vectorizer": model_dir / "vectorizer.pkl",
        "meta": model_dir / "meta.json",
        "onnx": model_dir / "model.onnx",
    }


def _resolve_unique_model_name(requested_name: str) -> str:
    def is_available(name: str) -> bool:
        paths = _model_paths(name)
        return not paths["meta"].exists() and not paths["weights"].exists() and not paths["onnx"].exists()

    if is_available(requested_name):
        return requested_name

    match = re.match(r"^(.*?)(?:_v)?(\d+)$", requested_name)
    if match:
        prefix = match.group(1).rstrip("_")
        counter = int(match.group(2))
        while True:
            counter += 1
            candidate = f"{prefix}_{counter}"
            if is_available(candidate):
                return candidate

    counter = 2
    while True:
        candidate = f"{requested_name}_{counter}"
        if is_available(candidate):
            return candidate
        counter += 1


def _chart_label_for_line(index: int, line: str) -> str:
    endpoint = infer_endpoint(line)
    if endpoint:
        return endpoint
    compact_line = normalize_log_line(line)
    compact_line = re.sub(r"\s+", " ", compact_line.strip())
    if len(compact_line) > 36:
        compact_line = f"{compact_line[:36]}..."
    return f"#{index} {compact_line}" if compact_line else f"#{index}"


def _line_is_error(line: str, severity: str) -> bool:
    lowered = line.lower()
    if severity in {"critical", "error"}:
        return True
    return any(token in lowered for token in (" exception", " traceback", " failed", " error -", " 500", " 502", " 503", " 504"))


def _classify_error_context(line: str, severity: str) -> str:
    if not _line_is_error(line, severity):
        return "none"

    lowered = line.lower()

    if re.search(r"\b4\d\d\b", lowered):
        return "client"

    client_markers = (
        "missing authorization header",
        "noauthorizationerror",
        "unauthorized",
        "forbidden",
        "permission denied",
        "validationerror",
        "string should match pattern",
        "badrequest",
        "invalid",
    )
    if any(marker in lowered for marker in client_markers):
        return "client"

    server_markers = (
        " traceback",
        " 500",
        " 502",
        " 503",
        " 504",
        "connectionerror",
        "timeout",
        "failed to resolve",
    )
    if any(marker in lowered for marker in server_markers):
        return "server"

    if severity in {"critical", "error"}:
        return "server"

    return "client"


def _is_benign_operational_event(line: str, severity: str, weak_label: str) -> bool:
    if severity != "info":
        return False
    if weak_label != "endpoint_normal":
        return False
    if _line_is_error(line, severity):
        return False
    lowered = line.lower()
    benign_tokens = (
        "socket disconnect",
        "[ws] verbunden",
        "jwt-auth ok",
        "connect start",
        "session lookup",
        "cookie-auth ok",
    )
    return any(token in lowered for token in benign_tokens)


def _is_likely_normal_endpoint_info(
    line: str,
    severity: str,
    weak_label: str,
    score: float,
    threshold: float,
    endpoint_zscore: float,
) -> bool:
    if severity != "info" or weak_label != "endpoint_normal":
        return False
    if _line_is_error(line, severity):
        return False

    if score <= threshold * 1.35:
        return True

    return endpoint_zscore < 4.0


def _build_endpoint_insights(lines: list[str], anomaly_indices: set[int]) -> dict[str, Any]:
    stats: dict[str, dict[str, float | int | str]] = {}
    global_anomaly_rate = len(anomaly_indices) / max(1, len(lines))
    model_signal_reliable = 0.02 <= global_anomaly_rate <= 0.70
    reliability_reason = "ok"
    if global_anomaly_rate > 0.70:
        reliability_reason = "saturated_high"
    elif global_anomaly_rate < 0.02:
        reliability_reason = "saturated_low"

    for index, line in enumerate(lines):
        endpoint_key = infer_endpoint(line)
        if not endpoint_key:
            continue

        severity = infer_severity(line)
        is_error = _line_is_error(line, severity)
        is_anomaly = index in anomaly_indices

        if endpoint_key not in stats:
            stats[endpoint_key] = {
                "endpoint": endpoint_key,
                "total_hits": 0,
                "error_hits": 0,
                "server_error_hits": 0,
                "client_error_hits": 0,
                "anomaly_hits": 0,
                "error_rate": 0.0,
                "server_error_rate": 0.0,
                "client_error_rate": 0.0,
                "anomaly_rate": 0.0,
            }

        stats[endpoint_key]["total_hits"] = int(stats[endpoint_key]["total_hits"]) + 1
        if is_error:
            stats[endpoint_key]["error_hits"] = int(stats[endpoint_key]["error_hits"]) + 1
            error_context = _classify_error_context(line, severity)
            if error_context == "server":
                stats[endpoint_key]["server_error_hits"] = int(stats[endpoint_key]["server_error_hits"]) + 1
            elif error_context == "client":
                stats[endpoint_key]["client_error_hits"] = int(stats[endpoint_key]["client_error_hits"]) + 1
        if is_anomaly:
            stats[endpoint_key]["anomaly_hits"] = int(stats[endpoint_key]["anomaly_hits"]) + 1

    entries = list(stats.values())
    for entry in entries:
        total_hits = int(entry["total_hits"])
        error_hits = int(entry["error_hits"])
        server_error_hits = int(entry.get("server_error_hits", 0))
        client_error_hits = int(entry.get("client_error_hits", 0))
        anomaly_hits = int(entry["anomaly_hits"])
        entry["error_rate"] = error_hits / max(1, total_hits)
        entry["server_error_rate"] = server_error_hits / max(1, total_hits)
        entry["client_error_rate"] = client_error_hits / max(1, total_hits)
        entry["anomaly_rate"] = anomaly_hits / max(1, total_hits)
        entry["anomaly_lift"] = float(entry["anomaly_rate"]) - global_anomaly_rate

        if float(entry["server_error_rate"]) >= 0.10 and total_hits >= 3:
            entry["stability_status"] = "risky"
            entry["status_reason"] = "high_server_error_rate"
        elif float(entry["server_error_rate"]) >= 0.03 and total_hits >= 3:
            entry["stability_status"] = "warn"
            entry["status_reason"] = "elevated_server_error_rate"
        elif float(entry["error_rate"]) > 0 and float(entry["server_error_rate"]) == 0.0 and total_hits >= 3:
            entry["stability_status"] = "unknown"
            entry["status_reason"] = "client_error_dominant"
        elif total_hits < 10:
            entry["stability_status"] = "unknown"
            entry["status_reason"] = "insufficient_samples"
        elif float(entry["anomaly_rate"]) > min(0.25, global_anomaly_rate + 0.10):
            entry["stability_status"] = "warn"
            entry["status_reason"] = "anomaly_high_without_errors"
        elif model_signal_reliable and float(entry["anomaly_lift"]) >= 0.15 and anomaly_hits >= 3 and total_hits >= 5:
            entry["stability_status"] = "warn"
            entry["status_reason"] = "anomaly_above_baseline"
        elif total_hits >= 10 and float(entry["error_rate"]) == 0.0:
            entry["stability_status"] = "healthy"
            entry["status_reason"] = "no_errors_observed"
        else:
            entry["stability_status"] = "unknown"
            entry["status_reason"] = "insufficient_signal"

    frequent_endpoints = sorted(entries, key=lambda item: int(item["total_hits"]), reverse=True)[:10]
    healthy_endpoints = sorted(
        [
            item for item in entries
            if str(item.get("stability_status", "")) == "healthy"
        ],
        key=lambda item: int(item["total_hits"]),
        reverse=True,
    )[:10]
    risky_endpoints = sorted(
        [
            item for item in entries
            if str(item.get("stability_status", "")) in {"warn", "risky"}
        ],
        key=lambda item: (
            float(item.get("server_error_rate", 0.0)),
            float(item["error_rate"]),
            float(item.get("anomaly_lift", 0.0)),
            int(item.get("server_error_hits", 0)),
            int(item["error_hits"]),
            float(item["anomaly_rate"]),
        ),
        reverse=True,
    )[:10]

    return {
        "total_detected_endpoints": len(entries),
        "global_anomaly_rate": global_anomaly_rate,
        "model_signal_reliable": model_signal_reliable,
        "model_signal_reason": reliability_reason,
        "frequent_endpoints": frequent_endpoints,
        "healthy_endpoints": healthy_endpoints,
        "risky_endpoints": risky_endpoints,
    }


def _load_logs(selected_logs: list[str]) -> list[str]:
    lines, _ = _load_logs_with_sources(selected_logs)
    return lines


def _load_logs_with_sources(selected_logs: list[str]) -> tuple[list[str], list[str]]:
    print(f"[DEBUG] _load_logs aufgerufen mit: {selected_logs}")
    if not selected_logs:
        raise HTTPException(status_code=400, detail="Bitte mindestens eine Log-Datei auswählen.")

    merged_lines: list[str] = []
    line_sources: list[str] = []
    empty_or_invalid_files: list[str] = []
    for log_file in selected_logs:
        source = UPLOAD_DIR / log_file
        print(f"[DEBUG] Prüfe Log-Datei: {source} (exists: {source.exists()})")
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"Log-Datei nicht gefunden: {log_file}")
        lines_for_file = read_log_file(source)
        if not lines_for_file:
            empty_or_invalid_files.append(log_file)
            continue
        merged_lines.extend(lines_for_file)
        line_sources.extend([log_file] * len(lines_for_file))

    lines = merged_lines
    print(f"[DEBUG] Anzahl gelesener Zeilen: {len(lines)}")
    if not lines:
        if empty_or_invalid_files:
            files_str = ", ".join(empty_or_invalid_files)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Aus den gewählten Logs konnten keine Zeilen gelesen werden. "
                    f"Leere/ungültige Dateien: {files_str}."
                ),
            )
        raise HTTPException(status_code=400, detail="Aus den gewählten Logs konnten keine Zeilen gelesen werden.")
    return lines, line_sources


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-logs")
async def upload_logs(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    saved: list[str] = []
    skipped_empty: list[str] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".csv", ".json"}:
            continue
        file_bytes = await file.read()
        if not file_bytes or not file_bytes.strip():
            skipped_empty.append(file.filename or "upload.log")
            continue
        destination = UPLOAD_DIR / (file.filename or "upload.log")
        with destination.open("wb") as output:
            output.write(file_bytes)
        saved.append(destination.name)

    if not saved:
        if skipped_empty:
            skipped_str = ", ".join(skipped_empty)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Keine gültigen Dateien hochgeladen. "
                    f"Leere Dateien übersprungen: {skipped_str}"
                ),
            )
        raise HTTPException(status_code=400, detail="Keine gültigen Dateien hochgeladen (.txt, .csv, .json).")

    response: dict[str, Any] = {"uploaded": saved}
    if skipped_empty:
        response["skipped_empty"] = skipped_empty
    return response


@app.get("/logs")
def list_logs() -> dict[str, list[str]]:
    logs = [f.name for f in sorted(UPLOAD_DIR.glob("*")) if f.is_file()]
    return {"logs": logs}


@app.get("/logs/{log_name}/lines")
def get_log_lines(log_name: str) -> dict[str, Any]:
    safe_name = Path(log_name).name
    if safe_name != log_name:
        raise HTTPException(status_code=400, detail="Ungültiger Dateiname.")

    source = UPLOAD_DIR / safe_name
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=404, detail=f"Log-Datei nicht gefunden: {safe_name}")

    lines = read_log_file(source)
    if not lines:
        raise HTTPException(status_code=400, detail=f"Keine lesbaren Zeilen in: {safe_name}")

    return {"log_name": safe_name, "lines": lines}


@app.get("/models")
def list_models(request: Request) -> dict[str, Any]:
    models = [d.name for d in sorted(ARTIFACT_DIR.glob("*")) if d.is_dir() and (d / "meta.json").exists()]
    onnx_download_urls = {
        model_name: str(request.url_for("model_onnx", model_name=model_name))
        for model_name in models
    }
    return {"models": models, "onnx_download_urls": onnx_download_urls}


@app.get("/models/{model_name}/meta")
def model_meta(model_name: str) -> dict[str, Any]:
    paths = _model_paths(model_name)
    if not paths["meta"].exists():
        raise HTTPException(status_code=404, detail="Modell-Metadaten nicht gefunden.")

    metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if "tfidf" not in metadata and paths["vectorizer"].exists():
        try:
            with paths["vectorizer"].open("rb") as fp:
                vectorizer = pickle.load(fp)
            metadata["tfidf"] = {
                "terms": [term for term, _ in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])],
                "idf": [float(value) for value in vectorizer.idf_.tolist()],
            }
            paths["meta"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return metadata


@app.get("/models/{model_name}/onnx")
def model_onnx(model_name: str) -> FileResponse:
    paths = _model_paths(model_name)
    if not paths["onnx"].exists():
        raise HTTPException(status_code=404, detail="ONNX-Datei nicht gefunden.")
    return FileResponse(path=paths["onnx"], filename=f"{model_name}.onnx", media_type="application/octet-stream")


@app.post("/preprocess-preview")
def preprocess_preview(payload: PreprocessPreviewRequest) -> dict[str, Any]:
    lines = _load_logs(payload.selected_logs)
    summary = summarize_preprocessing(lines, sample_limit=payload.sample_limit)
    return {
        "selected_logs": payload.selected_logs,
        "summary": summary,
    }


@app.post("/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    print(f"[DEBUG] Training-Request erhalten: {payload.model_dump()}")
    if payload.model_type == "transformer" and payload.transformer_d_model % payload.transformer_num_heads != 0:
        raise HTTPException(
            status_code=400,
            detail="transformer_d_model muss durch transformer_num_heads teilbar sein.",
        )

    resolved_model_name = _resolve_unique_model_name(payload.model_name)
    if resolved_model_name != payload.model_name:
        print(f"[DEBUG] Modellname existiert bereits, verwende neuen Namen: {resolved_model_name}")

    lines = _load_logs(payload.selected_logs)
    preprocessing_summary = summarize_preprocessing(lines)
    prepared_lines = prepare_lines_for_model(lines)
    vectorizer = build_vectorizer(payload.max_features)
    matrix = vectorizer.fit_transform(prepared_lines).toarray().astype(np.float32)
    try:
        train_device, train_device_backend = resolve_torch_device(require_directml=False)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    print(f"[DEVICE] Training Device: {train_device} ({train_device_backend})", flush=True)

    weak_positive = np.array([
        infer_weak_label(line) in {"endpoint_error", "endpoint_warn", "error_without_endpoint", "warn_without_endpoint"}
        for line in lines
    ])

    sample_count = matrix.shape[0]
    indices = np.arange(sample_count)
    split_warning: str | None = None

    if sample_count < 2:
        train_idx = indices
        test_idx = indices
        split_warning = "Zu wenige Samples für Train/Test-Split; Evaluation erfolgt auf Trainingsdaten."
    else:
        weak_positive_int = weak_positive.astype(int)
        class_counts = np.bincount(weak_positive_int, minlength=2)
        can_stratify = np.count_nonzero(class_counts) > 1 and int(np.min(class_counts[class_counts > 0])) >= 2
        stratify_targets = weak_positive_int if can_stratify else None

        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=payload.test_size,
                random_state=payload.random_state,
                shuffle=True,
                stratify=stratify_targets,
            )
        except ValueError as exc:
            train_idx = indices
            test_idx = indices
            split_warning = (
                "Train/Test-Split fehlgeschlagen; Evaluation erfolgt auf Trainingsdaten. "
                f"Grund: {str(exc)}"
            )

    train_matrix = matrix[train_idx]
    test_matrix = matrix[test_idx]
    weak_positive_train = weak_positive[train_idx]
    weak_positive_test = weak_positive[test_idx]

    best_run: dict[str, Any] | None = None
    round_metrics: list[dict[str, Any]] = []

    lr_candidates = [
        max(1e-6, payload.learning_rate * 0.5),
        payload.learning_rate,
        min(0.1, payload.learning_rate * 2.0),
    ]
    threshold_candidates = [
        max(0.5, payload.threshold_quantile - 0.05),
        payload.threshold_quantile,
        min(0.999, payload.threshold_quantile + 0.02),
    ]

    dedup_lr_candidates: list[float] = []
    for candidate in lr_candidates:
        if not any(np.isclose(candidate, existing) for existing in dedup_lr_candidates):
            dedup_lr_candidates.append(float(candidate))

    dedup_threshold_candidates: list[float] = []
    for candidate in threshold_candidates:
        if not any(np.isclose(candidate, existing) for existing in dedup_threshold_candidates):
            dedup_threshold_candidates.append(float(candidate))

    hyperparam_candidates: list[tuple[float, float]] = [
        (payload.learning_rate, payload.threshold_quantile)
    ]
    for lr_candidate in dedup_lr_candidates:
        for threshold_candidate in dedup_threshold_candidates:
            if np.isclose(lr_candidate, payload.learning_rate) and np.isclose(
                threshold_candidate,
                payload.threshold_quantile,
            ):
                continue
            hyperparam_candidates.append((lr_candidate, threshold_candidate))

    for round_index in range(payload.training_rounds):
        round_no = round_index + 1
        active_learning_rate, active_threshold_quantile = hyperparam_candidates[
            round_index % len(hyperparam_candidates)
        ]
        round_prefix = f"{payload.model_type.upper()} Round {round_no}/{payload.training_rounds}"
        print(
            f"[TRAIN] {round_prefix} gestartet (lr={active_learning_rate:.6f}, "
            f"threshold_q={active_threshold_quantile:.3f})",
            flush=True,
        )

        if payload.model_type == "transformer":
            trained_candidate = train_transformer_autoencoder(
                features=train_matrix,
                d_model=payload.transformer_d_model,
                num_heads=payload.transformer_num_heads,
                num_layers=payload.transformer_num_layers,
                d_ff=payload.transformer_d_ff,
                dropout=payload.transformer_dropout,
                epochs=payload.epochs,
                batch_size=payload.batch_size,
                learning_rate=active_learning_rate,
                threshold_quantile=active_threshold_quantile,
                progress_prefix=round_prefix,
                device=train_device,
            )
            train_scores = transformer_reconstruction_errors(
                trained_candidate.model,
                test_matrix,
                device=train_device,
                batch_size=payload.batch_size,
            )
        else:
            trained_candidate = train_autoencoder(
                features=train_matrix,
                hidden_dim=payload.hidden_dim,
                epochs=payload.epochs,
                batch_size=payload.batch_size,
                learning_rate=active_learning_rate,
                threshold_quantile=active_threshold_quantile,
                progress_prefix=round_prefix,
                device=train_device,
            )
            train_scores = reconstruction_errors(trained_candidate.model, test_matrix, device=train_device)

        predicted_anomaly = train_scores > trained_candidate.threshold

        true_positive = int(np.sum(predicted_anomaly & weak_positive_test))
        false_positive = int(np.sum(predicted_anomaly & ~weak_positive_test))
        false_negative = int(np.sum(~predicted_anomaly & weak_positive_test))

        precision_test = true_positive / max(1, true_positive + false_positive)
        recall_test = true_positive / max(1, true_positive + false_negative)
        f1_score_test = (2 * precision_test * recall_test) / max(1e-12, precision_test + recall_test)

        first_loss = float(trained_candidate.losses[0]) if trained_candidate.losses else 0.0
        last_loss = float(trained_candidate.losses[-1]) if trained_candidate.losses else 0.0
        run = {
            "round": round_no,
            "trained": trained_candidate,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "threshold": float(trained_candidate.threshold),
            "predicted_anomaly_count": int(np.sum(predicted_anomaly)),
            "predicted_anomaly_rate": float(np.mean(predicted_anomaly)) if len(predicted_anomaly) else 0.0,
            "weak_label_precision": precision_test,
            "weak_label_recall": recall_test,
            "weak_label_f1": f1_score_test,
            "weak_label_precision_test": precision_test,
            "weak_label_recall_test": recall_test,
            "weak_label_f1_test": f1_score_test,
            "learning_rate": float(active_learning_rate),
            "threshold_quantile": float(active_threshold_quantile),
        }
        print(
            f"[TRAIN] {round_prefix} abgeschlossen - last_loss={last_loss:.6f}, "
            f"threshold={run['threshold']:.6f}, weak_f1_test={f1_score_test:.4f}, "
            f"lr={active_learning_rate:.6f}, threshold_q={active_threshold_quantile:.3f}",
            flush=True,
        )
        round_metrics.append({
            "round": run["round"],
            "first_loss": run["first_loss"],
            "last_loss": run["last_loss"],
            "threshold": run["threshold"],
            "predicted_anomaly_count": run["predicted_anomaly_count"],
            "predicted_anomaly_rate": run["predicted_anomaly_rate"],
            "weak_label_precision": run["weak_label_precision"],
            "weak_label_recall": run["weak_label_recall"],
            "weak_label_f1": run["weak_label_f1"],
            "weak_label_precision_test": run["weak_label_precision_test"],
            "weak_label_recall_test": run["weak_label_recall_test"],
            "weak_label_f1_test": run["weak_label_f1_test"],
            "learning_rate": run["learning_rate"],
            "threshold_quantile": run["threshold_quantile"],
        })

        if best_run is None:
            best_run = run
        else:
            current_f1 = float(run["weak_label_f1"])
            best_f1 = float(best_run["weak_label_f1"])
            current_last_loss = float(run["last_loss"])
            best_last_loss = float(best_run["last_loss"])
            if current_f1 > best_f1 or (np.isclose(current_f1, best_f1) and current_last_loss < best_last_loss):
                best_run = run

    if best_run is None:
        raise HTTPException(status_code=500, detail="Training fehlgeschlagen: Kein gültiger Trainingslauf.")

    trained = best_run["trained"]
    first_loss = float(best_run["first_loss"])
    last_loss = float(best_run["last_loss"])
    loss_improvement = first_loss - last_loss
    loss_improvement_pct = (loss_improvement / max(1e-12, first_loss)) * 100.0

    training_quality = {
        "training_rounds": payload.training_rounds,
        "selected_round": int(best_run["round"]),
        "selected_learning_rate": float(best_run["learning_rate"]),
        "selected_threshold_quantile": float(best_run["threshold_quantile"]),
        "candidate_learning_rates": dedup_lr_candidates,
        "candidate_threshold_quantiles": dedup_threshold_candidates,
        "rounds": round_metrics,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_improvement": loss_improvement,
        "loss_improvement_pct": loss_improvement_pct,
        "predicted_anomaly_count": int(best_run["predicted_anomaly_count"]),
        "predicted_anomaly_rate": float(best_run["predicted_anomaly_rate"]),
        "sample_counts": {
            "all": int(sample_count),
            "train": int(train_matrix.shape[0]),
            "test": int(test_matrix.shape[0]),
        },
        "split": {
            "test_size": float(payload.test_size),
            "random_state": int(payload.random_state),
            "is_holdout_enabled": bool(split_warning is None and sample_count >= 2),
            "warning": split_warning,
        },
        "weak_label_positive_count_train": int(np.sum(weak_positive_train)),
        "weak_label_positive_rate_train": float(np.mean(weak_positive_train)) if len(weak_positive_train) else 0.0,
        "weak_label_positive_count_test": int(np.sum(weak_positive_test)),
        "weak_label_positive_rate_test": float(np.mean(weak_positive_test)) if len(weak_positive_test) else 0.0,
        "weak_label_precision": float(best_run["weak_label_precision"]),
        "weak_label_recall": float(best_run["weak_label_recall"]),
        "weak_label_f1": float(best_run["weak_label_f1"]),
        "weak_label_precision_test": float(best_run["weak_label_precision_test"]),
        "weak_label_recall_test": float(best_run["weak_label_recall_test"]),
        "weak_label_f1_test": float(best_run["weak_label_f1_test"]),
        "note": (
            "Transformer wird in mehreren PyTorch-Runden auf Train-Daten trainiert; die Auswahl erfolgt über Weak-Label-F1 auf Test-Daten."
            if payload.model_type == "transformer"
            else "Autoencoder wird in mehreren PyTorch-Runden auf Train-Daten trainiert; die Auswahl erfolgt über Weak-Label-F1 auf Test-Daten."
        ),
    }

    paths = _model_paths(resolved_model_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    trained_model_cpu = trained.model.to(torch.device("cpu"))

    torch.save(
        {
            "model_type": payload.model_type,
            "input_dim": matrix.shape[1],
            "hidden_dim": payload.hidden_dim,
            "transformer_d_model": payload.transformer_d_model,
            "transformer_num_heads": payload.transformer_num_heads,
            "transformer_num_layers": payload.transformer_num_layers,
            "transformer_d_ff": payload.transformer_d_ff,
            "transformer_dropout": payload.transformer_dropout,
            "state_dict": trained_model_cpu.state_dict(),
        },
        paths["weights"],
    )
    with paths["vectorizer"].open("wb") as fp:
        pickle.dump(vectorizer, fp)

    print(f"[EXPORT] ONNX-Export gestartet für Modell '{resolved_model_name}' ({payload.model_type})", flush=True)
    try:
        if payload.model_type == "transformer":
            export_transformer_onnx(trained_model_cpu, input_dim=matrix.shape[1], onnx_path=paths["onnx"])
        else:
            export_onnx(trained_model_cpu, input_dim=matrix.shape[1], onnx_path=paths["onnx"])
        print(f"[EXPORT] ONNX-Export abgeschlossen: {paths['onnx'].name}", flush=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    metadata = {
        "model_name": resolved_model_name,
        "requested_model_name": payload.model_name,
        "model_type": payload.model_type,
        "input_dim": matrix.shape[1],
        "hidden_dim": payload.hidden_dim,
        "transformer": {
            "d_model": payload.transformer_d_model,
            "num_heads": payload.transformer_num_heads,
            "num_layers": payload.transformer_num_layers,
            "d_ff": payload.transformer_d_ff,
            "dropout": payload.transformer_dropout,
        },
        "epochs": payload.epochs,
        "batch_size": payload.batch_size,
        "learning_rate": payload.learning_rate,
        "test_size": payload.test_size,
        "random_state": payload.random_state,
        "selected_learning_rate": float(best_run["learning_rate"]),
        "selected_threshold_quantile": float(best_run["threshold_quantile"]),
        "max_features": payload.max_features,
        "training_rounds": payload.training_rounds,
        "device_backend": train_device_backend,
        "threshold": trained.threshold,
        "losses": trained.losses,
        "trained_on_logs": payload.selected_logs,
        "preprocessing": {
            "normalized_patterns": ["timestamp", "ip", "uuid", "hex", "path", "id", "number"],
            "weak_labels": "endpoint+severity",
            "summary": preprocessing_summary,
        },
        "training_quality": training_quality,
        "onnx_path": str(paths["onnx"].name),
        "tfidf": {
            "terms": [term for term, _ in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])],
            "idf": [float(value) for value in vectorizer.idf_.tolist()],
        },
    }
    paths["meta"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "message": "Training abgeschlossen",
        "model": resolved_model_name,
        "requested_model_name": payload.model_name,
        "model_type": payload.model_type,
        "threshold": trained.threshold,
        "selected_learning_rate": float(best_run["learning_rate"]),
        "selected_threshold_quantile": float(best_run["threshold_quantile"]),
        "losses": trained.losses,
        "onnx_file": str(paths["onnx"].name),
        "input_features": int(matrix.shape[1]),
        "preprocessing": preprocessing_summary,
        "training_quality": training_quality,
    }


@app.post("/analyze")
def analyze_logs(payload: AnalyzeRequest) -> dict[str, Any]:
    paths = _model_paths(payload.model_name)
    if not paths["meta"].exists() or not paths["weights"].exists() or not paths["vectorizer"].exists():
        raise HTTPException(status_code=404, detail="Modell nicht gefunden.")

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    with paths["vectorizer"].open("rb") as fp:
        vectorizer = pickle.load(fp)

    checkpoint = torch.load(paths["weights"], map_location="cpu")
    model_type = str(meta.get("model_type") or checkpoint.get("model_type") or "autoencoder")
    analyze_device, analyze_device_backend = resolve_torch_device()
    print(f"[DEVICE] Analyze Device: {analyze_device} ({analyze_device_backend})", flush=True)

    if model_type == "transformer":
        model = Transformer(
            d_model=int(checkpoint.get("transformer_d_model", 64)),
            num_heads=int(checkpoint.get("transformer_num_heads", 8)),
            num_layers=int(checkpoint.get("transformer_num_layers", 2)),
            d_ff=int(checkpoint.get("transformer_d_ff", 256)),
            max_seq_length=int(checkpoint["input_dim"]),
            dropout=float(checkpoint.get("transformer_dropout", 0.1)),
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = LogAutoencoder(input_dim=checkpoint["input_dim"], hidden_dim=checkpoint["hidden_dim"])
        model.load_state_dict(checkpoint["state_dict"])

    lines, line_sources = _load_logs_with_sources(payload.selected_logs)
    prepared_lines = prepare_lines_for_model(lines)
    matrix = vectorizer.transform(prepared_lines).toarray().astype(np.float32)
    if model_type == "transformer":
        scores = transformer_reconstruction_errors(model, matrix, device=analyze_device, batch_size=64)
    else:
        scores = reconstruction_errors(model, matrix, device=analyze_device)
    threshold = float(meta["threshold"])
    global_score_std = float(np.std(scores))
    zscore_std_floor = max(global_score_std * 0.35, 1e-6)
    anomaly_indices = set(np.where(scores > threshold)[0].tolist())

    severity_cache = [infer_severity(line) for line in lines]
    weak_label_cache = [infer_weak_label(line) for line in lines]
    error_semantic_indices = {
        index
        for index, line in enumerate(lines)
        if _line_is_error(line, severity_cache[index])
        or weak_label_cache[index] in {"endpoint_error", "error_without_endpoint"}
    }

    ranked_candidate_indices = sorted(
        anomaly_indices.union(error_semantic_indices),
        key=lambda idx: float(scores[idx]),
        reverse=True,
    )

    endpoint_for_index = [infer_endpoint(line) or "unknown" for line in lines]
    endpoint_score_map: dict[str, list[float]] = {}
    for index, endpoint_key in enumerate(endpoint_for_index):
        endpoint_score_map.setdefault(endpoint_key, []).append(float(scores[index]))

    endpoint_stats: dict[str, tuple[float, float]] = {}
    for endpoint_key, values in endpoint_score_map.items():
        mean_val = float(np.mean(values)) if values else 0.0
        std_val = float(np.std(values)) if values else 0.0
        endpoint_stats[endpoint_key] = (mean_val, std_val)

    anomalies = []
    for idx in ranked_candidate_indices:
        line = lines[idx]
        endpoint_key = endpoint_for_index[idx]
        severity = severity_cache[idx]
        weak_label = weak_label_cache[idx]
        endpoint_mean, endpoint_std = endpoint_stats.get(endpoint_key, (0.0, 0.0))
        score = float(scores[idx])
        endpoint_zscore = (score - endpoint_mean) / max(endpoint_std, zscore_std_floor)
        operational_risk = max(0.0, endpoint_zscore)
        error_context = _classify_error_context(line, severity)
        if error_context == "server":
            operational_risk += 2.0
        elif error_context == "client":
            operational_risk += 0.5
        elif severity in {"warn", "critical", "error"}:
            operational_risk += 1.0

        benign_event = _is_benign_operational_event(line, severity, weak_label)
        likely_normal = _is_likely_normal_endpoint_info(line, severity, weak_label, score, threshold, endpoint_zscore)
        suppressed_as_benign = (benign_event and endpoint_zscore < 2.5) or likely_normal

        normalized_line = normalize_log_line(line)
        normalized_line = remove_placeholder_tokens(normalized_line)
        compact_normalized = re.sub(r"\s+", " ", normalized_line).strip()
        if len(compact_normalized) > 260:
            compact_normalized = f"{compact_normalized[:260]}..."
        anomalies.append(
            {
                "line_index": int(idx),
                "line": line,
                "line_normalized": compact_normalized,
                "endpoint": endpoint_key,
                "severity": severity,
                "weak_label": weak_label,
                "score": score,
                "error_context": error_context,
                "endpoint_mean_score": endpoint_mean,
                "endpoint_zscore": float(endpoint_zscore),
                "operational_risk": float(operational_risk),
                "suppressed_as_benign": suppressed_as_benign,
            }
        )

    actionable_anomalies = [item for item in anomalies if not bool(item.get("suppressed_as_benign", False))]
    actionable_anomalies.sort(
        key=lambda item: (float(item.get("operational_risk", 0.0)), float(item.get("score", 0.0))),
        reverse=True,
    )
    if not actionable_anomalies:
        actionable_anomalies = sorted(anomalies, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    deduped_actionable: list[dict[str, Any]] = []
    dedup_index_map: dict[str, int] = {}
    for item in actionable_anomalies:
        dedup_key = "||".join(
            [
                str(item.get("endpoint", "")),
                str(item.get("weak_label", "")),
                str(item.get("severity", "")),
                str(item.get("line_normalized", "")),
            ]
        )
        if dedup_key not in dedup_index_map:
            entry = dict(item)
            entry["occurrences"] = 1
            entry["sample_line_indices"] = [int(item.get("line_index", 0))]
            dedup_index_map[dedup_key] = len(deduped_actionable)
            deduped_actionable.append(entry)
            continue

        existing = deduped_actionable[dedup_index_map[dedup_key]]
        existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
        sample_indices = list(existing.get("sample_line_indices", []))
        line_idx = int(item.get("line_index", 0))
        if line_idx not in sample_indices and len(sample_indices) < 5:
            sample_indices.append(line_idx)
        existing["sample_line_indices"] = sample_indices
        if float(item.get("operational_risk", 0.0)) > float(existing.get("operational_risk", 0.0)):
            existing["line_index"] = line_idx
            existing["line"] = item.get("line")
            existing["line_normalized"] = item.get("line_normalized")
            existing["score"] = item.get("score")
            existing["endpoint_mean_score"] = item.get("endpoint_mean_score")
            existing["endpoint_zscore"] = item.get("endpoint_zscore")
            existing["operational_risk"] = item.get("operational_risk")

    deduped_actionable.sort(
        key=lambda item: (
            float(item.get("operational_risk", 0.0)),
            int(item.get("occurrences", 1)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )

    top_chart_source = deduped_actionable[:10]

    actionable_line_indices = {int(item.get("line_index", -1)) for item in actionable_anomalies}
    per_file_analysis: list[dict[str, Any]] = []
    for log_file in payload.selected_logs:
        file_indices = [idx for idx, src in enumerate(line_sources) if src == log_file]
        if not file_indices:
            continue

        file_total = len(file_indices)
        file_anomaly_hits = sum(1 for idx in file_indices if idx in actionable_line_indices)

        endpoint_hits: dict[str, int] = {}
        endpoint_anomaly_hits: dict[str, int] = {}
        for idx in file_indices:
            endpoint_key = endpoint_for_index[idx]
            endpoint_hits[endpoint_key] = endpoint_hits.get(endpoint_key, 0) + 1
            if idx in actionable_line_indices:
                endpoint_anomaly_hits[endpoint_key] = endpoint_anomaly_hits.get(endpoint_key, 0) + 1

        top_endpoints = sorted(endpoint_hits.items(), key=lambda item: item[1], reverse=True)[:5]
        top_endpoints_summary = [
            {
                "endpoint": endpoint,
                "hits": hits,
                "anomaly_hits": int(endpoint_anomaly_hits.get(endpoint, 0)),
                "anomaly_rate": float(endpoint_anomaly_hits.get(endpoint, 0)) / max(1, hits),
            }
            for endpoint, hits in top_endpoints
        ]

        per_file_analysis.append(
            {
                "file_name": log_file,
                "total_lines": file_total,
                "anomaly_count": file_anomaly_hits,
                "anomaly_rate": file_anomaly_hits / max(1, file_total),
                "top_endpoints": top_endpoints_summary,
            }
        )

    top_chart = [
        {
            "label": _chart_label_for_line(int(item["line_index"]), lines[int(item["line_index"])]),
            "score": float(item["score"]),
            "operational_risk": float(item.get("operational_risk", 0.0)),
        }
        for item in top_chart_source
    ]

    endpoint_insights = _build_endpoint_insights(lines, anomaly_indices)

    return {
        "model": payload.model_name,
        "model_type": model_type,
        "device_backend": analyze_device_backend,
        "total_logs": len(lines),
        "threshold": threshold,
        "anomaly_count": len(anomalies),
        "anomaly_rate": (len(anomalies) / max(1, len(lines))),
        "actionable_anomaly_count": len(actionable_anomalies),
        "actionable_unique_count": len(deduped_actionable),
        "suppressed_benign_count": max(0, len(anomalies) - len(actionable_anomalies)),
        "top_scores": top_chart,
        "top_anomalies": deduped_actionable[:50],
        "per_file_analysis": per_file_analysis,
        "endpoint_insights": endpoint_insights,
    }