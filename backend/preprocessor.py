from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TIMESTAMP_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    re.compile(r"\b\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"),
]
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
HEX_PATTERN = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
LONG_ID_PATTERN = re.compile(r"\b[a-z0-9_-]{12,}\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+\b")
PATH_PATTERN = re.compile(r"(?:[a-zA-Z]:\\\\|/)[^\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PLACEHOLDER_TOKEN_PATTERN = re.compile(r"<\s*(?:ts|ip|uuid|hex|path|id|num|jwt|secret_kv|payload)\s*>", re.IGNORECASE)
METADATA_TOKEN_PATTERN = re.compile(
    r"\b(?:ts|time|timestamp|lvl|level|msg|logger|details|func|file|line|app|info|debug|warning|warn|error|critical|traceback|py)\b:?",
    re.IGNORECASE,
)
JWT_PATTERN = re.compile(r"\beyJ[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){2}\b")
SENSITIVE_KV_PATTERN = re.compile(
    r"\b(?:token|access_token|refresh_token|authorization|raw_cookies|cookie_sid|session_id|log_sid|log_csrf_token|csrf|sid)\s*[=:]\s*(?:'[^']*'|\"[^\"]*\"|[^\s|,;]+)",
    re.IGNORECASE,
)
AUTH_PAYLOAD_PATTERN = re.compile(r"auth_payload\s*=\s*\{.*?\}(?=\s+logger:|\s+details:|\s+func:|$)", re.IGNORECASE)
ENDPOINT_PATTERNS = [
    re.compile(r"exception on\s+([^\s]+)\s+\[(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\]", re.IGNORECASE),
    re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+https?://[^/\s]+([^\s\]\"']*)", re.IGNORECASE),
    re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\s+(/[^\s\]\"']*)", re.IGNORECASE),
]

SEVERITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "critical": ("critical", "fatal", "panic", "emergency", "segfault"),
    "error": ("error", "exception", "failed", "failure", "traceback"),
    "warn": ("warn", "warning", "deprecated", "retry"),
    "info": ("info", "started", "startup", "listening", "connected"),
    "debug": ("debug", "verbose", "trace"),
}


def _json_to_lines(content: str) -> list[str]:
    lines: list[str] = []

    def collect_records(node: object, prefix: str = "") -> None:
        if isinstance(node, list):
            for item in node:
                collect_records(item, prefix=prefix)
            return

        if isinstance(node, dict):
            scalar_pairs = []
            nested_items = []
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    nested_items.append((key, value))
                else:
                    scalar_pairs.append(f"{key}:{value}")

            if scalar_pairs:
                if prefix:
                    lines.append(f"{prefix} {' '.join(scalar_pairs)}")
                else:
                    lines.append(" ".join(scalar_pairs))

            for key, value in nested_items:
                nested_prefix = f"{prefix}.{key}" if prefix else str(key)
                collect_records(value, prefix=nested_prefix)
            return

        if node is not None:
            text = str(node)
            if text.strip():
                lines.append(f"{prefix}:{text}" if prefix else text)

    payload = None
    stripped_content = content.strip()
    candidates: list[str] = [stripped_content]

    embedded_candidate = _extract_embedded_json_candidate(stripped_content)
    if embedded_candidate and embedded_candidate not in candidates:
        candidates.append(embedded_candidate)

    for candidate in candidates:
        normalized_candidate = candidate.strip()
        if normalized_candidate.endswith(";"):
            normalized_candidate = normalized_candidate[:-1].rstrip()
        try:
            payload = json.loads(normalized_candidate)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        return []

    collect_records(payload)

    return [line.strip() for line in lines if line and str(line).strip()]


def _extract_embedded_json_candidate(content: str) -> str | None:
    first_bracket = content.find("[")
    last_bracket = content.rfind("]")
    if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
        return None
    candidate = content[first_bracket : last_bracket + 1].strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        return candidate
    return None


def read_log_file(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text_content = path.read_text(encoding="utf-8", errors="ignore")
        stripped_content = text_content.strip()
        if stripped_content.startswith("{") or stripped_content.startswith("["):
            json_lines = _json_to_lines(stripped_content)
            if json_lines:
                return json_lines
        embedded_json = _extract_embedded_json_candidate(stripped_content)
        if embedded_json:
            json_lines = _json_to_lines(embedded_json)
            if json_lines:
                return json_lines
        return [line.strip() for line in text_content.splitlines() if line.strip()]

    if suffix == ".csv":
        frame = pd.read_csv(path)
        rows = frame.fillna("").astype(str).agg(" ".join, axis=1).tolist()
        return [row.strip() for row in rows if row.strip()]

    if suffix == ".json":
        content = path.read_text(encoding="utf-8", errors="ignore")
        stripped_content = content.strip()
        parsed_lines = _json_to_lines(stripped_content)
        if parsed_lines:
            return parsed_lines

        embedded_json = _extract_embedded_json_candidate(stripped_content)
        if embedded_json:
            parsed_lines = _json_to_lines(embedded_json)
            if parsed_lines:
                return parsed_lines

        return []

    raise ValueError(f"Nicht unterstütztes Dateiformat: {suffix}")


def merge_log_lines(collection: Iterable[list[str]]) -> list[str]:
    merged: list[str] = []
    for lines in collection:
        merged.extend(lines)
    return merged


def normalize_log_line(line: str) -> str:
    normalized = line.strip().lower()
    normalized = AUTH_PAYLOAD_PATTERN.sub(" auth_payload=<payload> ", normalized)
    normalized = JWT_PATTERN.sub(" <jwt> ", normalized)
    normalized = SENSITIVE_KV_PATTERN.sub(" <secret_kv> ", normalized)
    for pattern in TIMESTAMP_PATTERNS:
        normalized = pattern.sub(" <ts> ", normalized)
    normalized = IPV4_PATTERN.sub(" <ip> ", normalized)
    normalized = UUID_PATTERN.sub(" <uuid> ", normalized)
    normalized = HEX_PATTERN.sub(" <hex> ", normalized)
    normalized = PATH_PATTERN.sub(" <path> ", normalized)
    normalized = LONG_ID_PATTERN.sub(" <id> ", normalized)
    normalized = NUMBER_PATTERN.sub(" <num> ", normalized)
    normalized = re.sub(r"\b(?:ts|time|timestamp|msg)\s*:\s*<ts>", " <ts> ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def _normalize_with_stats(line: str) -> tuple[str, dict[str, int]]:
    normalized = line.strip().lower()
    replacement_counts = {
        "timestamp": 0,
        "ip": 0,
        "uuid": 0,
        "hex": 0,
        "path": 0,
        "id": 0,
        "number": 0,
    }

    normalized, payload_hits = AUTH_PAYLOAD_PATTERN.subn(" auth_payload=<payload> ", normalized)
    normalized, jwt_hits = JWT_PATTERN.subn(" <jwt> ", normalized)
    normalized, secret_hits = SENSITIVE_KV_PATTERN.subn(" <secret_kv> ", normalized)
    replacement_counts["id"] += payload_hits + jwt_hits + secret_hits

    for pattern in TIMESTAMP_PATTERNS:
        normalized, hits = pattern.subn(" <ts> ", normalized)
        replacement_counts["timestamp"] += hits
    normalized, hits = IPV4_PATTERN.subn(" <ip> ", normalized)
    replacement_counts["ip"] += hits
    normalized, hits = UUID_PATTERN.subn(" <uuid> ", normalized)
    replacement_counts["uuid"] += hits
    normalized, hits = HEX_PATTERN.subn(" <hex> ", normalized)
    replacement_counts["hex"] += hits
    normalized, hits = PATH_PATTERN.subn(" <path> ", normalized)
    replacement_counts["path"] += hits
    normalized, hits = LONG_ID_PATTERN.subn(" <id> ", normalized)
    replacement_counts["id"] += hits
    normalized, hits = NUMBER_PATTERN.subn(" <num> ", normalized)
    replacement_counts["number"] += hits
    normalized = re.sub(r"\b(?:ts|time|timestamp|msg)\s*:\s*<ts>", " <ts> ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized, replacement_counts


def infer_severity(line: str) -> str:
    lowered = line.lower()
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if any(token in lowered for token in keywords):
            return severity
    return "unknown"


def _normalize_endpoint_path(path: str) -> str:
    sanitized_path = path.split("?", 1)[0].strip() or "/"
    if not sanitized_path.startswith("/"):
        sanitized_path = f"/{sanitized_path}"

    raw_segments = [segment for segment in sanitized_path.split("/") if segment]
    normalized_segments: list[str] = []
    for segment in raw_segments:
        lowered = segment.lower()
        if UUID_PATTERN.fullmatch(lowered):
            normalized_segments.append("<uuid>")
        elif HEX_PATTERN.fullmatch(lowered):
            normalized_segments.append("<hex>")
        elif lowered.isdigit():
            normalized_segments.append("<num>")
        elif LONG_ID_PATTERN.fullmatch(lowered):
            normalized_segments.append("<id>")
        else:
            normalized_segments.append(lowered)

    if not normalized_segments:
        return "/"
    return "/" + "/".join(normalized_segments)


def infer_endpoint(line: str) -> str | None:
    stripped = line.strip()

    ws_tag_match = re.search(r"\[ws\]\s*([^|]+)", stripped, flags=re.IGNORECASE)
    if ws_tag_match:
        raw_event = ws_tag_match.group(1).strip().lower()
        event_slug = re.sub(r"[^a-z0-9]+", "_", raw_event).strip("_") or "event"
        return f"WS /socket/{event_slug}"

    if re.search(r"socket\s+disconnect", stripped, flags=re.IGNORECASE):
        return "WS /socket/disconnect"

    func_match = re.search(r"func:(handle_[a-z0-9_]+)", stripped, flags=re.IGNORECASE)
    if func_match and ("app.socket" in stripped.lower() or "logger:socket" in stripped.lower()):
        func_name = func_match.group(1).lower().replace("handle_", "")
        func_slug = re.sub(r"[^a-z0-9_]+", "_", func_name).strip("_") or "event"
        return f"WS /socket/{func_slug}"

    action_method_match = re.search(
        r"\b([a-z0-9][a-z0-9 _/\-]{2,80}?)\s+(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\b",
        stripped,
        flags=re.IGNORECASE,
    )
    if action_method_match:
        action = action_method_match.group(1).strip().lower()
        method = action_method_match.group(2).upper()
        if "http" not in action and "logger:" not in action and "details:" not in action:
            action_slug = re.sub(r"[^a-z0-9]+", "_", action).strip("_")
            if action_slug:
                return f"{method} /event/{action_slug}"

    for pattern in ENDPOINT_PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue

        if pattern.pattern.startswith("exception on"):
            path = match.group(1)
            method = match.group(2)
        else:
            method = match.group(1)
            path = match.group(2)

        normalized_path = _normalize_endpoint_path(path)
        return f"{method.upper()} {normalized_path}"

    endpoint_match = re.search(r"(?:endpoint|path|route)[:=]\s*([^\s,;]+)", stripped, flags=re.IGNORECASE)
    if endpoint_match:
        normalized_path = _normalize_endpoint_path(endpoint_match.group(1))
        return f"UNK {normalized_path}"

    logger_match = re.search(r"\blogger:([a-z0-9_.-]+)", stripped, flags=re.IGNORECASE)
    func_match = re.search(r"\bfunc:([a-z0-9_]+)", stripped, flags=re.IGNORECASE)
    if logger_match or func_match:
        logger_name = (logger_match.group(1).lower() if logger_match else "unknown_logger").replace(".", "_")
        func_name = func_match.group(1).lower() if func_match else "unknown_func"
        logger_slug = re.sub(r"[^a-z0-9_]+", "_", logger_name).strip("_") or "unknown_logger"
        func_slug = re.sub(r"[^a-z0-9_]+", "_", func_name).strip("_") or "unknown_func"
        return f"INT /internal/{logger_slug}/{func_slug}"

    return None


def infer_weak_label(line: str) -> str:
    severity = infer_severity(line)
    endpoint = infer_endpoint(line)
    has_endpoint = endpoint is not None

    if severity in {"critical", "error"}:
        return "endpoint_error" if has_endpoint else "error_without_endpoint"
    if severity == "warn":
        return "endpoint_warn" if has_endpoint else "warn_without_endpoint"
    if has_endpoint:
        return "endpoint_normal"
    return "unknown"


def _remove_placeholders_for_model(normalized_line: str) -> str:
    compact = PLACEHOLDER_TOKEN_PATTERN.sub(" ", normalized_line)
    compact = METADATA_TOKEN_PATTERN.sub(" ", compact)
    compact = re.sub(r"[{}\[\]()<>=|]+", " ", compact)
    compact = re.sub("[.,;:\\\\\"'`]+", " ", compact)
    compact = re.sub(r"\b[a-zA-Z]\b", " ", compact)
    compact = WHITESPACE_PATTERN.sub(" ", compact).strip()
    return compact


def remove_placeholder_tokens(text: str) -> str:
    compact = PLACEHOLDER_TOKEN_PATTERN.sub(" ", text)
    compact = re.sub(r"[<>]+", " ", compact)
    compact = WHITESPACE_PATTERN.sub(" ", compact).strip()
    return compact


def prepare_lines_for_model(lines: list[str]) -> list[str]:
    prepared: list[str] = []
    for line in lines:
        normalized = normalize_log_line(line)
        normalized_for_model = _remove_placeholders_for_model(normalized)
        severity = infer_severity(line)
        endpoint = infer_endpoint(line)
        weak_label = infer_weak_label(line)
        endpoint_token = "endpoint_unknown"
        if endpoint:
            method, path = endpoint.split(" ", 1)
            token_path = re.sub(r"[^a-z0-9_<>/]+", "_", path.lower()).replace("/", "_").strip("_")
            endpoint_token = f"endpoint_{method.lower()}_{token_path or 'root'}"
        if normalized_for_model:
            prepared.append(f"severity_{severity} weak_{weak_label} {endpoint_token} {normalized_for_model}")
        else:
            prepared.append(f"severity_{severity} weak_{weak_label} {endpoint_token} empty_line")
    return prepared


def summarize_preprocessing(lines: list[str], sample_limit: int = 25) -> dict[str, object]:
    severity_counter: Counter[str] = Counter()
    weak_label_counter: Counter[str] = Counter()
    endpoint_counter: Counter[str] = Counter()
    replacement_counter: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for index, line in enumerate(lines):
        normalized, replacements = _normalize_with_stats(line)
        normalized_for_model = _remove_placeholders_for_model(normalized)
        severity = infer_severity(line)
        weak_label = infer_weak_label(line)
        endpoint = infer_endpoint(line)
        endpoint_token = "endpoint_unknown"
        if endpoint:
            method, path = endpoint.split(" ", 1)
            token_path = re.sub(r"[^a-z0-9_<>/]+", "_", path.lower()).replace("/", "_").strip("_")
            endpoint_token = f"endpoint_{method.lower()}_{token_path or 'root'}"
            endpoint_counter[endpoint] += 1

        model_input = (
            f"severity_{severity} weak_{weak_label} {endpoint_token} {normalized_for_model}"
            if normalized_for_model
            else f"severity_{severity} weak_{weak_label} {endpoint_token} empty_line"
        )

        severity_counter[severity] += 1
        weak_label_counter[weak_label] += 1
        replacement_counter.update(replacements)

        if index < sample_limit:
            samples.append(
                {
                    "original": line,
                    "normalized": normalized or "empty_line",
                    "normalized_for_model": normalized_for_model or "empty_line",
                    "severity": severity,
                    "weak_label": weak_label,
                    "endpoint": endpoint or "unknown",
                    "model_input": model_input,
                }
            )

    return {
        "total_lines": len(lines),
        "sample_count": len(samples),
        "severity_distribution": dict(sorted(severity_counter.items())),
        "weak_label_distribution": dict(sorted(weak_label_counter.items())),
        "endpoint_distribution": dict(endpoint_counter.most_common(20)),
        "replacement_counts": dict(replacement_counter),
        "samples": samples,
        "steps": [
            "Raw Log-Zeilen laden",
            "Strings normalisieren (Timestamp/IP/UUID/Hex/Path/ID/Number)",
            "Placeholder-Tokens für Modell-Input entfernen (<ts>/<id>/<num>/...) ",
            "Endpoint pro Zeile extrahieren und dynamische Path-Segmente normalisieren",
            "Endpoint-Weak-Label pro Zeile ableiten (error/warn/normal/unknown)",
            "Model-Input zusammensetzen: severity_<label> + weak_<endpoint_label> + endpoint_<token> + normalisierte Zeile",
            "TF-IDF-Vektorisierung für Autoencoder",
        ],
    }


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
