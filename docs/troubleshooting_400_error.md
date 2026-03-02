# Diagnose und Behebung von 400 Bad Request Fehlern

## Problem
Sie erhalten einen `HTTP/1.1 400 Bad Request` Fehler beim POST-Request an `/train`.

## Mögliche Ursachen

### 0. ONNX-Abhängigkeit fehlt (führt zu 500)
**Symptom:** Training startet, bricht aber beim ONNX-Export ab
**Fehlermeldung:** `ModuleNotFoundError: No module named 'onnxscript'`
**Lösung:** Im `backend/` Verzeichnis ausführen:

```bash
pip install -r requirements.txt
```

Danach den Backend-Server neu starten.

### 1. Keine Log-Dateien ausgewählt
**Symptom:** `selected_logs` Array ist leer
**Fehlermeldung:** "Bitte mindestens eine Log-Datei auswählen."
**Lösung:** Wählen Sie mindestens eine Log-Datei aus

### 2. Log-Dateien existieren nicht
**Symptom:** Die angegebenen Log-Dateien sind nicht im `uploads/` Ordner
**Fehlermeldung:** "Log-Datei nicht gefunden: {filename}"
**Lösung:** Laden Sie zuerst Log-Dateien hoch

### 3. Validierungsfehler
**Symptom:** Request-Daten entsprechen nicht dem Schema
**Fehlermeldung:** Detaillierte Validierungsfehler (ab jetzt mit neuem Handler)
**Lösung:** Überprüfen Sie die Request-Daten

## Verbesserte Fehlerbehandlung

Die Datei `app.py` wurde mit folgenden Verbesserungen aktualisiert:

### 1. Besserer Exception Handler für Validierungsfehler

```python
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
```

**Vorteil:** Statt nur "Unprocessable Entity" erhalten Sie jetzt detaillierte Informationen über ALLE Validierungsfehler.

### 2. Debug-Logging

```python
@app.post("/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    print(f"[DEBUG] Training-Request erhalten: {payload.model_dump()}")
    lines = _load_logs(payload.selected_logs)
    # ...

def _load_logs(selected_logs: list[str]) -> list[str]:
    print(f"[DEBUG] _load_logs aufgerufen mit: {selected_logs}")
    # ...
    for log_file in selected_logs:
        source = UPLOAD_DIR / log_file
        print(f"[DEBUG] Prüfe Log-Datei: {source} (exists: {source.exists()})")
        # ...
```

**Vorteil:** Sie sehen im Server-Log genau, was ankommt und wo das Problem liegt.

## Diagnose-Schritte

### Schritt 1: Server-Logs überprüfen

Starten Sie den Server mit:
```bash
cd backend
python -m uvicorn app:app --reload
```

Schauen Sie sich die Konsolen-Ausgabe an, wenn der Fehler auftritt.

### Schritt 2: Test-Skript ausführen

```bash
cd backend
python test_train_request.py
```

Dieses Skript testet verschiedene Szenarien und zeigt genau, wo das Problem liegt.

### Schritt 3: Frontend-Payload überprüfen

Öffnen Sie die Browser-Entwicklerkonsole (F12) und schauen Sie sich den Request im Network-Tab an:

```javascript
// Erwarteter Payload (aus App.jsx):
{
  "model_name": "log_autoencoder_v1",
  "selected_logs": ["Untitled-1.txt"],  // MUSS mindestens 1 Element haben!
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "hidden_dim": 64,
  "max_features": 256,
  "threshold_quantile": 0.95
}
```

### Schritt 4: Verfügbare Logs prüfen

```bash
curl http://127.0.0.1:8000/logs
```

oder im Browser:
```
http://127.0.0.1:8000/logs
```

## Häufige Lösungen

### Lösung 1: Log-Dateien hochladen

1. Öffnen Sie das Frontend
2. Klicken Sie auf "Datei auswählen"
3. Wählen Sie .txt, .csv oder .json Dateien
4. Klicken Sie auf "Upload"
5. Warten Sie auf "Upload erfolgreich"
6. Wählen Sie die hochgeladenen Dateien im Training-Bereich aus

### Lösung 2: Prüfen Sie das uploads/ Verzeichnis

```bash
ls backend/uploads/
```

Wenn leer, laden Sie Dateien hoch!

### Lösung 3: Request-Daten korrigieren

Im Frontend (App.jsx), stellen Sie sicher dass:
- `selectedLogs` nicht leer ist (wird in `onTrain()` geprüft)
- `modelName` gesetzt ist
- Alle Werte in den gültigen Bereichen liegen

## TrainRequest Schema

```python
class TrainRequest(BaseModel):
    model_name: str                         # 2-60 Zeichen
    selected_logs: list[str]                # MUSS mindestens 1 Element haben (geprüft in _load_logs)
    epochs: int = 10                        # 1-200
    batch_size: int = 32                    # 4-1024
    learning_rate: float = 0.001            # >0.0, <=0.1
    hidden_dim: int = 64                    # 16-2048
    max_features: int = 256                 # 32-10000
    threshold_quantile: float = 0.95        # 0.5-0.999
```

## Test mit curl

```bash
# Minimal (mit Standardwerten)
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "test_v1",
    "selected_logs": ["Untitled-1.txt"]
  }'

# Vollständig
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "test_v2",
    "selected_logs": ["Untitled-1.txt", "Untitled-2.json"],
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.002,
    "hidden_dim": 128,
    "max_features": 512,
    "threshold_quantile": 0.92
  }'
```

## Erwartete Antworten

### Erfolg (200 OK)
```json
{
  "message": "Training abgeschlossen",
  "model": "test_v1",
  "threshold": 0.123,
  "losses": [0.5, 0.4, 0.3, ...],
  "onnx_file": "model.onnx"
}
```

### Fehler: Keine Logs ausgewählt (400 Bad Request)
```json
{
  "detail": "Bitte mindestens eine Log-Datei auswählen."
}
```

### Fehler: Log nicht gefunden (404 Not Found)
```json
{
  "detail": "Log-Datei nicht gefunden: xyz.txt"
}
```

### Fehler: Validierung (422 Unprocessable Entity)
```json
{
  "detail": "Validierungsfehler bei der Anfrage",
  "errors": [
    "body -> epochs: ensure this value is less than or equal to 200 (type: value_error.number.not_le)",
    "body -> model_name: ensure this value has at least 2 characters (type: value_error.any_str.min_length)"
  ],
  "body": "..."
}
```

## Nächste Schritte

1. Starten Sie den Server neu (falls noch nicht geschehen)
2. Führen Sie `test_train_request.py` aus
3. Überprüfen Sie die Debug-Ausgaben
4. Wenn das Problem weiterhin besteht, teilen Sie die Server-Logs und die Fehlerausgabe

