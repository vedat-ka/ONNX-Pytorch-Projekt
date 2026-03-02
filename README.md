# PyTorch + ONNX Server Log Analyzer

Dieses Projekt enthält:
- **Python Backend (FastAPI)** für Upload, Training, ONNX-Export und Analyse
- **React Frontend (Vite)** für Datei-Upload, Trainingseinstellungen, Modell-/Log-Auswahl und Diagramme
- **Android App (Kotlin)** mit WLAN-Backend-Verbindung zur Analyse

## 1) Backend starten (Python)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Start mit Arc-Umgebung

```powershell
cd backend
if (-not (Test-Path .\.venv311\Scripts\Activate.ps1)) { python -m venv .venv311 }
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Optional bei mehreren GPUs: Arc-Adapter wählen (0/1/...)
$env:TORCH_DIRECTML_DEVICE_INDEX="0"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Hinweis: `POST /train` nutzt DirectML (Arc), wenn `torch-directml` verfügbar ist. Falls DirectML nicht verfügbar ist, wird automatisch auf CPU zurückgefallen.

### Wichtige API-Endpunkte
- `POST /upload-logs` (Dateien: `.json`, `.txt`, `.csv`)
- `GET /logs`
- `POST /train`
- `GET /models`
- `POST /analyze`
- `GET /models/{model_name}/meta`
- `GET /models/{model_name}/onnx`

Model-Artefakte landen unter `backend/artifacts/<model_name>/`:
- `model.pt` (PyTorch)
- `model.onnx` (Web/Android/Embedded)
- `vectorizer.pkl`
- `meta.json`

## 2) Frontend starten (React)

```bash
cd frontend
npm install
npm run dev
```

Dann im Browser öffnen (Standard): `http://localhost:5173`

Im Frontend:
1. Backend URL setzen (lokal oder WLAN-IP)
2. Logs hochladen
3. Training-Parameter setzen
4. Modell trainieren
5. Modelle/Logs auswählen und Analyse starten

## 3) Android App (WLAN)

Android-Projekt liegt unter `android-app/`.

1. `android-app` in Android Studio öffnen
2. Gradle Sync ausführen
3. In der App Backend-URL auf WLAN-IP setzen, z. B. `http://192.168.0.10:8000`
4. Backend muss mit `--host 0.0.0.0` laufen
5. Smartphone und Backend-Rechner im selben WLAN

## 4) ONNX Nutzung für Web / Android / Embedded

- ONNX-Datei über `GET /models/{model_name}/onnx` abrufen
- In **Web** z. B. mit `onnxruntime-web`
- In **Android** z. B. mit `onnxruntime-android`
- In **Embedded** (Linux/Edge) z. B. mit `onnxruntime`

## 5) Beispiel-Trainingspayload

```json
{
  "model_name": "log_autoencoder_v1",
  "selected_logs": ["server-1.txt", "server-2.csv"],
  "epochs": 12,
  "batch_size": 32,
  "learning_rate": 0.001,
  "hidden_dim": 64,
  "max_features": 256,
  "threshold_quantile": 0.95
}
```

## 6) Preprocessing vor dem Training

Das Training ist **unsupervised** (Autoencoder), daher gibt es keine manuell gelabelten Zielklassen wie bei klassischer Klassifikation.

Damit das Modell trotzdem robuster lernt, werden Logzeilen vor TF-IDF normalisiert:
- Timestamps → `<ts>`
- IP-Adressen → `<ip>`
- UUIDs/Hex/IDs/Zahlen/Pfade → Platzhalter

Zusätzlich wird ein **schwaches Label als Feature** ergänzt (`severity_critical`, `severity_error`, `severity_warn`, `severity_info`, `severity_debug`, `severity_unknown`), abgeleitet aus Schlüsselwörtern.

Das verbessert Generalisierung auf neue Runs mit anderen Zeitstempeln, IDs und Hosts.
