# Android App README

## Überblick
Die Android-App in diesem Ordner ist ein Client für den `Server Log Analyzer`.
Sie kann:
- Modelle und Logs vom Backend laden
- Log-Auswertung starten
- ONNX-Download-URLs pro Modell anzeigen

## Projektstruktur
- `app/` Android-App-Modul
- `build.gradle`, `settings.gradle` Gradle-Konfiguration

## Voraussetzungen
- Android Studio (aktuell)
- Android SDK installiert (inkl. Emulator + platform-tools)
- Backend läuft auf Port `8000`

## Backend starten
Im Backend-Ordner:

```powershell
cd ..\backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Wichtig: Für Emulator/Device muss das Backend über `0.0.0.0` erreichbar sein.

## Android-Projekt öffnen
1. Android Studio öffnen
2. `android-app/` als Projekt öffnen
3. Gradle Sync abwarten

## Emulator starten
Beispiel per CLI:

```powershell
C:\Users\mail\AppData\Local\Android\Sdk\emulator\emulator.exe -list-avds
C:\Users\mail\AppData\Local\Android\Sdk\emulator\emulator.exe -avd Pixel_9_Pro_XL
```

Wenn `adb` offline ist:

```powershell
C:\Users\mail\AppData\Local\Android\Sdk\platform-tools\adb.exe kill-server
C:\Users\mail\AppData\Local\Android\Sdk\platform-tools\adb.exe start-server
C:\Users\mail\AppData\Local\Android\Sdk\platform-tools\adb.exe devices
```

## App starten (Android Studio)
1. Run Configuration: `app`
2. Zielgerät: laufender Emulator oder echtes Device
3. Run drücken

## Backend URL in der App
Im UI-Feld `Backend URL (WLAN)`:
- Emulator (Android Studio):
	- `http://10.0.2.2:8000`
- Echtes Smartphone im gleichen WLAN:
	- `http://<PC-LAN-IP>:8000` (z. B. `http://192.168.0.10:8000`)

## Modell-Download-URLs im UI
Nach `Modelle & Logs laden` zeigt die App:
- Modellnamen
- ONNX-Download-URLs pro Modell (aus `GET /models`)

Beispiel URL:
- `http://<backend>/models/prod_log_autoencoder_v1/onnx`

## Modell-Überschreiben verhindert
Das Backend überschreibt bestehende Modelle nicht.
Wenn ein Name schon existiert, wird automatisch hochgezählt:
- Beispiel: `prod_log_autoencoder_v1` → `prod_log_autoencoder_2`

## Produktionsmodell-Dateien
Im Backend unter `backend/artifacts/<model_name>/`:
- `model.onnx` (Single-File, Android-ready)
- `model.pt` (PyTorch)
- `meta.json`
- `vectorizer.pkl`

## Troubleshooting

### 1) `adb device offline`
- Emulator neu starten
- `adb kill-server` / `adb start-server`

### 2) App erreicht Backend nicht
- Backend läuft wirklich?
- Richtige URL verwendet? (`10.0.2.2` im Emulator)
- Windows Firewall Port `8000` freigeben

### 3) Keine Modelle in der App
- Erst ein Modell trainieren (`/train`)
- Dann in der App `Modelle & Logs laden`

