"""
Test-Skript für den /train Endpoint
Hilft dabei, 400 Bad Request Fehler zu diagnostizieren
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Prüfe ob der Server läuft"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code} - {response.json()}")
    return response.status_code == 200

def list_logs():
    """Liste verfügbare Log-Dateien auf"""
    response = requests.get(f"{BASE_URL}/logs")
    if response.status_code == 200:
        logs = response.json().get("logs", [])
        print(f"\nVerfügbare Log-Dateien ({len(logs)}):")
        for log in logs:
            print(f"  - {log}")
        return logs
    else:
        print(f"Fehler beim Abrufen der Logs: {response.status_code}")
        return []

def test_train_minimal():
    """Teste /train mit minimalen Parametern"""
    print("\n" + "="*60)
    print("TEST 1: Training mit minimalen Parametern")
    print("="*60)

    logs = list_logs()
    if not logs:
        print("❌ FEHLER: Keine Log-Dateien verfügbar!")
        print("   Bitte laden Sie zuerst Log-Dateien hoch.")
        return

    payload = {
        "model_name": "test_model_v1",
        "selected_logs": [logs[0]],  # Erste verfügbare Log-Datei
        # Alle anderen Felder verwenden Standardwerte
    }

    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    if response.status_code == 200:
        print("\n✅ Training erfolgreich!")
    else:
        print("\n❌ Training fehlgeschlagen!")

def test_train_full():
    """Teste /train mit allen Parametern"""
    print("\n" + "="*60)
    print("TEST 2: Training mit allen Parametern")
    print("="*60)

    logs = list_logs()
    if not logs:
        print("❌ FEHLER: Keine Log-Dateien verfügbar!")
        return

    payload = {
        "model_name": "test_model_full",
        "selected_logs": logs[:min(2, len(logs))],  # Bis zu 2 Log-Dateien
        "epochs": 15,
        "batch_size": 64,
        "learning_rate": 0.0015,
        "hidden_dim": 128,
        "max_features": 512,
        "threshold_quantile": 0.92
    }

    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    if response.status_code == 200:
        print("\n✅ Training erfolgreich!")
    else:
        print("\n❌ Training fehlgeschlagen!")

def test_train_empty_logs():
    """Teste /train mit leerer Log-Liste (sollte 400 geben)"""
    print("\n" + "="*60)
    print("TEST 3: Training mit leerer Log-Liste (Fehlerfall)")
    print("="*60)

    payload = {
        "model_name": "test_model_error",
        "selected_logs": [],  # Leer!
    }

    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    if response.status_code == 400:
        print("\n✅ Fehler wie erwartet (400 Bad Request)")
    else:
        print("\n❌ Unerwartete Antwort!")

def test_train_invalid_log():
    """Teste /train mit nicht existierender Log-Datei (sollte 404 geben)"""
    print("\n" + "="*60)
    print("TEST 4: Training mit nicht existierender Log-Datei (Fehlerfall)")
    print("="*60)

    payload = {
        "model_name": "test_model_error2",
        "selected_logs": ["does_not_exist.txt"],
    }

    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))

    response = requests.post(f"{BASE_URL}/train", json=payload)
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Body:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    if response.status_code == 404:
        print("\n✅ Fehler wie erwartet (404 Not Found)")
    else:
        print("\n❌ Unerwartete Antwort!")

if __name__ == "__main__":
    print("="*60)
    print("FastAPI /train Endpoint Test Suite")
    print("="*60)

    # Prüfe ob Server läuft
    if not test_health():
        print("\n❌ Server ist nicht erreichbar!")
        print("   Bitte starten Sie den Server mit:")
        print("   python -m uvicorn app:app --reload")
        exit(1)

    # Führe Tests aus
    test_train_empty_logs()
    test_train_invalid_log()
    test_train_minimal()
    # test_train_full()  # Auskommentiert da Training länger dauert

    print("\n" + "="*60)
    print("Tests abgeschlossen!")
    print("="*60)

