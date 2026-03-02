import requests
import json

# Test 1: Health Check
print("=" * 60)
print("Test: Health Check")
print("=" * 60)
try:
    response = requests.get("http://127.0.0.1:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"FEHLER: {e}")
    exit(1)

# Test 2: Liste Logs
print("\n" + "=" * 60)
print("Test: Liste verfügbare Logs")
print("=" * 60)
try:
    response = requests.get("http://127.0.0.1:8000/logs")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Logs: {data.get('logs', [])}")
    logs = data.get('logs', [])
except Exception as e:
    print(f"FEHLER: {e}")
    logs = []

# Test 3: Train mit leerer Liste (sollte 400 geben)
print("\n" + "=" * 60)
print("Test: Train mit leerer Log-Liste")
print("=" * 60)
payload = {
    "model_name": "test_empty",
    "selected_logs": []
}
try:
    response = requests.post("http://127.0.0.1:8000/train", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"FEHLER: {e}")

# Test 4: Train mit gültigen Logs (wenn verfügbar)
if logs:
    print("\n" + "=" * 60)
    print("Test: Train mit gültiger Log-Datei")
    print("=" * 60)
    payload = {
        "model_name": "test_valid",
        "selected_logs": [logs[0]]
    }
    print(f"Payload: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post("http://127.0.0.1:8000/train", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Response Text: {response.text}")
    except Exception as e:
        print(f"FEHLER: {e}")

print("\n" + "=" * 60)
print("Tests abgeschlossen")
print("=" * 60)

