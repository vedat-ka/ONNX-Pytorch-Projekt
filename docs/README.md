# Lokales MCP mit Ollama in VS Code

Diese Workspace-README beschreibt die lokale MCP-Einrichtung für VS Code mit Ollama.

## Ziel

- MCP-Server lokal in VS Code nutzen
- Lokale Modelle aus Ollama verwenden
- Bestehenden Projektcode unverändert lassen

## Bereits eingerichtet

- Workspace-Konfiguration: `.vscode/mcp.json`
- Globale User-Konfiguration: `C:/Users/mail/AppData/Roaming/Code/User/mcp.json`
- MCP-Server: `ollama-local`
- Startkommando: `npx -y ollama-mcp`
- Ollama-Host: `http://127.0.0.1:11434`

## Voraussetzungen

- Ollama installiert
- Node.js + npx installiert
- Lokaler Ollama-Dienst läuft
- VS Code mit **beiden** Extensions:
	- `GitHub Copilot` (`github.copilot`)
	- `GitHub Copilot Chat` (`github.copilot-chat`)

Prüfen:

```powershell
ollama --version
node --version
npx --version
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/tags
```

## VS Code: MCP starten

1. `Ctrl+Shift+P` öffnen
2. `MCP: Add Server` ausführen (sicherster Einstieg)
3. `ollama-local` auswählen und starten
4. Trust-Dialog bestätigen

Optional hilfreich:

- `MCP: Open Workspace Folder Configuration`
- `MCP: Open User Configuration`
- `MCP: List Servers`
- `MCP: Show Installed Servers`
- `MCP: Reset Trust`

Wenn `MCP: List Servers` und `MCP: Show Installed Servers` keine Treffer liefern:

1. `Ctrl+Shift+P` -> `MCP: Open Workspace Folder MCP Configuration` öffnen
2. Prüfen, dass `.vscode/mcp.json` vorhanden ist
3. Extensions prüfen: `GitHub Copilot` **und** `GitHub Copilot Chat` müssen beide installiert sein
4. Falls `GitHub Copilot` fehlt: installieren und bei GitHub anmelden
5. VS Code neu laden: `Developer: Reload Window`
6. Danach testen: `MCP: Add Server`, `MCP: List Servers`, `MCP: Show Installed Servers`
7. Chat öffnen (`Ctrl+Alt+I`) und bei Tools/Configure Tools nach `ollama-local` suchen

Schnellcheck per Terminal:

```powershell
code --version
code --list-extensions --show-versions
```

Wenn nur `github.copilot-chat@0.37.x` angezeigt wird, ist die MCP-Funktion oft noch nicht verfügbar.
Dann VS Code + Copilot Chat im Extensions-View aktualisieren und VS Code neu starten.

## Modelle prüfen

```powershell
ollama list
```

Beispiel vorhandene Modelle (abhängig von deiner Maschine):

- `qwen2.5-coder:7b`
- `deepseek-r1:7b`
- `glm-4.7-flash:latest`

## Schnelltest im Chat

Beispiel-Prompt in VS Code Chat:

> Nutze das Tool zum Auflisten der Ollama-Modelle und gib mir die Ausgabe kompakt zurück.

Wenn der Server korrekt läuft, sollte der Agent Tool-Aufrufe von `ollama-local` anbieten.

## Fehlerbehebung

### Server startet nicht

- Prüfen, ob `npx -y ollama-mcp` im Terminal startet
- Prüfen, ob Ollama läuft (`http://127.0.0.1:11434/api/tags`)
- In VS Code: `MCP: List Servers` -> Server auswählen -> `Show Output`

### Keine Tools sichtbar

- Server neu starten
- Trust zurücksetzen (`MCP: Reset Trust`) und erneut bestätigen
- Prüfen, ob `mcp.json` gültiges JSON enthält

## Hinweis zur Konfigurations-Priorität

Sowohl Workspace als auch User-Config enthalten `ollama-local`. Für dieses Projekt wird üblicherweise die Workspace-Konfiguration genutzt.
