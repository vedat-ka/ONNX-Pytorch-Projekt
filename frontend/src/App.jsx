import { useEffect, useMemo, useState } from 'react';
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { createClient } from './services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const defaultConfig = {
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.001,
  hidden_dim: 64,
  transformer_d_model: 64,
  transformer_num_heads: 8,
  transformer_num_layers: 2,
  transformer_d_ff: 256,
  transformer_dropout: 0.1,
  max_features: 256,
  threshold_quantile: 0.95,
  training_rounds: 5,
};

const settingHelpText = {
  epochs: 'Anzahl Trainingsdurchläufe (Epochen) pro Trainingsrunde.',
  batch_size: 'Anzahl Samples, die gleichzeitig pro Optimierungsschritt verarbeitet werden.',
  learning_rate: 'Schrittweite des Optimizers. Zu hoch kann instabil sein, zu niedrig trainiert langsam.',
  hidden_dim: 'Größe der versteckten Schicht im Autoencoder.',
  transformer_d_model: 'Modell-Dimension des Transformers (muss durch num_heads teilbar sein).',
  transformer_num_heads: 'Anzahl Attention-Köpfe im Multi-Head-Attention-Block.',
  transformer_num_layers: 'Anzahl Encoder-/Decoder-Layer im Transformer.',
  transformer_d_ff: 'Dimension des Feed-Forward-Netzwerks im Transformer-Layer.',
  transformer_dropout: 'Dropout-Rate zur Regularisierung (0.0 bis 0.8).',
  max_features: 'Maximale Anzahl TF-IDF-Features für die Log-Vektorisierung.',
  threshold_quantile: 'Quantil zur Ableitung des Anomalie-Schwellenwerts aus Trainingsscores.',
  training_rounds: 'Wie oft komplett trainiert wird; bester Lauf wird übernommen.',
};

const TRAIN_MODEL_BASENAME_BY_TYPE = {
  autoencoder: 'prod_log_autoencoder',
  transformer: 'prod_log_transformer',
};

export default function App() {
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:8000');
  const [modelType, setModelType] = useState('autoencoder');
  const [modelName, setModelName] = useState('');
  const [settings, setSettings] = useState(defaultConfig);
  const [models, setModels] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedTrainLogs, setSelectedTrainLogs] = useState([]);
  const [selectedAnalyzeLogs, setSelectedAnalyzeLogs] = useState([]);
  const [files, setFiles] = useState([]);
  const [trainResult, setTrainResult] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [preprocessPreview, setPreprocessPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('Bereit');
  const [dialogMessage, setDialogMessage] = useState('');

  const showDialog = (text) => {
    setMessage(text);
    setDialogMessage(text);
  };

  const client = useMemo(() => createClient(baseUrl), [baseUrl]);

  const toggleTrainLog = (name) => {
    setSelectedTrainLogs((prev) =>
      prev.includes(name) ? prev.filter((item) => item !== name) : [...prev, name]
    );
  };

  const toggleAnalyzeLog = (name) => {
    setSelectedAnalyzeLogs((prev) =>
      prev.includes(name) ? prev.filter((item) => item !== name) : [...prev, name]
    );
  };

  const refreshLists = async () => {
    const [modelsRes, logsRes] = await Promise.all([client.get('/models'), client.get('/logs')]);
    const loadedModels = modelsRes.data.models || [];
    const loadedLogs = logsRes.data.logs || [];

    setModels(loadedModels);
    setLogs(loadedLogs);
    setSelectedTrainLogs((prev) => prev.filter((item) => loadedLogs.includes(item)));
    setSelectedAnalyzeLogs((prev) => prev.filter((item) => loadedLogs.includes(item)));

    if (loadedModels.length && !loadedModels.includes(modelName)) {
      setModelName(loadedModels[0]);
    }
  };

  useEffect(() => {
    refreshLists().catch((error) => {
      showDialog(error.response?.data?.detail || `Modelle/Logs laden fehlgeschlagen: ${error.message}`);
    });
  }, [client]);

  const onUpload = async () => {
    if (!files.length) {
      showDialog('Bitte Dateien auswählen.');
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      [...files].forEach((file) => form.append('files', file));
      await client.post('/upload-logs', form, { headers: { 'Content-Type': 'multipart/form-data' } });
      await refreshLists();
      setMessage('Upload erfolgreich.');
    } catch (error) {
      showDialog(error.response?.data?.detail || error.message);
    } finally {
      setLoading(false);
    }
  };

  const onPreviewPreprocessing = async () => {
    if (!selectedTrainLogs.length) {
      showDialog('Bitte mindestens eine Training-Log-Datei auswählen.');
      return;
    }

    setLoading(true);
    try {
      const res = await client.post('/preprocess-preview', {
        selected_logs: selectedTrainLogs,
        sample_limit: 25,
      });
      setPreprocessPreview(res.data);
      setMessage('Vorverarbeitung geladen.');
    } catch (error) {
      showDialog(error.response?.data?.detail || error.message);
    } finally {
      setLoading(false);
    }
  };

  const onTrain = async () => {
    if (!selectedTrainLogs.length) {
      showDialog('Bitte mindestens eine Training-Log-Datei auswählen.');
      return;
    }

    setLoading(true);
    try {
      const previewRes = await client.post('/preprocess-preview', {
        selected_logs: selectedTrainLogs,
        sample_limit: 25,
      });
      setPreprocessPreview(previewRes.data);

      const payload = {
        model_name: TRAIN_MODEL_BASENAME_BY_TYPE[modelType],
        model_type: modelType,
        selected_logs: selectedTrainLogs,
        ...settings,
      };
      const res = await client.post('/train', payload);
      setTrainResult(res.data);

      if (res.data?.model) {
        setModelName(res.data.model);
      }
      setSelectedAnalyzeLogs((prev) => (prev.length ? prev : [...selectedTrainLogs]));

      await refreshLists();
      setMessage('Training abgeschlossen.');
    } catch (error) {
      showDialog(error.response?.data?.detail || error.message);
    } finally {
      setLoading(false);
    }
  };

  const onAnalyze = async () => {
    if (!modelName) {
      showDialog('Bitte ein Modell auswählen.');
      return;
    }
    if (!selectedAnalyzeLogs.length) {
      showDialog('Bitte mindestens eine Analyse-Log-Datei auswählen.');
      return;
    }

    setLoading(true);
    try {
      const res = await client.post('/analyze', {
        model_name: modelName,
        selected_logs: selectedAnalyzeLogs,
      });
      setAnalysis(res.data);
      setMessage('Analyse abgeschlossen.');
    } catch (error) {
      showDialog(error.response?.data?.detail || error.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = {
    labels: (analysis?.top_scores || []).map((item) => item.label),
    datasets: [
      {
        label: 'Anomaly Score',
        data: (analysis?.top_scores || []).map((item) => item.score),
        backgroundColor: '#3b82f6',
      },
    ],
  };

  const settingKeys =
    modelType === 'transformer'
      ? [
          'epochs',
          'batch_size',
          'learning_rate',
          'transformer_d_model',
          'transformer_num_heads',
          'transformer_num_layers',
          'transformer_d_ff',
          'transformer_dropout',
          'max_features',
          'threshold_quantile',
          'training_rounds',
        ]
      : [
          'epochs',
          'batch_size',
          'learning_rate',
          'hidden_dim',
          'max_features',
          'threshold_quantile',
          'training_rounds',
        ];

  return (
    <div className="container">
      <h1>Server Log Analyzer (PyTorch + ONNX)</h1>
      <p className="mono">Status: {message}</p>

      <div className="card row">
        <div>
          <label>Backend URL (WLAN-fähig)</label>
          <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </div>
      </div>

      <div className="card">
        <h3>Logs hochladen (.json/.txt/.csv)</h3>
        <input type="file" multiple accept=".json,.txt,.csv" onChange={(e) => setFiles(e.target.files)} />
        <div className="actions">
          <button onClick={onUpload} disabled={loading}>Upload</button>
          <button onClick={refreshLists} disabled={loading}>Modelle/Logs laden</button>
          <button onClick={onPreviewPreprocessing} disabled={loading}>Vorverarbeitung anzeigen</button>
        </div>
      </div>

      <div className="card">
        <h3>Logs für Training</h3>
        <div className="list">
          {logs.map((log) => (
            <label key={`tr-${log}`} className="list-item">
              <input type="checkbox" checked={selectedTrainLogs.includes(log)} onChange={() => toggleTrainLog(log)} />
              <span>{log}</span>
            </label>
          ))}
        </div>
        <p className="mono">Ausgewählt für Training: {selectedTrainLogs.length}</p>
      </div>

      {preprocessPreview?.summary && (
        <div className="card">
          <h3>Vor dem Training: Datenpipeline</h3>
          <p>Ausgewählte Logs: {(preprocessPreview.selected_logs || []).join(', ') || '-'}</p>
          <p>Zeilen gesamt: {preprocessPreview.summary.total_lines} | Beispielzeilen: {preprocessPreview.summary.sample_count}</p>

          <h4>Vorher/Nachher + Model-Input</h4>
          <table className="table">
            <thead>
              <tr>
                <th>Original</th>
                <th>Normalisiert</th>
                <th>Severity</th>
                <th>Weak Label</th>
                <th>Endpoint</th>
                <th>Model-Input</th>
              </tr>
            </thead>
            <tbody>
              {(preprocessPreview.summary.samples || []).map((sample, idx) => (
                <tr key={idx}>
                  <td>{sample.original}</td>
                  <td>{sample.normalized}</td>
                  <td>{sample.severity}</td>
                  <td>{sample.weak_label}</td>
                  <td>{sample.endpoint}</td>
                  <td>{sample.model_input}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <h3>Training-Einstellungen</h3>
        <h4>Modellarchitektur (eindeutig auswählbar)</h4>
        <div className="list">
          <label className="list-item">
            <input
              type="radio"
              name="modelType"
              checked={modelType === 'autoencoder'}
              onChange={() => setModelType('autoencoder')}
            />
            <span>Autoencoder (Linear + ReLU)</span>
          </label>
          <label className="list-item">
            <input
              type="radio"
              name="modelType"
              checked={modelType === 'transformer'}
              onChange={() => setModelType('transformer')}
            />
            <span>Transformer (MultiHeadAttention + Encoder/Decoder)</span>
          </label>
        </div>
        <p className="mono">
          Gewählt: {modelType} | Modellname-Basis: {TRAIN_MODEL_BASENAME_BY_TYPE[modelType]}
        </p>
        <p className="mono">Training nutzt nur "Logs für Training".</p>
        <div className="row">
          {settingKeys.map((key) => (
            <div key={key}>
              <label>
                {key}{' '}
                <span
                  title={settingHelpText[key] || 'Keine Beschreibung verfügbar.'}
                  aria-label={`Info zu ${key}`}
                  style={{ cursor: 'help' }}
                >
                  ⓘ
                </span>
              </label>
              <input
                type="number"
                step={key.includes('learning') || key.includes('quantile') || key.includes('dropout') ? '0.001' : '1'}
                value={settings[key]}
                onChange={(e) => setSettings((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
              />
            </div>
          ))}
        </div>
        <div className="actions">
          <button onClick={onTrain} disabled={loading}>Training starten</button>
        </div>
      </div>

      <div className="card row">
        <div>
          <h3>Logs für Analyse</h3>
          <div className="list">
            {logs.map((log) => (
              <label key={`an-${log}`} className="list-item">
                <input type="checkbox" checked={selectedAnalyzeLogs.includes(log)} onChange={() => toggleAnalyzeLog(log)} />
                <span>{log}</span>
              </label>
            ))}
          </div>
          <p className="mono">Ausgewählt für Analyse: {selectedAnalyzeLogs.length}</p>
        </div>

        <div>
          <h3>Verfügbare Modelle (für Analyse)</h3>
          <div className="list">
            {models.map((model) => (
              <label key={model} className="list-item">
                <input type="radio" name="model" checked={modelName === model} onChange={() => setModelName(model)} />
                <span>{model}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Analyse (unabhängig vom Training)</h3>
        <p>Modell: {modelName || '-'} | Analyse-Logs: {selectedAnalyzeLogs.length}</p>
        <p className="mono">
          Ausgewählte Analyse-Logs: {selectedAnalyzeLogs.length ? selectedAnalyzeLogs.join(', ') : '-'}
        </p>
        <div className="actions">
          <button onClick={onAnalyze} disabled={loading || !modelName || !selectedAnalyzeLogs.length}>
            Ausgewähltes Modell auf Analyse-Logs anwenden
          </button>
        </div>
      </div>

      {trainResult && (
        <div className="card">
          <h3>Training Ergebnis</h3>
          <p>
            <span className="badge">Modell: {trainResult.model}</span>{' '}
            <span className="badge">Typ: {trainResult.model_type || 'autoencoder'}</span>{' '}
            <span className="badge">ONNX: {trainResult.onnx_file}</span>
          </p>
          <p>Threshold: {trainResult.threshold?.toFixed(6)}</p>
          <p>Input-Features (TF-IDF): {trainResult.input_features}</p>
          {trainResult.training_quality && <p className="mono">{trainResult.training_quality.note}</p>}
        </div>
      )}

      {analysis && (
        <div className="card">
          <h3>Auswertung</h3>
          <p>Modelltyp: {analysis.model_type || 'autoencoder'}</p>
          <p className="mono">Analysierte Logs: {selectedAnalyzeLogs.length ? selectedAnalyzeLogs.join(', ') : '-'}</p>
          <p>Gesamt: {analysis.total_logs} | Anomalien: {analysis.anomaly_count} | Rate: {(analysis.anomaly_rate * 100).toFixed(2)}%</p>
          <p>
            Operativ relevant: {analysis.actionable_anomaly_count ?? analysis.anomaly_count} |
            Eindeutig (dedupliziert): {analysis.actionable_unique_count ?? (analysis.top_anomalies || []).length} |
            Ausgefiltert (benigne Muster): {analysis.suppressed_benign_count ?? 0}
          </p>
          <Bar data={chartData} options={{ responsive: true }} />

          <h4>Top Anomalien</h4>
          <table className="table">
            <thead>
              <tr>
                <th>Line #</th>
                <th>Endpoint</th>
                <th>Weak Label</th>
                <th>Severity</th>
                <th>Score</th>
                <th>Risk</th>
                <th>Endpoint Z</th>
                <th>Occurrences</th>
                <th>Log-Zeile</th>
              </tr>
            </thead>
            <tbody>
              {(analysis.top_anomalies || []).map((item, idx) => (
                <tr key={idx}>
                  <td>{item.line_index}</td>
                  <td>{item.endpoint}</td>
                  <td>{item.weak_label}</td>
                  <td>{item.severity}</td>
                  <td>{item.score.toFixed(6)}</td>
                  <td>{(item.operational_risk ?? 0).toFixed(3)}</td>
                  <td>{(item.endpoint_zscore ?? 0).toFixed(3)}</td>
                  <td>{item.occurrences ?? 1}</td>
                  <td>{item.line_normalized || item.line}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {analysis.endpoint_insights && (
            <>
              <h4>Endpoint-Überblick</h4>
              <p>
                Erkannte Endpunkte: {analysis.endpoint_insights.total_detected_endpoints} | Modell-Signal:{' '}
                {analysis.endpoint_insights.model_signal_reliable ? 'zuverlässig' : 'nicht zuverlässig'}
              </p>

              <h4>Häufig verwendete Endpunkte</h4>
              <table className="table">
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Grund</th>
                    <th>Hits</th>
                    <th>Fehler-Hits</th>
                    <th>Fehler-Rate</th>
                    <th>Anomaly-Rate</th>
                    <th>Anomaly-Lift</th>
                  </tr>
                </thead>
                <tbody>
                  {(analysis.endpoint_insights.frequent_endpoints || []).map((item) => (
                    <tr key={`freq-${item.endpoint}`}>
                      <td>{item.endpoint}</td>
                      <td>{item.stability_status}</td>
                      <td>{item.status_reason}</td>
                      <td>{item.total_hits}</td>
                      <td>{item.error_hits}</td>
                      <td>{(item.error_rate * 100).toFixed(2)}%</td>
                      <td>{(item.anomaly_rate * 100).toFixed(2)}%</td>
                      <td>{(item.anomaly_lift * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4>Gut funktionierende Endpunkte</h4>
              <table className="table">
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Grund</th>
                    <th>Hits</th>
                    <th>Fehler-Rate</th>
                    <th>Anomaly-Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {(analysis.endpoint_insights.healthy_endpoints || []).map((item) => (
                    <tr key={`healthy-${item.endpoint}`}>
                      <td>{item.endpoint}</td>
                      <td>{item.stability_status}</td>
                      <td>{item.status_reason}</td>
                      <td>{item.total_hits}</td>
                      <td>{(item.error_rate * 100).toFixed(2)}%</td>
                      <td>{(item.anomaly_rate * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4>Fehleranfällige Endpunkte</h4>
              <table className="table">
                <thead>
                  <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Grund</th>
                    <th>Hits</th>
                    <th>Fehler-Hits</th>
                    <th>Fehler-Rate</th>
                    <th>Anomaly-Rate</th>
                    <th>Anomaly-Lift</th>
                  </tr>
                </thead>
                <tbody>
                  {(analysis.endpoint_insights.risky_endpoints || []).map((item) => (
                    <tr key={`risky-${item.endpoint}`}>
                      <td>{item.endpoint}</td>
                      <td>{item.stability_status}</td>
                      <td>{item.status_reason}</td>
                      <td>{item.total_hits}</td>
                      <td>{item.error_hits}</td>
                      <td>{(item.error_rate * 100).toFixed(2)}%</td>
                      <td>{(item.anomaly_rate * 100).toFixed(2)}%</td>
                      <td>{(item.anomaly_lift * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {analysis.per_file_analysis && analysis.per_file_analysis.length > 0 && (
            <>
              <h4>Per-File-Auswertung</h4>
              <table className="table">
                <thead>
                  <tr>
                    <th>Datei</th>
                    <th>Zeilen</th>
                    <th>Anomalien</th>
                    <th>Rate</th>
                    <th>Top-Endpunkte</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.per_file_analysis.map((item) => (
                    <tr key={item.file_name}>
                      <td>{item.file_name}</td>
                      <td>{item.total_lines}</td>
                      <td>{item.anomaly_count}</td>
                      <td>{(item.anomaly_rate * 100).toFixed(2)}%</td>
                      <td>
                        {(item.top_endpoints || []).map((endpointItem) => (
                          <div key={`${item.file_name}-${endpointItem.endpoint}`}>
                            {endpointItem.endpoint} ({endpointItem.hits} | A:{endpointItem.anomaly_hits})
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      {dialogMessage && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-card">
            <h3>Hinweis</h3>
            <p>{dialogMessage}</p>
            <div className="actions">
              <button onClick={() => setDialogMessage('')}>Schließen</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
