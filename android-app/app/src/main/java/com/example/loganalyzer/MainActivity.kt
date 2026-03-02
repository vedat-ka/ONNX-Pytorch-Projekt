package com.example.loganalyzer

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ListView
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.google.android.material.textfield.TextInputEditText
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.max
import kotlin.math.sqrt
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class MainActivity : AppCompatActivity() {

    /**
     * MODEL-BASED PIPELINE (On-Device)
     *
     * - Backend liefert nur Artefakte/Daten: Meta, ONNX, Logzeilen.
     * - Android berechnet lokal: TF-IDF Features -> ONNX Inferenz -> Scores -> Endpoint-Insights.
     * - Keine Backend-/analyze-Auswertung für den eigentlichen Score-Pfad.
     */

    private lateinit var etBaseUrl: TextInputEditText
    private lateinit var spinnerModels: Spinner
    private lateinit var lvLogs: ListView
    private lateinit var tvModels: TextView
    private lateinit var tvModelUrls: TextView
    private lateinit var tvLogs: TextView
    private lateinit var tvResult: TextView
    private lateinit var tvSummary: TextView
    private lateinit var chartTopScores: BarChart
    private lateinit var chartEndpointRisk: BarChart
    private lateinit var tvTopAnomalies: TextView
    private lateinit var tvFrequentEndpoints: TextView
    private lateinit var tvHealthyEndpoints: TextView
    private lateinit var tvRiskyEndpoints: TextView
    private lateinit var tvPerFile: TextView

    private lateinit var modelAdapter: ArrayAdapter<String>
    private lateinit var logsAdapter: ArrayAdapter<String>

    private var cachedModels: List<String> = emptyList()
    private var cachedLogs: List<String> = emptyList()

    private val timestampRegex = Regex("\\b\\d{4}[-/]\\d{2}[-/]\\d{2}[ T]\\d{2}:\\d{2}:\\d{2}(?:[.,]\\d+)?(?:Z|[+-]\\d{2}:?\\d{2})?\\b")
    private val ipRegex = Regex("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b")
    private val uuidRegex = Regex("\\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\b", RegexOption.IGNORE_CASE)
    private val hexRegex = Regex("\\b0x[0-9a-f]+\\b", RegexOption.IGNORE_CASE)
    private val pathRegex = Regex("(?:[a-zA-Z]:\\\\\\\\|/)[^\\s]+")
    private val longIdRegex = Regex("\\b[a-z0-9_-]{12,}\\b", RegexOption.IGNORE_CASE)
    private val numberRegex = Regex("\\b\\d+\\b")
    private val spacesRegex = Regex("\\s+")
    private val removePlaceholdersRegex = Regex("<\\s*(?:ts|ip|uuid|hex|path|id|num|jwt|secret_kv|payload)\\s*>", RegexOption.IGNORE_CASE)
    private val removeMetaRegex = Regex("\\b(?:ts|time|timestamp|lvl|level|msg|logger|details|func|file|line|app|info|debug|warning|warn|error|critical|traceback|py)\\b:?")
    private val tokenRegex = Regex("\\b\\w\\w+\\b")
    private val wsTagRegex = Regex("\\[ws\\]\\s*([^|]+)", RegexOption.IGNORE_CASE)
    private val socketDisconnectRegex = Regex("socket\\s+disconnect", RegexOption.IGNORE_CASE)
    private val socketFuncRegex = Regex("func:(handle_[a-z0-9_]+)", RegexOption.IGNORE_CASE)
    private val actionMethodRegex = Regex("\\b([a-z0-9][a-z0-9 _/\\-]{2,80}?)\\s+(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\\b", RegexOption.IGNORE_CASE)
    private val exceptionOnRegex = Regex("exception on\\s+([^\\s]+)\\s+\\[(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\\]", RegexOption.IGNORE_CASE)
    private val methodUrlRegex = Regex("\\b(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\\s+https?://[^/\\s]+([^\\s\\]\\\"']*)", RegexOption.IGNORE_CASE)
    private val methodPathRegex = Regex("\\b(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\\s+(/[^\\s\\]\\\"']*)", RegexOption.IGNORE_CASE)
    private val endpointKeyRegex = Regex("(?:endpoint|path|route)[:=]\\s*([^\\s,;]+)", RegexOption.IGNORE_CASE)
    private val loggerRegex = Regex("\\blogger:([a-z0-9_.-]+)", RegexOption.IGNORE_CASE)
    private val funcRegex = Regex("\\bfunc:([a-z0-9_]+)", RegexOption.IGNORE_CASE)
    private val displaySafeRegex = Regex("[^\\p{L}\\p{N}\\p{P}\\p{Zs}]+")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        etBaseUrl = findViewById(R.id.etBaseUrl)
        spinnerModels = findViewById(R.id.spinnerModels)
        lvLogs = findViewById(R.id.lvLogs)
        tvModels = findViewById(R.id.tvModels)
        tvModelUrls = findViewById(R.id.tvModelUrls)
        tvLogs = findViewById(R.id.tvLogs)
        tvResult = findViewById(R.id.tvResult)
        tvSummary = findViewById(R.id.tvSummary)
        chartTopScores = findViewById(R.id.chartTopScores)
        chartEndpointRisk = findViewById(R.id.chartEndpointRisk)
        tvTopAnomalies = findViewById(R.id.tvTopAnomalies)
        tvFrequentEndpoints = findViewById(R.id.tvFrequentEndpoints)
        tvHealthyEndpoints = findViewById(R.id.tvHealthyEndpoints)
        tvRiskyEndpoints = findViewById(R.id.tvRiskyEndpoints)
        tvPerFile = findViewById(R.id.tvPerFile)

        modelAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, mutableListOf())
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerModels.adapter = modelAdapter

        logsAdapter = ArrayAdapter(this, android.R.layout.simple_list_item_multiple_choice, mutableListOf())
        lvLogs.adapter = logsAdapter

        spinnerModels.setOnItemSelectedListener(object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val selected = modelAdapter.getItem(position).orEmpty()
                if (selected.isNotBlank()) {
                    tvResult.text = "Ausgewähltes Modell: $selected"
                }
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {
            }
        })

        findViewById<Button>(R.id.btnLoad).setOnClickListener { loadModelsAndLogs() }
        findViewById<Button>(R.id.btnAnalyze).setOnClickListener { analyzeLogs() }
    }

    /**
     * Erstellt den Retrofit-API-Client mit der aktuell gesetzten Backend-URL.
     *
     * @return Konfigurierter [ApiService] für alle Backend-Aufrufe.
     */
    private fun apiService(): ApiService {
        val base = normalizedBaseUrl()
        val retrofit = Retrofit.Builder()
            .baseUrl(base)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        return retrofit.create(ApiService::class.java)
    }

    /**
     * Normalisiert die Basis-URL aus dem Eingabefeld.
     *
     * @return URL mit garantiertem abschließendem `/`, damit Retrofit-Pfade korrekt aufgelöst werden.
     */
    private fun normalizedBaseUrl(): String {
        val base = etBaseUrl.text?.toString()?.trim().orEmpty()
        return if (base.endsWith("/")) base else "$base/"
    }

    /**
     * Lädt verfügbare Modelle und Logs vom Backend und aktualisiert die UI-Auswahl.
     *
     * @return `Unit` (UI-State wird direkt aktualisiert).
     */
    private fun loadModelsAndLogs() {
        lifecycleScope.launch {
            try {
                val api = apiService()
                val modelsResponse = api.getModels()
                val models = modelsResponse.models
                val logs = api.getLogs().logs
                cachedModels = models
                cachedLogs = logs
                modelAdapter.clear()
                modelAdapter.addAll(models)
                modelAdapter.notifyDataSetChanged()
                if (models.isNotEmpty()) {
                    spinnerModels.setSelection(0)
                }

                logsAdapter.clear()
                logsAdapter.addAll(logs)
                logsAdapter.notifyDataSetChanged()
                for (i in logs.indices) {
                    lvLogs.setItemChecked(i, false)
                }

                tvModels.text = "Modelle: ${if (models.isEmpty()) "-" else models.joinToString()}"
                val base = normalizedBaseUrl().removeSuffix("/")
                val onnxUrls = modelsResponse.onnx_download_urls.ifEmpty {
                    models.associateWith { "$base/models/$it/onnx" }
                }
                tvModelUrls.text = if (onnxUrls.isEmpty()) {
                    "Model Download URLs: -"
                } else {
                    "Model Download URLs:\n" + onnxUrls.entries.joinToString("\n") { (name, url) -> "$name -> $url" }
                }
                tvLogs.text = "Logs: ${if (logs.isEmpty()) "-" else logs.joinToString()}"
                tvResult.text = "Analyse: Bereit"
                tvSummary.text = "Zusammenfassung: Bitte Analyse starten"
            } catch (ex: Exception) {
                tvResult.text = "Fehler: ${ex.message}"
            }
        }
    }

    /**
     * Startet die Analyse mit aktuell ausgewähltem Modell und ausgewählten Logs.
     *
     * Validiert Auswahlzustand und führt danach die lokale modellbasierte Pipeline aus.
     *
     * @return `Unit` (Ergebnis wird in die UI geschrieben).
     */
    private fun analyzeLogs() {
        lifecycleScope.launch {
            try {
                val selectedPosition = spinnerModels.selectedItemPosition
                val modelName = if (selectedPosition >= 0) {
                    modelAdapter.getItem(selectedPosition).orEmpty()
                } else {
                    ""
                }
                if (modelName.isBlank()) {
                    tvResult.text = "Bitte ein Modell auswählen."
                    return@launch
                }
                if (cachedModels.isNotEmpty() && !cachedModels.contains(modelName)) {
                    tvResult.text = "Modell nicht geladen: $modelName"
                    return@launch
                }
                if (cachedLogs.isEmpty()) {
                    tvResult.text = "Bitte zuerst Modelle/Logs laden."
                    return@launch
                }

                val selectedLogs = mutableListOf<String>()
                for (i in cachedLogs.indices) {
                    if (lvLogs.isItemChecked(i)) {
                        selectedLogs.add(cachedLogs[i])
                    }
                }
                if (selectedLogs.isEmpty()) {
                    tvResult.text = "Bitte mindestens einen Log auswählen."
                    return@launch
                }

                val api = apiService()
                tvResult.text = "Lade Modell-Metadaten..."
                // Modellbasierte Analyse läuft lokal auf dem Gerät:
                // Metadaten + ONNX + Logzeilen laden -> Features bauen -> ONNX-Inferenz -> UI-Resultat
                val result = withContext(Dispatchers.IO) {
                    runLocalOnnxAnalysis(api = api, modelName = modelName, selectedLogs = selectedLogs)
                }
                renderAnalysis(result, selectedLogs)
            } catch (ex: Exception) {
                tvResult.text = "Fehler: ${ex.message}"
            }
        }
    }

    /**
     * Führt die komplette lokale Modell-Auswertung aus.
     *
     * 1) Meta laden (`input_dim`, `threshold`, `tfidf`)
     * 2) ONNX lokal bereitstellen (Cache/Download)
     * 3) Logzeilen laden und in Modell-Input transformieren
     * 4) ONNX-Scores berechnen und als Analyseobjekt aufbereiten
        *
        * @param api Retrofit-Client für Metadaten-, Modell- und Logabruf.
        * @param modelName Gewählter Modellname aus dem Dropdown.
        * @param selectedLogs Vom Benutzer ausgewählte Logdateien für die Analyse.
        * @return Vollständig berechnetes [AnalyzeResponse]-Objekt für die UI.
     */
    private suspend fun runLocalOnnxAnalysis(api: ApiService, modelName: String, selectedLogs: List<String>): AnalyzeResponse {
        val meta = api.getModelMeta(modelName)
        val tfidf = meta.tfidf ?: throw IllegalStateException("TF-IDF Metadaten fehlen im Modell.")
        if (meta.input_dim <= 0 || tfidf.terms.isEmpty() || tfidf.idf.isEmpty()) {
            throw IllegalStateException("Ungültige Modell-Metadaten für lokale ONNX-Analyse.")
        }

        val modelFile = ensureOnnxModelFile(api, modelName)

        val rawLines = mutableListOf<String>()
        val lineSources = mutableListOf<String>()
        for (logName in selectedLogs) {
            val payload = api.getLogLines(logName)
            val lines = payload.lines.filter { it.isNotBlank() }
            rawLines.addAll(lines)
            lineSources.addAll(List(lines.size) { logName })
        }
        if (rawLines.isEmpty()) {
            throw IllegalStateException("Keine Logzeilen für Analyse geladen.")
        }

        val prepared = rawLines.map { prepareLineForModel(it) }
        val features = buildTfidfMatrix(prepared, tfidf, meta.input_dim)
        val scores = runOnnxInference(modelFile, features, rawLines.size, meta.input_dim)
        return buildLocalAnalysisResponse(
            modelName = modelName,
            lines = rawLines,
            sources = lineSources,
            preparedLines = prepared,
            scores = scores,
            threshold = meta.threshold,
            selectedLogs = selectedLogs,
        )
    }

    /**
     * ONNX-Datei-Handling für das Gerät:
     * - nutzt vorhandene Datei aus App-Storage
     * - lädt nur bei Bedarf neu vom Backend
        *
        * @param api Retrofit-Client für den ONNX-Download.
        * @param modelName Name des Modells, das lokal vorliegen muss.
        * @return Lokale [File]-Referenz auf das verwendete ONNX-Modell.
     */
    private suspend fun ensureOnnxModelFile(api: ApiService, modelName: String): File {
        val modelDir = File(filesDir, "models/$modelName")
        if (!modelDir.exists()) {
            modelDir.mkdirs()
        }
        val modelFile = File(modelDir, "$modelName.onnx")
        if (modelFile.exists() && modelFile.length() > 0) {
            return modelFile
        }

        val body = api.downloadModelOnnx(modelName)
        body.byteStream().use { input ->
            FileOutputStream(modelFile).use { output ->
                input.copyTo(output)
            }
        }
        if (!modelFile.exists() || modelFile.length() == 0L) {
            throw IllegalStateException("ONNX-Modell konnte nicht gespeichert werden.")
        }
        return modelFile
    }

    /**
     * ONNX-Forward-Pass und Score-Bildung.
     * Score je Zeile = MSE(Input-Feature, Rekonstruktion).
        *
        * @param modelFile Lokale ONNX-Datei.
        * @param featureMatrix Flaches Feature-Array der Form `[rowCount * inputDim]`.
        * @param rowCount Anzahl der analysierten Zeilen.
        * @param inputDim Feature-Dimension pro Zeile.
        * @return Score-Array mit einem Rekonstruktionsfehler pro Zeile.
     */
    private fun runOnnxInference(modelFile: File, featureMatrix: FloatArray, rowCount: Int, inputDim: Int): DoubleArray {
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        session.use { ortSession ->
            val shape = longArrayOf(rowCount.toLong(), inputDim.toLong())
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(featureMatrix), shape)
            inputTensor.use { tensor ->
                val output = ortSession.run(mapOf("input" to tensor))
                output.use { ortResult ->
                    val reconstructed = ortResult[0].value as Array<FloatArray>
                    val scores = DoubleArray(rowCount)
                    for (row in 0 until rowCount) {
                        var mse = 0.0
                        val reconstructedRow = reconstructed[row]
                        val base = row * inputDim
                        for (col in 0 until inputDim) {
                            val diff = featureMatrix[base + col] - reconstructedRow[col]
                            mse += (diff * diff)
                        }
                        scores[row] = mse / max(1, inputDim)
                    }
                    return scores
                }
            }
        }
    }

    /**
     * Baut die TF-IDF Matrix exakt nach Modell-Meta (`terms`, `idf`) auf.
     * Diese Indexreihenfolge muss zum trainierten ONNX-Eingang passen.
        *
        * @param lines Vorverarbeitete Modell-Input-Zeilen.
        * @param tfidf Modell-Metadaten mit `terms` und `idf`.
        * @param inputDim Erwartete ONNX-Input-Dimension.
        * @return Flaches `FloatArray` mit TF-IDF-Features.
     */
    private fun buildTfidfMatrix(lines: List<String>, tfidf: TfidfConfig, inputDim: Int): FloatArray {
        val terms = tfidf.terms
        val idf = tfidf.idf
        val termToIndex = HashMap<String, Int>(terms.size)
        terms.forEachIndexed { index, token ->
            if (index < inputDim) {
                termToIndex[token] = index
            }
        }

        val matrix = FloatArray(lines.size * inputDim)
        for ((rowIndex, line) in lines.withIndex()) {
            val counts = HashMap<Int, Int>()
            val matches = tokenRegex.findAll(line)
            for (m in matches) {
                val token = m.value.lowercase(Locale.getDefault())
                val idx = termToIndex[token] ?: continue
                counts[idx] = (counts[idx] ?: 0) + 1
            }

            var norm2 = 0.0
            val rowOffset = rowIndex * inputDim
            for ((idx, tf) in counts) {
                val idfValue = if (idx < idf.size) idf[idx] else 1.0
                val value = tf * idfValue
                matrix[rowOffset + idx] = value.toFloat()
                norm2 += value * value
            }

            if (norm2 > 0.0) {
                val norm = sqrt(norm2).toFloat()
                for ((idx, _) in counts) {
                    matrix[rowOffset + idx] = matrix[rowOffset + idx] / norm
                }
            }
        }
        return matrix
    }

    /**
     * Leitet alle sichtbaren Analysewerte aus Modell-Scores ab:
     * - Anomalien über `score > threshold`
     * - Endpoint-Risiko über modellbasierte `anomaly_rate` je Endpoint
     * - Ausgabe für Charts, Tabellen und Per-File-Blöcke
        *
        * @param modelName Name des analysierten Modells.
        * @param lines Originale Logzeilen (zusammengeführt aus den gewählten Dateien).
        * @param sources Dateiquellen pro Zeile (gleiche Länge wie `lines`).
        * @param preparedLines Modell-Input pro Zeile nach Vorverarbeitung.
        * @param scores Rekonstruktionsfehler je Zeile aus ONNX-Inferenz.
        * @param threshold Schwellwert für Anomalie-Erkennung.
        * @param selectedLogs Vom Nutzer ausgewählte Logdateien.
        * @return Vollständige Analyse-Struktur für alle UI-Bereiche.
     */
    private fun buildLocalAnalysisResponse(
        modelName: String,
        lines: List<String>,
        sources: List<String>,
        preparedLines: List<String>,
        scores: DoubleArray,
        threshold: Double,
        selectedLogs: List<String>,
    ): AnalyzeResponse {
        val anomalyIndices = scores.indices.filter { scores[it] > threshold }.toSet()
        val globalAnomalyRate = anomalyIndices.size.toDouble() / max(1, lines.size)
        val scoreMean = scores.average()
        val scoreStd = sqrt(scores.map { (it - scoreMean) * (it - scoreMean) }.average()).coerceAtLeast(1e-9)

        val endpointPerLine = lines.map { inferEndpoint(it) ?: "unknown" }
        val severityPerLine = lines.map { inferSeverity(it) }
        val weakPerLine = lines.mapIndexed { idx, _ -> inferWeakLabel(severityPerLine[idx], endpointPerLine[idx]) }

        val topScoreItems = scores.indices
            .sortedByDescending { scores[it] }
            .take(10)
            .map { idx ->
                TopScoreItem(
                    label = chartLabel(idx, lines[idx], endpointPerLine[idx]),
                    score = scores[idx],
                    operational_risk = max(0.0, (scores[idx] - threshold) / max(1e-9, threshold)),
                )
            }

        val topAnomalies = scores.indices
            .filter { scores[it] > threshold }
            .sortedByDescending { scores[it] }
            .take(50)
            .map { idx ->
                val sev = severityPerLine[idx]
                val risk = max(0.0, (scores[idx] - scoreMean) / scoreStd)
                TopAnomalyItem(
                    line_index = idx,
                    endpoint = endpointPerLine[idx],
                    weak_label = weakPerLine[idx],
                    severity = sev,
                    score = scores[idx],
                    operational_risk = risk,
                    endpoint_zscore = risk,
                    occurrences = 1,
                    line_normalized = shorten(preparedLines[idx], 260),
                )
            }

        val endpointHits = mutableMapOf<String, Int>()
        val endpointAnomalyHits = mutableMapOf<String, Int>()
        endpointPerLine.forEachIndexed { idx, endpoint ->
            endpointHits[endpoint] = (endpointHits[endpoint] ?: 0) + 1
            if (idx in anomalyIndices) {
                endpointAnomalyHits[endpoint] = (endpointAnomalyHits[endpoint] ?: 0) + 1
            }
        }

        val candidateRates = endpointHits.entries
            .filter { it.key != "unknown" && it.value >= 3 }
            .map { endpointAnomalyHits.getOrDefault(it.key, 0).toDouble() / max(1, it.value) }
            .sorted()

        val lowQuantile = percentile(candidateRates, 0.25)
        val highQuantile = percentile(candidateRates, 0.75)

        val endpointItems = endpointHits.entries.map { (endpoint, hits) ->
            val anomalyHits = endpointAnomalyHits[endpoint] ?: 0
            val anomalyRate = anomalyHits.toDouble() / max(1, hits)
            val status = when {
                hits < 3 -> "unknown"
                endpoint == "unknown" -> "unknown"
                anomalyRate >= highQuantile -> "risky"
                anomalyRate <= lowQuantile -> "healthy"
                else -> "unknown"
            }
            val reason = when (status) {
                "risky" -> "model_quantile_high_anomaly_rate"
                "healthy" -> "model_quantile_low_anomaly_rate"
                else -> if (hits < 3) "insufficient_samples" else "model_mid_quantile"
            }
            EndpointItem(
                endpoint = endpoint,
                stability_status = status,
                status_reason = reason,
                total_hits = hits,
                error_hits = 0,
                error_rate = 0.0,
                anomaly_rate = anomalyRate,
                anomaly_lift = anomalyRate - globalAnomalyRate,
            )
        }

        val endpointInsights = EndpointInsights(
            total_detected_endpoints = endpointItems.size,
            model_signal_reliable = globalAnomalyRate in 0.02..0.70,
            model_signal_reason = when {
                globalAnomalyRate > 0.70 -> "saturated_high"
                globalAnomalyRate < 0.02 -> "saturated_low"
                else -> "ok"
            },
            global_anomaly_rate = globalAnomalyRate,
            frequent_endpoints = endpointItems.sortedByDescending { it.total_hits }.take(10),
            healthy_endpoints = endpointItems
                .filter { it.endpoint != "unknown" && it.total_hits >= 3 }
                .sortedWith(compareBy<EndpointItem> { it.anomaly_rate }.thenByDescending { it.total_hits })
                .take(10)
                .map { item ->
                    item.copy(
                        stability_status = "healthy",
                        status_reason = "model_lowest_anomaly_rate"
                    )
                },
            risky_endpoints = endpointItems.filter { it.stability_status == "risky" && it.endpoint != "unknown" }
                .sortedByDescending { it.anomaly_lift }
                .take(10),
        )

        val perFile = selectedLogs.mapNotNull { fileName ->
            val fileIndices = sources.indices.filter { sources[it] == fileName }
            if (fileIndices.isEmpty()) return@mapNotNull null
            val fileAnomalyCount = fileIndices.count { it in anomalyIndices }
            val fileEndpointHits = mutableMapOf<String, Int>()
            val fileEndpointAnomaly = mutableMapOf<String, Int>()
            for (idx in fileIndices) {
                val endpoint = endpointPerLine[idx]
                fileEndpointHits[endpoint] = (fileEndpointHits[endpoint] ?: 0) + 1
                if (idx in anomalyIndices) {
                    fileEndpointAnomaly[endpoint] = (fileEndpointAnomaly[endpoint] ?: 0) + 1
                }
            }

            val topEndpoints = fileEndpointHits.entries.sortedByDescending { it.value }.take(5).map { (endpoint, hits) ->
                val anomalyHits = fileEndpointAnomaly[endpoint] ?: 0
                PerFileEndpointItem(
                    endpoint = endpoint,
                    hits = hits,
                    anomaly_hits = anomalyHits,
                    anomaly_rate = anomalyHits.toDouble() / max(1, hits),
                )
            }

            PerFileAnalysisItem(
                file_name = fileName,
                total_lines = fileIndices.size,
                anomaly_count = fileAnomalyCount,
                anomaly_rate = fileAnomalyCount.toDouble() / max(1, fileIndices.size),
                top_endpoints = topEndpoints,
            )
        }

        return AnalyzeResponse(
            model = modelName,
            total_logs = lines.size,
            threshold = threshold,
            anomaly_count = anomalyIndices.size,
            anomaly_rate = globalAnomalyRate,
            actionable_anomaly_count = anomalyIndices.size,
            actionable_unique_count = topAnomalies.size,
            suppressed_benign_count = 0,
            top_scores = topScoreItems,
            top_anomalies = topAnomalies,
            per_file_analysis = perFile,
            endpoint_insights = endpointInsights,
        )
    }

    /**
     * Normalisiert eine einzelne Logzeile in das Modell-Featureformat.
     *
     * @param line Roh-Logzeile.
     * @return Modell-Input-String mit severity/weak/endpoint Token + bereinigtem Text.
     */
    private fun prepareLineForModel(line: String): String {
        val severity = inferSeverity(line)
        val endpoint = inferEndpoint(line)
        val weak = inferWeakLabel(severity, endpoint ?: "unknown")

        var normalized = line.lowercase(Locale.getDefault()).trim()
        normalized = timestampRegex.replace(normalized, " <ts> ")
        normalized = ipRegex.replace(normalized, " <ip> ")
        normalized = uuidRegex.replace(normalized, " <uuid> ")
        normalized = hexRegex.replace(normalized, " <hex> ")
        normalized = pathRegex.replace(normalized, " <path> ")
        normalized = longIdRegex.replace(normalized, " <id> ")
        normalized = numberRegex.replace(normalized, " <num> ")

        var compact = removePlaceholdersRegex.replace(normalized, " ")
        compact = removeMetaRegex.replace(compact, " ")
        compact = compact.replace(Regex("[{}\\[\\]()<>=|]+"), " ")
        compact = compact.replace(Regex("[.,;:]+"), " ")
        compact = compact.replace("\\", " ")
        compact = compact.replace("\"", " ")
        compact = compact.replace("'", " ")
        compact = compact.replace("`", " ")
        compact = compact.replace(Regex("\\b[a-zA-Z]\\b"), " ")
        compact = spacesRegex.replace(compact, " ").trim()

        val endpointToken = endpointToken(endpoint)
        val payload = if (compact.isBlank()) "empty_line" else compact
        return "severity_${severity} weak_${weak} $endpointToken $payload"
    }

    /**
     * Kodiert einen Endpoint als Feature-Token (`endpoint_<method>_<path>`).
     *
     * @param endpoint Erkannter Endpoint oder `null`.
     * @return Endpoint-Token für den Modell-Input.
     */
    private fun endpointToken(endpoint: String?): String {
        if (endpoint.isNullOrBlank() || !endpoint.contains(" ")) {
            return "endpoint_unknown"
        }
        val parts = endpoint.split(" ", limit = 2)
        val method = parts[0].lowercase(Locale.getDefault())
        val path = parts[1].lowercase(Locale.getDefault())
            .replace(Regex("[^a-z0-9_<>/]+"), "_")
            .replace("/", "_")
            .trim('_')
        return "endpoint_${method}_${if (path.isBlank()) "root" else path}"
    }

    /**
     * Leitet eine Severity-Klasse aus einer Logzeile ab.
     *
     * @param line Roh-Logzeile.
     * @return Eine Klasse aus `critical|error|warn|info|debug|unknown`.
     */
    private fun inferSeverity(line: String): String {
        val lowered = line.lowercase(Locale.getDefault())
        if (listOf("critical", "fatal", "panic", "emergency", "segfault").any { lowered.contains(it) }) return "critical"
        if (listOf("error", "exception", "failed", "failure", "traceback").any { lowered.contains(it) }) return "error"
        if (listOf("warn", "warning", "deprecated", "retry").any { lowered.contains(it) }) return "warn"
        if (listOf("info", "started", "startup", "listening", "connected").any { lowered.contains(it) }) return "info"
        if (listOf("debug", "verbose", "trace").any { lowered.contains(it) }) return "debug"
        return "unknown"
    }

    /**
     * Erzeugt ein Weak-Label aus Severity und Endpoint-Verfügbarkeit.
     *
     * @param severity Aus [inferSeverity] abgeleitete Klasse.
     * @param endpoint Erkannter Endpoint oder `unknown`.
     * @return Weak-Label für Feature- und Tabellenkontext.
     */
    private fun inferWeakLabel(severity: String, endpoint: String): String {
        val hasEndpoint = endpoint != "unknown"
        if (severity == "critical" || severity == "error") return if (hasEndpoint) "endpoint_error" else "error_without_endpoint"
        if (severity == "warn") return if (hasEndpoint) "endpoint_warn" else "warn_without_endpoint"
        if (hasEndpoint) return "endpoint_normal"
        return "unknown"
    }

    /**
     * Extrahiert den Endpoint aus HTTP/WS/Logger-Mustern.
     *
     * @param line Roh-Logzeile.
     * @return Normalisierter Endpoint oder `null`, wenn nichts erkannt wurde.
     */
    private fun inferEndpoint(line: String): String? {
        val wsMatch = wsTagRegex.find(line)
        if (wsMatch != null) {
            val event = wsMatch.groupValues[1].trim().lowercase(Locale.getDefault())
                .replace(Regex("[^a-z0-9]+"), "_")
                .trim('_')
            return "WS /socket/${if (event.isBlank()) "event" else event}"
        }

        if (socketDisconnectRegex.containsMatchIn(line)) {
            return "WS /socket/disconnect"
        }

        val socketFuncMatch = socketFuncRegex.find(line)
        val loweredLine = line.lowercase(Locale.getDefault())
        if (socketFuncMatch != null && (loweredLine.contains("app.socket") || loweredLine.contains("logger:socket"))) {
            val funcName = socketFuncMatch.groupValues[1].lowercase(Locale.getDefault()).replace("handle_", "")
            val funcSlug = funcName.replace(Regex("[^a-z0-9_]+"), "_").trim('_').ifBlank { "event" }
            return "WS /socket/$funcSlug"
        }

        val actionMethodMatch = actionMethodRegex.find(line)
        if (actionMethodMatch != null) {
            val action = actionMethodMatch.groupValues[1].trim().lowercase(Locale.getDefault())
            val method = actionMethodMatch.groupValues[2].uppercase(Locale.getDefault())
            if (!action.contains("http") && !action.contains("logger:") && !action.contains("details:")) {
                val actionSlug = action.replace(Regex("[^a-z0-9]+"), "_").trim('_')
                if (actionSlug.isNotBlank()) {
                    return "$method /event/$actionSlug"
                }
            }
        }

        val exceptionMatch = exceptionOnRegex.find(line)
        if (exceptionMatch != null) {
            val path = exceptionMatch.groupValues[1]
            val method = exceptionMatch.groupValues[2].uppercase(Locale.getDefault())
            return "$method ${normalizeEndpointPath(path)}"
        }

        val methodPath = methodPathRegex.find(line)
        if (methodPath != null) {
            val method = methodPath.groupValues[1].uppercase(Locale.getDefault())
            val path = normalizeEndpointPath(methodPath.groupValues[2])
            return "$method $path"
        }

        val methodUrl = methodUrlRegex.find(line)
        if (methodUrl != null) {
            val method = methodUrl.groupValues[1].uppercase(Locale.getDefault())
            val rawPath = if (methodUrl.groupValues[2].isBlank()) "/" else methodUrl.groupValues[2]
            val path = normalizeEndpointPath(rawPath)
            return "$method $path"
        }

        val endpointMatch = endpointKeyRegex.find(line)
        if (endpointMatch != null) {
            return "UNK ${normalizeEndpointPath(endpointMatch.groupValues[1])}"
        }

        val loggerMatch = loggerRegex.find(line)
        val funcMatch = funcRegex.find(line)
        if (loggerMatch != null || funcMatch != null) {
            val loggerName = (loggerMatch?.groupValues?.get(1)?.lowercase(Locale.getDefault()) ?: "unknown_logger").replace('.', '_')
            val funcName = funcMatch?.groupValues?.get(1)?.lowercase(Locale.getDefault()) ?: "unknown_func"
            val loggerSlug = loggerName.replace(Regex("[^a-z0-9_]+"), "_").trim('_').ifBlank { "unknown_logger" }
            val funcSlug = funcName.replace(Regex("[^a-z0-9_]+"), "_").trim('_').ifBlank { "unknown_func" }
            return "INT /internal/$loggerSlug/$funcSlug"
        }

        return null
    }

    /**
     * Normalisiert Endpoint-Pfade (IDs/UUID/Hex usw. werden abstrahiert).
     *
     * @param path Roh-Pfad aus der Logzeile.
     * @return Normalisierter Pfad für Gruppierung und Vergleich.
     */
    private fun normalizeEndpointPath(path: String): String {
        val clean = path.substringBefore("?").trim().ifBlank { "/" }
        val normalized = if (clean.startsWith("/")) clean else "/$clean"
        val segments = normalized.split("/").filter { it.isNotBlank() }.map { segment ->
            val s = segment.lowercase(Locale.getDefault())
            when {
                uuidRegex.matches(s) -> "<uuid>"
                hexRegex.matches(s) -> "<hex>"
                s.all { it.isDigit() } -> "<num>"
                longIdRegex.matches(s) -> "<id>"
                else -> s
            }
        }
        return if (segments.isEmpty()) "/" else "/" + segments.joinToString("/")
    }

    /**
     * Baut ein kurzes Label für Chart-Balken.
     *
     * @param index Zeilenindex in der Gesamtanalyse.
     * @param rawLine Originale Logzeile.
     * @param endpoint Erkannter Endpoint.
     * @return Endpoint oder gekürztes Fallback-Label.
     */
    private fun chartLabel(index: Int, rawLine: String, endpoint: String): String {
        if (endpoint != "unknown") return endpoint
        return "#$index ${shorten(rawLine, 28)}"
    }

    /**
     * Legacy-Hilfsfunktion für Severity-basierte Risikoaufschläge.
     *
     * @param severity Severity-Wert.
     * @return Numerischer Bonuswert.
     */
    private fun severityRiskBonus(severity: String): Double = when (severity) {
        "critical" -> 2.5
        "error" -> 2.0
        "warn" -> 1.0
        else -> 0.0
    }

    /**
     * Legacy-Hilfsfunktion zur heuristischen Fehlererkennung im Logtext.
     *
     * @param line Roh-Logzeile.
     * @param severity Vorher bestimmte Severity.
     * @return `true`, wenn die Zeile als Fehler gewertet wird.
     */
    private fun isErrorLine(line: String, severity: String): Boolean {
        if (severity == "critical" || severity == "error") return true
        val lowered = line.lowercase(Locale.getDefault())
        return listOf("exception", "traceback", "failed", " error ", " 500", " 502", " 503", " 504").any { lowered.contains(it) }
    }

    /**
     * Rendert alle Analyseergebnisse in die Android-UI-Komponenten.
     *
     * @param result Modellbasierte Analysewerte.
     * @param selectedLogs Vom Nutzer ausgewählte Logdateien.
     * @return `Unit`.
     */
    private fun renderAnalysis(result: AnalyzeResponse, selectedLogs: List<String>) {
        tvResult.text = "Analyse: Modell=${result.model} | Logs=${selectedLogs.size}"
        tvSummary.text = (
            "Gesamt=${result.total_logs} | Anomalien=${result.anomaly_count} (${formatPercent(result.anomaly_rate)})\n" +
                "Operativ relevant=${result.actionable_anomaly_count} | Eindeutig=${result.actionable_unique_count} | " +
                "Benigne gefiltert=${result.suppressed_benign_count}"
            )

        renderTopScoresChart(result.top_scores)
        renderEndpointRiskChart(result.endpoint_insights?.risky_endpoints ?: emptyList())

        tvTopAnomalies.text = if (result.top_anomalies.isEmpty()) {
            "-"
        } else {
            result.top_anomalies.take(20).joinToString("\n\n") { item ->
                "#${item.line_index} ${sanitizeDisplayText(item.endpoint)} | ${item.severity} | ${item.weak_label}\n" +
                    "score=${formatDecimal(item.score)} risk=${formatDecimal(item.operational_risk)} z=${formatDecimal(item.endpoint_zscore)} occ=${item.occurrences}\n" +
                    shorten(sanitizeDisplayText(item.line_normalized), 180)
            }
        }

        val insights = result.endpoint_insights
        if (insights == null) {
            tvFrequentEndpoints.text = "-"
            tvHealthyEndpoints.text = "-"
            tvRiskyEndpoints.text = "-"
        } else {
            val header = "Erkannte Endpunkte=${insights.total_detected_endpoints} | Modellsignal=${if (insights.model_signal_reliable) "zuverlässig" else "nicht zuverlässig"} (${insights.model_signal_reason})"

            tvFrequentEndpoints.text = header + "\n\n" + endpointBlock(insights.frequent_endpoints, withLift = true)
            tvHealthyEndpoints.text = endpointBlock(insights.healthy_endpoints, withLift = false)
            tvRiskyEndpoints.text = endpointBlock(insights.risky_endpoints, withLift = true)
        }

        tvPerFile.text = if (result.per_file_analysis.isEmpty()) {
            "-"
        } else {
            result.per_file_analysis.joinToString("\n\n") { file ->
                val topEndpoints = if (file.top_endpoints.isEmpty()) {
                    "-"
                } else {
                    file.top_endpoints.joinToString("; ") {
                        "${sanitizeDisplayText(it.endpoint)} (hits=${it.hits}, anom=${it.anomaly_hits}, rate=${formatPercent(it.anomaly_rate)})"
                    }
                }
                "${file.file_name}: lines=${file.total_lines}, anomalies=${file.anomaly_count}, rate=${formatPercent(file.anomaly_rate)}\nTop: $topEndpoints"
            }
        }
    }

    /**
     * Formatiert eine Endpoint-Liste für Textdarstellung.
     *
     * @param items Endpoint-Einträge.
     * @param withLift Gibt an, ob `anomaly_lift` angezeigt werden soll.
     * @return Mehrzeiliger String für UI-Textfelder.
     */
    private fun endpointBlock(items: List<EndpointItem>, withLift: Boolean): String {
        if (items.isEmpty()) return "-"
        return items.take(12).joinToString("\n") { item ->
            val base = "${sanitizeDisplayText(item.endpoint)} | status=${item.stability_status} (${item.status_reason}) | hits=${item.total_hits} | err=${formatPercent(item.error_rate)} | anom=${formatPercent(item.anomaly_rate)}"
            if (withLift) "$base | lift=${formatPercent(item.anomaly_lift)}" else base
        }
    }

    /**
     * Rendert das Top-Score Balkendiagramm.
     *
     * @param items Top-Score Einträge.
     * @return `Unit`.
     */
    private fun renderTopScoresChart(items: List<TopScoreItem>) {
        val top = items.take(10)
        if (top.isEmpty()) {
            chartTopScores.clear()
            chartTopScores.invalidate()
            return
        }

        val labels = top.map { shorten(it.label, 18) }
        val entries = top.mapIndexed { index, item -> BarEntry(index.toFloat(), item.score.toFloat()) }
        val dataSet = BarDataSet(entries, "Anomaly Score").apply {
            color = 0xFF3B82F6.toInt()
            valueTextSize = 10f
        }
        val barData = BarData(dataSet)
        barData.barWidth = 0.9f

        chartTopScores.data = barData
        configureBarChart(chartTopScores, labels)
    }

    /**
     * Rendert das Endpoint-Risiko-Diagramm (modellbasierte Anomaly-Rate).
     *
     * @param items Endpoint-Einträge für das Risiko-Chart.
     * @return `Unit`.
     */
    private fun renderEndpointRiskChart(items: List<EndpointItem>) {
        val top = items.take(8)
        if (top.isEmpty()) {
            chartEndpointRisk.clear()
            chartEndpointRisk.invalidate()
            return
        }

        val labels = top.map { shorten(it.endpoint, 16) }
        val entries = top.mapIndexed { index, item -> BarEntry(index.toFloat(), (item.anomaly_rate * 100.0).toFloat()) }
        val dataSet = BarDataSet(entries, "Anomaly-Rate %").apply {
            color = 0xFFEF4444.toInt()
            valueTextSize = 10f
        }
        val barData = BarData(dataSet)
        barData.barWidth = 0.9f

        chartEndpointRisk.data = barData
        configureBarChart(chartEndpointRisk, labels)
    }

    /**
     * Wendet gemeinsame Stil- und Achsenkonfiguration auf ein Balkendiagramm an.
     *
     * @param chart Ziel-Chart.
     * @param labels X-Achsenlabels.
     * @return `Unit`.
     */
    private fun configureBarChart(chart: BarChart, labels: List<String>) {
        chart.description.isEnabled = false
        chart.setFitBars(true)
        chart.axisRight.isEnabled = false
        chart.legend.isEnabled = true
        chart.setScaleEnabled(false)
        chart.setPinchZoom(false)

        val xAxis = chart.xAxis
        xAxis.position = XAxis.XAxisPosition.BOTTOM
        xAxis.valueFormatter = IndexAxisValueFormatter(labels)
        xAxis.granularity = 1f
        xAxis.labelRotationAngle = -35f
        xAxis.setDrawGridLines(false)

        chart.axisLeft.axisMinimum = 0f
        chart.invalidate()
    }

    /**
     * Formatiert einen Anteil als Prozentstring.
     *
     * @param value Anteil in [0..1].
     * @return Prozentformat, z. B. `12.34%`.
     */
    private fun formatPercent(value: Double): String = String.format("%.2f%%", value * 100.0)

    /**
     * Formatiert eine Gleitkommazahl auf drei Nachkommastellen.
     *
     * @param value Eingabewert.
     * @return Formatierter Dezimalstring.
     */
    private fun formatDecimal(value: Double): String = String.format("%.3f", value)

    /**
     * Quantil-Funktion (linear interpoliert) für modellbasierte Schwellen.
     * Wird genutzt, um Endpoint-Status aus Score-Verteilungen abzuleiten.
        *
        * @param sortedValues Aufsteigend sortierte Werte.
        * @param q Ziel-Quantil in [0, 1].
        * @return Interpolierter Quantilswert.
     */
    private fun percentile(sortedValues: List<Double>, q: Double): Double {
        if (sortedValues.isEmpty()) return 0.0
        if (sortedValues.size == 1) return sortedValues[0]
        val clamped = q.coerceIn(0.0, 1.0)
        val pos = clamped * (sortedValues.size - 1)
        val low = pos.toInt()
        val high = kotlin.math.ceil(pos).toInt()
        if (low == high) return sortedValues[low]
        val w = pos - low
        return sortedValues[low] * (1.0 - w) + sortedValues[high] * w
    }

    /**
     * Entfernt nicht darstellbare/unerwünschte Sonderzeichen für die UI-Ausgabe.
     *
     * @param text Eingabetext.
     * @return Bereinigter, lesbarer Text.
     */
    private fun sanitizeDisplayText(text: String): String {
        val cleaned = displaySafeRegex.replace(text, " ")
        return spacesRegex.replace(cleaned, " ").trim()
    }

    /**
     * Kürzt lange Strings mit `...` für kompakte Darstellung.
     *
     * @param value Eingabetext.
     * @param max Maximale Länge ohne Suffix.
     * @return Gekürzter oder unveränderter Text.
     */
    private fun shorten(value: String, max: Int): String {
        if (value.length <= max) return value
        return value.take(max) + "..."
    }
}
