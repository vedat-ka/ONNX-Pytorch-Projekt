package com.example.loganalyzer

data class ModelsResponse(
    val models: List<String> = emptyList(),
    val onnx_download_urls: Map<String, String> = emptyMap()
)

data class LogsResponse(val logs: List<String> = emptyList())

data class LogLinesResponse(
    val log_name: String = "",
    val lines: List<String> = emptyList()
)

data class ModelMetaResponse(
    val model_name: String = "",
    val input_dim: Int = 0,
    val threshold: Double = 0.0,
    val tfidf: TfidfConfig? = null
)

data class TfidfConfig(
    val terms: List<String> = emptyList(),
    val idf: List<Double> = emptyList()
)

data class AnalyzeRequest(
    val model_name: String,
    val selected_logs: List<String>
)

data class AnalyzeResponse(
    val model: String,
    val total_logs: Int,
    val threshold: Double,
    val anomaly_count: Int,
    val anomaly_rate: Double,
    val actionable_anomaly_count: Int = 0,
    val actionable_unique_count: Int = 0,
    val suppressed_benign_count: Int = 0,
    val top_scores: List<TopScoreItem> = emptyList(),
    val top_anomalies: List<TopAnomalyItem> = emptyList(),
    val per_file_analysis: List<PerFileAnalysisItem> = emptyList(),
    val endpoint_insights: EndpointInsights? = null
)

data class TopScoreItem(
    val label: String = "",
    val score: Double = 0.0,
    val operational_risk: Double = 0.0
)

data class TopAnomalyItem(
    val line_index: Int = 0,
    val endpoint: String = "unknown",
    val weak_label: String = "",
    val severity: String = "",
    val score: Double = 0.0,
    val operational_risk: Double = 0.0,
    val endpoint_zscore: Double = 0.0,
    val occurrences: Int = 1,
    val line_normalized: String = ""
)

data class EndpointInsights(
    val total_detected_endpoints: Int = 0,
    val model_signal_reliable: Boolean = false,
    val model_signal_reason: String = "",
    val global_anomaly_rate: Double = 0.0,
    val frequent_endpoints: List<EndpointItem> = emptyList(),
    val healthy_endpoints: List<EndpointItem> = emptyList(),
    val risky_endpoints: List<EndpointItem> = emptyList()
)

data class EndpointItem(
    val endpoint: String = "",
    val stability_status: String = "unknown",
    val status_reason: String = "",
    val total_hits: Int = 0,
    val error_hits: Int = 0,
    val error_rate: Double = 0.0,
    val anomaly_rate: Double = 0.0,
    val anomaly_lift: Double = 0.0
)

data class PerFileAnalysisItem(
    val file_name: String = "",
    val total_lines: Int = 0,
    val anomaly_count: Int = 0,
    val anomaly_rate: Double = 0.0,
    val top_endpoints: List<PerFileEndpointItem> = emptyList()
)

data class PerFileEndpointItem(
    val endpoint: String = "",
    val hits: Int = 0,
    val anomaly_hits: Int = 0,
    val anomaly_rate: Double = 0.0
)
