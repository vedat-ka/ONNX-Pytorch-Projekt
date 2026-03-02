package com.example.loganalyzer

import okhttp3.ResponseBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Streaming

interface ApiService {
    @GET("/models")
    suspend fun getModels(): ModelsResponse

    @GET("/logs")
    suspend fun getLogs(): LogsResponse

    @GET("/logs/{log_name}/lines")
    suspend fun getLogLines(@Path("log_name") logName: String): LogLinesResponse

    @GET("/models/{model_name}/meta")
    suspend fun getModelMeta(@Path("model_name") modelName: String): ModelMetaResponse

    @Streaming
    @GET("/models/{model_name}/onnx")
    suspend fun downloadModelOnnx(@Path("model_name") modelName: String): ResponseBody

    @POST("/analyze")
    suspend fun analyze(@Body request: AnalyzeRequest): AnalyzeResponse
}
