# DEPENDENCIES
from enum import Enum
from typing import List
from typing import Dict
from typing import Literal
from pydantic import Field
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from pydantic import validator


# ENUMS
class EmotionMode(str, Enum):
    BASE     = 'base'
    GRANULAR = 'granular'
    BOTH     = 'both'


class ExportFormat(str, Enum):
    JSON = 'json'
    CSV  = 'csv'
    PDF  = 'pdf'


class AnalysisStatus(str, Enum):
    PENDING    = 'pending'
    PROCESSING = 'processing'
    COMPLETED  = 'completed'
    FAILED     = 'failed'


# REQUEST MODELS
class AnalysisRequest(BaseModel):
    """
    Request model for audio analysis
    """
    filepath      : str                    = Field(..., description = "Path to audio file")
    emotion_mode  : EmotionMode            = Field(EmotionMode.BOTH, description = "Emotion analysis mode")
    language      : Optional[str]          = Field(None, description = "Target language for transcription")
    export_format : Optional[ExportFormat] = Field(None, description = "Export format")
    

    @validator('filepath')
    def validate_filepath(cls, v):
        if (not v or (len(v.strip()) == 0)):
            raise ValueError('Filepath cannot be empty')

        return v



class BatchAnalysisRequest(BaseModel):
    """
    Request model for batch analysis
    """
    filepaths           : List[str]              = Field(..., min_items=1, max_items = 50, description = "List of audio file paths")
    emotion_mode        : EmotionMode            = Field(EmotionMode.BOTH, description = "Emotion analysis mode")
    export_format       : Optional[ExportFormat] = Field(None, description = "Export format")
    parallel_processing : bool                   = Field(True, description = "Enable parallel processing")


class StreamingChunkRequest(BaseModel):
    """
    Request model for streaming audio chunk
    """
    chunk_index : int   = Field(..., ge = 0, description = "Index of the audio chunk")
    audio_data  : bytes = Field(..., description = "Raw audio data")


class ExportRequest(BaseModel):
    """
    Request model for exporting analysis results
    """
    analysis_id            : str          = Field(..., description = "Analysis ID")
    format                 : ExportFormat = Field(ExportFormat.JSON, description = "Export format")
    include_visualizations : bool         = Field(False, description = "Include attention visualizations")



# RESPONSE MODELS
class EmotionScore(BaseModel):
    """
    Emotion score model
    """
    label      : str   = Field(..., description = "Emotion label")
    score      : float = Field(..., ge = 0.0, le = 1.0, description = "Confidence score")
    percentage : str   = Field(..., description = "Formatted percentage")


class GranularEmotion(BaseModel):
    """
    Granular emotion model
    """
    emotions   : List[str] = Field(..., description = "List of granular emotion labels")
    confidence : float     = Field(..., ge = 0.0, le = 1.0, description = "Confidence score")
    percentage : str       = Field(..., description = "Formatted percentage")


class ComplexEmotion(BaseModel):
    """
    Complex emotion model
    """
    name       : str       = Field(..., description = "Complex emotion name")
    components : List[str] = Field(..., description = "Component emotions")
    confidence : float     = Field(..., ge = 0.0, le = 1.0, description = "Confidence score")
    percentage : str       = Field(..., description = "Formatted percentage")


class EmotionResult(BaseModel):
    """
    Emotion analysis result
    """
    base      : List[EmotionScore]             = Field(..., description = "Base emotion scores")
    primary   : Optional[GranularEmotion]      = Field(None, description = "Primary granular emotions")
    secondary : Optional[GranularEmotion]      = Field(None, description = "Secondary granular emotions")
    complex   : Optional[List[ComplexEmotion]] = Field(None, description = "Complex emotions")


class TranscriptionResult(BaseModel):
    """
    Transcription result
    """
    text          : str             = Field(..., description = "Transcribed text")
    language      : str             = Field(..., description = "Detected language")
    language_code : str             = Field(..., description = "Language code")
    confidence    : Optional[float] = Field(None, description = "Transcription confidence")


class AnalysisResult(BaseModel):
    """
    Complete analysis result
    """
    transcription   : TranscriptionResult = Field(..., description = "Transcription results")
    emotions        : EmotionResult       = Field(..., description = "Emotion analysis results")
    metadata        : Dict                = Field(default_factory = dict, description = "Additional metadata")
    processing_time : float               = Field(..., description = "Processing time in seconds")


class ChunkResult(BaseModel):
    """
    Chunk analysis result
    """
    chunk_index  : int            = Field(..., description = "Chunk index")
    total_chunks : int            = Field(..., description = "Total number of chunks")
    result       : AnalysisResult = Field(..., description = "Analysis result for chunk")


class AnalysisResponse(BaseModel):
    """
    Response model for analysis
    """
    analysis_id  : str                      = Field(..., description = "Unique analysis ID")
    status       : AnalysisStatus           = Field(..., description = "Analysis status")
    result       : Optional[AnalysisResult] = Field(None, description = "Analysis result")
    error        : Optional[str]            = Field(None, description = "Error message if failed")
    created_at   : datetime                 = Field(default_factory = datetime.utcnow, description = "Creation timestamp")
    completed_at : Optional[datetime]       = Field(None, description = "Completion timestamp")


class BatchAnalysisResponse(BaseModel):
    """
    Response model for batch analysis
    """
    batch_id        : str                    = Field(..., description = "Unique batch ID")
    status          : AnalysisStatus         = Field(..., description = "Batch status")
    total_files     : int                    = Field(..., description = "Total number of files")
    completed_files : int                    = Field(0, description = "Number of completed files")
    results         : List[AnalysisResponse] = Field(default_factory = list, description = "Individual results")
    created_at      : datetime               = Field(default_factory = datetime.utcnow, description = "Creation timestamp")


class ExportResponse(BaseModel):
    """
    Response model for export
    """
    export_id    : str          = Field(..., description = "Unique export ID")
    format       : ExportFormat = Field(..., description = "Export format")
    filepath     : str          = Field(..., description = "Path to exported file")
    download_url : str          = Field(..., description = "Download URL")
    created_at   : datetime     = Field(default_factory = datetime.utcnow, description = "Creation timestamp")



# ERROR MODELS
class ErrorDetail(BaseModel):
    """
    Error detail model
    """
    field      : Optional[str] = Field(None, description = "Field that caused the error")
    message    : str           = Field(..., description = "Error message")
    error_type : str           = Field(..., description = "Error type")


class ErrorResponse(BaseModel):
    """
    Error response model
    """
    status    : Literal['error']            = Field('error', description = "Status")
    message   : str                         = Field(..., description = "Error message")
    details   : Optional[List[ErrorDetail]] = Field(None, description = "Error details")
    timestamp : datetime                    = Field(default_factory = datetime.utcnow, description = "Error timestamp")



# STATUS MODELS
class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    status          : Literal['healthy', 'unhealthy'] = Field(..., description = "Service status")
    version         : str                             = Field(..., description = "API version")
    models_loaded   : bool                            = Field(..., description = "Whether models are loaded")
    redis_connected : bool                            = Field(..., description = "Redis connection status")
    timestamp       : datetime                        = Field(default_factory = datetime.utcnow, description = "Check timestamp")


class StatusResponse(BaseModel):
    """
    Status response for async tasks
    """
    task_id  : str                      = Field(..., description = "Task ID")
    status   : AnalysisStatus           = Field(..., description = "Task status")
    progress : Optional[float]          = Field(None, ge = 0.0, le = 100.0, description = "Progress percentage")
    message  : Optional[str]            = Field(None, description = "Status message")
    result   : Optional[AnalysisResult] = Field(None, description = "Result if completed")



# EXPLAINABILITY MODELS
class AttentionWeight(BaseModel):
    """
    Attention weight for explainability
    """
    layer   : int         = Field(..., description = "Layer index")
    head    : int         = Field(..., description = "Attention head index")
    weights : List[float] = Field(..., description = "Attention weights")
    tokens  : List[str]   = Field(..., description = "Corresponding tokens")


class ExplainabilityResult(BaseModel):
    """
    Explainability analysis result
    """
    analysis_id        : str                   = Field(..., description = "Associated analysis ID")
    attention_weights  : List[AttentionWeight] = Field(..., description = "Attention weights")
    feature_importance : Dict[str, float]      = Field(..., description = "Feature importance scores")
    visualization_urls : Dict[str, str]        = Field(..., description = "URLs to visualizations")



# BENCHMARK MODELS
class BenchmarkResult(BaseModel):
    """
    Benchmark result model
    """
    dataset            : str              = Field(..., description = "Dataset name")
    accuracy           : float            = Field(..., ge = 0.0, le = 1.0, description = "Overall accuracy")
    per_class_accuracy : Dict[str, float] = Field(..., description = "Per-class accuracy")
    confusion_matrix   : List[List[int]]  = Field(..., description = "Confusion matrix")
    avg_inference_time : float            = Field(..., description = "Average inference time (ms)")
    total_samples      : int              = Field(..., description = "Total samples processed")


class BenchmarkResponse(BaseModel):
    """
    Benchmark response model
    """
    benchmark_id : str                   = Field(..., description = "Benchmark run ID")
    results      : List[BenchmarkResult] = Field(..., description = "Benchmark results")
    timestamp    : datetime              = Field(default_factory = datetime.utcnow, description = "Benchmark timestamp")
