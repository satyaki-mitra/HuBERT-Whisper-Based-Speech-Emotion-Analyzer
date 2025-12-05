# DEPENDENCIES
import os
import uuid
from pathlib import Path
from flask import request
from flask import jsonify
from flask import send_file
from flask import Blueprint
from datetime import datetime
from api.tasks import celery_app
from config.settings import API_PREFIX
from config.settings import EXPORTS_DIR
from config.models import ErrorResponse
from config.models import AnalysisStatus
from config.models import AnalysisRequest
from werkzeug.utils import secure_filename
from config.models import AnalysisResponse
from config.settings import is_allowed_file
from utils.logging_util import setup_logger
from services.exporters import ExportService
from config.models import HealthCheckResponse
from config.settings import UPLOADED_FILES_DIR
from services.transcriber import get_transcriber
from services.explainer import ExplainabilityService
from benchmarks.benchmark_runner import run_benchmark
from utils.error_handlers import handle_generic_error
from utils.error_handlers import handle_validation_error
from api.tasks import analyze_audio_task, get_task_status
from utils.error_handlers import handle_file_not_found_error
from services.emotion_predictor import get_emotion_predictor
        

# SETUP LOGGING
logger = setup_logger(__name__)


# CREATE BLUEPRINT
api_bp = Blueprint('api', __name__, url_prefix = API_PREFIX)


# HEALTH & STATUS
@api_bp.route('/health', methods = ['GET'])
def health_check():
    """
    Health check endpoint
    """
    try:
        # Check models
        try:
            get_emotion_predictor()
            get_transcriber()
            models_loaded = True

        except:
            models_loaded = False
        
        # Check Redis
        try:
            celery_app.backend.get('test')
            redis_connected = True

        except:
            redis_connected = False
        
        status   = 'healthy' if (models_loaded and redis_connected) else 'unhealthy'
        
        response = HealthCheckResponse(status          = status,
                                       version         = '1.0.0',
                                       models_loaded   = models_loaded,
                                       redis_connected = redis_connected,
                                      )
        
        return jsonify(response.dict()), 200 if (status == 'healthy') else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {repr(e)}")
        error_response = handle_generic_error(e)
        return jsonify(error_response.dict()), 500


@api_bp.route('/status/<task_id>', methods = ['GET'])
def get_status(task_id: str):
    """
    Get async task status
    """
    try:
        status_data = get_task_status(task_id)
        
        if not status_data:
            return jsonify({'error'   : 'Task not found', 
                            'task_id' : task_id,
                          }), 404
        
        return jsonify(status_data), 200
        
    except Exception as e:
        logger.error(f"Status check failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 500



# UPLOAD AUDIO FILE 
@api_bp.route('/upload', methods = ['POST'])
def upload_file():
    """
    Upload audio file
    """
    try:
        if ('file' not in request.files):
            error_response = ErrorResponse(message   = 'No file provided',
                                           timestamp = datetime.utcnow(),
                                          )

            return jsonify(error_response.dict()), 400
        
        file = request.files['file']
        
        if (file.filename == ''):
            error_response = ErrorResponse(message   = 'No file selected',
                                           timestamp = datetime.utcnow(),
                                          )

            return jsonify(error_response.dict()), 400
        
        if not is_allowed_file(file.filename):
            error_response = ErrorResponse(message   = 'Invalid file type',
                                           timestamp = datetime.utcnow(),
                                          )

            return jsonify(error_response.dict()), 400
        
        # Save file
        filename        = secure_filename(file.filename)
        unique_filename = f"{Path(filename).stem}_{uuid.uuid4().hex}{Path(filename).suffix}"
        filepath        = os.path.join(UPLOADED_FILES_DIR, unique_filename)
        
        file.save(filepath)
        
        logger.info(f"File uploaded: {filepath}")
        
        return jsonify({'success'  : True,
                        'filepath' : filepath,
                        'filename' : unique_filename,
                      }), 200
        
    except Exception as e:
        logger.error(f"Upload failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 500



# ANALYZE AUDIO FILE
@api_bp.route('/analyze', methods = ['POST'])
def analyze_audio():
    """
    Analyze single audio file (async)
    """
    try:
        # Parse request
        data = request.get_json()
        
        # Validate using Pydantic
        try:
            analysis_request = AnalysisRequest(**data)

        except Exception as e:
            error_response = handle_validation_error(e)

            return jsonify(error_response.dict()), 400
        
        # Check file exists
        if not os.path.exists(analysis_request.filepath):
            error_response = handle_file_not_found_error(analysis_request.filepath)

            return jsonify(error_response.dict()), 404
        
        # Create analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Queue async task
        task        = analyze_audio_task.delay(analysis_id  = analysis_id,
                                               filepath     = analysis_request.filepath,
                                               emotion_mode = analysis_request.emotion_mode.value,
                                               language     = analysis_request.language,
                                              )
        
        # Create response
        response    = AnalysisResponse(analysis_id = analysis_id,
                                       status      = AnalysisStatus.PENDING,
                                       created_at  = datetime.utcnow(),
                                      )
        
        logger.info(f"Analysis queued: {analysis_id}, task: {task.id}")
        
        return jsonify({**response.dict(),
                        'task_id'    : task.id,
                        'status_url' : f"{API_PREFIX}/status/{task.id}"
                      }), 202
        
    except Exception as e:
        logger.error(f"Analysis request failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 400



# EXPLAINABILITY OF RESULTS
@api_bp.route('/explain/<analysis_id>', methods = ['GET'])
def get_explainability(analysis_id: str):
    """
    Get explainability results
    """
    try:
        explainability_service = ExplainabilityService()
        result                 = explainability_service.get_explanation(analysis_id)
        
        if (not result or (not result.get('available'))):

            return jsonify({'error'       : 'Explainability data not found',
                            'analysis_id' : analysis_id,
                          }), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Explainability request failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 500



# EXPORT RESULTS
@api_bp.route('/export', methods = ['POST'])
def export_results():
    """
    Export analysis results
    """
    try:
        data        = request.get_json()
        analysis_id = data.get('analysis_id')
        format      = data.get('format', 'json')
        
        if not analysis_id:
            return jsonify({'error': 'analysis_id required'}), 400
        
        # Initialize export service
        export_service = ExportService()
        
        # Mock result data (in production, retrieve from database)
        result_data    = data.get('result_data', {})
        
        export_path    = export_service.export(analysis_id, format, result_data)
        
        return jsonify({'success'      : True,
                        'export_path'  : str(export_path),
                        'download_url' : f"{API_PREFIX}/download/{export_path.name}",
                      }), 200
        
    except Exception as e:
        logger.error(f"Export failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 500


@api_bp.route('/download/<filename>', methods = ['GET'])
def download_file(filename: str):
    """
    Download exported file
    """
    try:
        filepath = EXPORTS_DIR / filename
        
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment = True)
        
    except Exception as e:
        logger.error(f"Download failed: {repr(e)}")
        return jsonify({'error' : str(e)}), 500



# BENCHMARKING ENDPOINT
@api_bp.route('/benchmark', methods = ['POST'])
def benchmark_endpoint():
    """
    Run benchmark on dataset
    """
    try:
        data         = request.get_json()
        dataset      = data.get('dataset', 'IEMOCAP')
        benchmark_id = str(uuid.uuid4())
        results      = run_benchmark(dataset)
        
        return jsonify({'benchmark_id' : benchmark_id,
                        'results'      : results,
                      }), 200
        
    except Exception as e:
        logger.error(f"Benchmark failed: {repr(e)}")
        error_response = handle_generic_error(e)

        return jsonify(error_response.dict()), 500



# ERROR HANDLERS
@api_bp.errorhandler(404)
def not_found(error):
    error_response = ErrorResponse(message   = 'Resource not found',
                                   timestamp = datetime.utcnow(),
                                  )

    return jsonify(error_response.dict()), 404


@api_bp.errorhandler(500)
def internal_error(error):
    error_response = ErrorResponse(message   = 'Internal server error',
                                   timestamp = datetime.utcnow(),
                                  )

    return jsonify(error_response.dict()), 500
