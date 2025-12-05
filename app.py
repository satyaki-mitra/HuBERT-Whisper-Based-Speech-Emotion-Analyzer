# DEPENDENCIES
import os
import shutil
from flask import Flask
from pathlib import Path
from api.routes import api_bp
from flask_socketio import emit
from config.settings import HOST
from config.settings import PORT
from config.settings import DEBUG
from flask import render_template
from flask_socketio import SocketIO
from config.settings import SECRET_KEY
from werkzeug.utils import secure_filename
from config.settings import TEMP_FILES_DIR
from config.settings import validate_models
from utils.logging_util import setup_logger
from utils.audio_utils import convert_to_wav
from config.settings import RECORDED_FILES_DIR
from utils.audio_utils import natural_sort_key
from config.settings import create_directories
from config.settings import SOCKETIO_PING_TIMEOUT
from config.settings import SOCKETIO_PING_INTERVAL
from utils.audio_utils import split_audio_on_silence
from utils.audio_utils import create_unique_filename
from services.explainer import ExplainabilityService
from utils.error_handlers import handle_generic_error
from services.audio_analyzer import get_audio_analyzer


# SETUP LOGGING
logger = setup_logger(__name__)


# VALIDATE MODELS BEFORE STARTUP
try:
    validate_models()
    logger.info("Models validated successfully")

except FileNotFoundError as e:
    logger.error(str(e))
    exit(1)


# CRATE REQUIRED DIRECTORIES
create_directories()


# INITIALIZE FLASK APP
app                              = Flask(__name__)
app.config['SECRET_KEY']         = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB


# REGISTER API BLUEPRINT
app.register_blueprint(api_bp)


# INITIALIZE SocketIO
socketio                         = SocketIO(app,
                                            ping_timeout         = SOCKETIO_PING_TIMEOUT,
                                            ping_interval        = SOCKETIO_PING_INTERVAL,
                                            cors_allowed_origins = "*",
                                            async_mode           = 'threading',
                                            max_http_buffer_size = 100 * 1024 * 1024,
                                           )

logger.info("Application initialized successfully")



# WEB ROUTES
@app.route('/')
def index():
    """
    Landing page
    """
    return render_template('index.html')


@app.route('/batch-analysis')
def batch_analysis():
    """
    Batch analysis interface
    """
    return render_template('batch_analysis.html')


@app.route('/live-streaming')
def live_streaming():
    """
    Live streaming interface
    """
    return render_template('live_streaming.html')


@app.route('/explainability')
def explainability_dashboard():
    """
    Explainability dashboard
    """
    return render_template('explainability.html')


@app.route('/favicon.ico')
def favicon():
    """
    Serve favicon
    """
    return '', 204



# SOCKETIO EVENTS
@socketio.on('connect')
def handle_connect():
    """
    Handle client connection
    """
    logger.info("Client connected")

    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle client disconnection
    """
    logger.info("Client disconnected")


@socketio.on('analyze_batch')
def handle_batch_analysis(data):
    """
    Handle batch analysis (WebSocket fallback)
    """
    try:
        filepath     = data.get('filepath')
        emotion_mode = data.get('emotion_mode', 'both')
        
        if not filepath or not os.path.exists(filepath):
            emit('error', {'message': 'File not found'})
            return
        
        # Create temp directory for chunks
        chunk_dir = os.path.join(TEMP_FILES_DIR, create_unique_filename(''))
        os.makedirs(chunk_dir, exist_ok = True)
        
        # Split audio
        emit('status', {'message'  : 'Splitting audio...', 
                        'progress' : 10,
                       }
            )

        chunk_paths  = split_audio_on_silence(filepath, chunk_dir)
        
        # Analyze chunks
        analyzer     = get_audio_analyzer()
        total_chunks = len(chunk_paths)
        
        for idx, chunk_path in enumerate(sorted(chunk_paths, key=natural_sort_key)):
            progress    = int(((idx + 1) / total_chunks) * 90) + 10
            
            emit('status', {'message'  : f'Analyzing segment {idx + 1}/{total_chunks}...', 
                            'progress' : progress,
                           }
                )
            
            result      = analyzer.analyze_complete(chunk_path, emotion_mode)
            formatted   = analyzer.format_results_for_display(result)
            
            analysis_id = result.get('metadata', {}).get('analysis_id')

            if analysis_id:
                try:
                    explainability = ExplainabilityService()
                    
                    # Generate emotion distribution chart
                    if (('emotions' in result) and ('base' in result['emotions'])):
                        explainability.generate_emotion_distribution(analysis_id    = analysis_id,
                                                                     emotion_scores = result['emotions']['base'],
                                                                    )

                except Exception as e:
                    logger.warning(f"Failed to generate explainability: {e}")
            
            # Add analysis_id to formatted result
            formatted['analysis_id'] = analysis_id
            
            emit('chunk_result', {'chunk_index'  : idx,
                                  'total_chunks' : total_chunks,
                                  'result'       : formatted,
                                 }
                )
        
        # Cleanup
        shutil.rmtree(chunk_dir, ignore_errors = True)
        emit('analysis_complete', {'message' : 'Complete'})
        
    except Exception as e:
        logger.error(f"Batch analysis error: {repr(e)}")
        error_response = handle_generic_error(e)
        
        emit('error', {'message': str(e)})


@socketio.on('streaming_chunk')
def handle_streaming_chunk(data):
    """
    Handle real-time streaming chunk
    """
    try:
        chunk_data  = data.get('audio_data')
        chunk_index = data.get('chunk_index', 0)
        
        if not chunk_data:
            emit('error', {'message' : 'No audio data'})
            return
        
        # Save chunk
        temp_filename = create_unique_filename('wav')
        temp_path     = os.path.join(TEMP_FILES_DIR, temp_filename)
        
        with open(temp_path, 'wb') as f:
            f.write(chunk_data)
        
        # Convert to WAV
        wav_path = str(Path(temp_path).with_suffix('.converted.wav'))

        convert_to_wav(temp_path, wav_path)
        
        # Analyze (streaming mode - faster)
        analyzer    = get_audio_analyzer()
        result      = analyzer.analyze_streaming(wav_path)
        formatted   = analyzer.format_results_for_display(result)
        
        analysis_id = result.get('metadata', {}).get('analysis_id')
        
        if analysis_id:
            try:
                explainability = ExplainabilityService()
                
                if (('emotions' in result) and ('base' in result['emotions'])):
                    explainability.generate_emotion_distribution(analysis_id    = analysis_id,
                                                                 emotion_scores = result['emotions']['base'],
                                                                )

            except Exception as e:
                logger.warning(f"Failed to generate explainability: {e}")
        
        formatted['analysis_id'] = analysis_id
        
        emit('streaming_result', {'chunk_index' : chunk_index,
                                  'result'      : formatted,
                                 }
            )
        
        # Cleanup
        os.remove(temp_path)
        os.remove(wav_path)
        
    except Exception as e:
        logger.error(f"Streaming chunk error: {repr(e)}")
        emit('error', {'message': str(e)})


@socketio.on('save_recording')
def handle_save_recording(data):
    """
    Save complete recording
    """
    try:
        recording_data = data.get('recording_data')
        
        if not recording_data:
            emit('error', {'message' : 'No recording data'})
            return
        
        # Save recording
        filename = create_unique_filename('wav')
        filepath = os.path.join(RECORDED_FILES_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(recording_data)
        
        # Convert to proper format
        wav_path = str(Path(filepath).with_suffix('.final.wav'))

        convert_to_wav(filepath, wav_path)
        
        os.remove(filepath)
        
        logger.info(f"Recording saved: {wav_path}")
        emit('save_complete', {'filepath': wav_path})
        
    except Exception as e:
        logger.error(f"Save recording error: {repr(e)}")
        emit('error', {'message': str(e)})



# ERROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return render_template('templates/index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {repr(error)}")
    return "Internal Server Error", 500



# RUN APPLICATION
if __name__ == '__main__':
    logger.info(f"Starting EmotiVoice on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"API endpoints available at /api/v1/")
    
    socketio.run(app,
                 host                  = HOST,
                 port                  = PORT,
                 debug                 = DEBUG,
                 allow_unsafe_werkzeug = True,
                )