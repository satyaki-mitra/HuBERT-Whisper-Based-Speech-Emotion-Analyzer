# DEPENDENCIES
import os
from pathlib import Path
from dotenv import load_dotenv


# LOAD ENVIRONMENT VARIABLES
load_dotenv()


# PROJECT PATHS
BASE_DIR                   = Path(__file__).parent.parent

# DATA DIRECTORIES
UPLOADED_FILES_DIR         = BASE_DIR / 'data' / 'uploaded_files'
RECORDED_FILES_DIR         = BASE_DIR / 'data' / 'recorded_files'
TEMP_FILES_DIR             = BASE_DIR / 'data' / 'temp_files'
LOGS_DIR                   = BASE_DIR / 'logs'
EXPORTS_DIR                = BASE_DIR / 'exports'

# MODEL DIRECTORIES
MODELS_DIR                 = BASE_DIR / 'models'
HUBERT_MODEL_PATH          = MODELS_DIR / 'hubert'
WHISPER_MODEL_PATH         = MODELS_DIR / 'whisper' / 'large-v3.pt'

# SERVER SETTINGS
HOST                       = os.getenv('HOST', 'localhost')
PORT                       = int(os.getenv('PORT', 8000))
DEBUG                      = os.getenv('DEBUG', 'True').lower() == 'true'
SECRET_KEY                 = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# UPLOAD SETTINGS
MAX_FILE_SIZE              = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS         = {'wav', 'mp3', 'webm', 'ogg', 'm4a', 'flac'}

# DEVICE SETTINGS
DEVICE                     = os.getenv('DEVICE', 'cpu')  # 'cpu', 'cuda', 'mps'
BATCH_SIZE                 = int(os.getenv('BATCH_SIZE', 8))
NUM_WORKERS                = int(os.getenv('NUM_WORKERS', 4))

# EMOTION SETTINGS
EMOTION_MODE               = os.getenv('EMOTION_MODE', 'granular')

# AUDIO PROCESSING
SAMPLE_RATE                = 16000
AUDIO_CHANNELS             = 1
SILENCE_THRESHOLD          = -40
MIN_SILENCE_LENGTH         = 500
KEEP_SILENCE               = 300
TARGET_CHUNK_LENGTH        = 10 * 1000

# REDIS & CELERY
REDIS_URL                  = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL          = REDIS_URL
CELERY_RESULT_BACKEND      = REDIS_URL
CELERY_TASK_TIME_LIMIT     = 600  # 10 minutes

# LOGGING
LOG_LEVEL                  = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE                   = LOGS_DIR / 'app.log'
LOG_FORMAT                 = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# SOCKETIO
SOCKETIO_PING_TIMEOUT      = 600
SOCKETIO_PING_INTERVAL     = 25
SOCKETIO_MAX_BUFFER_SIZE   = 100 * 1024 * 1024

# API SETTINGS
API_VERSION                = 'v1'
API_PREFIX                 = f'/api/{API_VERSION}'
RATE_LIMIT                 = '100/hour'

# EXPORT SETTINGS
EXPORT_FORMATS             = ['json', 'csv', 'pdf']
PDF_TEMPLATE               = 'exports/report_template.html'

# EMOTION LABELS
BASE_EMOTIONS              = {0 : "Anger", 
                              1 : "Fear", 
                              2 : "Happiness", 
                              3 : "Neutral", 
                              4 : "Sadness", 
                              5 : "Surprise",
                             }

GRANULAR_EMOTION_MAP       = {"Anger"     : {"primary"   : ["Anger", "Rage", "Fury"],
                                             "secondary" : ["Frustration", "Irritation", "Annoyance", "Resentment"],
                                             "threshold" : 0.3,
                                            },
                              "Fear"      : {"primary"   : ["Fear", "Terror", "Panic"],
                                             "secondary" : ["Anxiety", "Worry", "Nervousness", "Dread"],
                                             "threshold" : 0.3,
                                            },
                              "Happiness" : {"primary"   : ["Happiness", "Joy", "Delight"],
                                             "secondary" : ["Contentment", "Amusement", "Excitement", "Enthusiasm"],
                                             "threshold" : 0.3,
                                            },
                              "Neutral"   : {"primary"   : ["Neutral", "Calm"],
                                             "secondary" : ["Indifference", "Composure", "Serenity"],
                                             "threshold" : 0.4,
                                            },
                              "Sadness"   : {"primary"   : ["Sadness", "Grief", "Sorrow"],
                                             "secondary" : ["Melancholy", "Disappointment", "Despair", "Loneliness"],
                                             "threshold" : 0.3,
                                            },
                              "Surprise"  : {"primary"   : ["Surprise", "Amazement"],
                                             "secondary" : ["Shock", "Wonder", "Astonishment", "Disbelief"],
                                             "threshold" : 0.3,
                                            }
                             }

COMPLEX_EMOTIONS           = {("Happiness", "Surprise") : "Elation",
                              ("Sadness", "Anger")      : "Bitterness",
                              ("Fear", "Sadness")       : "Desperation",
                              ("Happiness", "Neutral")  : "Contentment",
                              ("Anger", "Fear")         : "Panic",
                              ("Sadness", "Neutral")    : "Resignation",
                             }

# EDGE CASE THRESHOLDS
SILENCE_DURATION_THRESHOLD = 5.0  # seconds
NOISE_SNR_THRESHOLD        = 10.0  # dB
MIN_SPEECH_DURATION        = 0.5  # seconds

# BENCHMARKING
BENCHMARK_DATASETS         = {'IEMOCAP' : {'path'     : BASE_DIR / 'benchmarks' / 'data' / 'iemocap',
                                           'emotions' : ['anger', 'happiness', 'sadness', 'neutral'],
                                          },
                              'RAVDESS' : {'path'     : BASE_DIR / 'benchmarks' / 'data' / 'ravdess',
                                           'emotions' : ['angry', 'happy', 'sad', 'neutral', 'fearful', 'surprised'],
                                          },
                             }


def create_directories():
    """
    Create all required directories
    """
    for directory in [UPLOADED_FILES_DIR, RECORDED_FILES_DIR, TEMP_FILES_DIR, LOGS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents = True, exist_ok = True)


def validate_models():
    """
    Check if model files exist
    """
    if not HUBERT_MODEL_PATH.exists():
        raise FileNotFoundError(f"HuBERT model not found at {HUBERT_MODEL_PATH}\n"
                                f"Download from: https://huggingface.co/facebook/hubert-base-ls960"
                               )
    
    if not WHISPER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Whisper model not found at {WHISPER_MODEL_PATH}\n"
                                f"Download from: https://github.com/openai/whisper"
                               )
    
    return True


def is_allowed_file(filename):
    """
    Check if file extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize directories on import
create_directories()