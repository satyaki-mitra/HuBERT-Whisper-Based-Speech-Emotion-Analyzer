# DEPENDENCIES
import numpy as np
from typing import Any
from typing import Dict
from scipy import signal
from typing import Optional
from datetime import datetime
from config.models import ErrorDetail
from config.models import ErrorResponse


# CUSTOM EXCEPTIONS
class EmotiVoiceException(Exception):
    """
    Base exception for EmotiVoice
    """
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        self.message     = message
        self.status_code = status_code
        self.details     = details or {}

        super().__init__(self.message)


class AudioProcessingError(EmotiVoiceException):
    """
    Audio processing errors
    """
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code = 422, details = details)


class ModelInferenceError(EmotiVoiceException):
    """
    Model inference errors
    """
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code = 500, details = details)


class ValidationError(EmotiVoiceException):
    """
    Validation errors
    """
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code = 400, details = details)


class FileNotFoundError(EmotiVoiceException):
    """
    File not found errors
    """
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code = 404, details = details)


class RateLimitError(EmotiVoiceException):
    """
    Rate limit errors
    """
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict] = None):
        super().__init__(message, status_code = 429, details = details)


class ServiceUnavailableError(EmotiVoiceException):
    """
    Service unavailable errors
    """
    def __init__(self, message: str = "Service temporarily unavailable", details: Optional[Dict] = None):
        super().__init__(message, status_code = 503, details = details)



# ERROR HANDLERS
def handle_audio_error(error: Exception, audio_path: str = None) -> ErrorResponse:
    """
    Handle audio processing errors
    """
    error_details = [ErrorDetail(field      = "audio_file",
                                 message    = str(error),
                                 error_type = "AudioProcessingError",
                                )
                    ]
    
    if audio_path:
        error_details[0].field = f"audio_file:{audio_path}"
    
    return ErrorResponse(message   = "Failed to process audio file",
                         details   = error_details,
                         timestamp = datetime.utcnow(),
                        )


def handle_model_error(error: Exception, model_name: str) -> ErrorResponse:
    """
    Handle model inference errors
    """
    return ErrorResponse(message   = f"Model inference failed: {model_name}",
                         details   = [ErrorDetail(field      = "model",
                                                  message    = str(error),
                                                  error_type = "ModelInferenceError",
                                                 )
                                     ],
                         timestamp = datetime.utcnow()
                        )


def handle_validation_error(error: Exception, field: str = None) -> ErrorResponse:
    """
    Handle validation errors
    """
    error_details = list()
    
    # Handle Pydantic validation errors
    if hasattr(error, 'errors'):
        for err in error.errors():
            error_details.append(ErrorDetail(field      = '.'.join(str(loc) for loc in err['loc']),
                                             message    = err['msg'],
                                             error_type = err['type'],
                                            )
                                )

    else:
        error_details.append(ErrorDetail(field      = field,
                                         message    = str(error),
                                         error_type = "ValidationError",
                                        )
                            )
    
    return ErrorResponse(message   = "Validation failed",
                         details   = error_details,
                         timestamp = datetime.utcnow(),
                        )


def handle_file_not_found_error(filepath: str) -> ErrorResponse:
    """
    Handle file not found errors
    """
    return ErrorResponse(message   = "File not found",
                         details   = [ErrorDetail(field      = "filepath",
                                                  message    = f"File not found: {filepath}",
                                                  error_type = "FileNotFoundError",
                                                 )
                                     ],
                         timestamp = datetime.utcnow(),
                        )


def handle_rate_limit_error(limit: str) -> ErrorResponse:
    """
    Handle rate limit errors
    """
    return ErrorResponse(message   = "Rate limit exceeded",
                         details   = [ErrorDetail(field      = "rate_limit",
                                                  message    = f"Exceeded rate limit: {limit}",
                                                  error_type = "RateLimitError",
                                                 )
                                     ],
                         timestamp = datetime.utcnow()
                        )


def handle_generic_error(error: Exception) -> ErrorResponse:
    """
    Handle generic errors
    """
    return ErrorResponse(message   = "An unexpected error occurred",
                         details   = [ErrorDetail(field      = None,
                                                  message    = str(error),
                                                  error_type = type(error).__name__,
                                                 )
                                     ],
                         timestamp = datetime.utcnow(),
                        )



# EDGE CASE HANDLERS
class EdgeCaseDetector:
    """
    Detect and handle edge cases in audio
    """
    @staticmethod
    def detect_silence(audio_array, threshold_duration: float = 5.0) -> Dict[str, Any]:
        """
        Detect prolonged silence
        """
        # Simple energy-based silence detection
        energy     = np.abs(audio_array).mean()
        is_silence = energy < 0.01
        
        return {'is_silence'   : is_silence,
                'energy_level' : float(energy),
                'duration'     : len(audio_array) / 16000,  # Assuming 16kHz
                'warning'      : 'Prolonged silence detected' if is_silence else None,
               }
    

    @staticmethod
    def detect_noise(audio_array, snr_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Detect high noise levels
        """
        # Estimate SNR
        signal_power = np.var(audio_array)

        # Use first 1000 samples as noise estimate
        noise_power  = np.var(audio_array[:1000]) 
        
        if (noise_power > 0):
            snr = 10 * np.log10(signal_power / noise_power)

        else:
            snr = float('inf')
        
        is_noisy     = snr < snr_threshold
        
        return {'is_noisy'     : is_noisy,
                'snr_db'       : float(snr),
                'signal_power' : float(signal_power),
                'noise_power'  : float(noise_power),
                'warning'      : f'High noise level detected (SNR: {snr:.2f} dB)' if is_noisy else None,
               }

    
    @staticmethod
    def detect_clipping(audio_array) -> Dict[str, Any]:
        """
        Detect audio clipping
        """
        # Check for samples at max amplitude
        clipped_samples = np.sum(np.abs(audio_array) >= 0.99)
        total_samples   = len(audio_array)
        clipping_ratio  = clipped_samples / total_samples
        
        # More than 1% clipped
        is_clipped      = clipping_ratio > 0.01  
        
        return {'is_clipped'      : is_clipped,
                'clipping_ratio'  : float(clipping_ratio),
                'clipped_samples' : int(clipped_samples),
                'warning'         : f'Audio clipping detected ({clipping_ratio*100:.2f}%)' if is_clipped else None,
               }

    
    @staticmethod
    def detect_multiple_speakers(audio_array) -> Dict[str, Any]:
        """
        Detect multiple speakers (simplified)
        """
        # Use spectral clustering on MFCCs (simplified version)
        # Calculate variance in short-time energy
        window_size = 4000  # ~0.25s at 16kHz
        num_windows = len(audio_array) // window_size
        
        energies    = list()

        for i in range(num_windows):
            start  = i * window_size
            end    = start + window_size
            energy = np.abs(audio_array[start:end]).mean()

            energies.append(energy)
        
        if (len(energies) > 1):
            energy_variance = np.var(energies)
            # High variance might indicate speaker changes
            has_multiple_speakers = energy_variance > 0.01
        else:
            has_multiple_speakers = False
        
        return {'has_multiple_speakers' : has_multiple_speakers,
                'confidence'            : 0.6 if has_multiple_speakers else 0.4,
                'warning'               : 'Multiple speakers possibly detected' if has_multiple_speakers else None,
               }
    

    @classmethod
    def analyze_audio_quality(cls, audio_array) -> Dict[str, Any]:
        """
        Comprehensive audio quality analysis
        """
        return {'silence'           : cls.detect_silence(audio_array),
                'noise'             : cls.detect_noise(audio_array),
                'clipping'          : cls.detect_clipping(audio_array),
                'multiple_speakers' : cls.detect_multiple_speakers(audio_array),
               }