# DEPENDENCIES
import torch
import whisper
from typing import Dict
from config.settings import DEVICE
from utils.logging_util import setup_logger
from config.settings import WHISPER_MODEL_PATH
from config.whisper_config import SUPPORTED_LANGUAGES
from config.whisper_config import DEFAULT_WHISPER_CONFIG


# SETUP LOGGING
logger = setup_logger(__name__)


class Transcriber:
    """
    Speech transcription service using Whisper
    """
    def __init__(self):
        self.device = torch.device(DEVICE)
        
        try:
            self.model = whisper.load_model(name          = str(WHISPER_MODEL_PATH),
                                            device        = self.device,
                                            download_root = None,
                                            in_memory     = False,
                                           )

            logger.info(f"Whisper model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {repr(e)}")
            raise
    

    def transcribe(self, audio_path: str, language: str = None) -> Dict[str, str]:
        """
        Transcribe audio file to text
        
        Arguments:
        ----------
            audio_path { str } : Path to audio file

            language   { str } : Target language code (None for auto-detect)
        
        Returns:
        --------
                { dict }       : Dictionary with text, language, and language_code
        """
        try:
            config    = DEFAULT_WHISPER_CONFIG
            
            result    = self.model.transcribe(audio                       = audio_path,
                                              task                        = config.task,
                                              language                    = language or config.language,
                                              beam_size                   = config.beam_size,
                                              temperature                 = config.temperature,
                                              compression_ratio_threshold = config.compression_ratio_threshold,
                                              logprob_threshold           = config.logprob_threshold,
                                              no_speech_threshold         = config.no_speech_threshold,
                                              condition_on_previous_text  = config.condition_on_previous_text,
                                              verbose                     = config.verbose,
                                             )
            
            text      = result['text'].strip()
            lang_code = result.get('language', 'en')
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')
            
            logger.info(f"Transcription complete: {len(text)} chars, language: {lang_name}")
            
            return {'text'          : text,
                    'language'      : lang_name,
                    'language_code' : lang_code,
                   }
            
        except Exception as e:
            logger.error(f"Transcription error: {repr(e)}")
            raise
    

    def transcribe_streaming(self, audio_path: str) -> Dict[str, str]:
        """
        Fast transcription for streaming (reduced beam size)
        """
        try:
            result    = self.model.transcribe(audio                      = audio_path,
                                              task                       = 'transcribe',
                                              beam_size                  = 1,  # Faster
                                              temperature                = 0.0,
                                              no_speech_threshold        = 0.6,
                                              condition_on_previous_text = False,
                                              verbose                    = False,
                                             )
            
            text      = result['text'].strip()
            lang_code = result.get('language', 'en')
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')
            
            return {'text'          : text,
                    'language'      : lang_name,
                    'language_code' : lang_code,
                   }

        except Exception as e:
            logger.error(f"Streaming transcription error: {repr(e)}")
            raise



# Global instance
_transcriber_instance = None



def get_transcriber() -> Transcriber:
    """
    Get or create transcriber singleton
    """
    global _transcriber_instance

    if _transcriber_instance is None:
        _transcriber_instance = Transcriber()
        
    return _transcriber_instance