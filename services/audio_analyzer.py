# DEPENDENCIES
import uuid
import time
from typing import Dict
from utils.logging_util import setup_logger
from services.transcriber import get_transcriber
from services.emotion_predictor import get_emotion_predictor


# SETUP LOGGING
logger = setup_logger(__name__)


class AudioAnalyzer:
    """
    Combined audio analysis service
    """
    def __init__(self):
        self.emotion_predictor = get_emotion_predictor()
        self.transcriber       = get_transcriber()

        logger.info("Audio analyzer initialized")
    

    def analyze_complete(self, audio_path: str, emotion_mode: str = 'both') -> Dict:
        """
        Complete analysis: transcription + emotion recognition
        
        Arguments:
        ----------
            audio_path   { str } : Path to audio file

            emotion_mode { str } : 'base', 'granular', or 'both'
        
        Returns:
        --------
                 { dict }        : Dictionary with transcription and emotion results
        """
        try:
            start_time           = time.time()
            
            # Generate unique analysis ID
            analysis_id          = str(uuid.uuid4())

            # Transcribe
            transcription_result = self.transcriber.transcribe(audio_path)
            
            # Predict emotions
            emotion_result       = self.emotion_predictor.predict(audio_path, emotion_mode)
            
            processing_time      = time.time() - start_time
            
            return {'transcription' : transcription_result,
                    'emotions'      : emotion_result,
                    'metadata'      : {'analysis_id'     : analysis_id,
                                       'processing_time' : processing_time,
                                       'audio_path'      : audio_path,
                                      },
                    'status'        : 'success',
                   }
                
        except Exception as e:
            logger.error(f"Analysis error: {repr(e)}")
            return {'status'  : 'error',
                    'message' : str(e),
                   }

    
    def analyze_streaming(self, audio_path: str) -> Dict:
        """
        Fast analysis for streaming (optimized)
        """
        try:
            start_time           = time.time()

            # Generate unique analysis ID
            analysis_id          = str(uuid.uuid4())
            
            # Quick transcription
            transcription_result = self.transcriber.transcribe_streaming(audio_path)
            
            # Base emotions only (faster)
            emotion_result      = self.emotion_predictor.predict(audio_path, mode='base')
            
            processing_time     = time.time() - start_time
            
            return {'transcription' : transcription_result,
                    'emotions'      : emotion_result,
                    'metadata'      : {'analysis_id'     : analysis_id,
                                       'processing_time' : processing_time,
                                       'status'          : 'success',
                                      }
                   }
            
        except Exception as e:
            logger.error(f"Streaming analysis error: {repr(e)}")
            return {'status'  : 'error',
                    'message' : str(e),
                   }

    
    def format_results_for_display(self, analysis_result: Dict) -> Dict:
        """
        Format analysis results for frontend display
        """
        if (analysis_result.get('status') == 'error'):
            return analysis_result
        
        formatted = {'transcription' : analysis_result['transcription']['text'],
                     'language'      : analysis_result['transcription']['language'],
                     'emotions'      : {},
                    }
        
        emotions  = analysis_result['emotions']
        
        # Format base emotions
        if ('base' in emotions):
            sorted_emotions               = sorted(emotions['base'].items(),
                                                   key     = lambda x: x[1],
                                                   reverse = True,
                                                  )
            formatted['emotions']['base'] = [{'label'      : emotion,
                                              'score'      : score,
                                              'percentage' : f"{score * 100:.2f}%",
                                             }
                                             for emotion, score in sorted_emotions
                                            ]
        
        # Format granular emotions
        if (('primary' in emotions) and emotions['primary']):
            formatted['emotions']['primary'] = {'emotions'   : emotions['primary']['emotions'],
                                                'confidence' : f"{emotions['primary']['confidence'] * 100:.2f}%",
                                               }
        
        if (('secondary' in emotions) and emotions['secondary']):
            formatted['emotions']['secondary'] = {'emotions'   : emotions['secondary']['emotions'],
                                                  'confidence' : f"{emotions['secondary']['confidence'] * 100:.2f}%",
                                                 }
        
        # Format complex emotions
        if (('complex' in emotions) and emotions['complex']):
            formatted['emotions']['complex'] = [{'name'       : item['name'],
                                                 'components' : item['components'],
                                                 'confidence' : f"{item['confidence'] * 100:.2f}%",
                                                }
                                                for item in emotions['complex']
                                               ]
        
        # Add metadata
        if ('metadata' in analysis_result):
            formatted['metadata'] = analysis_result['metadata']
        
        return formatted


# Global instance
_analyzer_instance = None


def get_audio_analyzer() -> AudioAnalyzer:
    """
    Get or create audio analyzer singleton
    """
    global _analyzer_instance

    if _analyzer_instance is None:
        _analyzer_instance = AudioAnalyzer()
    
    return _analyzer_instance