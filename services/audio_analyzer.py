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
            transcription_result = self.transcriber.transcribe(audio_path = audio_path)
            
            # Predict emotions
            emotion_result       = self.emotion_predictor.predict(audio_path = audio_path, 
                                                                  mode       = emotion_mode,
                                                                 )
            
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
            emotion_result      = self.emotion_predictor.predict(audio_path = audio_path, 
                                                                 mode       = 'base',
                                                                )
            
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
        
        emotions = analysis_result['emotions']
        
        # Format base emotions
        if ('base' in emotions):
            sorted_emotions               = sorted(emotions['base'].items(),
                                                   key     = lambda x: x[1],
                                                   reverse = True,
                                                  )
            
            formatted['emotions']['base'] = list()

            for emotion, score in sorted_emotions:
                # Handle both float and string percentages
                if (isinstance(score, (int, float))):
                    # score is already a percentage
                    percentage = f"{score:.2f}%"  

                elif isinstance(score, str):
                    # Remove existing % if present and format
                    clean_score = score.replace('%', '').strip()
                    
                    try:
                        percentage = f"{float(clean_score):.2f}%"
                    
                    except ValueError:
                        percentage = f"{score}%"
                
                else:
                    percentage = f"{score}%"
                
                formatted['emotions']['base'].append({'label'      : emotion,
                                                      'score'      : score, 
                                                      'percentage' : percentage,
                                                    })
        
        # Format granular emotions 
        if (('primary' in emotions) and emotions['primary']):
            confidence = emotions['primary']['confidence']
            
            if (isinstance(confidence, (int, float))):
                confidence_str = f"{confidence:.2f}%"

            else:
                confidence_str = confidence if '%' in str(confidence) else f"{confidence}%"
            
            formatted['emotions']['primary'] = {'emotions'   : emotions['primary']['emotions'],
                                                'confidence' : confidence_str,
                                               }
        
        if (('secondary' in emotions) and emotions['secondary']):
            confidence = emotions['secondary']['confidence']
            
            if (isinstance(confidence, (int, float))):
                confidence_str = f"{confidence:.2f}%"

            else:
                confidence_str = confidence if '%' in str(confidence) else f"{confidence}%"
            
            formatted['emotions']['secondary'] = {'emotions'   : emotions['secondary']['emotions'],
                                                  'confidence' : confidence_str,
                                                 }
        
        # Format complex emotions
        if (('complex' in emotions) and emotions['complex']):
            formatted['emotions']['complex'] = []

            for item in emotions['complex']:
                confidence = item['confidence']

                if (isinstance(confidence, (int, float))):
                    confidence_str = f"{confidence:.2f}%"
                
                else:
                    confidence_str = confidence if '%' in str(confidence) else f"{confidence}%"
                
                formatted['emotions']['complex'].append({'name'       : item['name'],
                                                         'components' : item['components'],
                                                         'confidence' : confidence_str,
                                                       })
        
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