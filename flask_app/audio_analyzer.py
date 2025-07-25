# DEPENDENCIES
import warnings
from typing import List
from typing import Dict
from .transcriber import transcribe_audio
from .hubert_predictor import predict_emotion


# IGNORING ALL KIND OF WARNINGS RAISED AT RUN TIME
warnings.filterwarnings("ignore")


def analyze_audio(input_audio: str) -> List[Dict[str, str]]:
    """
    Analyze an audio file to extract speech-to-text (STT) and speech emotion recognition (SER) results

    Arguments:
    ----------
        input_audio {str}  : Path to the audio file to be analyzed

    Errors:
    -------
        AudioAnalysisError : If an error occurs during analysis of the input audio

    Returns:
    --------
            {list}         : Python list object containing transcription results and emotion analysis,
                             or an error message
    """
    try:
        # Transcribe the audio file to text and detect its language
        transcribed_text, audio_language = transcribe_audio(speech_file_path = input_audio)

        # Add line breaks to the transcribed text at sentence endings
        formatted_transcribed_text       = transcribed_text.replace('.', '.<br>')\
                                                           .replace(';', '.<br>')\
                                                           .replace('!', '!<br>')\
                                                           .replace('?', '?<br>')\
                                                           .replace('<br>.', '')
        
        # Prepare the transcription results
        transcription_results            = [{"Transcription"  : formatted_transcribed_text}, 
                                            {"Audio Language" : audio_language}
                                           ]

        # Analyze the audio file for emotions
        emotion_analysis_results         = predict_emotion(audio_path    = input_audio, 
                                                           sampling_rate = 16000)
        
        # Combine transcription and emotion results
        total_results                    = transcription_results + emotion_analysis_results
        

        # Return the combined results as a list object
        return total_results

    except Exception as AudioAnalysisError:
        return (f"AudioAnalysisError: An error occurred while analyzing the audio : {repr(AudioAnalysisError)}")

