# DEPENDENCIES
import sys
import torch
import whisper
import warnings
from typing import Tuple
sys.path.append('../')
from config import WHISPER_MODEL_PATH


# IGNORING ALL KIND OF WARNINGS RAISED AT RUN TIME
warnings.filterwarnings('ignore')

# Initialize the Whisper model large-v3
whisper_model = whisper.load_model(name          = WHISPER_MODEL_PATH,
                                   device        = torch.device('cpu'),
                                   download_root = None,
                                   in_memory     = False)


def transcribe_audio(speech_file_path: str) -> Tuple[str, str]:
    """
    Convert speech from an audio file to text and detects the language of the speech

    Arguments:
    ----------
        speech_file_path (str) : The file path to the audio file containing the speech

    Errors:
    -------
        TranscriptionError     : If there is any exception while transcribing the audio
        
    Returns:
    --------
           Tuple[str, str]     : A tuple containing the transcribed text and the language of the audio
    """
    try:
        # Transcribe using transcribe class of whisper
        transcription_result = whisper_model.transcribe(audio                       = speech_file_path,
                                                        task                        = 'transcribe',
                                                        beam_size                   = 5,
                                                        verbose                     = False,
                                                        temperature                 = 0.0,
                                                        compression_ratio_threshold = 2.4,
                                                        logprob_threshold           = None,
                                                        no_speech_threshold         = 0.5,
                                                        condition_on_previous_text  = True,
                                                        initial_prompt              = None,
                                                        word_timestamps             = False)
        
    except Exception as TranscriptionError:
        # Return a detailed error message if something goes wrong
        return (f"TranscriptionError: While transcribing the audio file, found : {repr(TranscriptionError)}")
    
    # Language abbreviation dictionary
    LANGUAGES         = {"en": "English",
                         "zh": "Chinese",
                         "de": "German",
                         "es": "Spanish",
                         "ru": "russian",
                         "ko": "Korean",
                         "fr": "French",
                         "ja": "Japanese",
                         "pt": "Portuguese",
                         "tr": "Turkish",
                         "pl": "Polish",
                         "ca": "Catalan",
                         "nl": "Dutch",
                         "ar": "Arabic",
                         "sv": "Swedish",
                         "it": "Italian",
                         "id": "Indonesian",
                         "hi": "Hindi",
                         "fi": "Finnish",
                         "vi": "Vietnamese",
                         "iw": "Hebrew",
                         "uk": "Ukrainian",
                         "el": "Greek",
                         "ms": "Malay",
                         "cs": "Czech",
                         "ro": "Romanian",
                         "da": "Danish",
                         "hu": "Hungarian",
                         "ta": "Tamil",
                         "no": "Norwegian",
                         "th": "Thai",
                         "ur": "Urdu",
                         "hr": "Croatian",
                         "bg": "Bulgarian",
                         "lt": "Lithuanian",
                         "la": "Latin",                           
                         "mi": "Maori",
                         "ml": "Malayalam",
                         "cy": "Welsh",
                         "sk": "Slovak",
                         "te": "Telugu",
                         "fa": "Persian",
                         "lv": "Latvian",
                         "bn": "Bengali",
                         "sr": "Serbian",                           
                         "az": "Azerbaijani",
                         "sl": "Slovenian",
                         "kn": "Kannada",
                         "et": "Estonian",
                         "mk": "Macedonian",
                         "br": "Breton",
                         "eu": "Basque",
                         "is": "Icelandic",
                         "hy": "Armenian",
                         "ne": "Nepali",
                         "mn": "Mongolian",
                         "bs": "Bosnian",
                         "kk": "Kazakh",
                         "sq": "Albanian",
                         "sw": "Swahili",
                         "gl": "Galician",
                         "mr": "Marathi",
                         "pa": "Punjabi",
                         "si": "Sinhala",
                         "km": "Khmer",
                         "sn": "Shona",
                         "yo": "Yoruba",
                         "so": "Somali",
                         "af": "Afrikaans",
                         "oc": "Occitan",
                         "ka": "Georgian",
                         "be": "Belarusian",
                         "tg": "Tajik",
                         "sd": "Sindhi",
                         "gu": "Gujarati",
                         "am": "Amharic",
                         "yi": "Yiddish",
                         "lo": "Lao",
                         "uz": "Uzbek",
                         "fo": "Faroese",
                         "ht": "Haitian creole",
                         "ps": "Pashto",
                         "tk": "Turkmen",
                         "nn": "Nynorsk",
                         "mt": "Maltese",
                         "sa": "Sanskrit",
                         "lb": "Luxembourgish",
                         "my": "Myanmar",
                         "bo": "Tibetan",
                         "tl": "Tagalog",
                         "mg": "Malagasy",
                         "as": "Assamese",
                         "tt": "Tatar",
                         "haw": "Hawaiian",
                         "ln": "Lingala",
                         "ha": "Hausa",
                         "ba": "Bashkir",
                         "jw": "Javanese",
                         "su": "Sundanese",
                        }
                
    # Extract transcripted text from the result
    transcripted_text  = transcription_result['text']

    # Also extract the abbribiation of the audio language
    raw_audio_language = transcription_result['language']
    audio_language     = LANGUAGES[raw_audio_language]
    
    return transcripted_text, audio_language
