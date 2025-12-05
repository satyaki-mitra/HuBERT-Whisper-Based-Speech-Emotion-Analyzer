# DEPENDENCIES
from typing import Literal
from typing import Optional
from dataclasses import dataclass


@dataclass
class WhisperConfig:
    """
    Whisper model configuration
    """
    # Model settings
    model_size                  : Literal['base', 'medium', 'large', 'large-v2', 'large-v3'] = 'large-v3'
    
    # Transcription parameters
    task                        : Literal['transcribe', 'translate']                         = 'transcribe'
    language                    : Optional[str]                                              = None   # Auto-detect if None
    beam_size                   : int                                                        = 5
    best_of                     : int                                                        = 5
    temperature                 : float                                                      = 0.0
    
    # Quality settings
    compression_ratio_threshold : float                                                      = 2.4
    logprob_threshold           : Optional[float]                                            = -1.0
    no_speech_threshold         : float                                                      = 0.6
    
    # Context
    condition_on_previous_text  : bool                                                       = True
    initial_prompt              : Optional[str]                                              = None
    
    # Timestamps
    word_timestamps             : bool                                                       = False
    
    # Processing
    fp16                        : bool                                                       = False  # Use FP16 for GPU
    verbose                     : bool                                                       = False
    
    # Streaming optimizations
    streaming_chunk_length      : int                                                        = 30     # seconds
    

    def to_dict(self):
        """
        Convert to dictionary for Whisper API
        """
        return {'task'                        : self.task,
                'language'                    : self.language,
                'beam_size'                   : self.beam_size,
                'best_of'                     : self.best_of,
                'temperature'                 : self.temperature,
                'compression_ratio_threshold' : self.compression_ratio_threshold,
                'logprob_threshold'           : self.logprob_threshold,
                'no_speech_threshold'         : self.no_speech_threshold,
                'condition_on_previous_text'  : self.condition_on_previous_text,
                'initial_prompt'              : self.initial_prompt,
                'word_timestamps'             : self.word_timestamps,
                'verbose'                     : self.verbose,
               }


# Default configuration
DEFAULT_WHISPER_CONFIG   = WhisperConfig()

# Fast configuration for streaming
STREAMING_WHISPER_CONFIG = WhisperConfig(beam_size                  = 1,
                                         best_of                    = 1,
                                         temperature                = 0.0,
                                         no_speech_threshold        = 0.6,
                                         condition_on_previous_text = False,
                                         word_timestamps            = False,
                                         verbose                    = False,
                                        )

# High-quality configuration for batch
BATCH_WHISPER_CONFIG     = WhisperConfig(beam_size                   = 5,
                                         best_of                     = 5,
                                         temperature                 = 0.0,
                                         compression_ratio_threshold = 2.4,
                                         logprob_threshold           = -1.0,
                                         condition_on_previous_text  = True,
                                         word_timestamps             = True,
                                        )

# Language-specific configs
MULTILINGUAL_CONFIG      = WhisperConfig(language    = None,
                                         beam_size   = 5,
                                         temperature = 0.0,
                                        )

ENGLISH_ONLY_CONFIG      = WhisperConfig(language    = 'en',
                                         beam_size   = 3,
                                         temperature = 0.0,
                                        )

# Supported languages
SUPPORTED_LANGUAGES      = {"en"  : "English",
                            "zh"  : "Chinese", 
                            "de"  : "German", 
                            "es"  : "Spanish", 
                            "ru"  : "Russian", 
                            "ko"  : "Korean", 
                            "fr"  : "French", 
                            "ja"  : "Japanese",
                            "pt"  : "Portuguese", 
                            "tr"  : "Turkish", 
                            "pl"  : "Polish", 
                            "ca"  : "Catalan",
                            "nl"  : "Dutch", 
                            "ar"  : "Arabic", 
                            "sv"  : "Swedish", 
                            "it"  : "Italian",
                            "id"  : "Indonesian", 
                            "hi"  : "Hindi", 
                            "fi"  : "Finnish", 
                            "vi"  : "Vietnamese",
                            "iw"  : "Hebrew", 
                            "uk"  : "Ukrainian", 
                            "el"  : "Greek", 
                            "ms"  : "Malay",
                            "cs"  : "Czech", 
                            "ro"  : "Romanian", 
                            "da"  : "Danish", 
                            "hu"  : "Hungarian",
                            "ta"  : "Tamil", 
                            "no"  : "Norwegian", 
                            "th"  : "Thai", 
                            "ur"  : "Urdu",
                            "hr"  : "Croatian", 
                            "bg"  : "Bulgarian", 
                            "lt"  : "Lithuanian", 
                            "la"  : "Latin",
                            "mi"  : "Maori", 
                            "ml"  : "Malayalam", 
                            "cy"  : "Welsh", 
                            "sk"  : "Slovak",
                            "te"  : "Telugu", 
                            "fa"  : "Persian", 
                            "lv"  : "Latvian", 
                            "bn"  : "Bengali",
                            "sr"  : "Serbian", 
                            "az"  : "Azerbaijani", 
                            "sl"  : "Slovenian", 
                            "kn"  : "Kannada",
                            "et"  : "Estonian", 
                            "mk"  : "Macedonian", 
                            "br"  : "Breton", 
                            "eu"  : "Basque",
                            "is"  : "Icelandic", 
                            "hy"  : "Armenian", 
                            "ne"  : "Nepali", 
                            "mn"  : "Mongolian",
                            "bs"  : "Bosnian", 
                            "kk"  : "Kazakh", 
                            "sq"  : "Albanian", 
                            "sw"  : "Swahili",
                            "gl"  : "Galician", 
                            "mr"  : "Marathi", 
                            "pa"  : "Punjabi", 
                            "si"  : "Sinhala",
                            "km"  : "Khmer", 
                            "sn"  : "Shona", 
                            "yo"  : "Yoruba", 
                            "so"  : "Somali",
                            "af"  : "Afrikaans", 
                            "oc"  : "Occitan", 
                            "ka"  : "Georgian", 
                            "be"  : "Belarusian",
                            "tg"  : "Tajik", 
                            "sd"  : "Sindhi", 
                            "gu"  : "Gujarati", 
                            "am"  : "Amharic",
                            "yi"  : "Yiddish", 
                            "lo"  : "Lao", 
                            "uz"  : "Uzbek", 
                            "fo"  : "Faroese",
                            "ht"  : "Haitian Creole", 
                            "ps"  : "Pashto", 
                            "tk"  : "Turkmen", 
                            "nn"  : "Nynorsk",
                            "mt"  : "Maltese",
                            "sa"  : "Sanskrit", 
                            "lb"  : "Luxembourgish", 
                            "my"  : "Myanmar",
                            "bo"  : "Tibetan", 
                            "tl"  : "Tagalog",
                            "mg"  : "Malagasy", 
                            "as"  : "Assamese",
                            "tt"  : "Tatar", 
                            "haw" : "Hawaiian", 
                            "ln"  : "Lingala", 
                            "ha"  : "Hausa",
                            "ba"  : "Bashkir",
                            "jw"  : "Javanese", 
                            "su"  : "Sundanese",
                           }