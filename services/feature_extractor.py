# sDEPENDENCIES
import librosa
import numpy as np
from typing import Dict
from utils.logging_util import setup_logger


# SETUP LOGGING
logger = setup_logger(__name__)


class AudioFeatureExtractor:
    """
    Extract acoustic features from audio for explainability
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    

    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract comprehensive audio features
        
        Arguments:
        ----------
            audio_path { str } : Path to the audio file

        Returns:
        --------
               { dict }        : Dictionary of feature names and values
        """
        try:
            # Load audio
            audio, sr              = librosa.load(path = audio_path, 
                                                  sr   = self.sample_rate,
                                                 )
            
            # Pitch features
            pitches, magnitudes    = librosa.piptrack(y=audio, sr=sr)
            pitch_values           = pitches[pitches > 0]
            pitch_variance         = float(np.var(pitch_values)) if (len(pitch_values) > 0) else 0.0
            
            # Energy features
            rms                    = librosa.feature.rms(y=audio)[0]
            energy_rms             = float(np.mean(rms))
            
            # Spectral features
            spectral_centroid      = librosa.feature.spectral_centroid(y  = audio, 
                                                                       sr = sr,
                                                                      )[0]

            spectral_centroid_mean = float(np.mean(spectral_centroid))
            
            # Zero crossing rate
            zcr                    = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean               = float(np.mean(zcr))
            
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs                  = librosa.feature.mfcc(y      = audio, 
                                                          sr     = sr, 
                                                          n_mfcc = 13,
                                                         )

            mfcc_features          = {f'MFCC-{i+1} (Timbre)' if (i < 2) else f'MFCC-{i+1}': float(np.mean(mfccs[i])) for i in range(min(3, len(mfccs)))}
            
            # Jitter and Shimmer (voice quality)
            jitter                 = self._calculate_jitter(audio, sr)
            shimmer                = self._calculate_shimmer(audio = audio)
            
            # Speaking rate (approximate)
            tempo, _               = librosa.beat.beat_track(y  = audio, 
                                                             sr = sr,
                                                            )
            speaking_rate          = float(tempo)
            
            # Formants (simplified using spectral peaks)
            formant_ratio          = self._calculate_formant_ratio(audio = audio, 
                                                                   sr    = sr,
                                                                  )
            
            # Combine all features
            features               = {'Pitch Variance'         : pitch_variance,
                                      'Energy (RMS)'           : energy_rms,
                                      'Speaking Rate'          : speaking_rate / 100,  # Normalize
                                      'Spectral Centroid'      : spectral_centroid_mean / 10000,  # Normalize
                                      'Zero Crossing Rate'     : zcr_mean,
                                       **mfcc_features,
                                      'Jitter (Voice Quality)' : jitter,
                                      'Shimmer (Amplitude)'    : shimmer,
                                      'Formant F1/F2 Ratio'    : formant_ratio,
                                     }
            
            logger.info(f"Extracted {len(features)} features from audio")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {repr(e)}")
            return {}

    
    def _calculate_jitter(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate jitter (period perturbation)
        """
        try:
            # Simplified jitter calculation
            f0       = librosa.yin(y     = audio,      
                                   fmin  = 50, 
                                   fmax  = 400, 
                                   sr    = sr,
                                  )

            valid_f0 = f0[f0 > 0]
            
            if (len(valid_f0) > 1):
                periods = 1.0 / valid_f0
                jitter  = np.std(np.diff(periods)) / np.mean(periods)
                
                # Cap at 1.0
                return float(min(jitter, 1.0))  

            return 0.0
        
        except:
            return 0.0
    

    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """
        Calculate shimmer (amplitude perturbation)
        """
        try:
            # Shimmer calculation
            amplitudes = np.abs(audio)

            if (len(amplitudes) > 1):
                shimmer = np.std(np.diff(amplitudes)) / np.mean(amplitudes)
                
                # Cap at 1.0
                return float(min(shimmer, 1.0))  
            
            return 0.0
        
        except:
            return 0.0
    

    def _calculate_formant_ratio(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate approximate F1/F2 formant ratio
        """
        try:
            # Use spectral peaks as formant approximation
            spectrum = np.abs(np.fft.rfft(audio))
            freqs    = np.fft.rfftfreq(len(audio), 1/sr)
            
            # Find peaks in typical formant ranges
            f1_range = (200, 1000)
            f2_range = (1000, 3000)
            
            f1_mask  = (freqs >= f1_range[0]) & (freqs <= f1_range[1])
            f2_mask  = (freqs >= f2_range[0]) & (freqs <= f2_range[1])
            
            f1_peak  = np.max(spectrum[f1_mask]) if np.any(f1_mask) else 1.0
            f2_peak  = np.max(spectrum[f2_mask]) if np.any(f2_mask) else 1.0
            
            ratio    = (f1_peak / f2_peak) if (f2_peak > 0) else 0.5
            
            return float(min(ratio, 1.0))
        
        except:
            return 0.5