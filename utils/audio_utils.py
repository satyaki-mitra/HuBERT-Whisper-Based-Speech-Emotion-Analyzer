# DEPENDENCIES
import os
import re
import uuid
import shutil
import torchaudio
import subprocess
import numpy as np
from typing import List
from typing import Tuple
from pathlib import Path
from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from config.settings import SAMPLE_RATE
from config.settings import KEEP_SILENCE
from config.settings import TEMP_FILES_DIR
from pydub.silence import split_on_silence
from utils.logging_util import setup_logger
from config.settings import SILENCE_THRESHOLD
from config.settings import MIN_SILENCE_LENGTH
from config.settings import TARGET_CHUNK_LENGTH


# SETUP LOGGING
logger = setup_logger(__name__)


def create_unique_filename(extension: str = 'wav') -> str:
    """
    Create a unique filename with timestamp and UUID
    
    Arguments:
    ----------
        extension { str } : File extension without dot
        
    Returns:
    --------
           { str }        : Unique filename string
    """
    unique_id        = uuid.uuid4().hex
    unique_filename = f"{unique_id}.{extension}"
    
    return unique_filename


def natural_sort_key(input_string: str) -> list:
    """
    Generate a natural sort key for sorting filenames
    
    Arguments:
    ----------
        input_string { str } : The string to generate sort key for
    
    Returns:
        List of integers and strings for natural sorting
    """
    sorted_keys = [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', input_string)]
    
    return sorted_keys


def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """
    Convert audio file to WAV format using FFmpeg
    
    Arguments:
    ----------
        input_path  { str } : Path to input audio file

        output_path { str } : Path for output WAV file (optional)
        
    Returns:
    --------
             { str }        : Path to converted WAV file
    """
    if output_path is None:
        output_path = str(TEMP_FILES_DIR / create_unique_filename('wav'))
    
    try:
        command = ['ffmpeg', '-y', '-i', input_path, '-ar', str(SAMPLE_RATE), '-ac', '1', '-sample_fmt', 's16', output_path]
        
        subprocess.run(command, 
                       check          = True, 
                       capture_output = True, 
                       text           = True,
                      )

        logger.info(f"Successfully converted {input_path} to WAV format")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode('utf-8')}")
        
        raise RuntimeError(f"Audio conversion failed: {repr(e)}")


def convert_to_wav_streaming(input_path: str, output_path: str = None) -> str:
    """
    Convert audio file to WAV format - optimized for STREAMING (fast!)
    """
    if output_path is None:
        output_path = str(TEMP_FILES_DIR / create_unique_filename('wav'))
    
    try:
        # STRATEGY 1: If it's already a WAV file, just copy it
        if input_path.lower().endswith('.wav'):
            shutil.copy2(input_path, output_path)
            logger.debug(f"Streaming: Copied WAV file directly")
            
            return output_path
        
        # STRATEGY 2: For webm/ogg from browser: optimized FFmpeg
        command = ['ffmpeg',
                   '-y',                     # Overwrite without asking
                   '-i', input_path,
                   '-ar', str(SAMPLE_RATE),  # Sample rate
                   '-ac', '1',               # Mono
                   '-c:a', 
                   'pcm_s16le',              # Audio codec
                   '-loglevel', 'error',     # Minimal logging
                   '-threads', '1',          # Single thread for faster startup
                    output_path
                   ]
        
        # Use Popen with timeout for streaming
        process = subprocess.Popen(command,
                                   stdout = subprocess.DEVNULL,  # Discard stdout
                                   stderr = subprocess.PIPE,
                                   text   = True,
                                  )
        
        # Wait with short timeout for streaming: 10 second maximum
        try:
            _, stderr = process.communicate(timeout = 10)  
            if (process.returncode != 0):
                raise RuntimeError(f"FFmpeg streaming failed: {stderr}")

        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning(f"Streaming conversion timed out, using fallback for: {input_path}")
            
            # Fall back to pydub which might be faster for small chunks
            return streaming_fallback_conversion(input_path, output_path)
        
        logger.debug(f"Streaming FFmpeg conversion successful: {input_path}")
        return output_path
        
    except Exception as e:
        logger.warning(f"Streaming conversion failed, using fallback: {str(e)}")
        # Emergency fallback
        return streaming_fallback_conversion(input_path, output_path)


def streaming_fallback_conversion(input_path: str, output_path: str) -> str:
    """
    Ultra-fast fallback conversion for streaming when FFmpeg is too slow
    """
    try:
        # Use pydub which is often faster for simple conversions
        audio = AudioSegment.from_file(input_path)
        
        # Minimal processing: just ensure mono, don't resample unless needed
        if (audio.channels > 1):
            audio = audio.set_channels(1)
        
        # Only resample if sample rate is very different
        if (abs(audio.frame_rate - SAMPLE_RATE) > 4000):
            audio = audio.set_frame_rate(SAMPLE_RATE)
        
        # Export with minimal options
        audio.export(output_path, 
                     format     = 'wav',
                     parameters = ['-acodec', 'pcm_s16le']  # 16-bit PCM
                    )
        
        logger.debug(f"Streaming fallback conversion successful: {input_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"All streaming conversion methods failed: {repr(e)}")
        
        # Last resort: if input is WAV-like, just rename it
        if input_path.lower().endswith('.wav'):
            try:
                shutil.copy2(input_path, output_path)
                return output_path
            
            except:
                pass
        
        raise RuntimeError(f"All audio conversion methods failed for streaming: {str(e)}")


def load_audio_as_tensor(audio_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and convert to tensor with target sample rate
    
    Arguments:
    ----------
        audio_path { str } : Path to audio file

        target_sr  { str } : Target sample rate
        
    Returns:
    --------
          { np.ndarray }   : Audio tensor
    """
    try:
        waveform, sample_rate = torchaudio.load(uri = audio_path)
        
        # Resample if necessary
        if (sample_rate != target_sr):
            resampler = torchaudio.transforms.Resample(orig_freq            = sample_rate, 
                                                       new_freq             = target_sr, 
                                                       resampling_method    = 'sinc_interp_hann', 
                                                       lowpass_filter_width = 6, 
                                                       rolloff              = 0.99,
                                                      )
            waveform  = resampler(waveform)
        
        # Convert to mono if stereo
        if (waveform.shape[0] > 1):
            waveform = waveform.mean(dim     = 0, 
                                     keepdim = True,
                                    )
        
        return waveform.squeeze().numpy()
        
    except Exception as e:
        logger.error(f"Error loading audio {audio_path}: {repr(e)}")
        
        raise


def split_audio_on_silence(audio_path: str, output_dir: str) -> List[str]:
    """
    Split audio file into chunks based on silence detection
    
    Arguments:
    ----------
        audio_path { str } : Path to input audio file

        output_dir { str } : Directory to save chunks
        
    Returns:
    --------
            { list }       : List of chunk file paths
    """
    try:
        # Read audio file
        sample_rate, audio_array = read(audio_path)
        num_channels             = 1 if (len(audio_array.shape) == 1) else audio_array.shape[1]
        audio_bytes              = audio_array.tobytes()
        sample_width             = audio_array.dtype.itemsize
        
        # Create AudioSegment
        audio_segment            = AudioSegment(data         = audio_bytes,
                                                frame_rate   = sample_rate,
                                                sample_width = sample_width,
                                                channels     = num_channels,
                                               )
        
        # Split on silence
        chunks                   = split_on_silence(audio_segment   = audio_segment,
                                                    min_silence_len = MIN_SILENCE_LENGTH,
                                                    silence_thresh  = SILENCE_THRESHOLD,
                                                    keep_silence    = KEEP_SILENCE,
                                                   )
        
        if not chunks:
            logger.warning("No chunks created, returning original file")
            return [audio_path]
        
        # Combine small chunks
        combined_chunks = [chunks[0]]

        for chunk in chunks[1:]:
            if (len(combined_chunks[-1]) < TARGET_CHUNK_LENGTH):
                combined_chunks[-1] += chunk
            
            else:
                combined_chunks.append(chunk)
        
        # Export chunks
        chunk_paths = list()

        os.makedirs(output_dir, exist_ok = True)
        
        for idx, chunk in enumerate(combined_chunks):
            chunk_path = os.path.join(output_dir, f"chunk_{idx:03d}.wav")

            chunk.export(chunk_path, format = 'wav')
            chunk_paths.append(chunk_path)
        
        logger.info(f"Split audio into {len(chunk_paths)} chunks")
        
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Error splitting audio: {repr(e)}")
        raise


def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio array to [-1, 1] range
    
    Arguments:
    ----------
        audio_array { np.ndarray } : Input audio array
        
    Returns:
    --------
             { np.ndarray }        : Normalized audio array
    """
    max_val = np.abs(audio_array).max()

    if (max_val > 0):
        return audio_array / max_val

    return audio_array


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds
    
    Arguments:
    ----------
        audio_path { str } : Path to audio file
        
    Returns:
    --------
           { float }       : Duration in seconds
    """
    try:
        audio          = AudioSegment.from_file(file = audio_path)

        # Convert ms to seconds
        audio_duration = len(audio) / 1000.0

        return audio_duration 
    
    except Exception as e:
        logger.error(f"Error getting audio duration: {repr(e)}")
        return 0.0


def save_audio_array(audio_array: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE):
    """
    Save numpy audio array to WAV file
    
    Arguments:
    ----------
        audio_array { np.ndarray } : Audio data as numpy array

        output_path    { str }     : Path to save WAV file
        
        sample_rate    { int }     : Sample rate of audio
    """
    try:
        # Normalize to 16-bit PCM range
        audio_int16 = (audio_array * 32767).astype(np.int16)

        write(filename = output_path, 
              rate     = sample_rate, 
              data     = audio_int16,
             )

        logger.info(f"Saved audio to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving audio: {repr(e)}")
        raise