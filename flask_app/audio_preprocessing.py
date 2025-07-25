# DEPENDENCIES
import os
from pydub import AudioSegment
from scipy.io.wavfile import read
from pydub.silence import split_on_silence


# Splitting Uploaded audio based on silence as a pre-processing step
def audio_split_on_silence(audio_path:str, dumping_path:str) -> str:
    """
    Split an audio file based on silence and save the chunks as separate WAV files

    Arguments:
    ----------
        audio_path        {str} : Path to the input audio file

        dumping_path      {str} : Directory where the split audio chunks will be saved

     Errors:
    -------
        ValueError              : If the audio file does not exist at the defined path

        SplittingOnSilenceError : If there's an issue reading, splitting or writing the
                                  audio file

    Returns:
    --------
              {str}             : 'Successful' if splitting and saving is successful, 
                                   otherwise returns an error message
    """
    # Check if the input audio file exists
    if not os.path.exists(audio_path):
            return ValueError(f"Audio file '{audio_path}' not found")
    
    try:
        # Read the audio file and extract required features
        sampling_rate, audio_array = read(filename = audio_path)
        number_of_channels         = 1 if len(audio_array.shape) == 1 else audio_array.shape[1]
        audio_data_bytes           = audio_array.tobytes()
        sample_width               = audio_array.dtype.itemsize
        # Get required sampling rate 
        #if ((sampling_rate == 192000) or (sampling_rate == 22050)):
        #    required_sampling_rate = sampling_rate
        #else:
        #    required_sampling_rate = (sampling_rate * 2)

        # Convert the audio array to a PyDub AudioSegment
        pydub_audio_segments       = AudioSegment(data         = audio_data_bytes,
                                                  frame_rate   = sampling_rate,
                                                  sample_width = sample_width,
                                                  channels     = number_of_channels)
        
        # Split the audio based on silence
        audio_chunks               = split_on_silence(audio_segment   = pydub_audio_segments,
                                                      min_silence_len = 500,  # split on silences longer than 500 ms (0.5 sec)
                                                      silence_thresh  = -40,  # anything under -40 dBFS is considered silence
                                                      keep_silence    = 300,  # keep 400 ms of leading/trailing silence
                                                     )

        # Recombine the chunks so that the parts are at least 10 seconds long
        target_length              = 10 * 1000
        output_chunks              = [audio_chunks[0]]
        
        for chunk in audio_chunks[1:]:
            if (len(output_chunks[-1]) < target_length):
                output_chunks[-1] += chunk
            else:
                # If the last output chunk is longer than the target length, start a new one
                output_chunks.append(chunk)

        # Export each chunk as a separate WAV file
        for index, chunk_item in enumerate(output_chunks):
            chunk_name = f"chunk_{index}.wav"
            chunk_path = os.path.join(dumping_path, chunk_name)
            chunk_item.export(out_f  = chunk_path, 
                              format = 'wav')
        
        return 'Successful'

    except Exception as SplittingOnSilenceError:
        return (f"SplittingOnSilenceError : While splitting the uploaded audio file based on silence,\
                 got : {repr(SplittingOnSilenceError)}".replace('  ', ''))




        
