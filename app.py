# DPEENDENCIES
import os
import shutil
import warnings
import subprocess
from config import HOST
from config import PORT
from flask import Flask
from flask import request
from flask import jsonify
from gevent import monkey
from flask_socketio import emit
from config import UPLOAD_FOLDER
from config import RECORD_FOLDER
from flask import render_template
from flask_socketio import SocketIO
from gevent.pywsgi import WSGIServer
from flask_app.helper import create_filename
from flask_app.helper import natural_sort_key
from flask_app.audio_analyzer import analyze_audio
from geventwebsocket.handler import WebSocketHandler
from flask_app.audio_preprocessing import audio_split_on_silence


# IGNORING ALL KIND OF WARNINGS RAISED AT RUN TIME
warnings.filterwarnings("ignore")


# Apply gevent monkey patch to ensure compatibility
monkey.patch_all()


# ENSURE THE UPLOAD AND RECORD FOLDERS EXISTS
os.makedirs(name     = UPLOAD_FOLDER, 
            exist_ok = True)

os.makedirs(name     = RECORD_FOLDER, 
            exist_ok = True)


# INITALIZING FLASK APPLICATION AND SOCKET CONNECTION
ai_app   = Flask(__name__)
ai_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ai_app.config['RECORD_FOLDER'] = RECORD_FOLDER


socketio = SocketIO(app           = ai_app, 
                    ping_timeout  = 600, 
                    ping_interval = 1)



# Endpoint for Home/Index page
@ai_app.route('/')
def index():
    """
    Render the index page with a brief description of the application

    Returns:
    --------
        Response : The HTML page rendered with Flask's `render_template`
    """
    description = ("<h4>This application allows you to analyze speech to determine the emotions it conveys and transcribe it in English.<br>"
                   "You can either upload an existing audio file or record new audio to analyze emotions.<br>"
                   "The application will process the voice clip and provide a detailed breakdown of the identified emotions with respective scores.<br></h4>"
                  )

    # Render the HTML index page with the description
    return render_template(template_name_or_list = "index.html",
                           description           = description)


# Endpoint for file upload and processing
@ai_app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    Handle the file upload and save it to the upload folder

    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"FileNotFoundError": f"No file found for this path: {file}"}), 400

        filename = create_filename()
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the uploaded audio file in the destination folder
        file.save(filepath)

        # Confirm file exists in the defined path
        if not os.path.isfile(filepath):
            return jsonify({"FileNotFoundError": f"File not found at path: {filepath}"}), 400

        return jsonify({"filePath": filepath})

    except Exception as AudioUploadingError:
        return jsonify({"AudioUploadingError": f"An unexpected error occurred: {repr(AudioUploadingError)}"}), 500


@socketio.on('analyze_uploaded_audio_by_chunks')
def analyze_uploaded_audio_by_chunks(data):
    """
    Analyze uploaded audio file by splitting it into chunks and processing each chunk

    """
    try:
        file_path = data.get('filePath')
        if not file_path:
            emit('error', {'message': 'No file path provided'})
            return

        # Create a new directory (if not exists) which will contain temporary chunk files
        chunk_dir = os.path.join(UPLOAD_FOLDER, 'chunks')
        os.makedirs(name     = chunk_dir, 
                    exist_ok = True)
        
        print(f"\n\nStarting the chunking process...\n")
        # Split the audio into chunks based on silence
        splitting_result = audio_split_on_silence(audio_path   = file_path, 
                                                  dumping_path = chunk_dir)
        
        # Check the splitting result and proceed accordingly
        if (splitting_result == 'Successful'):
            print(f"Got small audio chunks. Proceeding further...")
            # Process each chunk asynchronously
            for chunk_file in sorted(os.listdir(chunk_dir), key = natural_sort_key):
                chunk_path            = os.path.join(chunk_dir, chunk_file)
                
                # Analyze the small audio chunk and get the results
                chunk_analysis_result = analyze_audio(input_audio = chunk_path)

                print(f'Emitting chunk_result for {chunk_file}: {chunk_analysis_result}')
                emit('chunk_result', chunk_analysis_result)
        else:
            print(splitting_result)
            emit('error', {'message': splitting_result})

        print('Emitting processing_complete') 
        emit('processing_complete', {'message': 'Processing complete'})

    except Exception as e:
        emit('error', {'message': str(e)})
        print(f'Error: {e}') 
    
    finally:
        # Clean up the chunks directory
        try:
            shutil.rmtree(chunk_dir)
            print(f'Chunks directory {chunk_dir} has been deleted')

        except Exception as cleanup_error:
            print(f'Error during cleanup: {cleanup_error}')
    

@socketio.on('analyze_recorded_audio_chunk')
def analyze_recorded_audio_chunk(data):
    """
    Analyze recorded audio chunk and send the results back to the client

    """
    try:
        print('Got audio chunk...')
        chunk_data = data.get('recordedChunk')

        if ((not chunk_data)):
            emit('error', {'message': 'Recorded Chunk data not provided'})
            return
        
        #print('Chunk data length:', len(chunk_data))
            
        # Create a temporary folder to save intermidiate files
        raw_folder         = os.path.join(RECORD_FOLDER, 'temp', 'raw_recordings')
        wav_folder         = os.path.join(RECORD_FOLDER, 'temp', 'final_recordings')
            
        os.makedirs(name     = raw_folder, 
                    exist_ok = True)
            
        os.makedirs(name     = wav_folder,
                    exist_ok = True)
            
        # Create two wav files for saving the data is raw and wav format
        temp_filename      = create_filename()
        final_filename     = create_filename()
            
        # Create a temporary file path to store the raw audio data
        temp_file_path     = os.path.join(raw_folder, temp_filename)

        # Create another temporary file path to store audio in required wav format
        required_file_path = os.path.join(wav_folder, final_filename)

        # Write the raw data as a temporary file
        with open(temp_file_path, 'ab') as tmp:
            tmp.write(chunk_data)
            
        # Command to convert the original audio to WAV
        conversion_command = ["ffmpeg", 
                              "-i", 
                              temp_file_path, 
                              required_file_path]
                
        # Execute the conversion, raising an exception if it fails
        try:
            subprocess.run(args           = conversion_command, 
                           check          = True, 
                           capture_output = True)
            
            # Analyze the small audio chunk and get the results
            chunk_analysis_result = analyze_audio(input_audio = required_file_path)
            # Send the result back to the client
            print(f"Emitting chunk_result for {required_file_path}: {chunk_analysis_result}")
            emit('chunk_result', chunk_analysis_result)
        
        except Exception as expo: # subprocess.CalledProcessError as e:
            #error_message = f"Error converting audio: {e.stderr.decode('utf-8')}"
            #emit('error', {'message': error_message})
            #os.remove(temp_file_path)
            print(repr(expo))
            pass
        
    except Exception as e:
        emit('error', {'message': repr(e)})
        print(f'Error: {e}')


@socketio.on('save_full_recording')
def save_full_recording(data):
    """
    Save the full recording after the analysis is complete

    """
    try:
        recording_data = data.get('recordingData')
        if not recording_data:
            emit('error', {'message': 'No recording data provided'})
            return

        recorded_filename_temp  = create_filename()
        recorded_filename_final = create_filename()
        save_path_temp          = os.path.join(RECORD_FOLDER, recorded_filename_temp)
        save_path_final         = os.path.join(RECORD_FOLDER, recorded_filename_final)

        with open(save_path_temp, 'ab') as recording_file:
            recording_file.write(recording_data)

        # Command to convert the original audio to WAV
        conversion_command      = ["ffmpeg",
                                   "-i",
                                   save_path_temp,
                                   save_path_final]

        # Execute the conversion, raising an exception if it fails
        subprocess.run(args           = conversion_command,
                       check          = True,
                       capture_output = True)

        # Delete the raw file after successful conversion to wav
        os.remove(save_path_temp)

        print(f"Successfully saved complete recording at: {save_path_final}")

    except Exception as e:
        emit('error', {'message': repr(e)})
    
    finally:
        """
        # Clean up the chunks directory
        try:
            temp_folder = os.path.join(RECORD_FOLDER, 'temp')
            shutil.rmtree(temp_folder)

        except Exception as temp_cleanup_error:
            print(f'Error during cleanup: {temp_cleanup_error}')
        """

# Route to serve the favicon.ico file
@ai_app.route('/favicon.ico')
def favicon():
    """
    Endpoint for handling requests to '/favicon.ico' and it returns an empty
    response with a 204 status code
    
    Returns:
    --------
        {str} : Empty response
    """
    return '', 204





# Start the Flask app
if __name__ == "__main__":
    # Use gevent WSGIServer to serve the Flask app with WebSocket support
    http_server = WSGIServer(listener      = (HOST, PORT), 
                             application   = ai_app, 
                             handler_class = WebSocketHandler)
    
    socketio.run(app   = ai_app, 
                 port  = PORT, 
                 debug = True)