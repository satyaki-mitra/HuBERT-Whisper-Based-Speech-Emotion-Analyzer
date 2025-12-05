# DEPENDENCIES
import os
import sys
import socket
import shutil
import subprocess
from pathlib import Path


# COLOR CODES FOR TERMINAL OUTPUT
class Colors:
    HEADER  = '\033[95m'
    OKBLUE  = '\033[94m'
    OKCYAN  = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_python_version():
    """
    Check if Python version is 3.10+
    """
    print_info("Checking Python version...")

    version = sys.version_info
    if (version.major >= 3) and (version.minor >= 10):
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True

    else:
        print_error(f"Python {version.major}.{version.minor} detected. Python 3.10+ required!")
        return False


def check_ffmpeg():
    """
    Check if FFmpeg is installed
    """
    print_info("Checking FFmpeg installation...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                                capture_output = True, 
                                text           = True,
                               )

        if (result.returncode == 0):
            version = result.stdout.split('\n')[0]
            print_success(f"FFmpeg installed: {version}")

            return True
    
    except FileNotFoundError:
        print_error("FFmpeg not found!")
        print_info("Install FFmpeg:")
        print_info("  macOS: brew install ffmpeg")
        print_info("  Ubuntu: sudo apt install ffmpeg")
        print_info("  Windows: choco install ffmpeg")
        return False

    return False


def check_dependencies():
    """
    Check if required Python packages are installed
    """
    print_info("Checking Python dependencies...")

    required = ['flask', 
                'torch', 
                'scipy', 
                'pydub', 
                'librosa',
                'whisper',
                'torchaudio',
                'transformers',
                'python-dotenv',
                'flask_socketio',  
               ]

    missing  = list()
    
    for package in required:
        try:
            __import__(package)
            print_success(f"{package}")
        
        except ImportError:
            print_error(f"{package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print_warning(f"\nMissing packages: {', '.join(missing)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_models():
    """
    Check if model files exist
    """
    print_info("Checking model files...")
    
    hubert_path  = Path('models/hubert')
    whisper_path = Path('models/whisper/large-v3.pt')
    all_exist    = True
    
    if hubert_path.exists() and (hubert_path / 'config.json').exists():
        print_success(f"HuBERT model found at {hubert_path}")
    
    else:
        print_error(f"HuBERT model not found at {hubert_path}")
        print_info("Download from: https://huggingface.co/facebook/hubert-base-ls960")
        
        all_exist = False
    
    if whisper_path.exists():
        print_success(f"Whisper model found at {whisper_path}")
    
    else:
        print_error(f"Whisper model not found at {whisper_path}")
        print_info("Download from: https://github.com/openai/whisper")
        
        all_exist = False
    
    return all_exist


def check_directories():
    """
    Check and create required directories
    """
    print_info("Checking directory structure...")
    
    dirs = ['logs', 
            'models',
            'data/temp_files',
            'data/uploaded_files',
            'data/recorded_files',
           ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents = True, exist_ok = True)

        print_success(f"{dir_path}/")
    
    return True


def check_env_file():
    """
    Check if .env file exists
    """
    print_info("Checking configuration...")
    
    if Path('.env').exists():
        print_success(".env file found")
        return True
   
    else:
        print_warning(".env file not found")

        if Path('.env.example').exists():
            print_info("Creating .env from .env.example...")
            shutil.copy('.env.example', '.env')
            print_success("Created .env file - please review and update settings")
        
        else:
            print_error("No .env.example found!")
            return False

    return True


def check_port(port = 8000):
    """
    Check if port is available
    """
    print_info(f"Checking if port {port} is available...")
    
    sock   = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))

    sock.close()
    
    if (result == 0):
        print_warning(f"Port {port} is already in use!")
        print_info("Change PORT in .env or stop the other service")

        return False
    
    else:
        print_success(f"Port {port} is available")
        return True


def run_system_check():
    """
    Run all system checks
    """
    print_header("EmotiVoice System Check")
    
    checks = [("Python Version", check_python_version),
              ("FFmpeg", check_ffmpeg),
              ("Dependencies", check_dependencies),
              ("Model Files", check_models),
              ("Directories", check_directories),
              ("Configuration", check_env_file),
              ("Port Availability", lambda: check_port()),
             ]
    
    all_passed = True

    for name, check_func in checks:
        if not check_func():
            all_passed = False

        print()
    
    return all_passed


def launch_app():
    """
    Launch the Flask application
    """
    print_header("Launching EmotiVoice")
    
    try:
        print_info("Starting Flask server...")
        print_info("Access the application at: http://localhost:8000")
        print_info("Press Ctrl+C to stop the server\n")
        
        # Import and run the app
        from app import socketio, app
        from config.settings import HOST, PORT
        
        socketio.run(app,
                     host                  = HOST,
                     port                  = PORT,
                     debug                 = True,
                     allow_unsafe_werkzeug = True,
                    )
        
    except KeyboardInterrupt:
        print_info("\n\nShutting down gracefully...")
        print_success("EmotiVoice stopped")

    except Exception as e:
        print_error(f"Failed to start application: {str(e)}")
        sys.exit(1)


def main():
    """
    Main entry point
    """
    print_header("EmotiVoice - AI Speech Emotion Recognition")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run system checks
    if not run_system_check():
        print_error("\n- System check failed!")
        print_warning("Please fix the issues above before launching")
        sys.exit(1)
    
    print_success("\n- All system checks passed!")
    print_info("Ready to launch...\n")
    
    # Ask user to proceed
    response = input(f"{Colors.BOLD}Launch EmotiVoice? [Y/n]: {Colors.ENDC}").strip().lower()
    
    if response in ['', 'y', 'yes']:
        launch_app()

    else:
        print_info("Launch cancelled")


if __name__ == "__main__":
    main()
