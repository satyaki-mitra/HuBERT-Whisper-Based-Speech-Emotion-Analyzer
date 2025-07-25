# DEPENDENCIES
import re
import uuid
import warnings

# IGNORING ALL KIND OF WARNINGS RAISED AT RUN TIME
warnings.filterwarnings('ignore')


# HELPER FUNCTION TO CREATE A UNIQUE FILE NAME
def create_filename() -> str:
    """
    Create a unique filename based following a specific convention

    Errors:
    -------
        FileNameCreationError : If any exception happens during filename creation

    Returns:
    --------
               {str}          : A string representing the unique filename with .wav extension
    """
    try:
        # Generate a UUID
        unique_id = uuid.uuid4().hex

        # Create a unique filename using the unique_id
        filename  = f"{unique_id}.wav"
        
        return filename
    
    except Exception as FileNameCreationError:
        # Return a detailed error message if something goes wrong
        return (f"FileNameCreationError: While creating unique filename maintaining a\
                specific convention, got : {repr(FileNameCreationError)}".replace('  ', ''))


def natural_sort_key(input_string: str) -> list:
    """
    Generate a natural sort key for a string
    
    Arguments:
    ----------
        input_string {str} : The string to generate a sort key for
    
    Returns:
    --------
        {list}             : The natural sort key
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', input_string)]
