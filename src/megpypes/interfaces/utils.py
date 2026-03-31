import datetime
from pathlib import Path
import mne
from mne import BaseEpochs
from mne.io import BaseRaw

def abspath_with_time(base_filename):
    """Utility function to create new filename with a timestamp prefix and absolute path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp}-{base_filename}"
    new_out_file_str = str(Path(filename).absolute())
    
    return new_out_file_str
    