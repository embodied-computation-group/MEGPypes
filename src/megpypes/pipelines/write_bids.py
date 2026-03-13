
import string
import os

def build_bids_container(bids_dir_path, input_wf_dir, **iterables):
    """
    Dynamically constructs BIDS compliant directories and filename substitutions.
    (Nipype Function nodes must be completely self-contained).
    """
    # check the existence of the 
    import os

    # first create directory
    os.makedirs(bids_dir_path, exist_ok=True)

    # then look for current iterable
    print("Received iterables:", iterables)
    # find the correct folder based on iterables
    folders = os.listdir(input_wf_dir)
    print("Available folders in input workflow directory:", folders)
    # find the folder that matches the iterables
    target_folder = None
    for folder in folders:
        if all(f"{key}-{value}" in folder for key, value in iterables.items()):
            target_folder = folder
            break
    if target_folder is None:
        raise ValueError(f"No folder found in {input_wf_dir} matching iterables {iterables}")
    print(f"Target folder found: {target_folder}")

    # build the BIDS directory path
    # first build the subject (mandatory)
    try:
        subject = iterables.get("subject_id") or iterables.get("subject") or iterables.get("sub")
        session = iterables.get("session_id") or iterables.get("session") or iterables.get("ses")
    except AttributeError:
        subject = None
        session = None

    if subject is None:
        raise ValueError("Subject identifier not found in iterables. Expected keys: 'subject_id', 'subject', or 'sub'.")
    
    bids_sub = f"sub-{subject}"
    bids_dir = os.path.join(bids_dir_path, bids_sub)
    
    # then build the session if it exists
    if session:
        bids_ses = f"ses-{session}"
        bids_dir = os.path.join(bids_dir, bids_ses)

        # create the directory if it doesn't exist
    os.makedirs(bids_dir, exist_ok=True)

    # then find the datasink data 
    data_sink_path = os.path.join(input_wf_dir, target_folder, "datasink/output")
    if not os.path.exists(data_sink_path):
        raise ValueError(f"Datasink path not found: {data_sink_path}")
    
    # retrieve all file paths from the datasink directory
    file_paths = []
    for root, dirs, files in os.walk(data_sink_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    print(f"Files found in datasink: {file_paths}")

    # build the iteration subfiles
    # look for dir name after /output
    

    return bids_dir_path