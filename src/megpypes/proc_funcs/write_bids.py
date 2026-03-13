
import string
import os

def build_bids_paths(subject=None, session=None, task=None, run=None, event_label=None):
    """
    Dynamically constructs BIDS compliant directories and filename substitutions.
    (Nipype Function nodes must be completely self-contained).
    """
    parts_dir = []
    parts_name = []
    
    if subject is not None:
        sub = str(subject).replace("sub-", "")
        parts_dir.append(f"sub-{sub}")
        parts_name.append(f"sub-{sub}")
        
    if session is not None:
        ses = str(session).replace("ses-", "")
        parts_dir.append(f"ses-{ses}")
        parts_name.append(f"ses-{ses}")
        
    if task is not None:
        parts_name.append(f"task-{task}")
        
    if run is not None:
        parts_name.append(f"run-{run}")
        
    # The container folder: "sub-X/ses-Y"
    bids_dir = "/".join(parts_dir) if parts_dir else ""
    
    # The filename prefix: "sub-X_ses-Y_task-Z"
    bids_prefix = "_".join(parts_name) if parts_name else "bids"
        
    # DataSink substitutions to replace raw output names with BIDS-compliant names
    substitutions = [
        ("artifact_cleaned_raw.fif", f"{bids_prefix}_desc-cleaned_meg.fif"),
        ("ica-icasolution.fif", f"{bids_prefix}_desc-ica_meg.fif"),
        ("ica_comps_source_plot.png", f"{bids_prefix}_desc-icacomps_plot.png"),
        # In case some outputs write without extensions
        ("artifact_cleaned_raw", f"{bids_prefix}_desc-cleaned_meg"),
        ("ica-icasolution", f"{bids_prefix}_desc-ica_meg"),
        ("ica_comps_source_plot", f"{bids_prefix}_desc-icacomps_plot")
    ]
    
    return bids_dir, substitutions