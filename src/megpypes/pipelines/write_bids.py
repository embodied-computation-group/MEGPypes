
import string
import os

def build_bids_container(bids_dir_name, input_wf_dir, subject: str, session: str | None = None, **extra_tags):
    """
    Dynamically constructs BIDS compliant directories and filename substitutions.
    (Nipype Function nodes must be completely self-contained).
    """
    # check the existence of the 
    from pathlib import Path
    import os

    if not subject:
        raise ValueError("Subject identifier is required to build BIDS directory structure.")

    print(f"Building BIDS container with subject: {subject}, session: {session}, extra_tags: {extra_tags}")

    # first create output directory
    bids_output_dir = input_wf_dir / bids_dir_name
    bids_output_dir.mkdir(parents=True, exist_ok=True)

    # find the correct folder based on iterables
    folders = [f for f in Path(input_wf_dir).glob("*") if f.is_dir()]
    print("Available folders in input workflow directory:", list(folders))
    # find the folder that matches the iterables
    target_folder = None
    # collect all tags/iterables into a single dict 
    # (assumes that the pipeline iterates across all given iterables)
    iterables = {"subject": subject}
    if session:
        iterables["session"] = session
    iterables.update(extra_tags)
    # look for the pipeline iteration target folder 
    for folder in folders:
        if all(f"{key}_{value}" in folder.name for key, value in iterables.items()):
            target_folder = folder
            break
    if target_folder is None:
        raise ValueError(f"No folder found in {input_wf_dir} matching iterables {iterables}")
    print(f"Target folder found: {target_folder}")

    # build the BIDS directory path
    # first build the subject (mandatory)
    bids_sub_str = f"sub-{subject}"

    bids_dir = bids_output_dir / bids_sub_str
    
    # then build the optional session if more than 1
    if session:
        bids_ses_str = f"ses-{session}"
        bids_dir = bids_dir / bids_ses_str
    
    bids_dir.mkdir(parents=True, exist_ok=True)
    
    
    # then find the datasink data 
    datasink_path = input_wf_dir / target_folder / "datasink"
    if not datasink_path.exists():
        raise ValueError(f"Datasink directory not found in {target_folder}")
    
    # retrieve all file paths from the datasink directory
    sub_ses_output_files = list(datasink_path.glob("**/*"))
    sub_ses_output_files = [f for f in sub_ses_output_files if "/output" in str(f) and f.is_file()]
    print(f"Found {len(sub_ses_output_files)} files in datasink directory.")
    # build the BIDS compliant filename for each file and copy to the BIDS directory
    for file in sub_ses_output_files:
        if file.is_file():

            datatype_dir = file.parents[1].name
            print(f"datatype directory: {datatype_dir}")
            
            filename = f"{bids_sub_str}"
            if session:
                filename += f"_{bids_ses_str}"
            # add extra tags if provided
            if extra_tags:
                for key, value in extra_tags.items():
                    filename += f"_{key}-{value}"
            # add the original filename
            filename += f"_desc-{file.name}"
        
            # create the full path for the BIDS file
            bids_file_path = bids_dir / datatype_dir / filename
            bids_file_path.parent.mkdir(parents=True, exist_ok=True)
            # copy the file to the BIDS directory
            os.system(f"cp {file} {bids_file_path}")
            print(f"Copied {file} to {bids_file_path}")
    

    return bids_output_dir