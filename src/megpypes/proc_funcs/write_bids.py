def build_bids_container(bids_dir_name, input_wf_dir, subject: str, session: str | None = None, **extra_tags):
    """
    Dynamically constructs BIDS compliant directories and filename substitutions.
    (NiPype Function nodes must be completely self-contained).

    Arguments
    ----------
    bids_dir_name: str
        Name of the output BIDS directory to create within the workflow's working directory.
    input_wf_dir: Path
        Path to the workflow's working directory where the BIDS directory will be created.
    subject: str
        Subject identifier to use in the BIDS directory structure and filenames (e.g., "01").
    session: str | None
        Optional session identifier to include in the BIDS directory structure and filenames (e.g., "01").

    Returns
    -------
    bids_output_dir: Path
        Path to the created BIDS directory containing the organized output files.

    """
    # check the existence of the 
    from pathlib import Path
    import os
    import re

    if not isinstance(bids_dir_name, str):
        raise ValueError("bids_dir_name must be a string.")
    if not isinstance(input_wf_dir, Path):
        raise ValueError("input_wf_dir must be a Path object.")

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
    bids_ses_str = None
    
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

            # Determine the datatype directory
            parts = file.parts
            if "output" in parts:
                output_idx = parts.index("output")
                # Datatype is the first path component right after "output".
                if output_idx + 1 < len(parts) - 1:
                    datatype_dir = parts[output_idx + 1]
                else:
                    # Fallback for files placed directly under "output" with no datatype folder.
                    datatype_dir = "misc"
            else:
                # Defensive fallback if filtering changes upstream.
                datatype_dir = file.parent.name
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

    # Recurseively go through the BIDS directory and find folders with files
    # then sort the files and rename the date string according to order 0,1,2
    def normalize_bids_filenames(bids_root: Path):
        """
        Recursively walks through BIDS directory and replaces timestamp in filenames
        with ordered integers per directory.
        """

        # regex to capture timestamp after desc-
        pattern = re.compile(r"(desc-)(\d{8}T\d{6})(.*)")

        # iterate through all subdirectories
        for directory in [d for d in bids_root.rglob("*") if d.is_dir()]:
            files = [f for f in directory.iterdir() if f.is_file()]

            if not files:
                continue

            # filter only files that match expected pattern
            matched_files = []
            for f in files:
                match = pattern.search(f.name)
                if match:
                    matched_files.append((f, match))

            if not matched_files:
                continue

            # sort by timestamp string
            matched_files.sort(key=lambda x: x[1].group(2))

            # rename with index
            for idx, (file_path, match) in enumerate(matched_files):
                prefix, timestamp, suffix = match.groups()

                new_name = pattern.sub(f"{prefix}{idx}{suffix}", file_path.name)
                new_path = file_path.with_name(new_name)

                print(f"Renaming {file_path.name} -> {new_name}")
                file_path.rename(new_path)

    normalize_bids_filenames(bids_output_dir)

    return bids_output_dir

