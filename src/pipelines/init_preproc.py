from pathlib import Path
from nipype import Workflow, Node, IdentityInterface, SelectFiles, DataSink
from src.interfaces.initpreproc import InitialPreproc
from src.utils import set_node_inputs
import logging
logger = logging.getLogger(__name__)

def create_initial_preprocessing(
    basedir: str,
    workdir: str,
    output_dir: str,
    subject_list: list[str],
    crop_params: dict,
    filter_params: dict,
    gradcomp_params: dict,
):
    """
    MEG preprocessing pipeline using classic Nipype iterables pattern.
    
    Parallelization: infosource.iterables creates one workflow execution per subject.
    No MapNode needed—iterables handles subject-level iteration.
    """
    # set basedir to full path based on working dir
    raw_dir = Path(basedir)
    if not raw_dir.is_absolute():
        logger.debug("Data path is not absolute, setting to working dir.")
        raw_dir = Path().cwd() / basedir
        print(f"raw_dir {raw_dir}")

    # Create workflow
    wf = Workflow(name="megpreproc")
    wf.base_dir = workdir
    
    # === SUBJECT ITERATION (classic Nipype pattern) ===
    infosource = Node(
        IdentityInterface(fields=['subject_id']),
        name="infosource"
    )
    infosource.iterables = [('subject_id', subject_list)]  # ← This creates parallel executions
    
    # === FILE SELECTION ===
    templates = {"meg": "{subject_id}/meg/{subject_id}_task-MMNHCS_run-0_meg.fif"}
    selectraw = Node(
        SelectFiles(templates, base_directory=raw_dir),
        name="selectfiles"
    )
    
    # === PROCESSING NODE (regular Node, NOT MapNode) ===
    # iterables handles the iteration, so no iterfield needed
    initial_preproc = Node(
        InitialPreproc(),
        name='initial_preproc'
    )
    
    # Define parameters as a dictionary
    params = {
        "stim_channel": crop_params["stim_channel"],
        "min_buffer": crop_params["min_buffer"],
        "max_buffer": crop_params["max_buffer"],
        "l_freq": filter_params["l_freq"],
        "h_freq": filter_params["h_freq"],
        "gradcomp_auto": gradcomp_params["auto"],
        "gradcomp_order": gradcomp_params["order"],
        "out_file": "preproc_raw.fif",
    }

    # Use the helper function to set node inputs
    set_node_inputs(initial_preproc, params)
    
    # === OUTPUT ===
    datasink = Node(
        DataSink(
            base_directory=output_dir,
            container="preprocessed",
            parameterization=False  # Clean output paths
        ),
        name="datasink"
    )
    
    # === CONNECTIONS ===
    wf.connect([
        (infosource, selectraw, [("subject_id", "subject_id")]),
        (selectraw, initial_preproc, [("meg", "in_file")]),
        (initial_preproc, datasink, [("out_file", "megpreproc.@final")]),
    ])
    
    # Optional: log workflow structure (after creation, not during)
    logger.info(f"Created workflow with {len(subject_list)} subjects")
    logger.debug(f"Subject list: {subject_list}")
    
    return wf