from pathlib import Path
from nipype import Workflow, Node, IdentityInterface, SelectFiles, DataSink, config
from megpypes.interfaces.initpreproc import InitialPreproc
from megpypes.interfaces.artifact_rejection import ArtifactRejection
from megpypes.interfaces.auto_ica import AutoICA
from megpypes.interfaces.epoching import Epoching
from megpypes.pipelines.utils import apply_interface_config
import logging
logger = logging.getLogger(__name__)

def create_meg_preprocessing(
    basedir: str,
    workdir: str,
    output_dir: str,
    subject_list: list[str],
    pipeline_config: dict
):
    """
    MEG initial (basic) preprocessing pipeline using classic Nipype iterables pattern.
    
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
    initial_preproc = Node(InitialPreproc(), name='initial_preproc')
    apply_interface_config(initial_preproc, pipeline_config["initial_preproc"])

    # ==== Artifact Rejection ====
    artifact_rejection = Node(ArtifactRejection(), name="artifact_rejection")
    apply_interface_config(artifact_rejection, pipeline_config["artifact_rejection"])

    # === Auto ICA ====
    ica = Node(AutoICA(), name="auto_ica")
    apply_interface_config(ica, pipeline_config["auto_ica"])

    # === Epoching ====
    epoching = Node(Epoching(), name="epoching")
    apply_interface_config(epoching, pipeline_config["epoching"])
    
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
    
    # Log workflow structure (after creation, not during)
    logger.info(f"Created workflow with {len(subject_list)} subjects")
    logger.debug(f"Subject list: {subject_list}")
    
    return wf