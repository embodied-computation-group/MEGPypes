from pathlib import Path
import re

from nipype import JoinNode, Workflow, Node, IdentityInterface, SelectFiles, DataSink, config
from nipype.interfaces.utility import Function
from megpypes.interfaces.initpreproc import InitialPreproc
from megpypes.interfaces.artifact_rejection import ArtifactRejection
from megpypes.interfaces.auto_ica import AutoICA
from megpypes.interfaces.epoching import Epoching
from megpypes.pipelines.utils import apply_interface_config
from megpypes.proc_funcs.epoch import concatenate_epoch_files
from megpypes.proc_funcs.write_bids import build_bids_container
import logging
logger = logging.getLogger(__name__)

def create_meg_preprocessing(
    basedir: str,
    workdir: str,
    output_dir: str,
    file_templates: str,
    iterable_fields: list[str],
    iterable_values: dict[str, list[str]],
    pipeline_config: dict
):
    """
    MEG initial (basic) preprocessing pipeline using classic Nipype iterables pattern.
    
    Parallelization: infosource.iterables creates one workflow execution per subject.
    No MapNode needed—iterables handles subject-level iteration.

    Arguments
    ----------
    basedir: str
        Base directory where raw data is located (used for SelectFiles).
    workdir: str
        Directory where the workflow will store intermediate results and outputs.
    output_dir: str
        Name of the output BIDS directory to create within the workflow's working directory.
    file_templates: dict
        Dictionary of file templates for SelectFiles (e.g., {"meg": "sub-{subject}/meg/sub-{subject}_meg.fif"}).
    iterable_fields: list[str]
        List of fields to iterate over (e.g., ["subject", "session"]).
    iterable_values: dict[str, list[str]]
        Dictionary mapping iterable fields to their values (e.g., {"subject": ["01", "02"], "session": ["01"]}).
    pipeline_config: dict
        Dictionary containing configuration for each interface. Keys should match interface names.
        See the example config file for more info on expected config structure.
    
    Returns
    -------
    wf: Workflow
        Configured Nipype workflow ready for execution.
    """
    # set basedir to full path based on working dir
    raw_dir = Path(basedir)
    if not raw_dir.is_absolute():
        logger.debug("Data path is not absolute, setting to working dir.")
        raw_dir = Path().cwd() / basedir
        print(f"raw_dir {raw_dir}")

    # Create workflow
    wf_name = "megpreproc"
    wf = Workflow(name=wf_name)
    wf.base_dir = workdir

    # Ensure a subject axis is always available for BIDS naming.
    iterable_fields = list(iterable_fields)
    iterable_values = dict(iterable_values)
    if "subject" not in iterable_fields:
        logger.warning(
            "No 'subject' iterable field was provided. Injecting synthetic subject axis with default value '01'."
        )
        iterable_fields.append("subject")
    subject_values = iterable_values.get("subject")
    if not subject_values:
        logger.warning(
            "No subject iterable values were provided. Using default subject ['01'] for BIDS output naming."
        )
        iterable_values["subject"] = ["01"]

    infosource = Node(
        IdentityInterface(fields=iterable_fields),
        name="infosource"
    )

    # Build iterables dict from config/inputs
    iterables_dict = {}
    print(f"Iterating over fields: {iterable_fields}")
    print(f"Provided iterable values: {iterable_values}")
    for field in iterable_fields:
        print(f"Processing iterable field: {field}")
        if field in iterable_values.keys():
            print(f"Using custom values for field: {field}")
            # Allow config to override with custom values per field
            iterables_dict[field] = iterable_values[field]
        # TODO: Add support for dynamic discovery of values (e.g. from filesystem) if not provided in config
        else:
            print(iterable_values[field])
            raise ValueError(f"No values provided for iterable field: {field}")

    print(f"Final iterables dict: {iterables_dict}")
    # Transpose: [{'sub':'01', 'task':'A'}, {'sub':'01', 'task':'B'}] -> {'sub':['01','01'], 'task':['A','B']}
    infosource.iterables = [(field, values) for field, values in iterables_dict.items()]
    infosource.synchronize = False # 
    # === FILE SELECTION ===
    selectraw = Node(
        SelectFiles(file_templates, base_directory=raw_dir),
        name="selectfiles"
    )

    # Connect only fields that exist in SelectFiles templates.
    template_fields = set()
    template_values = file_templates.values() if isinstance(file_templates, dict) else [file_templates]
    for template in template_values:
        template_fields.update(re.findall(r"{([^{}]+)}", template))

    for field in iterable_fields:
        if field in template_fields:
            wf.connect(infosource, field, selectraw, field)
    
    # === PROCESSING NODE (regular Node, NOT MapNode) ===
    # iterables handles the iteration, so no iterfield needed
    initial_preproc = Node(InitialPreproc(), name='initial_preproc')
    initial_preproc = apply_interface_config(initial_preproc, pipeline_config["initial_preproc"])

    # ==== Artifact Rejection ====
    artifact_rejection = Node(ArtifactRejection(), name="artifact_rejection")
    artifact_rejection = apply_interface_config(artifact_rejection, pipeline_config["artifact_rejection"])

    # === Auto ICA ====
    auto_ica = Node(AutoICA(), name='auto_ica')
    auto_ica = apply_interface_config(auto_ica, pipeline_config["auto_ica"])

    # === Output (datasink) ===
    datasink = Node(
        DataSink(
            base_directory="output",
            parameterization=True
        ),
        name="datasink"
    )
    
    # === CONNECTIONS ===
    wf.connect([
        (selectraw, initial_preproc, [("meg", "in_file")]),
        (initial_preproc, artifact_rejection, [("out_file", "in_file")]),
        (artifact_rejection, datasink, [
            ("out_file", "meg.@final_raw"),
            ("ica_file", "meg.@final_ica"),
            ("ica_plot", "qc.@final_ica_plot"),
            ("psd_before", "qc.@psd_before"),
            ("psd_after", "qc.@psd_after")
        ])
    ])

    # === Epoching ====
    do_epoching = True
    if do_epoching:
        combine_event_epochs = bool(pipeline_config["epoching"].get("combine_event_epochs", False))

        # Tranform dict into list iterables
        event_mapping = pipeline_config["epoching"]["iterables"]["event_mapping"]
        event_ids = []
        event_labels = []
        event_tmins = []
        event_tmaxs = []

        for event_id, (label, tmin, tmax) in event_mapping.items():
            event_ids.append(int(event_id))
            event_labels.append(label)
            event_tmins.append(tmin)
            event_tmaxs.append(tmax)

        # setup epoching node

        epoching = Node(Epoching(), name="epoching")
        epoching.iterables = [
            ("event_id", event_ids),
            ("event_label", event_labels),
            ("event_tmin", event_tmins),
            ("event_tmax", event_tmaxs),
        ]
        epoching.synchronize = True
        apply_interface_config(epoching, pipeline_config["epoching"])

        # Collect all per-event outputs so the datasink keeps one file per event.
        join_fields = ["epo_files", "plots_raw_epochs", "plots_ar_reject_log", "plots_epochs_after_ar"]
        collect_epochs = JoinNode(
            IdentityInterface(fields=join_fields),
            joinsource="epoching",   # join within each subject/session branch
            joinfield=join_fields,
            name="collect_epochs",
        )

        wf.connect(
            [
                (artifact_rejection, epoching, [("out_file", "in_file")]),   
            ]
        )
        #(epoching, collect_epochs, [("out_file", "epo_files")])
        wf.connect(
            [
                (epoching, collect_epochs, [
                    ("out_file", "epo_files"),
                    ("plot_raw_epochs", "plots_raw_epochs"),
                    ("plot_ar_reject_log", "plots_ar_reject_log"),
                    ("plot_epochs_after_ar", "plots_epochs_after_ar")
                ])
            ]
        )
        # connect collected epochs to datasink
        wf.connect(
            [
                (collect_epochs, datasink, [
                    ("epo_files", "meg.@final_epo"),
                    ("plots_raw_epochs", "qc.@raw_epochs_plot"),
                    ("plots_ar_reject_log", "qc.@ar_reject_log_plot"),
                    ("plots_epochs_after_ar", "qc.@ar_epochs_plot")
                ])
            ]
        )

        if combine_event_epochs:
            merge_epochs = Node(
                Function(
                    input_names=["epo_files", "out_file"],
                    output_names="merged_file",
                    function=concatenate_epoch_files,
                ),
                name="merge_epochs",
            )
            merge_epochs.inputs.out_file = "combined-epoched_epo.fif"

            wf.connect(
                [
                    (collect_epochs, merge_epochs, [("epo_files", "epo_files")]),
                    (merge_epochs, datasink, [("merged_file", "meg.@final_epo_combined")]),
                ]
            )

    # Log workflow structure (after creation, not during)

    # === Build BIDS container ===
    build_bids_inputs = ["bids_dir_path", "input_wf_dir", "datasink_output"] + iterable_fields
    print(f"Building BIDS container with inputs: {build_bids_inputs}")
    build_bids = Node(
        Function(
            input_names=build_bids_inputs,
            output_names="bids_dir",
            function=build_bids_container
        ),
        name="build_bids_container"
    )
    print(f"datasink outputs: {datasink.outputs}")
    wf.connect(datasink, "out_file", build_bids, "datasink_output") # pseudo-connect datasink to bids to structure DAG flow

    build_bids.inputs.bids_dir_path = Path(output_dir).resolve()
    workflow_dir = Path(f"{wf.base_dir}/{wf_name}").absolute()
    print(f"workflow_dir: {workflow_dir}")
    build_bids.inputs.input_wf_dir = workflow_dir

    print(f"infosource iterables: {infosource.iterables}")
    for field in iterable_fields:
        if field in ["subject", "session"]:
            wf.connect(infosource, field, build_bids, field)
        else:
            print(f"Connecting extra tag {field}")
            wf.connect(infosource, field, build_bids, field)

    # connect datasink to build_bids
    
    return wf