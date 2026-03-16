from pathlib import Path
import string

from msgspec import field
from parse import compile as parse_compile
from nipype import Workflow, Node, IdentityInterface, SelectFiles, DataSink, config
from nipype.interfaces.utility import Function
from megpypes.proc_funcs.runs import RunFinder
from megpypes.interfaces.initpreproc import InitialPreproc
from megpypes.interfaces.artifact_rejection import ArtifactRejection
from megpypes.interfaces.auto_ica import AutoICA
from megpypes.interfaces.epoching import Epoching
from megpypes.pipelines.utils import apply_interface_config
from megpypes.pipelines.write_bids import build_bids_container
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

    # connect the dynamic fields (subject->subject)
    for field in iterable_fields:
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
            ("ica_plot", "qc.@final_ica_plot")
        ])
    ])

    # === Epoching ====
    do_epoching = True
    if do_epoching:
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

        wf.connect(
            [
                (artifact_rejection, epoching, [("out_file", "in_file")]),
                (epoching, datasink, [
                    ("out_file", "megpreproc.@final_epo")
                ])
            ]
        )

    # Log workflow structure (after creation, not during)

    
    # === Build BIDS container ===
    build_bids_inputs = ["bids_dir_name", "input_wf_dir"] + iterable_fields
    print(f"Building BIDS container with inputs: {build_bids_inputs}")
    build_bids = Node(
        Function(
            input_names=build_bids_inputs,
            output_names="bids_dir",
            function=build_bids_container
        ),
        name="build_bids_container"
    )
    wf.connect(datasink, "base_directory", build_bids, "input_wf_dir") # pseudo-connect datasink to bids to structure DAG flow

    build_bids.inputs.bids_dir_name = output_dir
    workflow_dir = Path(f"{wf.base_dir}/{wf_name}").absolute()
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