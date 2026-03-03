"""

"""
import os
import mne
import autoreject
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits
import logging

logger = logging.getLogger(__name__)


class EpochingInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True, desc="Input MEG BaseRaw file")
    
    # steps on/off
    enable_autoreject = traits.Bool(True, usedefault=True, desc="Flag to enable autoreject-based epoch rejection")

    # 1. Epoching based on events
    event_id = traits.Dict(traits.Str(), traits.Int(), mandatory=True, desc="Trigger code (e.g. 19)")
    event_label = traits.Dict(traits.Str(), mandatory=True, desc="Event label (e.g. 'cue_onset')")
    event_tmin = traits.Float(mandatory=True, desc="Epoch start time relative to event (s)")
    event_tmax = traits.Float( mandatory=True, desc="Epoch end time relative to event (s)")
    
    
    # output
    out_file = traits.Str("epoched_raw.fif", usedefault=True, desc="Output filename")

class EpochingOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc="Epoched MEG file")

class Epoching(BaseInterface):
    input_spec = EpochingInputSpec
    output_spec = EpochingOutputSpec

    def _run_interface(self, runtime):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(f"NODE: Epoching | In-File: {self.inputs.in_file}")

        # Load data
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        # 1. Epoching based on events
        events = mne.find_events(raw)

        mask = events[:, 2] == self.inputs.event_id[self.inputs.event_id]
        

        # 2. Autoreject-based epoch rejection
        if self.inputs.enable_autoreject:
            logger.info("Applying autoreject-based epoch rejection")

