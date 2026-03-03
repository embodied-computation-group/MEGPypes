"""

"""
import os
import mne
from autoreject import AutoReject
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
    out_file = traits.Str("epoched_raw-epo.fif", usedefault=True, desc="Output filename")

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
        selected_events = events[mask]

        if len(selected_events) == 0:
            raise RuntimeError(
                f"No events found for trigger {self.inputs.event_code}"
            )
        
        epochs = mne.Epochs(
            raw,
            events=selected_events,
            tmin=self.inputs.event_tmin,
            tmax=self.inputs.event_tmax,
            baseline=None
        )


        # 2. Autoreject-based epoch rejection
        if self.inputs.enable_autoreject:
            logger.info("Running autoreject on epochs")
            # TODO: picks should be created based on channel types present in the data 
            # AutoReject can only handle on type at a time
            picks = mne.pick_types(epochs.info, meg=True, eeg=False, eog=False, stim=False)
            ar = AutoReject(
                n_interpolates=[1, 4, 32], # TODO: What should these values be?
                consensus=[0.5, 1], # TODO: What should these values be?
                picks=picks,
                thresh_method="random_search",
                random_state=893
            )
            # TODO: Save the ar object for later QC of rejected epochs and channels
            # we can use the included ar.get_reject_log() function 
            ar_epochs = ar.transform(epochs)

        # Save epoched data
        logger.info(f"OUT FILE PATH: {self.inputs.out_file}")
        out_path = os.path.abspath(self.inputs.out_file)
        ar_epochs.save(out_path, overwrite=True)
        logger.info(f"Saved: {out_path}")

        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs