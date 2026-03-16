"""

"""
import os
import mne
from collections import Counter
from autoreject import AutoReject
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits
from megpypes.proc_funcs.epoch import handle_find_events
import logging
logger = logging.getLogger(__name__)


class EpochingInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True, desc="Input MEG BaseRaw file")
    
    # steps on/off
    epoch = traits.Bool(True, usedefault=True, desc="Flag to enable epoching of data")
    autoreject = traits.Bool(True, usedefault=True, desc="Flag to enable autoreject-based epoch rejection")

    # 1. Epoching based on events
    event_id = traits.Int(mandatory=True)
    event_label = traits.Str(mandatory=True)
    event_tmin = traits.Float(mandatory=True)
    event_tmax = traits.Float(mandatory=True)
    
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
        logger.info(f"Epoching for event: {self.inputs.event_label}")
        logger.info(f"id: {self.inputs.event_id}, tmin: {self.inputs.event_tmin}, tmax: {self.inputs.event_tmax}")

        # Load data
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        # 1. Epoching based on events
        events = handle_find_events(raw)
        logger.debug(f"events: {events}")

        mask = events[:, 2] == self.inputs.event_id
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
            baseline=None,
            preload=True
        )

        # TODO: Save Raw epochs for QC

        # 2. Autoreject-based epoch rejection
        if self.inputs.autoreject:
            logger.info("Running autoreject on epochs")
            # TODO: picks should be created based on channel types present in the data 
            # AutoReject can only handle on type at a time
            picks = mne.pick_types(
                epochs.info,
                meg="mag",
                eeg=False,
                eog=False,
                stim=False,
                ref_meg=False
            )

            picked_types = [mne.channel_type(epochs.info, i) for i in picks]
            logger.info(f"AutoReject picks: {Counter(picked_types)}")

            epochs_meg = epochs.copy().pick(picks) # Forces the picks abobe
            # TODO: Crosscheck whether these picks are sufficient or should be merged with grads e.g.
            # CTF system might operate differntly than elektromag

            logger.debug(f"epochs montage: {epochs.get_montage()}")
            logger.debug(len(epochs.info["dig"]) if epochs.info.get("dig") else "No dig points")

            ar = AutoReject(
                n_interpolate=[1, 4, 32], # TODO: What should these values be?
                consensus=[0.5, 1], # TODO: What should these values be?
                thresh_method="random_search",
                random_state=893
            )
            # TODO: Save the ar object for later QC of rejected epochs and channels
            # we can use the included ar.get_reject_log() function 
            ar.fit(epochs_meg)
            ar_epochs = ar.transform(epochs)

            final_epochs = ar_epochs
        else:
            final_epochs = epochs

        # Save epoched data
        logger.info(f"OUT FILE PATH: {self.inputs.out_file}")
        out_path = os.path.abspath(f"{self.inputs.event_label}-epo.fif")
        final_epochs.save(out_path, overwrite=True)
        logger.info(f"Saved: {out_path}")

        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs