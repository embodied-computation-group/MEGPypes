from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined, OutputMultiPath
)
import mne
from mne import find_events
import logging
import os

logger = logging.getLogger(__name__)

class ArtifactRejectionInputSpec(BaseInterfaceInputSpec):
    # TODO: Write all input traits here
    
class ArtifactRejectionOutputSpec(TraitedSpec):


class ArtifactRejection(BaseInterface):
