
# TODO

## March
6/3 March
- [X] I have to debug/test apply_interface_config function


- [ ]



# LOG

### 7/3 March

**RunFinder**
I am trying to create a RunFinder class procfuns.runs.
Its purpose it to handle Both BIDS and CTF-like directories or similar, discover all possible runs 
and then subsequently filtering these based on user inputs for the current pipeline.

This is neccesary to correctly handle CTF data, and experiments with several sessions/tasks in potentially nested structures.

**Pipeline (Current test state)**
In the pipeline I have currently tried to test artifact rejection but stumbles upon the fact that the opneneuro meg dataset is already fairly preprocessed with 1,40 bandpass filtering, excluding any line noise, therefore zapline is failing and generally the pipeline is unnecessary for this.

I will instead adapt the pipeline for Melinas dataset, however this require "Non-BIDS compliant" data handling.
