
# TODO

## March
6/3 March
- [X] I have to debug/test apply_interface_config function


- [ ]



# LOG

## March
### 7/3 

**RunFinder**
I am trying to create a RunFinder class procfuns.runs.
Its purpose it to handle Both BIDS and CTF-like directories or similar, discover all possible runs 
and then subsequently filtering these based on user inputs for the current pipeline.

This is neccesary to correctly handle CTF data, and experiments with several sessions/tasks in potentially nested structures.

**Pipeline (Current test state)**
In the pipeline I have currently tried to test artifact rejection but stumbles upon the fact that the opneneuro meg dataset is already fairly preprocessed with 1,40 bandpass filtering, excluding any line noise, therefore zapline is failing and generally the pipeline is unnecessary for this.

I will instead adapt the pipeline for Melinas dataset, however this require "Non-BIDS compliant" data handling.

### 11/3
**ICA Exclusion split**
I have to split up workflow to allow for intermediate user specification of ICA exlcusion based on source plots of compoenents.

### 13/3
**BIDS Writer**
I am trying to create a bids writer Node.
I have been messing around with path handling a bit weirdly, 
I think it makes much more sense to have pathlib find the absolute path outside of the node itself and then pass that to the node instead of passing relative path name which is then instanced at the node working dir and then looking for parents...