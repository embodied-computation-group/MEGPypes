import numpy as np
from meegkit.dss import dss_line_iter
import mne
from mne.io import BaseRaw, RawArray
import logging
logger = logging.getLogger(__name__)

def apply_zapline_denoising(
    raw,
    fline=50.0,
    n_chunks=10,
    spot_sz=7,
    win_sz=12,
    nfft=2048,
    n_iter_max=30,
    mag_only=True
):
    logger.info(f"Running apply_zapline_denoising")
    logger.debug(f"fline={fline}, n_chunks={n_chunks}")

    raw_copy = raw.copy().load_data()

    raw_data = raw_copy.get_data()
    info = raw_copy.info

    # Select channels
    if mag_only:
        mag_ix = np.array([
            i for i, ch_type in enumerate(raw_copy.get_channel_types())
            if ch_type == "mag"
        ])
        if len(mag_ix) == 0:
            raise ValueError("No magnetometer channels found.")
        data_to_denoise = raw_data[mag_ix]
    else:
        mag_ix = np.arange(raw_data.shape[0])
        data_to_denoise = raw_data

    sfreq = info["sfreq"]

    chunks = np.array_split(data_to_denoise, n_chunks, axis=1)
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

        chunk_T = np.moveaxis(chunk, 0, -1)

        cleaned_T, _ = dss_line_iter(
            chunk_T,
            fline=fline,
            sfreq=sfreq,
            spot_sz=spot_sz,
            win_sz=win_sz,
            nfft=nfft,
            n_iter_max=n_iter_max
        )

        cleaned = np.moveaxis(cleaned_T, -1, 0)
        cleaned_chunks.append(cleaned)

    cleaned_data = np.hstack(cleaned_chunks)

    new_data = raw_copy.get_data()
    new_data[mag_ix, :] = cleaned_data

    new_raw = RawArray(
        new_data,
        info,
        first_samp=raw_copy.first_samp
    )

    out_file = "zapline_raw.fif"

    logger.info(f"ZAPLINED raw file.")

    return new_raw