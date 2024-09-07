import numpy as np

import pandas as pd

np.float = np.float64
np.int = np.int_


def _range_vec(vec, th):
    """
    flag a vector depending on the threshold, th
    -1 if the value is below -th
    1 if the value is above th
    0 if it is between -th and th
    """
    flags = np.zeros(vec.size)
    _out = vec < -th
    _in = vec > th
    flags[_out] = -1
    flags[_in] = 1
    return flags


def _get_pulses_indexes(light_on, min_window_size=0, is_pad=True):
    """
    Get the start and end of a given pulse.
    """

    if is_pad:

        light_on = np.pad(light_on, (1, 1), "constant", constant_values=False)

    switches = np.diff(light_on.astype(np.int))
    (turn_on,) = np.where(switches == 1)
    (turn_off,) = np.where(switches == -1)

    if is_pad:
        turn_on -= 1
        turn_off -= 1
        turn_on = np.clip(turn_on, 0, light_on.size - 3)
        turn_off = np.clip(turn_off, 0, light_on.size - 3)

    assert turn_on.size == turn_off.size

    delP = turn_off - turn_on

    good = delP > min_window_size

    return turn_on[good], turn_off[good]


def _flag_regions(vec, central_th, extrema_th, smooth_window, min_frame_range):
    """
    Flag a frames into lower (-1), central (0) and higher (1) regions.
    If the quantity used to flag the frame is NaN, and the frame i smore than
    smooth_window away from the last non-NaN frame, return a NaN

    The strategy is
        1) Smooth the timeseries by smoothed window
        2) Find frames that are certainly lower or higher using extrema_th
        3) Find regions that are between (-central_th, central_th) and
            and last more than min_frame_range. This regions are certainly
            central regions.
        4) If a region was not identified as central, but contains
            frames labeled with a given extrema, label the whole region
            with the corresponding extrema.
    """
    # vv = pd.Series(vec).fillna(method='ffill').fillna(method='bfill')
    try:
        vv = pd.Series(vec).interpolate(method="nearest")
    except:
        # interpolate can fail if only one value is not nan.
        # just use ffill/bfill
        vv = pd.Series(vec).fillna(method="ffill").fillna(method="bfill")
    smoothed_vec = vv.rolling(window=smooth_window, center=True).mean()

    paused_f = (smoothed_vec > -central_th) & (smoothed_vec < central_th)
    turn_on, turn_off = _get_pulses_indexes(paused_f, min_frame_range)
    inter_pulses = zip([0] + list(turn_off), list(turn_on) + [paused_f.size - 1])

    flag_modes = _range_vec(smoothed_vec, extrema_th)

    for ini, fin in inter_pulses:
        dd = np.unique(flag_modes[ini : fin + 1])
        dd = [x for x in dd if x != 0]
        if len(dd) == 1:
            flag_modes[ini : fin + 1] = dd[0]
        elif len(dd) > 1:
            kk = flag_modes[ini : fin + 1]
            kk[kk == 0] = np.nan
            kk = pd.Series(kk).fillna(method="ffill").fillna(method="bfill")
            flag_modes[ini : fin + 1] = kk

    # the region is ill-defined if the frame was a NaN
    is_nan = (
        pd.Series(vec)
        .fillna(method="ffill", limit=smooth_window)
        .fillna(method="bfill", limit=smooth_window)
        .isna()
    )
    flag_modes[is_nan] = np.nan

    return flag_modes


# initialize data
def get_events(df, fps, mm_px, fly_length=None):
    smooth_window_s = 0.25
    min_paused_win_speed_s = 1 / 3

    if fly_length is None:
        fly_length = 280 * mm_px
    pause_th_lower = 0.05 * fly_length
    pause_th_higher = 0.1 * fly_length
    speed = df["thorax_vel"].values
    w_size = int(round(fps * smooth_window_s))
    smooth_window = w_size if w_size % 2 == 1 else w_size + 1
    min_paused_win_speed = fps * min_paused_win_speed_s
    motion_mode = _flag_regions(
        speed, pause_th_lower, pause_th_higher, smooth_window, min_paused_win_speed
    )
    return motion_mode
