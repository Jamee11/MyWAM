"""NumPy-only current-observation geometry shared by training and RoboTwin client."""
import numpy as np

ROBOTWIN_EE_TO_TCP_METERS = .12


def tcp_context_from_eef20(eef, offset=ROBOTWIN_EE_TO_TCP_METERS):
    """Official rot6d stores columns: TCP = endpose + R[:,0] * 0.12m.

    RoboTwin robot._trans_endpose subtracts 0.12m for action endpose only.
    This conversion uses observed endpose alone, never the oracle TCP field.
    """
    value = np.asarray(eef, dtype=np.float32)
    if value.shape[-1] != 20 or not np.isfinite(value).all():
        raise ValueError("Expected finite 20D official EEF")
    result = value.reshape(*value.shape[:-1], 2, 10).copy()
    result[..., :3] += float(offset) * result[..., 3:6]
    return result
