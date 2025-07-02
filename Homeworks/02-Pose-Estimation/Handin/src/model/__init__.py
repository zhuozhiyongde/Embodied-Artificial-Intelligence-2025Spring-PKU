import torch

from .est_coord import EstCoordNet
from .est_pose import EstPoseNet

from ..config import Config


def get_model(config: Config) -> torch.nn.Module:
    """
    Get model according to the model type in the config.

    Parameters
    ----------
    config: Config
        Configuration object containing the model type.

    Returns
    -------
    nn.Module
        The model instance.
    """
    model_type = config.model_type

    if model_type == "est_pose":
        return EstPoseNet(config)
    elif model_type == "est_coord":
        return EstCoordNet(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "get_model",
    "EstCoordNet",
    "EstPoseNet",
]
