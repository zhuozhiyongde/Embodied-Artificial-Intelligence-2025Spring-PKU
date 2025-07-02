from os.path import join, dirname


def get_exp_dir(exp_name: str) -> str:
    """
    Get the experiment directory path.

    Parameters
    ----------
    exp_name : str
        The name of the experiment.

    Returns
    -------
    str
        The path to the experiment directory.
    """
    return join("exps", exp_name)


def get_log_path(exp_name: str) -> str:
    """
    Get the log path for the experiment.

    Parameters
    ----------
    exp_name : str
        The name of the experiment.

    Returns
    -------
    str
        The path to the log directory.
    """
    return join(get_exp_dir(exp_name), "log")


def get_exp_config_path(exp_name: str) -> str:
    """
    Get the path to the experiment configuration file.

    Parameters
    ----------
    exp_name : str
        The name of the experiment.

    Returns
    -------
    str
        The path to the experiment configuration file.
    """
    return join(get_exp_dir(exp_name), "config.yaml")


def get_checkpoint_dir(exp_name: str) -> str:
    """
    Get the checkpoint directory for the experiment.

    Parameters
    ----------
    exp_name : str
        The name of the experiment.

    Returns
    -------
    str
        The path to the checkpoint directory.
    """
    return join(get_exp_dir(exp_name), "checkpoint")


def get_checkpoint_path(exp_name: str, step: int) -> str:
    """
    Get the path to a specific checkpoint file.

    Parameters
    ----------
    exp_name : str
        The name of the experiment.
    step : int
        The step number of the checkpoint.

    Returns
    -------
    str
        The path to the checkpoint file.
    """
    return join(get_checkpoint_dir(exp_name), f"checkpoint_{step}.pth")


def get_exp_config_from_checkpoint(checkpoint_path: str) -> str:
    """
    Get the experiment configuration file path from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint file.

    Returns
    -------
    str
        The path to the experiment configuration file.
    """
    return join(dirname(dirname(checkpoint_path)), "config.yaml")
