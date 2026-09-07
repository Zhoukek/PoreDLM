__all__ = [
    "PoreLMError",
    "PoreLMConfigurationError",
    "PoreLMCliError",
    "PoreLMEnvironmentError",
    "PoreLMNetworkError",
    "PoreLMCheckpointError",
]


class PoreLMError(Exception):
    """
    Base class for all custom PoreLM exceptions.
    """


class PoreLMConfigurationError(PoreLMError):
    """
    An error with a configuration file.
    """


class PoreLMCliError(PoreLMError):
    """
    An error from incorrect CLI usage.
    """


class PoreLMEnvironmentError(PoreLMError):
    """
    An error from incorrect environment variables.
    """


class PoreLMNetworkError(PoreLMError):
    """
    An error with a network request.
    """


class PoreLMCheckpointError(PoreLMError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class PoreLMThreadError(Exception):
    """
    Raised when a thread fails.
    """
