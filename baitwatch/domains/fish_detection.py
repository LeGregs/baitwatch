from enum import Enum


class FishDetectionEnum(str, Enum):
    """Supported type of fish detection.

    FONF: Fish Or No Fish
    IFSP = Individual Fish Species Prediction
    """
    FONF = "fonf"
    IFSP = "ifsp"
    # TODO: add support for WAW
    # WAW = "waw"
