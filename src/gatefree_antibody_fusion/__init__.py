"""Gate-free structural--evolutionary prior fusion."""

from .fusion import AMINO_ACIDS, fuse_log_probabilities, normalize_log_probabilities

__all__ = ["AMINO_ACIDS", "fuse_log_probabilities", "normalize_log_probabilities"]
__version__ = "0.1.0"

