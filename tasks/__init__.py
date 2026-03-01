# Re-export the CLEVR dataset so callers can write ``from tasks import CLEVRDataset``.
from .clevr import CLEVRDataset

__all__ = ["CLEVRDataset"]
