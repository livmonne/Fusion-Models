"""Fusion Model package (JAX/Flax implementation).

Re-exports the main model class and all sub-components so that callers
can write, e.g., ``from fusion_model import FusionModel``.
"""

from .decision import DecisionRouter
from .guess import GuessComponent
from .loss import fusion_loss
from .memory import RuleMemory
from .model import FusionModel
from .rule_engine import RuleGenerator

__all__ = [
    "DecisionRouter",
    "FusionModel",
    "fusion_loss",
    "GuessComponent",
    "RuleMemory",
    "RuleGenerator",
]
