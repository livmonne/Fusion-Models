"""Fusion Model package.

Re-exports the main model class and all sub-components so that callers
can write, e.g., ``from fusion_model import FusionModel``.
"""

from .decision import NUM_PATHWAYS, PATHWAY_NAMES, DecisionRouter
from .guess import GuessComponent
from .loss import FusionLoss
from .memory import RuleMemory
from .model import FusionModel
from .rule_engine import RuleGenerator

__all__ = [
    "NUM_PATHWAYS",
    "PATHWAY_NAMES",
    "DecisionRouter",
    "FusionModel",
    "FusionLoss",
    "GuessComponent",
    "RuleMemory",
    "RuleGenerator",
]
