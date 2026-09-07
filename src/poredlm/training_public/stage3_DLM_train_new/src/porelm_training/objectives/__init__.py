from .base import ObjectiveOutput, TrainingObjective
from .flow_matching import FlowMatchingObjective

OBJECTIVES = {"flow_matching": FlowMatchingObjective}

__all__ = ["OBJECTIVES", "FlowMatchingObjective", "ObjectiveOutput", "TrainingObjective"]
