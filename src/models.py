"""
src/models.py
Exports the 5 lobes of the RetroAGI brain based on the Universal Predictive Coding architecture.
This file ensures backwards compatibility with scripts importing from models.py.
"""

from src.brain.occipital import OccipitalLobe
from src.brain.temporal import TemporalLobe
from src.brain.hippocampus import HippocampusLobe as Hippocampus
from src.brain.prefrontal import PrefrontalLobe
from src.brain.motor import MotorLobe

__all__ = [
    'OccipitalLobe',
    'TemporalLobe',
    'Hippocampus',
    'PrefrontalLobe',
    'MotorLobe'
]
