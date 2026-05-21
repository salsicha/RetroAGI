"""
src/models.py
Exports the 5 lobes of the RetroAGI brain based on the Universal Predictive Coding architecture.
This file ensures backwards compatibility with scripts importing from models.py.
"""

from brain.occipital import OccipitalLobe
from brain.temporal import TemporalLobe
from brain.hippocampus import HippocampusLobe as Hippocampus
from brain.prefrontal import PrefrontalLobe
from brain.motor import MotorLobe

__all__ = [
    'OccipitalLobe',
    'TemporalLobe',
    'Hippocampus',
    'PrefrontalLobe',
    'MotorLobe'
]
