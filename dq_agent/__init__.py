"""Data Quality Agent - automated profiling, anomaly detection, and lineage tracking."""

from .agent import DQAgent
from .profiler import DataProfiler

__all__ = ["DQAgent", "DataProfiler"]
__version__ = "0.1.0"
