"""Built-in detectors. Importing this package registers them in pipeline order."""

from datasi.detectors.base import (
    BaseDetector,
    Context,
    Detector,
    register_detector,
    registry,
)
from datasi.detectors.categorical import CategoricalDetector, StringDetector
from datasi.detectors.duplicates import DuplicateDetector
from datasi.detectors.missingness import MissingnessDetector
from datasi.detectors.numeric import NumericDetector, OutlierDetector
from datasi.detectors.relationships import CorrelationDetector, RedundancyDetector
from datasi.detectors.rules import RulesDetector
from datasi.detectors.schema import SchemaDetector
from datasi.detectors.structure import ConstantDetector, IdentifierDetector
from datasi.detectors.target import TargetDetector
from datasi.detectors.temporal import DatetimeDetector, TemporalDetector
from datasi.detectors.types import TypeDetector

__all__ = [
    "BaseDetector",
    "CategoricalDetector",
    "ConstantDetector",
    "Context",
    "CorrelationDetector",
    "DatetimeDetector",
    "Detector",
    "DuplicateDetector",
    "IdentifierDetector",
    "MissingnessDetector",
    "NumericDetector",
    "OutlierDetector",
    "RedundancyDetector",
    "RulesDetector",
    "SchemaDetector",
    "StringDetector",
    "TargetDetector",
    "TemporalDetector",
    "TypeDetector",
    "register_detector",
    "registry",
]
