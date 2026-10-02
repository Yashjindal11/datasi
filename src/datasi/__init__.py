"""DataSI: find what is wrong with your data before your model does."""

from datasi._version import __version__
from datasi.configuration import ColumnRule, Config
from datasi.core.compare import compare, inspect_train_test
from datasi.core.dataset import Dataset
from datasi.core.pipeline import inspect, inspect_schema, investigate
from datasi.findings import Category, Finding, Severity
from datasi.loaders import load
from datasi.reports import Report

__all__ = [
    "Category",
    "ColumnRule",
    "Config",
    "Dataset",
    "Finding",
    "Report",
    "Severity",
    "__version__",
    "compare",
    "inspect",
    "inspect_schema",
    "inspect_train_test",
    "investigate",
    "load",
]
