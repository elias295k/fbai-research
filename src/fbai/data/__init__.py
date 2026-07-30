"""Canonical historical-match data layer for Football Intelligence Research."""

from fbai.data.audit import AuditResult, audit_canonical
from fbai.data.canonical import CanonicalWriteResult, write_canonical_partitions
from fbai.data.loader import canonicalize_source_frame, load_source_csv
from fbai.data.schema import CANONICAL_COLUMNS, NATURAL_KEY, validate_canonical_frame

__all__ = [
    "CANONICAL_COLUMNS",
    "NATURAL_KEY",
    "AuditResult",
    "CanonicalWriteResult",
    "audit_canonical",
    "canonicalize_source_frame",
    "load_source_csv",
    "validate_canonical_frame",
    "write_canonical_partitions",
]
