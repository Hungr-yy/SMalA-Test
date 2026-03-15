"""
data_filter.py
==============
Implements pre-filtering and truncation logic for large malware sandbox
detonation reports.

Hybrid Analysis reports can exceed 128 k tokens when ingested verbatim.
Research shows that retaining only a small set of high-signal fields
("total_processes", "mitre_attcks", "signatures") has a **negligible impact**
on model performance while dramatically reducing context window consumption.

This module provides the :class:`DataFilter` class used by both
:mod:`report_parser` and the ``--truncate-input`` flag in the eval pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Fields kept when running in ``essential_only`` mode.
#: Aligns with the CyberSOCEval truncation specification.
ESSENTIAL_FIELDS: tuple[str, ...] = (
    "sha256",
    "verdict",
    "threat_score",
    "submit_name",
    "type",
    "type_short",
    "total_processes",
    "mitre_attcks",
    "signatures",
    "tags",
    "environment_description",
)

#: Fields that are kept in the ``standard`` (non-essential-only) mode but
#: still truncated to avoid blowing the context window.
STANDARD_FIELDS: tuple[str, ...] = ESSENTIAL_FIELDS + (
    "domains",
    "hosts",
    "compromised_hosts",
    "extracted_files",
    "process_list",
    "network_traffic",
    "registry_operations",
    "file_operations",
)

#: Maximum number of list items to retain for any list-valued field.
DEFAULT_LIST_LIMIT: int = 50


# ---------------------------------------------------------------------------
# DataFilter
# ---------------------------------------------------------------------------

class DataFilter:
    """Filter and truncate a malware sandbox report dict.

    Parameters
    ----------
    mode:
        ``"essential"`` – keep only :data:`ESSENTIAL_FIELDS` (matches the
        ``--truncate-input`` behaviour used in CyberSOCEval benchmarking).
        ``"standard"`` – keep a broader set of fields but still truncate lists.
        ``"none"`` – pass through unchanged (useful for full-context ablations).
    list_limit:
        Maximum number of items kept per list-valued field.
    """

    def __init__(
        self,
        mode: str = "essential",
        list_limit: int = DEFAULT_LIST_LIMIT,
    ) -> None:
        mode = mode.lower()
        if mode not in ("essential", "standard", "none"):
            raise ValueError(f"mode must be 'essential', 'standard', or 'none', got {mode!r}")
        self.mode = mode
        self.list_limit = list_limit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, report: dict[str, Any]) -> dict[str, Any]:
        """Return a filtered copy of *report*.

        Parameters
        ----------
        report:
            Raw JSON-decoded report dict from a sandbox detonation.

        Returns
        -------
        dict
            A new dict containing only the allowed fields, with list fields
            truncated to at most :attr:`list_limit` items.
        """
        if self.mode == "none":
            return dict(report)

        allowed = ESSENTIAL_FIELDS if self.mode == "essential" else STANDARD_FIELDS
        filtered: dict[str, Any] = {}

        for key in allowed:
            value = self._extract(report, key)
            if value is None:
                continue
            if isinstance(value, list):
                value = value[: self.list_limit]
            filtered[key] = value

        original_size = _approx_tokens(report)
        filtered_size = _approx_tokens(filtered)
        if original_size == 0:
            logger.debug("DataFilter[%s]: empty report, nothing to filter", self.mode)
            return filtered
        logger.debug(
            "DataFilter[%s]: ~%d → ~%d tokens (%.1f%% reduction)",
            self.mode,
            original_size,
            filtered_size,
            100.0 * (1 - filtered_size / original_size),
        )
        return filtered

    def filter_string(self, json_str: str) -> str:
        """Filter a JSON-encoded report string and return a JSON string."""
        import json

        report = json.loads(json_str)
        filtered = self.filter(report)
        return json.dumps(filtered)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(report: dict[str, Any], key: str) -> Any:
        """Return *key* from *report*, searching one level of nesting."""
        if key in report:
            return report[key]
        # Check common wrapper keys used by different sandbox formats
        for wrapper in ("analysis", "result", "report", "data", "summary"):
            sub = report.get(wrapper)
            if isinstance(sub, dict) and key in sub:
                return sub[key]
        return None


def _approx_tokens(obj: Any) -> int:
    """Rough token estimate: len(str(obj)) // 4."""
    try:
        import json as _json
        return len(_json.dumps(obj)) // 4
    except Exception:
        return len(str(obj)) // 4
