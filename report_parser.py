"""
report_parser.py
================
Ingests JSON-formatted malware sandbox detonation reports (e.g., from Hybrid
Analysis / CrowdStrike Falcon Sandbox) and produces structured summaries
suitable for passing to the teacher LLM.

Supported inputs
----------------
* JSON report files (Hybrid Analysis format)
* Raw JSON strings
* Python dicts already loaded from JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReportParser
# ---------------------------------------------------------------------------

class ReportParser:
    """Parse a malware sandbox detonation report into a structured summary.

    Parameters
    ----------
    truncate:
        When ``True``, apply the same filtering as :class:`data_filter.DataFilter`
        before summarising (retains only essential fields to save context space).
    """

    # Fields surfaced at the top level of the summary
    TOP_LEVEL_FIELDS = [
        "verdict",
        "threat_score",
        "environment_description",
        "submit_name",
        "sha256",
        "type",
        "type_short",
        "analysis_start_time",
    ]

    # Behaviour / IOC fields extracted from nested structures
    BEHAVIOUR_FIELDS = [
        "total_processes",
        "mitre_attcks",
        "signatures",
        "domains",
        "hosts",
        "compromised_hosts",
        "extracted_files",
        "tags",
    ]

    def __init__(self, truncate: bool = False) -> None:
        self.truncate = truncate

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def parse_file(self, path: str | Path) -> dict[str, Any]:
        """Parse a JSON report file from *path*."""
        with open(path) as fh:
            raw = json.load(fh)
        return self._parse_dict(raw)

    def parse_string(self, json_str: str) -> dict[str, Any]:
        """Parse a JSON-encoded report string."""
        raw = json.loads(json_str)
        return self._parse_dict(raw)

    def parse_dict(self, report: dict[str, Any]) -> dict[str, Any]:
        """Parse an already-loaded report dict."""
        return self._parse_dict(report)

    def to_summary_string(self, parsed: dict[str, Any]) -> str:
        """Convert a parsed report to a human-readable summary string."""
        lines = []

        # Metadata
        for key in self.TOP_LEVEL_FIELDS:
            value = parsed.get(key)
            if value is not None:
                lines.append(f"{key}: {value}")

        # Process info
        if "total_processes" in parsed:
            lines.append(f"total_processes: {parsed['total_processes']}")
        if "processes" in parsed:
            lines.append("processes:")
            for proc in parsed["processes"][:20]:  # cap for context
                lines.append(f"  - {proc}")

        # MITRE ATT&CK
        if "mitre_attcks" in parsed and parsed["mitre_attcks"]:
            lines.append("mitre_attcks:")
            for technique in parsed["mitre_attcks"]:
                lines.append(f"  - {technique}")

        # Signatures
        if "signatures" in parsed and parsed["signatures"]:
            lines.append("signatures:")
            for sig in parsed["signatures"][:30]:
                lines.append(f"  - {sig}")

        # Network
        for net_key in ("domains", "hosts", "compromised_hosts"):
            if parsed.get(net_key):
                lines.append(f"{net_key}: {', '.join(str(x) for x in parsed[net_key][:20])}")

        # Tags
        if parsed.get("tags"):
            lines.append(f"tags: {', '.join(str(t) for t in parsed['tags'])}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_dict(self, report: dict[str, Any]) -> dict[str, Any]:
        """Extract structured fields from a raw report dict."""
        if self.truncate:
            from data_filter import DataFilter
            report = DataFilter().filter(report)

        parsed: dict[str, Any] = {}

        # Top-level metadata
        for key in self.TOP_LEVEL_FIELDS:
            if key in report:
                parsed[key] = report[key]

        # Behaviour fields (may be nested under various keys)
        for key in self.BEHAVIOUR_FIELDS:
            value = self._deep_get(report, key)
            if value is not None:
                parsed[key] = value

        # Process tree
        for proc_key in ("process_list", "processes", "process_tree"):
            if proc_key in report:
                raw_procs = report[proc_key]
                parsed["processes"] = self._extract_processes(raw_procs)
                break

        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_get(d: dict[str, Any], key: str) -> Any:
        """Return the first value found for *key* by BFS through nested dicts/lists."""
        queue = [d]
        while queue:
            current = queue.pop(0)
            if isinstance(current, dict):
                if key in current:
                    return current[key]
                queue.extend(current.values())
            elif isinstance(current, list):
                queue.extend(current)
        return None

    @staticmethod
    def _extract_processes(raw: Any) -> list[str]:
        """Flatten a process tree/list into a list of descriptive strings."""
        result: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("process_name") or item.get("image", "")
                    cmd = item.get("command_line") or item.get("cmdline") or ""
                    pid = item.get("pid") if item.get("pid") is not None else item.get("process_id")
                    desc = str(name)
                    if pid is not None:
                        desc += f" (pid={pid})"
                    if cmd:
                        desc += f" cmd={cmd!r}"
                    result.append(desc)
                elif isinstance(item, str):
                    result.append(item)
        elif isinstance(raw, str):
            result.append(raw)
        return result
