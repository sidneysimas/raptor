"""CISA Known-Exploited Vulnerabilities (KEV) lookup.

The full KEV catalog is one ~150 KB JSON document fetched from CISA. We
download it once per ``raptor-sca`` run (24h cache), build an in-memory set
of exploited CVE IDs, and answer ``contains(cve_id)`` in O(1).

Failure modes:
- Network down + cold cache → ``KevClient.is_loaded()`` is False;
  ``contains`` always returns False (degraded but harmless: KEV is a
  bonus signal layered on top of OSV's match).
- Stale cache + offline → load the cache anyway; the KEV list rarely
  changes day-to-day.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

from core.json import JsonCache
from core.http import HttpClient, HttpError

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_DEFAULT_TTL = 24 * 3600
_CACHE_KEY = "kev"


class KevClient:
    """In-memory KEV lookup; lazy-loads on first call."""

    def __init__(
        self,
        http: HttpClient,
        cache: JsonCache,
        *,
        offline: bool = False,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._http = http
        self._cache = cache
        self._offline = offline
        self._ttl = ttl_seconds
        self._loaded = False
        self._cve_set: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, cve_id: str) -> bool:
        """True if ``cve_id`` is in CISA's KEV list (case-insensitive)."""
        if not cve_id:
            return False
        if not self._loaded:
            self._load()
        return cve_id.upper() in self._cve_set

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        record = self._cache.get(_CACHE_KEY, ttl_seconds=self._ttl)
        if record is None and not self._offline:
            try:
                record = self._http.get_json(KEV_URL)
            except HttpError as e:
                logger.warning("sca.kev: fetch failed (%s); KEV unavailable", e)
                self._loaded = True
                return
            if isinstance(record, dict):
                self._cache.put(_CACHE_KEY, record, ttl_seconds=self._ttl)
        if record is None:
            # Offline + cold cache.
            self._loaded = True
            return

        self._cve_set = _extract_cves(record)
        self._loaded = True


def _extract_cves(record: object) -> Set[str]:
    """Pull the CVE-id set from a KEV catalog payload.

    Schema (relevant slice): ``{"vulnerabilities": [{"cveID": "CVE-..."},
    ...]}``. Anything else is silently ignored; a corrupt feed yields an
    empty set rather than a crash.
    """
    if not isinstance(record, dict):
        return set()
    vulns = record.get("vulnerabilities")
    if not isinstance(vulns, list):
        return set()
    out: Set[str] = set()
    for entry in vulns:
        if not isinstance(entry, dict):
            continue
        cve = entry.get("cveID") or entry.get("cve_id")
        if isinstance(cve, str) and cve:
            out.add(cve.upper())
    return out


__all__ = ["KevClient"]
