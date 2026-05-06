"""Mechanical supply-chain heuristics.

Each check emits a ``SupplyChainFinding`` consumed by the findings layer:

- ``install_hooks`` — npm ``package.json`` lifecycle scripts that fire
  at install time, with regex patterns for known-malicious shapes.
- ``typosquat`` — Damerau-Levenshtein distance against the bundled
  popular-name list per ecosystem.
- ``artefacts`` — four project-tree heuristics: ``.pth`` files,
  binary fixtures in test trees, ``disguised_filename`` (extension
  lies about content), ``large_obfuscated_artefact`` (minified /
  obfuscated source-tree files outside build dirs).
- ``python_imports`` — top-level executable code in ``.py`` files
  outside test trees (``subprocess`` / ``os.system`` / ``eval`` /
  ``__import__`` / network calls at import time).
- ``exfil_destinations`` — URLs in source matching curated lists of
  paste sites, anonymous file-share, URL shorteners, Tor, Discord
  webhooks, Telegram bots, raw-IP URLs.
- ``gha_drift`` — GitHub Actions workflows using mutable refs
  (``uses: foo/action@v1`` rather than 40-char SHA pins).
- ``git_drift`` — manifest-pinned git deps with branch/tag refs
  rather than SHAs.

Deferred to follow-ups:

- Recent-publish / maintainer-change checks (need registry metadata
  over the network — separate clients, separate cache).
- Walking ``node_modules`` for per-dep install hooks (most CI runs
  don't have ``node_modules`` materialised at scan time).
- LLM-assisted version-diff / postinstall / maintainer-trust reviews
  (Tier B; the curated lists in ``data/`` will be reused as exemplars).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from ..models import (
    Dependency,
    Manifest,
    SupplyChainFinding,
)
from . import artefacts as _artefacts
from . import exfil_destinations as _exfil
from . import gha_drift as _gha_drift
from . import git_drift as _git_drift
from . import install_hooks as _install_hooks
from . import python_imports as _python_imports
from . import registry_metadata as _registry_metadata
from . import sentinel as _sentinel
from . import typosquat as _typosquat
from . import typosquat_domain as _typosquat_domain

logger = logging.getLogger(__name__)


def evaluate(
    target: Path,
    manifests: Iterable[Manifest],
    deps: Iterable[Dependency],
    *,
    pypi_client=None,
    npm_client=None,
    cache=None,
) -> List[SupplyChainFinding]:
    """Run every mechanical supply-chain check.

    Args:
        target: project root (used by artefact / source walks).
        manifests: the discovery output (manifests + lockfiles).
        deps: the joined dep list — typically post-``join.join``.
        pypi_client / npm_client: optional registry clients used by the
            ``recent_publish``/``maintainer_change``/
            ``maintainer_account_change`` detectors. When absent, those
            detectors are no-ops so we don't make uncached HTTP calls
            from a unit test or in a context where the orchestrator
            doesn't have access to the registry layer.
    """
    manifests_list = list(manifests)
    deps_list = list(deps)
    out: List[SupplyChainFinding] = []

    for hit in _install_hooks.scan_manifests(manifests_list, deps_list):
        out.append(_install_hook_to_finding(hit))

    for sh in _sentinel.scan_deps(deps_list):
        out.append(_sentinel_to_finding(sh))

    for ts in _typosquat.scan_deps(deps_list):
        out.append(_typosquat_to_finding(ts))

    for art in _artefacts.scan_target(target, manifests_list):
        out.append(_artefact_to_finding(art))

    for it in _python_imports.scan_target(
        target, manifests_list, cache=cache,
    ):
        out.append(_python_import_to_finding(it))

    for ex in _exfil.scan_target(target, manifests_list):
        out.append(_exfil_to_finding(ex))

    for gha in _gha_drift.scan_target(target, manifests_list):
        out.append(_gha_drift_to_finding(gha))

    for gd in _git_drift.scan_deps(deps_list):
        out.append(_git_drift_to_finding(gd))

    for td in _typosquat_domain.scan_target(target, manifests_list):
        out.append(_typosquat_domain_to_finding(td))

    if pypi_client is not None or npm_client is not None:
        for rm in _registry_metadata.scan_deps(
            deps_list,
            pypi_client=pypi_client,
            npm_client=npm_client,
        ):
            out.append(_registry_meta_to_finding(rm))

    return out


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def _install_hook_to_finding(
    hit: _install_hooks.InstallHookFinding,
) -> SupplyChainFinding:
    why = ", ".join(hit.hit.reasons) if hit.hit.reasons else "hook present"
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:install_hook_suspicious:"
            f"{hit.dependency.ecosystem}:{hit.dependency.name}:"
            f"{hit.hit.script_key}:{hit.dependency.declared_in}"
        ),
        kind="install_hook_suspicious",
        dependency=hit.dependency,
        detail=(
            f"`scripts.{hit.hit.script_key}` runs at install time; "
            f"reason: {why}; body: {_truncate(hit.hit.script_body)}"
        ),
        evidence={
            "script_key": hit.hit.script_key,
            "script_body": _truncate(hit.hit.script_body),
            "reasons": list(hit.hit.reasons),
        },
        severity=hit.severity,             # type: ignore[arg-type]
        confidence=hit.confidence,
    )


def _sentinel_to_finding(
    sh: _sentinel.SentinelHit,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:sentinel_match:"
            f"{sh.dependency.ecosystem}:{sh.dependency.name}:"
            f"{sh.dependency.version or '*'}:{sh.ref}"
        ),
        kind="sentinel_match",
        dependency=sh.dependency,
        detail=(
            f"'{sh.dependency.name}' matches known-malicious package: "
            f"{sh.incident}"
        ),
        evidence={
            "incident": sh.incident,
            "ref": sh.ref,
        },
        severity=sh.severity,                 # type: ignore[arg-type]
        confidence=sh.confidence,
    )


def _typosquat_to_finding(
    ts: _typosquat.TyposquatFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:typosquat_candidate:"
            f"{ts.dependency.ecosystem}:{ts.dependency.name}:"
            f"{ts.dependency.declared_in}"
        ),
        kind="typosquat_candidate",
        dependency=ts.dependency,
        detail=(
            f"name '{ts.dependency.name}' is distance {ts.distance} from "
            f"popular package '{ts.nearest_popular}' — verify the spelling"
        ),
        evidence={
            "nearest_popular": ts.nearest_popular,
            "distance": ts.distance,
        },
        severity=ts.severity,              # type: ignore[arg-type]
        confidence=ts.confidence,
    )


def _artefact_to_finding(
    art: _artefacts.ArtefactFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:{art.kind}:"
            f"{art.dependency.ecosystem}:{art.path}"
        ),
        kind=art.kind,                     # type: ignore[arg-type]
        dependency=art.dependency,
        detail=art.detail,
        evidence={"path": str(art.path)},
        severity=art.severity,             # type: ignore[arg-type]
        confidence=art.confidence,
    )


def _python_import_to_finding(
    it: _python_imports.ImportTimeFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:python_import_time_execution:"
            f"{it.path}:{it.line}"
        ),
        kind="python_import_time_execution",
        dependency=it.dependency,
        detail=it.detail,
        evidence={"path": str(it.path), "line": it.line},
        severity=it.severity,                  # type: ignore[arg-type]
        confidence=it.confidence,
    )


def _exfil_to_finding(
    ex: _exfil.ExfilFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:known_exfil_destination:"
            f"{ex.path}:{ex.line}:{ex.category}"
        ),
        kind="known_exfil_destination",
        dependency=ex.dependency,
        detail=ex.detail,
        evidence={"path": str(ex.path), "line": ex.line,
                   "category": ex.category},
        severity=ex.severity,                  # type: ignore[arg-type]
        confidence=ex.confidence,
    )


def _gha_drift_to_finding(
    gha: _gha_drift.GhaDriftFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:gha_action_ref_drift:"
            f"{gha.path}:{gha.line}:{gha.action}"
        ),
        kind="gha_action_ref_drift",
        dependency=gha.dependency,
        detail=gha.detail,
        evidence={
            "path": str(gha.path), "line": gha.line,
            "action": gha.action, "ref": gha.ref, "ref_kind": gha.ref_kind,
        },
        severity=gha.severity,                 # type: ignore[arg-type]
        confidence=gha.confidence,
    )


def _git_drift_to_finding(
    gd: _git_drift.GitDriftFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:git_tag_drift:"
            f"{gd.dependency.ecosystem}:{gd.dependency.name}:"
            f"{gd.dependency.declared_in}"
        ),
        kind="git_tag_drift",
        dependency=gd.dependency,
        detail=gd.detail,
        evidence={"ref": gd.ref, "ref_kind": gd.ref_kind},
        severity=gd.severity,                  # type: ignore[arg-type]
        confidence=gd.confidence,
    )


def _typosquat_domain_to_finding(
    td: _typosquat_domain.TyposquatDomainFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:typosquat_domain:"
            f"{td.path}:{td.line}:{td.suspect_host}"
        ),
        kind="typosquat_domain",
        dependency=td.dependency,
        detail=td.detail,
        evidence={
            "path": str(td.path),
            "line": td.line,
            "suspect_host": td.suspect_host,
            "nearest_popular": td.nearest_popular,
            "distance": td.distance,
        },
        severity=td.severity,                  # type: ignore[arg-type]
        confidence=td.confidence,
    )


def _registry_meta_to_finding(
    rm: _registry_metadata.RegistryMetaFinding,
) -> SupplyChainFinding:
    return SupplyChainFinding(
        finding_id=(
            f"sca:supplychain:{rm.kind}:"
            f"{rm.dependency.ecosystem}:{rm.dependency.name}:"
            f"{rm.dependency.declared_in}"
        ),
        kind=rm.kind,                          # type: ignore[arg-type]
        dependency=rm.dependency,
        detail=rm.detail,
        evidence=dict(rm.evidence),
        severity=rm.severity,                  # type: ignore[arg-type]
        confidence=rm.confidence,
    )


def _truncate(s: str, limit: int = 200) -> str:
    s = s.strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


__all__ = ["evaluate"]
