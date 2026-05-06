"""Transitive-dependency expansion orchestrator.

Mode (b) — cascade resolver in sandbox — is the default. The matching
resolver from ``packages/sca/resolvers/`` runs against the manifest's
parent dir and emits a real lockfile; we then parse those lockfile
bytes to extract the transitive dep set with exact resolved versions.

Mode (c) — registry-metadata recursive walk — is the opt-in fallback
(``--fallback-registry-metadata``) for when (b) can't run (toolchain
not in PATH, sandbox unavailable, resolver fails or times out, network
refusal at proxy). Approximation; emits findings tagged
``source_kind="metadata_walk"`` with low parser_confidence so
operators know to triage with caution.

Per-manifest, ecosystem-by-ecosystem decision tree:

  1. Sibling lockfile already present? → already parsed by the
     pipeline; skip orchestration.
  2. ``--no-resolve-transitive``? → skip; emit hygiene reason
     ``"resolver disabled by flag"``.
  3. Run (b) cascade resolver in sandbox.
     - success → parse generated lockfile bytes; emit transitives
       with ``source_kind="cascade_resolver"`` and high confidence.
     - skipped (toolchain unavailable) / failed (registry refused,
       resolver couldn't satisfy, timeout) → log specific reason,
       fall through.
  4. ``--fallback-registry-metadata`` enabled? → run (c). Approximation
     with low confidence.
  5. Else → skip; the existing ``lockfile_missing`` hygiene finding
     surfaces the gap.

Returns ``(transitive_deps, [TransitiveStatus, ...])``. Status rows
feed the operator-facing report so they can see, per ecosystem,
whether transitive coverage was real / approximate / missing-and-why.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
)

from core.http import HttpClient
from core.json import JsonCache

from .models import Confidence, Dependency, Manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitiveStatus:
    """Per-(manifest, ecosystem) record of how transitive expansion went."""

    manifest: Path
    ecosystem: str
    method: str          # "cascade_resolver" / "metadata_walk" /
                          # "skipped_lockfile_present" /
                          # "skipped_resolver_disabled" /
                          # "skipped_no_method_succeeded"
    reason: Optional[str]
    deps_added: int = 0
    failures: int = 0


def expand_missing_transitives(
    manifests: Sequence[Manifest],
    direct_deps: Sequence[Dependency],
    *,
    http: Optional[HttpClient] = None,
    cache: Optional[JsonCache] = None,
    enable_resolver: bool = True,
    enable_metadata_fallback: bool = False,
) -> Tuple[List[Dependency], List[TransitiveStatus]]:
    """Expand transitive deps for manifests that don't have a sibling
    lockfile already in ``manifests``.

    Args:
      manifests: every Manifest discovered by ``find_manifests``,
        including any lockfiles already on disk.
      direct_deps: every dep parsed from those manifests (post
        ``parse_manifest``, pre ``join_deps``). Used to dedup the
        new transitives against direct deps already declared.
      http: HttpClient for the metadata-walk fallback. ``None``
        disables (c) entirely.
      cache: JsonCache for metadata-walk caching. Optional.
      enable_resolver: when False, skip mode (b) entirely. Operators
        running ``--no-resolve-transitive`` get this. Useful for
        fast CI gates on lockfile-equipped projects.
      enable_metadata_fallback: when True, fall through to mode (c)
        when (b) fails. Operators running
        ``--fallback-registry-metadata`` get this. Default False —
        approximate findings hurt operator trust by default.

    Returns ``(transitive_deps, statuses)``. Caller merges
    ``transitive_deps`` into ``direct_deps`` before ``join_deps``.
    """
    statuses: List[TransitiveStatus] = []
    new_transitives: List[Dependency] = []

    # Build the "lockfile already on disk" set keyed by ecosystem-
    # parent-dir so we can detect "this manifest has a sibling
    # lockfile, skip transitive expansion".
    lockfile_dirs: Dict[Tuple[str, Path], Manifest] = {}
    for m in manifests:
        if m.is_lockfile:
            lockfile_dirs[(m.ecosystem, m.path.parent)] = m

    # We expand per (ecosystem, project_dir) — one cascade run covers
    # every manifest of that ecosystem in that dir. Group accordingly.
    # ``Inline`` manifests (Dockerfile / GHA / devcontainer / shell)
    # are skipped: they don't have an ecosystem in the cascade-resolver
    # sense (pip / npm / cargo). Inline-extracted deps still join the
    # direct-dep set; they just don't trigger lockfile generation.
    by_eco_dir: Dict[Tuple[str, Path], List[Manifest]] = {}
    for m in manifests:
        if m.is_lockfile:
            continue
        if m.ecosystem == "Inline":
            continue
        by_eco_dir.setdefault((m.ecosystem, m.path.parent), []).append(m)

    for (eco, project_dir), eco_manifests in by_eco_dir.items():
        if (eco, project_dir) in lockfile_dirs:
            statuses.append(TransitiveStatus(
                manifest=eco_manifests[0].path, ecosystem=eco,
                method="skipped_lockfile_present",
                reason=f"sibling lockfile: {lockfile_dirs[(eco, project_dir)].path.name}",
            ))
            continue
        if not enable_resolver and not enable_metadata_fallback:
            statuses.append(TransitiveStatus(
                manifest=eco_manifests[0].path, ecosystem=eco,
                method="skipped_resolver_disabled",
                reason="--no-resolve-transitive set and metadata-walk fallback off",
            ))
            continue

        added: List[Dependency] = []
        method = "skipped_no_method_succeeded"
        reason: Optional[str] = None
        failures = 0
        cascade_reason: Optional[str] = None

        # Cascade resolver — produces a real lockfile via the
        # ecosystem's own toolchain.
        if enable_resolver:
            cascade_deps, cascade_reason = _try_cascade(
                eco, project_dir, eco_manifests[0].path,
            )
            if cascade_deps is not None:
                added = cascade_deps
                method = "cascade_resolver"
                reason = None

        # Registry-metadata walk — opt-in fallback, approximate.
        if not added and enable_metadata_fallback and http is not None:
            walk_deps, c_failures = _try_metadata_walk(
                eco, [d for d in direct_deps if d.ecosystem == eco],
                http=http, cache=cache,
            )
            if walk_deps:
                added = walk_deps
                method = "metadata_walk"
                reason = (
                    "approximate transitive set: cascade resolver "
                    "unavailable, fell back to registry metadata "
                    "(less accurate; declared deps not resolved)"
                )
                failures = c_failures

        # When nothing succeeded, surface the most-informative reason
        # we have — cascade's specific failure beats the generic
        # "no method succeeded" message.
        if not added and method == "skipped_no_method_succeeded":
            reason = cascade_reason or (
                "no transitive-expansion method succeeded"
            )

        # Dedup against direct_deps — emit only NEW deps the project
        # didn't already declare directly.
        direct_keys = {(d.ecosystem, d.name, d.version or "*")
                       for d in direct_deps}
        deduped = [
            d for d in added
            if (d.ecosystem, d.name, d.version or "*") not in direct_keys
        ]
        new_transitives.extend(deduped)

        statuses.append(TransitiveStatus(
            manifest=eco_manifests[0].path, ecosystem=eco,
            method=method, reason=reason,
            deps_added=len(deduped), failures=failures,
        ))

    return new_transitives, statuses


# ---------------------------------------------------------------------------
# Mode (b): cascade resolver in sandbox
# ---------------------------------------------------------------------------

def _try_cascade(
    ecosystem: str, project_dir: Path, host_manifest_path: Path,
) -> Tuple[Optional[List[Dependency]], Optional[str]]:
    """Run the matching cascade resolver. Return
    ``(deps, None)`` on success and ``(None, reason)`` on any failure
    (no toolchain, resolver couldn't satisfy, no lockfile parser, etc.)
    so the orchestrator can surface a specific operator-facing message
    rather than a generic "no method succeeded".
    """
    _ensure_lockfile_parsers_loaded()
    from .resolvers import get_resolver
    resolver = get_resolver(ecosystem, project_dir=project_dir)
    if resolver is None:
        return None, f"no cascade resolver registered for {ecosystem}"
    if not resolver.is_available():
        # Resolver implementations encode the toolchain name in
        # `is_available()` via a `<tool> --version` probe; surface
        # it explicitly so operators don't have to guess which
        # tool is missing.
        return None, (
            f"{ecosystem} toolchain not installed (cascade resolver "
            f"requires it for transitive resolution)"
        )
    result = resolver.dry_run(project_dir)
    if not result.success:
        return None, (
            f"{ecosystem} resolver failed: "
            f"{(result.error or 'unknown error')[:140]}"
        )
    if result.proposed_lockfile is None:
        return None, (
            f"{ecosystem} resolver succeeded but produced no "
            f"lockfile bytes (transitive ingest not possible)"
        )
    parser = _LOCKFILE_PARSERS.get(ecosystem)
    if parser is None:
        return None, (
            f"no lockfile parser wired for {ecosystem}; transitive "
            f"data was generated but cannot be ingested"
        )
    deps = _parse_lockfile_bytes(
        ecosystem, result.proposed_lockfile, parser, host_manifest_path,
    )
    if deps is None:
        return None, (
            f"{ecosystem} lockfile parse failed for cascade output "
            f"(unexpected format from resolver)"
        )
    # Re-tag every dep with cascade_resolver source_kind + direct=False.
    # Subtle: pip-compile output is a flat pinned-requirements file;
    # the requirements.txt parser has no signal to mark anything
    # transitive (it parses a manifest shape, defaults direct=True).
    # But cascade output IS the transitive closure by definition —
    # the operator's actual direct deps are already in ``direct_deps``
    # at the orchestrator's caller. The orchestrator's dedup against
    # direct_deps strips overlap; what's left here is purely transitive,
    # so direct=False is correct for the whole list.
    #
    # ``declared_in`` is also re-tagged to the host manifest path —
    # the cascade lockfile lives in a per-call ``TemporaryDirectory``
    # whose path looks like ``/tmp/raptor-sca-cascade-<rand>/...``
    # and is gone before the report renders. Pointing operators at
    # the host manifest (the file they can actually edit) is more
    # useful than at a vanished temp.
    tagged = [
        _with_cascade_source(d, host_manifest_path) for d in deps
    ]
    return tagged, None


def _parse_lockfile_bytes(
    ecosystem: str, blob: bytes,
    parser: Callable[[Path], List[Dependency]],
    host_manifest_path: Path,
) -> Optional[List[Dependency]]:
    """Most lockfile parsers take a ``Path``. Write the bytes to a
    temp file matching the parser's expected name, parse, return."""
    expected_name = _CASCADE_LOCKFILE_NAMES[ecosystem]
    try:
        with tempfile.TemporaryDirectory(prefix="raptor-sca-cascade-") as td:
            tmp = Path(td) / expected_name
            tmp.write_bytes(blob)
            return list(parser(tmp))
    except Exception as e:                          # noqa: BLE001
        logger.debug(
            "transitive: cascade lockfile parse failed for %s: %s",
            ecosystem, e,
        )
        return None


def _with_cascade_source(
    d: Dependency, host_manifest_path: Path,
) -> Dependency:
    """Re-tag a Dependency as cascade_resolver-sourced.

    Three re-tags happen here:
      - ``source_kind="cascade_resolver"`` — the bytes came from an
        in-sandbox cascade run, not a checked-in lockfile; the
        report distinction matters for triage trust.
      - ``direct=False`` — cascade output is the transitive closure;
        the orchestrator dedups against the operator's actual
        direct deps before the result reaches ``join_deps``.
      - ``declared_in=host_manifest_path`` — the original
        ``declared_in`` from the parser points at the cascade temp
        lockfile (``/tmp/raptor-sca-cascade-<rand>/...``), which is
        deleted before the report renders. The operator-actionable
        path is the host manifest that triggered the cascade — what
        they need to edit to change the resolution input.
    """
    return Dependency(
        ecosystem=d.ecosystem, name=d.name, version=d.version,
        declared_in=host_manifest_path, scope=d.scope,
        is_lockfile=d.is_lockfile, pin_style=d.pin_style,
        direct=False,
        purl=d.purl,
        parser_confidence=d.parser_confidence,
        declared_license=d.declared_license,
        commented_out=d.commented_out,
        source_kind="cascade_resolver",
    )


# ---------------------------------------------------------------------------
# Mode (c): registry metadata walk
# ---------------------------------------------------------------------------

def _try_metadata_walk(
    ecosystem: str, eco_direct_deps: List[Dependency],
    *, http: HttpClient, cache: Optional[JsonCache],
) -> Tuple[List[Dependency], int]:
    """Walk the registry metadata for an ecosystem's direct deps.
    Returns (deps, failures). Empty deps + non-zero failures is
    "tried, couldn't get data"; empty deps + zero failures is
    "tried, found no transitives"."""
    from .registry_metadata_walk import walk_transitive
    result = walk_transitive(
        eco_direct_deps, http=http, cache=cache,
        ecosystems={ecosystem},
    )
    return (result.deps_added, result.failures)


# ---------------------------------------------------------------------------
# Per-ecosystem cascade-output parsing
# ---------------------------------------------------------------------------

# Filename the cascade resolver's bytes should be written under so
# the matching parser recognises the format. Different per ecosystem
# because parsers do filename-based dispatch internally.
_CASCADE_LOCKFILE_NAMES: Dict[str, str] = {
    "PyPI": "requirements.txt",        # pip-compile output is a pinned reqs
    "npm": "package-lock.json",
    "crates.io": "Cargo.lock",
    "Go": "go.sum",
    "RubyGems": "Gemfile.lock",
    "Packagist": "composer.lock",
    "NuGet": "packages.lock.json",
    # "Maven", "Maven (Gradle)" — the resolvers emit dep-tree text
    # rather than a structured lockfile; cascade-result parsing for
    # these is a separate follow-up. Default behaviour: skip with
    # "no lockfile parser wired" log.
}


def _import_lockfile_parser(ecosystem: str) -> Optional[Callable]:
    """Lazy-import the matching lockfile parser. Avoids import-time
    cycles between transitive.py and the parsers package."""
    from . import parsers
    if ecosystem == "PyPI":
        from .parsers.requirements import parse as _parse
        return _parse
    if ecosystem == "npm":
        from .parsers.package_lock_json import parse as _parse
        return _parse
    if ecosystem == "crates.io":
        from .parsers.cargo import parse_lockfile as _parse
        return _parse
    if ecosystem == "Go":
        from .parsers.gomod import parse_lockfile as _parse
        return _parse
    if ecosystem == "RubyGems":
        from .parsers.gemfile import parse_lockfile as _parse
        return _parse
    if ecosystem == "Packagist":
        from .parsers.composer import parse_lockfile as _parse
        return _parse
    if ecosystem == "NuGet":
        from .parsers.nuget import parse_lockfile as _parse
        return _parse
    return None


# Initialised lazily on first orchestrator call to avoid import-time
# parser imports leaking into modules that don't need them.
_LOCKFILE_PARSERS: Dict[str, Callable] = {}


def _ensure_lockfile_parsers_loaded() -> None:
    if _LOCKFILE_PARSERS:
        return
    for eco in _CASCADE_LOCKFILE_NAMES:
        p = _import_lockfile_parser(eco)
        if p is not None:
            _LOCKFILE_PARSERS[eco] = p


__all__ = [
    "TransitiveStatus",
    "expand_missing_transitives",
]
