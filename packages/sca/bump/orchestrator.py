"""Bumper orchestrator — walks opinion surfaces, proposes
target versions, evaluates verdicts, optionally applies edits.

Phase 2.d MVP covers ONE surface: Dockerfile ARG version pins.
Future surfaces (manifest deps, FROM image refs, GHA `uses:`,
Helm chart deps, git submodules) plug in via the same shape —
each adds a walker + an upstream-source lookup + a rewriter
registration.

Operator-facing flow:

  raptor-sca bump <target>
    → walks every Dockerfile under <target>
    → for each ARG pin with a known upstream source, fetches
      the latest stable version
    → for each proposed bump, runs ``evaluate_bump_supply_chain``
      + ``_compute_verdict`` to produce a Block / Review / Clean
      verdict
    → prints a verdict table (default)
    → optionally writes the changes (``--apply``)
    → optionally emits a proposed/ directory (``--out``) instead
      of in-place writes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import List, Optional, Tuple

from ..models import SupplyChainFinding
from ..parsers.inline_installs._arg_version_pins import (
    _BUILTIN_ARG_MAP,
    _ARG_RE,
)
from ..registries.npm import NpmClient
from ..registries.pypi import PyPIClient
from ..rewriters import RewriteEdit, RewriteResult, rewrite
from .evaluator import evaluate_bump_supply_chain
from .upstream_map import UpstreamSource, lookup_upstream

logger = logging.getLogger(__name__)

# Verdict ladder constants (mirroring ``review.py``).
_VERDICT_CLEAN = 0
_VERDICT_REVIEW = 1
_VERDICT_BLOCK = 2

_VERDICT_LABEL = {
    _VERDICT_CLEAN: "Clean",
    _VERDICT_REVIEW: "Review",
    _VERDICT_BLOCK: "Block",
}


@dataclass(frozen=True)
class BumpCandidate:
    """One proposed bump: where it lives + what we'd change it to.

    ``kind`` discriminates the surface:
      * ``"arg"`` — Dockerfile ARG version pin.  ``locator`` is
        the ARG name (``SEMGREP_VERSION``); ``upstream`` is the
        ``UpstreamSource`` to query for the target.
      * ``"from_image"`` — Dockerfile FROM image ref.
        ``locator`` is ``"{registry}/{repository}"``; ``upstream``
        is None (the target comes from
        ``core.upstream_latest.oci_tags``).

    More kinds plug in by adding to this enum + a walker block in
    ``_enumerate_candidates``."""

    kind: str
    locator: str
    file: Path
    current_version: str
    target_version: str
    upstream: Optional[UpstreamSource] = None
    # Kind-specific metadata (e.g. SHA pair for SHA-pinned GHA
    # uses lines). The apply path forwards this to
    # ``RewriteEdit.extra`` so rewriters can read it.
    extra: Optional[dict] = None

    @property
    def arg_name(self) -> str:
        """Back-compat alias for ARG-pin candidates. Tests + the
        legacy JSON output read this; keep the name reachable
        without breaking the refactor."""
        return self.locator


@dataclass
class BumpResult:
    """Per-candidate outcome — what verdict we computed and
    whether we applied the rewrite."""

    candidate: BumpCandidate
    verdict: int
    verdict_label: str
    bump_supply_chain_findings: List[SupplyChainFinding]
    error: Optional[str] = None
    rewrite_result: Optional[RewriteResult] = None


@dataclass
class BumpReport:
    """Aggregate report from a ``run_bump`` call."""

    target: Path
    candidates: List[BumpCandidate]
    results: List[BumpResult]
    skipped: List[Tuple[str, Path, str]] = field(default_factory=list)
    # ``(arg_name, file, reason)`` for ARGs we couldn't bump
    # (no upstream mapping, current version not parseable, etc.)


def run_bump(
    target: Path,
    *,
    http,
    pypi_client: Optional[PyPIClient] = None,
    npm_client: Optional[NpmClient] = None,
    apply: bool = False,
    now: Optional[datetime] = None,
    cache=None,
    github_token: Optional[str] = None,
) -> BumpReport:
    """Walk Dockerfiles under ``target``, propose ARG bumps,
    compute verdicts, optionally apply.

    ``apply=False`` is dry-run: candidates + verdicts only, no
    file writes. ``apply=True`` rewrites in place via the
    Dockerfile-ARG rewriter — only edits where the verdict is
    Clean are applied (Review and Block surface in the report
    but don't auto-apply, per the project's "suggest-only"
    posture documented in
    project_sca_dependabot_plus_plus.md).
    """
    now = now or datetime.now(timezone.utc)
    candidates, skipped = _enumerate_candidates(
        target, http=http, cache=cache, github_token=github_token,
    )
    results: List[BumpResult] = []
    for cand in candidates:
        result = _evaluate_one(
            cand,
            pypi_client=pypi_client, npm_client=npm_client,
            now=now,
        )
        if apply and result.verdict == _VERDICT_CLEAN:
            edit = RewriteEdit(
                locator=cand.locator,
                old_value=cand.current_version,
                new_value=cand.target_version,
                extra=cand.extra,
            )
            rewrites = rewrite(cand.file, [edit])
            if rewrites:
                result.rewrite_result = rewrites[0]
        results.append(result)
    return BumpReport(
        target=target,
        candidates=candidates,
        results=results,
        skipped=skipped,
    )


def _enumerate_candidates(
    target: Path,
    *,
    http,
    cache,
    github_token: Optional[str],
) -> Tuple[List[BumpCandidate], List[Tuple[str, Path, str]]]:
    """Find every (Dockerfile, ARG) pair under ``target`` with a
    built-in upstream-source mapping, query the upstream, and
    build a candidate list."""
    from core.upstream_latest.github_releases import (
        NoStableVersionsFound,
        UpstreamLookupError,
        latest_release,
        latest_tag,
    )
    candidates: List[BumpCandidate] = []
    skipped: List[Tuple[str, Path, str]] = []
    target = target.resolve()
    if not target.exists():
        return candidates, skipped
    dockerfiles = _find_dockerfiles(target)
    # Cache upstream lookups per (kind, coordinate) — multiple
    # Dockerfiles may pin the same tool.
    latest_cache: dict = {}
    for dockerfile in dockerfiles:
        try:
            text = dockerfile.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("sca.bump: read failed for %s: %s",
                            dockerfile, e)
            continue
        for line in text.splitlines():
            match = _ARG_RE.match(line)
            if match is None:
                continue
            arg_name = match.group(1)
            current = match.group(2).strip('"').strip("'")
            upstream = lookup_upstream(arg_name)
            if upstream is None:
                # No upstream source — silent skip (operator can
                # add via the inline-comment override path).
                continue
            cache_key = (upstream.kind, upstream.coordinate)
            if cache_key in latest_cache:
                target_version = latest_cache[cache_key]
            else:
                try:
                    if upstream.kind == "github_release":
                        raw = latest_release(
                            upstream.coordinate,
                            http=http, cache=cache,
                            github_token=github_token,
                        )
                    elif upstream.kind == "github_tag":
                        raw = latest_tag(
                            upstream.coordinate,
                            http=http, cache=cache,
                            github_token=github_token,
                        )
                    else:
                        raw = None
                except (UpstreamLookupError, NoStableVersionsFound) as e:
                    skipped.append(
                        (arg_name, dockerfile,
                         f"upstream lookup failed: {e}")
                    )
                    latest_cache[cache_key] = None
                    continue
                target_version = (raw or "").lstrip("v")
                latest_cache[cache_key] = target_version
            if not target_version:
                skipped.append(
                    (arg_name, dockerfile, "no upstream version")
                )
                continue
            if target_version == current:
                # Already at latest — not a bump candidate.
                continue
            candidates.append(BumpCandidate(
                kind="arg",
                locator=arg_name,
                file=dockerfile,
                current_version=current,
                target_version=target_version,
                upstream=upstream,
            ))

    # FROM image refs — bump candidates from ``FROM
    # <registry>/<repository>:<tag>`` lines. Tag must be clean-
    # semver shape (refuses ``python:latest``, ``python:3.12-
    # bookworm`` — variants aren't bump candidates without a
    # variant-tag map we don't have).
    for dockerfile in dockerfiles:
        try:
            text = dockerfile.read_text(encoding="utf-8")
        except OSError:
            continue
        from_candidates, from_skipped = _enumerate_from_image_candidates(
            text=text, dockerfile=dockerfile,
            http=http, cache=cache,
            from_cache=latest_cache,
        )
        candidates.extend(from_candidates)
        skipped.extend(from_skipped)

    # GitHub Actions ``uses:`` refs — bump candidates from
    # ``.github/workflows/*.yml`` files. Phase 3.b ships tag-
    # pinned support only; SHA-pinned refs (raptor's convention)
    # need a tag→SHA resolver and ship in 3.b.2.
    workflow_files = _find_gha_workflows(target)
    for wf in workflow_files:
        try:
            text = wf.read_text(encoding="utf-8")
        except OSError:
            continue
        gha_candidates, gha_skipped = _enumerate_gha_uses_candidates(
            text=text, workflow=wf,
            http=http, cache=cache,
            github_token=github_token,
            uses_cache=latest_cache,
        )
        candidates.extend(gha_candidates)
        skipped.extend(gha_skipped)
    return candidates, skipped


def _enumerate_from_image_candidates(
    *,
    text: str,
    dockerfile: Path,
    http,
    cache,
    from_cache: dict,
) -> Tuple[List[BumpCandidate], List[Tuple[str, Path, str]]]:
    """Walk ``FROM`` instructions in ``text``; emit candidates
    for image refs with bumpable stable-semver tags.

    Skipped silently:
      * Digest-pinned FROM (``image@sha256:...``) — immutable
      * Multi-stage ``FROM x AS y`` where ``x`` references a
        previous stage by name (no registry component)
      * Tag is ``latest`` / branch-shaped / variant
        (``3.12-bookworm``)

    Skipped with explanation:
      * Upstream lookup fails (registry 404 / network / no
        stable tag at all)
    """
    from core.dockerfile.parser import parse_dockerfile
    from core.oci.image_ref import parse_image_ref
    from core.upstream_latest._version_filter import parse_stable
    from core.upstream_latest.github_releases import (
        NoStableVersionsFound,
        UpstreamLookupError,
    )
    from core.upstream_latest.oci_tags import latest_tag as oci_latest_tag

    candidates: List[BumpCandidate] = []
    skipped: List[Tuple[str, Path, str]] = []
    try:
        instructions = parse_dockerfile(text)
    except Exception:                # noqa: BLE001 — parsers must not crash
        logger.warning("sca.bump: Dockerfile parse failed for %s",
                        dockerfile, exc_info=True)
        return candidates, skipped
    # Track stage names from prior FROM lines so we can skip
    # ``FROM <stage>`` refs (those reuse a previous stage, not a
    # real image to bump).
    stage_names: set = set()
    for inst in instructions:
        if inst.directive != "FROM":
            continue
        args = inst.args.strip()
        # ``FROM image AS stage`` — record the stage name + bump
        # the image portion.
        as_split = args.split(" AS ", 1)
        if len(as_split) == 1:
            as_split = args.split(" as ", 1)
        image_ref_str = as_split[0].strip()
        if len(as_split) > 1:
            stage_names.add(as_split[1].strip())
        # Reusing a prior stage by name — not a bump target.
        if image_ref_str in stage_names:
            continue
        try:
            ref = parse_image_ref(image_ref_str)
        except Exception:            # noqa: BLE001
            skipped.append((
                image_ref_str, dockerfile,
                f"unparseable FROM ref: {image_ref_str}"
            ))
            continue
        # Digest-pinned → immutable, not a bump candidate.
        if ref.digest:
            continue
        if not ref.tag:
            continue
        # Tag must be clean stable semver to be bumpable. Variants
        # (``3.12-bookworm``) and aliases (``latest``) are silently
        # skipped — we don't have a variant-tag map that says
        # ``3.12-bookworm`` and ``3.13-bookworm`` are equivalent.
        if parse_stable(ref.tag) is None:
            continue
        locator = f"{ref.registry}/{ref.repository}"
        cache_key = ("oci_tag", locator)
        if cache_key in from_cache:
            target_tag = from_cache[cache_key]
        else:
            try:
                target_tag = oci_latest_tag(
                    image_ref_str, http=http, cache=cache,
                )
            except (UpstreamLookupError, NoStableVersionsFound) as e:
                skipped.append((
                    locator, dockerfile,
                    f"OCI tag lookup failed: {e}",
                ))
                from_cache[cache_key] = None
                continue
            from_cache[cache_key] = target_tag
        if not target_tag or target_tag == ref.tag:
            continue
        candidates.append(BumpCandidate(
            kind="from_image",
            locator=locator,
            file=dockerfile,
            current_version=ref.tag,
            target_version=target_tag,
            upstream=None,
        ))
    return candidates, skipped


def _evaluate_one(
    cand: BumpCandidate,
    *,
    pypi_client: Optional[PyPIClient],
    npm_client: Optional[NpmClient],
    now: datetime,
) -> BumpResult:
    """Compute the verdict for one bump candidate.

    ARG-kind candidates with a ``_BUILTIN_ARG_MAP`` entry get the
    full bump-tier verdict (recent_publish via registry metadata,
    maintainer_change / install_hook for npm). FROM-image-kind
    and ARG-kind without an eco-map fall through to Clean (no
    bump-tier signals available for OCI yet — operator review on
    the suggest-only PR is the gate).
    """
    eco_map = None
    if cand.kind == "arg":
        eco_map = _BUILTIN_ARG_MAP.get(cand.locator)
    findings: List[SupplyChainFinding] = []
    if eco_map is not None:
        ecosystem, package_name = eco_map
        try:
            findings = evaluate_bump_supply_chain(
                ecosystem=ecosystem, name=package_name,
                current_version=cand.current_version,
                target_version=cand.target_version,
                pypi_client=pypi_client, npm_client=npm_client,
                now=now,
            )
        except Exception as e:                # noqa: BLE001
            return BumpResult(
                candidate=cand,
                verdict=_VERDICT_REVIEW,    # err on the side of human-review
                verdict_label=_VERDICT_LABEL[_VERDICT_REVIEW],
                bump_supply_chain_findings=[],
                error=f"evaluator raised: {e}",
            )
    from ..review import _compute_verdict
    verdict = _compute_verdict(
        vuln_findings=[],
        typo_findings=[],
        bump_supply_chain_findings=findings,
    )
    return BumpResult(
        candidate=cand,
        verdict=verdict,
        verdict_label=_VERDICT_LABEL.get(verdict, str(verdict)),
        bump_supply_chain_findings=findings,
    )


_USES_RE = re.compile(
    r"^\s*(?:-\s+)?uses:\s*"             # optional YAML list marker
    r"(?P<repo>[\w.-]+/[\w.-]+)"        # owner/repo
    r"(?P<subpath>(?:/[\w./-]+)?)"       # optional sub-action path
    r"@"
    r"(?P<ref>[^\s#]+)"                   # ref (up to ws / comment)
)

# Phase 3.b.2 — SHA-pinned with ``# was vX`` comment. The comment
# carries the human-readable tag so the bumper can compute a new
# tag → new SHA on the same axis.
_USES_SHA_COMMENT_RE = re.compile(
    r"^\s*(?:-\s+)?uses:\s*"
    r"(?P<repo>[\w.-]+/[\w.-]+)"
    r"(?P<subpath>(?:/[\w./-]+)?)"
    r"@"
    r"(?P<sha>[a-f0-9]{40})"
    r"\s+#\s*was\s+"
    r"(?P<tag>[^\s#]+)"
)


def _enumerate_gha_uses_candidates(
    *,
    text: str,
    workflow: Path,
    http,
    cache,
    github_token: Optional[str],
    uses_cache: dict,
) -> Tuple[List[BumpCandidate], List[Tuple[str, Path, str]]]:
    """Walk ``uses:`` lines in a GHA workflow file; emit
    candidates for tag-pinned refs whose upstream has a newer
    stable tag.

    Skipped silently:
      * SHA-pinned refs (40-char hex) — Phase 3.b.2 territory
      * Branch-pinned refs (``@main``, ``@master``) — out of
        scope for auto-bumper
      * Refs that aren't clean stable-semver (handled via the
        github_releases.latest_release path which already
        filters pre-releases)

    Skipped with explanation:
      * Upstream lookup fails
    """
    from core.upstream_latest._version_filter import parse_stable
    from core.upstream_latest.github_releases import (
        NoStableVersionsFound,
        UpstreamLookupError,
        latest_release,
        latest_tag,
        resolve_tag_to_sha,
    )

    candidates: List[BumpCandidate] = []
    skipped: List[Tuple[str, Path, str]] = []
    for line in text.splitlines():
        # Phase 3.b.2: SHA-pinned with ``# was vX`` comment.
        # Detect first; if matched, propose new tag + new SHA.
        sha_match = _USES_SHA_COMMENT_RE.match(line)
        if sha_match is not None:
            repo = sha_match.group("repo")
            current_sha = sha_match.group("sha")
            current_tag = sha_match.group("tag")
            if parse_stable(current_tag) is None:
                # Comment tag isn't semver — operator's not using
                # the convention we recognize. Skip silently.
                continue
            target_tag = _lookup_latest_release_or_tag(
                repo, http=http, cache=cache,
                github_token=github_token,
                uses_cache=uses_cache,
                skipped=skipped, workflow=workflow,
            )
            if not target_tag:
                continue
            # Same-major-pin filter: ``# was v6`` and target
            # ``v6.2.1`` would surface as a noisy same-major
            # update (operator chose major-only). Skip those.
            if _same_major_pin(current_tag, target_tag):
                continue
            if current_tag == target_tag:
                continue
            # Resolve target tag → commit SHA. Cache per
            # (repo, target_tag) — multiple workflows often pin
            # the same actions to the same SHAs.
            sha_cache_key = ("tag_to_sha", repo, target_tag)
            if sha_cache_key in uses_cache:
                target_sha = uses_cache[sha_cache_key]
            else:
                try:
                    target_sha = resolve_tag_to_sha(
                        repo, target_tag, http=http, cache=cache,
                        github_token=github_token,
                    )
                except UpstreamLookupError as e:
                    skipped.append((
                        repo, workflow,
                        f"tag→SHA resolution failed: {e}",
                    ))
                    uses_cache[sha_cache_key] = None
                    continue
                uses_cache[sha_cache_key] = target_sha
            if not target_sha:
                continue
            candidates.append(BumpCandidate(
                kind="gha_uses",
                locator=repo,
                file=workflow,
                current_version=current_tag,
                target_version=target_tag,
                upstream=None,
                extra={
                    "old_sha": current_sha,
                    "new_sha": target_sha,
                },
            ))
            continue
        match = _USES_RE.match(line)
        if match is None:
            continue
        repo = match.group("repo")
        ref = match.group("ref")
        # SHA-pinned without our # was vX comment → can't
        # safely bump (no human-readable anchor). Skip.
        if re.fullmatch(r"[a-f0-9]{40}", ref):
            continue
        # Branch-shaped → skip (auto-bumper doesn't surface
        # branch-to-tag transitions yet).
        if ref in ("main", "master", "develop") or "/" in ref:
            continue
        # Must be parseable as semver (``v4``, ``v4.1.0``, ``1.0``).
        # We use the relaxed `parse_stable` filter — accepts 1-4
        # part numeric + optional v-prefix.
        if parse_stable(ref) is None:
            continue
        target_ref = _lookup_latest_release_or_tag(
            repo, http=http, cache=cache,
            github_token=github_token,
            uses_cache=uses_cache,
            skipped=skipped, workflow=workflow,
        )
        if not target_ref:
            continue
        # Normalise both to compare like-shapes. If the current
        # ref is a major-only (``v4``) and the latest is full
        # (``v4.2.1``), we'd want to either:
        #   (a) propose ``v5`` once it exists (major-only roll)
        #   (b) propose the full version (specific roll)
        # Renovate uses (a); for our suggest-only flow (a) is
        # less noisy.
        if ref == target_ref:
            continue
        # If current ref is major-only and target's major is the
        # same, no candidate (we'd be proposing a same-major
        # specific roll, which renovate considers a no-op for
        # major-only pins).
        if _same_major_pin(ref, target_ref):
            continue
        candidates.append(BumpCandidate(
            kind="gha_uses",
            locator=repo,
            file=workflow,
            current_version=ref,
            target_version=target_ref,
            upstream=None,
        ))
    return candidates, skipped


def _lookup_latest_release_or_tag(
    repo: str,
    *,
    http,
    cache,
    github_token: Optional[str],
    uses_cache: dict,
    skipped: List[Tuple[str, Path, str]],
    workflow: Path,
) -> Optional[str]:
    """Look up the latest stable upstream version for a GitHub
    repo. Tries ``/releases/latest`` first (proper GitHub
    Releases); falls back to ``/tags`` (projects that tag without
    releases). Caches per-repo via ``uses_cache``.

    Stability filter: GitHub's ``releases/latest`` endpoint
    returns whatever tag the publisher marked as the latest
    release, without enforcing stable-semver shape. For example,
    ``github/codeql-action`` publishes ``codeql-bundle-vX.Y.Z``
    bundle releases that don't match the ``v?N.N.N`` shape we
    expect for auto-bumping a ``v4`` → ``vN`` pin. This function
    validates the upstream-latest result through ``parse_stable``
    and falls through to the tag-listing path if the
    ``releases/latest`` tag doesn't pass the filter. If neither
    path produces a stable-semver tag, the repo is recorded as
    skipped with reason — operator sees the gap explicitly.
    """
    from core.upstream_latest._version_filter import parse_stable
    from core.upstream_latest.github_releases import (
        NoStableVersionsFound, UpstreamLookupError,
        latest_release, latest_tag,
    )
    cache_key = ("gha_uses", repo)
    if cache_key in uses_cache:
        return uses_cache[cache_key]
    target_ref = None
    try:
        candidate = latest_release(
            repo, http=http, cache=cache, github_token=github_token,
        )
    except UpstreamLookupError:
        candidate = None
    # Validate the /releases/latest result. Some projects publish
    # non-semver bundle tags (codeql-bundle-vX.Y.Z); we can't
    # substitute those for a vN pin shape, so fall through to
    # /tags which DOES filter to stable.
    if candidate is not None and parse_stable(candidate) is not None:
        target_ref = candidate
    else:
        try:
            target_ref = latest_tag(
                repo, http=http, cache=cache, github_token=github_token,
            )
        except (UpstreamLookupError, NoStableVersionsFound) as e:
            skipped.append((
                repo, workflow,
                (f"upstream lookup found non-semver release "
                 f"{candidate!r} and tag-listing also failed: {e}")
                if candidate is not None
                else f"upstream lookup failed: {e}",
            ))
            uses_cache[cache_key] = None
            return None
    uses_cache[cache_key] = target_ref
    return target_ref


def _same_major_pin(current: str, target: str) -> bool:
    """True if ``current`` is a major-only pin (``v4``) and the
    target is in the same major (``v4.2.1``). Avoids proposing
    a major-only roll TO a specific version — operators using
    major-only pins explicitly chose that level."""
    from core.upstream_latest._version_filter import parse_stable
    cur = parse_stable(current)
    tgt = parse_stable(target)
    if cur is None or tgt is None:
        return False
    if len(cur) != 1:
        return False
    return cur[0] == tgt[0]


def _find_gha_workflows(target: Path) -> List[Path]:
    """Walk ``target/.github/workflows/`` for YAML files."""
    workflows_dir = target / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return []
    out: List[Path] = []
    for path in workflows_dir.iterdir():
        if path.is_file() and path.suffix in (".yml", ".yaml"):
            out.append(path)
    return sorted(out)


def _find_dockerfiles(target: Path) -> List[Path]:
    """Walk ``target`` for files the Dockerfile-ARG rewriter
    knows how to handle. Mirrors the inline-installs parser's
    discovery predicate so the bumper sees every ARG-bearing
    file that the rest of SCA does."""
    if target.is_file():
        return [target] if _is_dockerfile(target) else []
    out: List[Path] = []
    for path in target.rglob("*"):
        if path.is_file() and _is_dockerfile(path):
            out.append(path)
    return sorted(out)


def _is_dockerfile(path: Path) -> bool:
    name = path.name
    if name in ("Dockerfile", "Containerfile"):
        return True
    if name.startswith("Dockerfile.") or name.endswith(".Dockerfile"):
        return True
    if path.suffix == ".dockerfile":
        return True
    return False


def render_report(report: BumpReport) -> str:
    """Operator-readable table summarising the bump report.

    Format chosen for terminal-readability; the bumper CLI prints
    it to stdout. PR-comment rendering is a separate codepath
    (the existing ``diff --pr-comment`` machinery, when wired in
    a future commit)."""
    lines: List[str] = []
    lines.append(f"raptor-sca bump: target {report.target}")
    if not report.candidates and not report.skipped:
        lines.append("  no bump candidates found")
        return "\n".join(lines) + "\n"
    if report.candidates:
        lines.append("")
        lines.append(
            f"  {'Kind':<11} {'Locator':<35} "
            f"{'Current':<14} {'Target':<22} {'Verdict':<8} Result"
        )
        # Dedup the display by (kind, locator, current_version,
        # target_version). The underlying ``results`` list still
        # has one entry per file (so --apply iterates all files)
        # but the human-readable table folds identical proposals
        # into one row with a file-count suffix. Pre-fix on
        # raptor: 8 CODEQL_VERSION rows + 3 github/codeql-action
        # rows (one per file each); operators read it as noise.
        groups: "dict[tuple, List[BumpResult]]" = {}
        order: List[tuple] = []
        for r in report.results:
            key = (
                r.candidate.kind, r.candidate.locator,
                r.candidate.current_version, r.candidate.target_version,
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(r)
        for key in order:
            group = groups[key]
            head = group[0]
            n_files = len(group)
            # ``Result`` field aggregates the per-file outcomes:
            # if all applied → "applied (N files)"; if mixed →
            # "applied N, skipped M"; if none applied → just
            # the dominant reason from the first.
            applied_count = sum(
                1 for r in group
                if r.rewrite_result is not None and r.rewrite_result.applied
            )
            if applied_count > 0:
                if applied_count == n_files:
                    result_label = (
                        f"applied ({n_files} file)"
                        if n_files == 1 else f"applied ({n_files} files)"
                    )
                else:
                    result_label = (
                        f"applied {applied_count}/{n_files}"
                    )
            elif head.rewrite_result is not None:
                result_label = f"skipped ({head.rewrite_result.reason})"
            elif head.error:
                result_label = f"error: {head.error}"
            else:
                result_label = "" if n_files == 1 else f"({n_files} files)"
            lines.append(
                f"  {head.candidate.kind:<11} "
                f"{head.candidate.locator:<35} "
                f"{head.candidate.current_version:<14} "
                f"{head.candidate.target_version:<22} "
                f"{head.verdict_label:<8} {result_label}"
            )
            # Surface the supply-chain findings inline so operators
            # know WHY a verdict isn't Clean. (One copy per group;
            # identical proposals would emit identical findings.)
            for sf in head.bump_supply_chain_findings:
                lines.append(f"      [{sf.severity}] {sf.kind}: {sf.detail}")
    if report.skipped:
        lines.append("")
        lines.append("  Skipped:")
        for arg, path, reason in report.skipped:
            lines.append(
                f"    {arg} ({path.name}): {reason}"
            )
    return "\n".join(lines) + "\n"
