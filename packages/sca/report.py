"""Markdown report renderer for ``raptor-sca`` runs.

The report is the human-facing artefact: scannable summary at the top,
full per-finding detail in the body. Operators read this on PRs;
findings.json is for tools.

Layout:

    # SCA Report — <target>

    ## Summary
    | Severity | Count | KEV | Top advisory |
    | ...      | ...   | ... | ...          |

    Hygiene: <N findings>
    Dependencies analysed: <N>
    Cache hit rate: <pct>

    ## Vulnerable dependencies
    ### CRITICAL — lodash 4.17.20 → fix: 5.0.0
    - Advisory: GHSA-... (CVE-2021-44228)
    - KEV: yes  /  EPSS: 0.97
    - Reachability: not_evaluated (mechanical-layer scope)
    - References: ...
    - Detail: <markdown>

    ## Hygiene findings
    ### lockfile_drift — npm:lodash
    ...

Design notes:
- Status text is *Title Case* (per CLAUDE.md output rules).
- KEV/EPSS columns surface the operationally most-actionable signals.
- Long advisory bodies are truncated with a "see findings.json" pointer.
- All output is markdown, no ANSI colour, no emoji.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from core.security.log_sanitisation import escape_nonprintable

from .findings import severity_rank
from .models import (
    Advisory,
    Dependency,
    HygieneFinding,
    Severity,
    SupplyChainFinding,
    VulnFinding,
)

logger = logging.getLogger(__name__)

# Cap on the length of the truncated detail block (chars).
_DETAIL_TRUNCATE = 600

# Severity → display label (Title Case per CLAUDE.md).
_SEV_LABEL: dict[str, str] = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "info": "Info",
    "none": "None",
}


def render_markdown_report(
    *,
    target: Path,
    deps_analysed: int,
    vuln_findings: Sequence[VulnFinding],
    hygiene_findings: Sequence[HygieneFinding],
    supply_chain_findings: Sequence[SupplyChainFinding] = (),
    cache_hits: Optional[int] = None,
    cache_misses: Optional[int] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Return the full report as a single markdown string."""
    generated_at = generated_at or datetime.now(timezone.utc)
    sorted_vulns = sorted(
        vuln_findings,
        key=lambda f: (-severity_rank(f.severity),
                       not f.in_kev,
                       -(f.epss or 0.0),
                       f.dependency.name),
    )
    sorted_hygiene = sorted(
        hygiene_findings,
        key=lambda f: (-severity_rank(f.severity), f.kind, f.dependency.name),
    )
    sorted_supply_chain = sorted(
        supply_chain_findings,
        key=lambda f: (-severity_rank(f.severity), f.kind, f.dependency.name),
    )

    parts: List[str] = []
    parts.append(_render_header(target, generated_at))
    parts.append(_render_summary(
        deps_analysed=deps_analysed,
        vuln_findings=sorted_vulns,
        hygiene_findings=sorted_hygiene,
        supply_chain_findings=sorted_supply_chain,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    ))
    if sorted_vulns:
        parts.append(_render_vuln_section(sorted_vulns))
    if sorted_supply_chain:
        parts.append(_render_supply_chain_section(sorted_supply_chain))
    if sorted_hygiene:
        parts.append(_render_hygiene_section(sorted_hygiene))
    if not sorted_vulns and not sorted_hygiene and not sorted_supply_chain:
        parts.append("## Findings\n\nNo vulnerabilities, hygiene, or "
                     "supply-chain issues detected for the analysed "
                     "dependency set.\n")
    return "\n".join(parts).rstrip() + "\n"


def write_markdown_report(path: Path, content: str) -> None:
    """Atomically write ``content`` to ``path``.

    Thin wrapper over the canonical helper so legacy callers don't
    have to update imports.
    """
    from ._atomic import atomic_write_text
    atomic_write_text(path, content)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _render_header(target: Path, generated_at: datetime) -> str:
    return (
        f"# SCA Report — {target}\n\n"
        f"_Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}_\n"
    )


def _render_summary(
    *,
    deps_analysed: int,
    vuln_findings: Sequence[VulnFinding],
    hygiene_findings: Sequence[HygieneFinding],
    supply_chain_findings: Sequence[SupplyChainFinding],
    cache_hits: Optional[int],
    cache_misses: Optional[int],
) -> str:
    severity_counts: Counter[str] = Counter()
    kev_count = 0
    suppressed_count = 0
    for f in vuln_findings:
        if f.suppressed:
            suppressed_count += 1
            continue
        severity_counts[f.severity] += 1
        if f.in_kev:
            kev_count += 1

    rows = [
        "## Summary\n",
        "| Severity | Count |",
        "|---|---|",
    ]
    for sev in ("critical", "high", "medium", "low", "info"):
        if severity_counts.get(sev):
            rows.append(f"| {_SEV_LABEL[sev]} | {severity_counts[sev]} |")
    if not any(severity_counts.values()):
        rows.append("| (none) | 0 |")

    rows.append("")
    rows.append(f"- Dependencies analysed: **{deps_analysed}**")
    rows.append(f"- Vulnerable findings: **{len(vuln_findings)}**")
    rows.append(f"- KEV-listed: **{kev_count}**")
    rows.append(f"- Supply-chain findings: **{len(supply_chain_findings)}**")
    rows.append(f"- Hygiene findings: **{len(hygiene_findings)}**")
    if suppressed_count:
        rows.append(f"- Suppressed: **{suppressed_count}** (operator-marked, "
                    "see `.raptor-sca-suppress.yml`)")
    if cache_hits is not None and cache_misses is not None:
        total = cache_hits + cache_misses
        rate = (cache_hits * 100 // total) if total else 0
        rows.append(
            f"- Advisory cache: **{cache_hits} hits / {cache_misses} misses "
            f"({rate}%)**"
        )
    rows.append("")
    return "\n".join(rows)


def _render_vuln_section(findings: Sequence[VulnFinding]) -> str:
    lines: List[str] = ["## Vulnerable dependencies\n"]
    for f in findings:
        lines.append(_render_one_vuln(f))
    return "\n".join(lines)


def _render_one_vuln(f: VulnFinding) -> str:
    dep = f.dependency
    primary: Optional[Advisory] = f.advisories[0] if f.advisories else None
    label = _SEV_LABEL.get(f.severity, f.severity.title())
    # Dep name comes from the operator's manifest — sanitise defensively
    # against ANSI / BIDI / control-character smuggling in package names.
    head = f"### {label} — {escape_nonprintable(dep.name)} " \
           f"{escape_nonprintable(dep.version or '*')}"
    if f.fixed_version:
        head += f" → fix: {escape_nonprintable(f.fixed_version)}"
    if f.suppressed:
        reason = escape_nonprintable(f.suppression_reason or 'no reason')
        head += f" _(suppressed: {reason})_"

    bullets: List[str] = []
    if primary is not None:
        aliases = ", ".join(escape_nonprintable(a) for a in primary.aliases[:3]) \
            if primary.aliases else "—"
        bullets.append(
            f"- Advisory: **{escape_nonprintable(primary.osv_id)}** "
            f"(aliases: {aliases})"
        )
        if primary.summary:
            bullets.append(
                f"- Summary: {escape_nonprintable(primary.summary)}"
            )

    badges = _badges(f)
    if badges:
        bullets.append(f"- {' / '.join(badges)}")

    if dep.is_lockfile:
        bullets.append(f"- Source: lockfile (`{dep.declared_in}`)")
    else:
        bullets.append(f"- Source: manifest (`{dep.declared_in}`)")
    bullets.append(f"- Direct: {'yes' if dep.direct else 'no'}; "
                   f"scope: {dep.scope}; pin: {dep.pin_style.value}")

    reach_reason = f.reachability.confidence.reason
    reach_line = (f"- Reachability: {f.reachability.verdict} "
                   f"(confidence {f.reachability.confidence.level}"
                   + (f" — {escape_nonprintable(reach_reason)}"
                      if reach_reason else "")
                   + ")")
    bullets.append(reach_line)

    # Inline-confidence display: the design specifies that operators
    # should see at a glance whether a finding is rock-solid (`high`
    # everywhere) or uncertain (`low — Gradle DSL parser is heuristic`).
    vmc_reason = f.version_match_confidence.reason
    bullets.append(
        f"- Version match: {f.version_match_confidence.level}"
        + (f" — {escape_nonprintable(vmc_reason)}"
           if vmc_reason else "")
    )
    pc_reason = dep.parser_confidence.reason
    bullets.append(
        f"- Parser: {dep.parser_confidence.level}"
        + (f" — {escape_nonprintable(pc_reason)}"
           if pc_reason else "")
    )

    if primary and primary.references:
        refs = ", ".join(f"<{escape_nonprintable(r)}>"
                          for r in primary.references[:3])
        bullets.append(f"- References: {refs}")

    detail = (primary.details if primary else "") or ""
    if detail:
        clipped = detail.strip()
        if len(clipped) > _DETAIL_TRUNCATE:
            clipped = clipped[:_DETAIL_TRUNCATE].rstrip() + (
                f"… (truncated; see findings.json `{f.finding_id}`)"
            )
        # Advisory detail is the largest attacker-influenced text in the
        # report; sanitise it before rendering.
        clipped = escape_nonprintable(clipped)
        bullets.append("\n<details><summary>Advisory detail</summary>\n\n"
                       f"{clipped}\n\n</details>")

    return head + "\n" + "\n".join(bullets) + "\n"


def _badges(f: VulnFinding) -> List[str]:
    out: List[str] = []
    if f.cvss_score is not None and f.cvss_vector:
        out.append(f"CVSS {f.cvss_score:.1f}")
    if f.in_kev:
        out.append("**KEV**")
    if f.epss is not None:
        out.append(f"EPSS {f.epss:.2f}")
    return out


def _render_supply_chain_section(
    findings: Sequence[SupplyChainFinding],
) -> str:
    lines: List[str] = ["## Supply-chain findings\n"]
    for f in findings:
        dep = f.dependency
        label = _SEV_LABEL.get(f.severity, f.severity.title())
        head = (
            f"### {label} — {f.kind}: "
            f"{dep.ecosystem}:{escape_nonprintable(dep.name)}"
        )
        bullets = [
            f"- Detail: {escape_nonprintable(f.detail)}",
            f"- Source: `{dep.declared_in}`",
            f"- Confidence: {f.confidence.level} "
            f"({escape_nonprintable(f.confidence.reason)})"
            if f.confidence.reason else f"- Confidence: {f.confidence.level}",
        ]
        lines.append(head + "\n" + "\n".join(bullets) + "\n")
    return "\n".join(lines)


def _render_hygiene_section(findings: Sequence[HygieneFinding]) -> str:
    lines: List[str] = ["## Hygiene findings\n"]
    for f in findings:
        lines.append(_render_one_hygiene(f))
    return "\n".join(lines)


def _render_one_hygiene(f: HygieneFinding) -> str:
    dep = f.dependency
    label = _SEV_LABEL.get(f.severity, f.severity.title())
    head = (
        f"### {label} — {f.kind}: "
        f"{dep.ecosystem}:{escape_nonprintable(dep.name)}"
    )
    bullets = [
        f"- Detail: {escape_nonprintable(f.detail)}",
        f"- Source: `{dep.declared_in}`",
        f"- Confidence: {f.confidence.level} "
        f"({escape_nonprintable(f.confidence.reason)})"
        if f.confidence.reason else f"- Confidence: {f.confidence.level}",
    ]
    return head + "\n" + "\n".join(bullets) + "\n"


__all__ = ["render_markdown_report", "write_markdown_report"]
