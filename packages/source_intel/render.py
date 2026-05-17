"""Render :class:`SourceIntelResult` evidence into prompt-friendly
strings for Stage D / `/exploit` / `/agentic` consumers.

The output is a list of human-readable lines; ordering puts the
strongest signal first (literal observations before alias-only).
Consumers concatenate the lines into a structured block under
TaintedString / UntrustedBlock envelopes per the project's prompt-
envelope discipline.

Three styles surfaced in Phase 2:
  * ``stage_d`` — evidence supporting/against a Stage D ruling
  * ``exploit_plan`` — constraints to plan around for /exploit
  * ``agentic_variant`` — seed candidates for variant hunting

For substrate, all three render the same content with style-specific
phrasing. Axes 2-7 may diverge per style when their evidence
classes have distinct interpretations per consumer.

Also provides ``derive_mitigations_found()`` — structured list of
``Mitigation`` records per design strict invariant ("mitigations_found:
[...] shape; no boolean hardened. Absence ≠ unhardened.").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.build.build_flags import BuildFlagsContext
from packages.source_intel.analyze import (
    GRADE_DOMINATES,
    GRADE_SAME_FUNCTION,
    GRADE_SAME_PATH,
    KIND_ACCESS,
    KIND_ALLOC_SIZE,
    KIND_MALLOC,
    KIND_NO_STACK_PROTECTOR,
    KIND_NONNULL,
    KIND_NORETURN,
    KIND_RETURNS_NONNULL,
    KIND_WUR,
    AbortEvidence,
    AllocationEvidence,
    AttributeEvidence,
    SourceIntelResult,
)


@dataclass(frozen=True)
class Mitigation:
    """Structured mitigation entry per design strict invariant.

    ``name`` is the canonical mitigation kind (abort_dominates,
    fortify_blocks, etc.). ``axis`` is the source_intel axis id
    that detected it. ``confidence`` is one of:
        * ``"high"`` — strong evidence (DOMINATES grade, exact
          line match, FORTIFY blocks intercepted call site)
        * ``"medium"`` — moderate (same_path grade, near-line)
        * ``"low"`` — same_function grade with proximity,
          informational signals

    Absence of a mitigation in the list does NOT imply unhardened —
    it means source_intel didn't detect that mitigation (which may
    be a coverage gap, not real absence). Per the design strict
    invariant: never emit ``hardened: True/False``; only the
    structured-list shape.
    """

    name: str
    axis: str  # "axis_1" through "axis_8"
    confidence: str  # "low" | "medium" | "high"
    detail: str
    location: Optional[tuple] = None  # (file, line) or None


_STYLES = ("stage_d", "exploit_plan", "agentic_variant")


# Stage E binary-verdict values that supersede source_intel's
# EXPLOITABLE-leaning signal. Per design: "Binary observation
# supersedes source intent when both available (Stage E binary wins)."
# When the binary side says the bug can't reach exploitable runtime
# (RELRO blocks GOT overwrite, no usable ROP, sanitizer in production
# build, etc.), source_intel's structural evidence is reframed as
# informational rather than verdict-bearing.
#
# Verdicts from ``packages.exploit_feasibility.api:analyze_binary``:
#   "exploitable" | "likely_exploitable"  — binary agrees with EXPLOITABLE
#   "blocked"                              — binary says NO
#   "requires_environment"                 — binary says probably-NO
_BINARY_SUPERSEDING_VERDICTS = frozenset({
    "blocked",
    "requires_environment",
})


def _supersession_prefix(binary_verdict: Optional[str]) -> Optional[str]:
    """Return a one-line SUPERSEDED marker when the binary verdict
    overrides source_intel; ``None`` otherwise.

    Stage E semantics: binary observation always wins over source
    intent when both are available. This prefix tells the consumer
    LLM: "the following source_intel observations are factually
    correct, but the binary side already proved the path isn't
    exploitable — weigh them as context, not as exploitability
    evidence."
    """
    if binary_verdict is None:
        return None
    if binary_verdict not in _BINARY_SUPERSEDING_VERDICTS:
        return None
    return (
        f"SUPERSEDED: binary verdict `{binary_verdict}` from "
        f"packages.exploit_feasibility — the following source_intel "
        f"observations are STRUCTURALLY CORRECT but DO NOT change "
        f"exploitability. Binary side already proved the primitive "
        f"can't reach exploitable runtime. Treat the lines below as "
        f"context for the verdict explanation, not as evidence "
        f"for/against EXPLOITABLE."
    )


def derive_evidence_strings(
    result: SourceIntelResult,
    finding_function: Optional[str] = None,
    build_flags: Optional[BuildFlagsContext] = None,
    style: str = "stage_d",
    max_lines: Optional[int] = None,
    binary_verdict: Optional[str] = None,
) -> List[str]:
    """Render source_intel evidence for a finding into prompt lines.

    Args:
      result: the per-target SourceIntelResult
      finding_function: the function the finding cites (used to filter
        WUR evidence to relevant functions; when None, all observations
        surface)
      build_flags: per-target build-flag context (for compile-enforcement
        interpretation of WUR — `__must_check` is binding only if
        `-Werror=unused-result` was on)
      style: "stage_d" | "exploit_plan" | "agentic_variant" — chooses
        framing. Substrate ships identical content per style; axis-N
        PRs can diverge.
      max_lines: cap the number of returned lines (for context-tight
        prompt budgets); None = no cap.
      binary_verdict: optional Stage E binary-side verdict from
        :mod:`packages.exploit_feasibility`. Per design Stage E,
        the binary observation supersedes source intent when both
        are available. When this verdict is ``"blocked"`` or
        ``"requires_environment"`` (binary says NOT exploitable),
        the rendered output is prefixed with a SUPERSEDED marker
        and reframed as informational-only: the LLM should weigh
        the binary verdict over any source_intel EXPLOITABLE signal.
        ``None`` (default): no binary side; emit unchanged.

    Returns an empty list when the result is skipped or carries no
    relevant evidence — consumers can render "no source_intel signal"
    or omit the block entirely.
    """
    if style not in _STYLES:
        raise ValueError(f"unknown style: {style!r} (expected one of {_STYLES})")

    lines: List[str] = []

    if result.is_skipped:
        # Surface the skip reason so consumers know there was no
        # evidence at all — distinct from "evidence ran and found
        # nothing." This is critical: consumers MUST NOT interpret
        # an empty block as "unhardened".
        lines.append(
            f"Source_intel skipped: {result.skipped_reason}. "
            f"No evidence either way."
        )
        # Stage E supersession still applies even when source_intel
        # was skipped — the consumer needs to see the binary verdict
        # disposition regardless.
        prefix = _supersession_prefix(binary_verdict)
        if prefix is not None:
            lines = [prefix] + lines
        return _truncate(lines, max_lines)

    # Filter attributes to the finding's function when supplied.
    observations = list(result.attributes)
    if finding_function:
        observations = [
            ev for ev in observations
            if ev.function_name == finding_function
        ]
    # Literal observations first, then known-alias.
    observations.sort(key=lambda ev: 0 if ev.match_source == "literal" else 1)

    for ev in observations:
        line = _render_attribute_line(ev, build_flags, style)
        if line is not None:
            lines.append(line)

    # Abort evidence (axis 2). Filter to the finding's function when
    # supplied (same composition as attributes). Strongest grade
    # first so dominate-grade signal appears at the top.
    aborts = list(result.aborts)
    if finding_function:
        aborts = [
            ab for ab in aborts
            if ab.enclosing_function == finding_function
            or ab.enclosing_function is None
        ]
    _GRADE_ORDER = {
        GRADE_DOMINATES: 0,
        GRADE_SAME_PATH: 1,
        GRADE_SAME_FUNCTION: 2,
    }
    aborts.sort(key=lambda ab: _GRADE_ORDER.get(ab.grade, 99))
    for ab in aborts:
        lines.append(_render_abort_line(ab, style))

    # Allocation evidence (axis 3). Filter to the finding's function
    # when supplied. Phase 6a: only the field-assignment shape lands;
    # later shapes are added as axis-3-expansion ships.
    allocations = list(result.allocations)
    if finding_function:
        allocations = [
            ae for ae in allocations
            if ae.enclosing_function == finding_function
            or ae.enclosing_function is None
        ]
    for ae in allocations:
        lines.append(_render_allocation_line(ae, style))

    # Axis-6 sanitizer context. Surfaced once per finding (target-wide,
    # not per-call-site) when build_flags carries observed sanitizers.
    # The LLM weighs this in two opposing directions per consumer:
    #   * Production-equivalent builds: a memory bug in code compiled
    #     with -fsanitize=address / KASAN is caught at the cost of a
    #     panic — bug becomes DoS, not RCE.
    #   * Test / CI builds with sanitizers: bug surface is wider than
    #     production, but the finding may be a sanitizer-only artefact.
    sanitizer_line = _render_sanitizers_line(build_flags, style)
    if sanitizer_line is not None:
        lines.append(sanitizer_line)

    # When source_intel ran but found nothing relevant — emit an
    # explicit "no signal" line so the consumer prompt template
    # carries the absence acknowledgement.
    if not lines:
        lines.append(
            "Source_intel ran; no attribute or proximity evidence for "
            f"{finding_function or '<finding function>'}. "
            f"Absence of evidence is NOT evidence of unhardened code."
        )

    # Stage E binary-supersedes (Phase C PR2). When the binary side
    # says NOT exploitable, prepend a SUPERSEDED marker reframing
    # everything below as informational. Always applies — even when
    # only the "no signal" line was emitted — so the consumer sees
    # the consistent "binary wins" disposition.
    prefix = _supersession_prefix(binary_verdict)
    if prefix is not None:
        lines = [prefix] + lines

    return _truncate(lines, max_lines)


def _render_allocation_line(ae: AllocationEvidence, style: str) -> str:
    """Render one unchecked-allocation observation."""
    fn_text = (
        f"function `{ae.enclosing_function}`"
        if ae.enclosing_function
        else f"in {ae.location[0]} near line {ae.location[1]}"
    )
    field_text = (
        f"field `->{ae.target_field}`"
        if ae.target_field
        else "the assigned location"
    )

    if style == "stage_d":
        prefix = "Allocator-result not checked"
    elif style == "exploit_plan":
        prefix = "Primitive — unchecked allocator result"
    else:
        prefix = "Variant hint — unchecked alloc shape"

    caveat = ""
    if ae.conditional_on:
        caveat = (
            f" (CONDITIONAL: gated by `#if* {ae.conditional_on}` — "
            f"downweight unless the actual build enables this.)"
        )

    return (
        f"{prefix}: `{ae.allocator}` at {ae.location[0]}:{ae.location[1]} "
        f"{fn_text} stores into {field_text} with NO subsequent NULL "
        f"check on that location. Allocation failure → NULL stored → "
        f"downstream deref crashes (CWE-476).{caveat}"
    )


def _render_abort_line(ab: AbortEvidence, style: str) -> str:
    """Render one abort-evidence observation."""
    fn_text = (
        f"function `{ab.enclosing_function}`"
        if ab.enclosing_function
        else f"in {ab.location[0]} near line {ab.location[1]}"
    )
    grade_phrase = {
        GRADE_DOMINATES: "DOMINATES the function body (depth-1, no early exit precedes)",
        GRADE_SAME_PATH: "appears on a nested control-flow path (depth>1, inside if/loop/switch)",
        GRADE_SAME_FUNCTION: "shares the function with the sink",
    }.get(ab.grade, ab.grade)

    if style == "stage_d":
        prefix = "Control-flow signal — abort-class call near sink"
    elif style == "exploit_plan":
        prefix = "DoS-only constraint — abort proximate to sink"
    else:
        prefix = "Variant hint — abort proximity"

    caveat = ""
    if ab.conditional_on:
        caveat = (
            f" (CONDITIONAL: gated by `#if* {ab.conditional_on}` — "
            f"downweight unless the actual build enables this.)"
        )
    if ab.grade == GRADE_SAME_PATH:
        # Phase C PR1: explicit weaker-than-dominates caveat.
        # `same_path` means the abort sits inside a nested branch
        # (depth>1) — execution that enters that branch DOES hit the
        # abort, but other branches in the function bypass it. This
        # is mid-strength evidence, NOT a guarantee like `dominates`.
        caveat += (
            " Grade `same_path` is mid-strength: the abort sits on "
            "SOME control-flow path through the function but other "
            "branches (else / loop fall-through / different switch "
            "arms) bypass it. The bug primitive may reach runtime via "
            "an unguarded branch. Stronger grade `dominates` (depth-1, "
            "no early exit) would be needed to prove DoS-only outcome."
        )
    if ab.grade == GRADE_SAME_FUNCTION:
        caveat += (
            " Grade `same_function` is weak: the abort may be on an "
            "unrelated path within the function. Stronger grades "
            "(`same_path`, `dominates`) require axis-2-expansion."
        )
    return (
        f"{prefix}: `{ab.macro}` call at {ab.location[0]}:{ab.location[1]} "
        f"{fn_text} — {grade_phrase}. If this abort is reached before "
        f"the bug primitive, the program halts and the bug becomes "
        f"DoS-only.{caveat}"
    )


def _render_attribute_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> Optional[str]:
    """Dispatch to the per-kind renderer. Unknown kinds return None
    (silently dropped — render is best-effort).

    When ``conditional_on`` is set, the rendered line is suffixed with
    a caveat: matches under unknown ``#ifdef`` blocks may not apply
    to the binary that was actually built.
    """
    if ev.kind == KIND_WUR:
        line = _render_wur_line(ev, build_flags, style)
    elif ev.kind == KIND_NONNULL:
        line = _render_nonnull_line(ev, build_flags, style)
    elif ev.kind == KIND_ALLOC_SIZE:
        line = _render_alloc_size_line(ev, build_flags, style)
    elif ev.kind == KIND_RETURNS_NONNULL:
        line = _render_returns_nonnull_line(ev, build_flags, style)
    elif ev.kind == KIND_NORETURN:
        line = _render_noreturn_line(ev, build_flags, style)
    elif ev.kind == KIND_MALLOC:
        line = _render_malloc_line(ev, build_flags, style)
    elif ev.kind == KIND_NO_STACK_PROTECTOR:
        line = _render_no_stack_protector_line(ev, build_flags, style)
    elif ev.kind == KIND_ACCESS:
        line = _render_access_line(ev, build_flags, style)
    else:
        return None
    return _append_conditional_caveat(line, ev)


def _append_conditional_caveat(
    line: str,
    ev: AttributeEvidence,
) -> str:
    """Append the ``conditional_on`` caveat when the match is under
    an ``#if*`` block. Caller-side ``derive_evidence_strings`` consumes
    the suffix as part of the single evidence string."""
    if not ev.conditional_on:
        return line
    return (
        f"{line} (CONDITIONAL: this annotation is gated by "
        f"`#if* {ev.conditional_on}` — downweight unless the actual "
        f"build is known to enable this config.)"
    )


# =====================================================================
# Per-evidence-kind line builders
# =====================================================================


def _render_wur_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """One line of WUR evidence, framed per consumer style.

    The enforcement-status caveat depends on build flags:
      * `-Werror=unused-result` known True → "compile-enforced"
      * `-Werror=unused-result` known False → "author intent only;
        warning was suppressed"
      * None / build_flags absent → "advisory; enforcement depends
        on build flags not in evidence"
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )
    src_text = (
        "literal __attribute__((warn_unused_result))"
        if ev.match_source == "literal"
        else f"known alias `{ev.raw_match}`"
    )

    enforcement = _enforcement_phrase(build_flags)

    if style == "stage_d":
        prefix = "Author intent — must-check contract"
    elif style == "exploit_plan":
        prefix = "Constraint — caller-must-check contract"
    else:  # agentic_variant
        prefix = "Variant hint — must-check signal"

    return (
        f"{prefix}: {fn_text} annotated as warn_unused_result via "
        f"{src_text}. {enforcement}"
    )


def _render_nonnull_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render nonnull evidence.

    Nonnull is a TWO-EDGED signal for memory corruption:
      * Author intent — caller MUST pass non-null pointers.
      * Compiler behaviour — when -O2+ AND -fdelete-null-pointer-checks
        is ON (GCC userspace default), the compiler may eliminate
        redundant null-checks inside the annotated function. A real
        NULL reaching the function then dereferences without the
        defensive branch the author may have written.
      * In the kernel, -fno-delete-null-pointer-checks is in CFLAGS
        since 4.9, so the elimination doesn't happen — defensive null
        checks are preserved.

    The Stage D consumer reads this evidence WITH the build-flag
    context to determine effective semantics.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    null_check_phrase = _nonnull_null_check_phrase(build_flags)

    if style == "stage_d":
        prefix = "Author intent + compiler signal — nonnull"
    elif style == "exploit_plan":
        prefix = "Constraint — caller-must-be-non-null contract"
    else:
        prefix = "Variant hint — nonnull annotation"

    return (
        f"{prefix}: {fn_text} annotated nonnull (caller must pass "
        f"non-null). {null_check_phrase}"
    )


def _nonnull_null_check_phrase(build_flags: Optional[BuildFlagsContext]) -> str:
    """Compose the dead-code-elimination caveat for nonnull."""
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "Compiler-elimination status unknown (build flags not in "
            "evidence); a NULL reaching this function may be more or "
            "less exploitable depending on -fdelete-null-pointer-checks."
        )
    if build_flags.delete_null_pointer_checks is False:
        return (
            "Build flags include -fno-delete-null-pointer-checks — "
            "defensive null checks inside the function are preserved; "
            "any NULL dereference behaves as the source code shows."
        )
    if build_flags.delete_null_pointer_checks is True:
        return (
            "Build flags explicitly enable -fdelete-null-pointer-checks "
            "— compiler may dead-code-eliminate redundant null checks "
            "inside the function; a real NULL would reach the deref."
        )
    return (
        "Compiler-elimination status not pinned by observed flags — "
        "default depends on -O level and compiler version."
    )


def _render_alloc_size_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render alloc_size evidence.

    The annotation tells the compiler the return buffer's byte size.
    When FORTIFY_SOURCE is on, this unlocks __builtin_object_size and
    fortified intrinsics (memcpy_chk etc.) — operations on the return
    value get bounds-checked at runtime. Without FORTIFY_SOURCE, the
    annotation is mostly hint-for-analyzers.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    fortify_phrase = _alloc_size_fortify_phrase(build_flags)

    if style == "stage_d":
        prefix = "Author intent + compiler signal — alloc_size"
    elif style == "exploit_plan":
        prefix = "Constraint — alloc_size advertises returned buffer size"
    else:
        prefix = "Variant hint — alloc_size annotation"

    return (
        f"{prefix}: {fn_text} returns a buffer whose byte size equals "
        f"the value of the annotated parameter(s). {fortify_phrase}"
    )


def _alloc_size_fortify_phrase(build_flags: Optional[BuildFlagsContext]) -> str:
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "FORTIFY_SOURCE status unknown (build flags not in evidence); "
            "any runtime bounds-checking on the returned buffer depends "
            "on _FORTIFY_SOURCE level at compile time."
        )
    level = build_flags.fortify_source_level
    if level is None:
        return (
            "FORTIFY_SOURCE not set in observed flags; the alloc_size "
            "annotation gives the static-analyzer hint but no runtime "
            "bounds-check on the buffer."
        )
    if level >= 2:
        return (
            f"FORTIFY_SOURCE=_{level}_ — fortified intrinsics will "
            f"bounds-check operations against the returned buffer at "
            f"runtime; some overflows in the caller would be caught."
        )
    if level == 1:
        return (
            "FORTIFY_SOURCE=1 — limited runtime bounds-checking active; "
            "caller-side memcpy_chk catches overflows when the source "
            "length is also known statically."
        )
    return (
        "_FORTIFY_SOURCE=0 (explicitly disabled); annotation is "
        "static-analyzer-only, no runtime protection."
    )


def _render_returns_nonnull_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render returns_nonnull evidence.

    Author promises the function never returns NULL. Callers may
    legitimately skip null checks. If the annotation is wrong AND
    -fdelete-null-pointer-checks is enabled (gcc userspace default),
    the compiler may also dead-code-eliminate any defensive null
    checks the caller DID write — making a wrong annotation actively
    dangerous.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    caveat = _returns_nonnull_caveat_phrase(build_flags)

    if style == "stage_d":
        prefix = "Author claim — returns_nonnull"
    elif style == "exploit_plan":
        prefix = "Constraint — caller may skip null check on return"
    else:
        prefix = "Variant hint — returns_nonnull annotation"

    return (
        f"{prefix}: {fn_text} promises never to return NULL. {caveat}"
    )


def _returns_nonnull_caveat_phrase(
    build_flags: Optional[BuildFlagsContext],
) -> str:
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "Compiler-elimination status unknown (build flags not in "
            "evidence); if the annotation is wrong, a returned NULL "
            "may bypass defensive caller checks depending on "
            "-fdelete-null-pointer-checks."
        )
    if build_flags.delete_null_pointer_checks is False:
        return (
            "Build flags include -fno-delete-null-pointer-checks — "
            "defensive null checks in the caller are preserved even if "
            "the annotation is incorrect."
        )
    if build_flags.delete_null_pointer_checks is True:
        return (
            "Build flags enable -fdelete-null-pointer-checks — if the "
            "annotation is wrong, compiler may eliminate caller-side "
            "null checks, making a returned NULL a real deref."
        )
    return (
        "Compiler-elimination status not pinned by observed flags — "
        "default depends on -O level and compiler version."
    )


def _render_noreturn_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render noreturn evidence.

    Marks the function as a guaranteed abort (panic, _Exit, BUG-style).
    Strong DoS-vs-RCE discriminator: if the abort sits on the path
    between source and bug primitive, exploitation collapses to DoS.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    if style == "stage_d":
        prefix = "Control-flow signal — noreturn"
    elif style == "exploit_plan":
        prefix = "DoS-only constraint — noreturn function"
    else:
        prefix = "Variant hint — noreturn annotation"

    return (
        f"{prefix}: {fn_text} is declared noreturn. If this function "
        f"is invoked on the path between source and the bug primitive, "
        f"the program aborts before exploitation; the bug becomes DoS-only."
    )


def _render_malloc_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render malloc evidence.

    The annotation declares the function as an allocator — returned
    pointer is fresh and unaliased. The gcc 11+ paramised form
    `malloc(free_fn[, n])` pairs the allocator with its deallocator.
    Source_intel records the annotation; combined with alloc_size,
    the LLM can recognise allocator semantics even when the function
    name doesn't say "malloc".
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    if style == "stage_d":
        prefix = "Allocator signal — malloc"
    elif style == "exploit_plan":
        prefix = "Constraint — function declared as allocator"
    else:
        prefix = "Variant hint — malloc annotation"

    return (
        f"{prefix}: {fn_text} declared as an allocator — returned "
        f"pointer is fresh and unaliased per the annotation. May be "
        f"paired with a deallocator on gcc 11+ (`malloc(free_fn)`)."
    )


def _render_no_stack_protector_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render no_stack_protector evidence.

    This is an explicit HARDENING HOLE — the function opts out of the
    stack-canary insertion that -fstack-protector* would normally add.
    A stack buffer overflow in such a function bypasses the canary
    check entirely.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    sp_phrase = _stack_protector_phrase(build_flags)

    if style == "stage_d":
        prefix = "Hardening hole — no_stack_protector"
    elif style == "exploit_plan":
        prefix = "Constraint relaxed — no canary on this function"
    else:
        prefix = "Variant hint — no_stack_protector annotation"

    return (
        f"{prefix}: {fn_text} explicitly OPTS OUT of -fstack-protector. "
        f"A stack buffer overflow in this function bypasses the canary "
        f"check; saved return address reaches via overflow without "
        f"defence. {sp_phrase}"
    )


def _stack_protector_phrase(build_flags: Optional[BuildFlagsContext]) -> str:
    """Phrase describing what the build-wide stack protector level is —
    the no_stack_protector attribute matters most when the build was
    otherwise enabling canary insertion."""
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "Build-wide stack-protector level unknown; the opt-out "
            "matters most when the rest of the binary was canary-"
            "protected."
        )
    level = build_flags.stack_protector_level
    if level in ("strong", "all"):
        return (
            f"Build flags include -fstack-protector-{level} — most of "
            f"the binary has canary protection that this function "
            f"explicitly disables."
        )
    if level == "weak":
        return (
            "Build flags include -fstack-protector (weak); the opt-out "
            "matters for functions that would have qualified for "
            "canary insertion."
        )
    if level == "none":
        return (
            "Build-wide stack-protector is disabled (-fno-stack-protector); "
            "this function's opt-out is redundant — canary wasn't "
            "present anyway."
        )
    return (
        "Build-wide stack-protector status not pinned by observed flags."
    )


def _render_access_line(
    ev: AttributeEvidence,
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> str:
    """Render access evidence.

    Declares which pointer parameters are read-only / write-only /
    read-write, and optionally ties access width to another parameter.
    Combined with FORTIFY_SOURCE, this unlocks runtime bounds-checking
    on the annotated parameters.
    """
    fn_text = (
        f"function `{ev.function_name}`"
        if ev.function_name
        else f"function in {ev.location[0]} at line {ev.location[1]}"
    )

    fortify_phrase = _access_fortify_phrase(build_flags)

    if style == "stage_d":
        prefix = "Author intent + compiler signal — access"
    elif style == "exploit_plan":
        prefix = "Constraint — declared parameter access pattern"
    else:
        prefix = "Variant hint — access annotation"

    return (
        f"{prefix}: {fn_text} declares parameter access pattern "
        f"(read_only / write_only / read_write, possibly tied to a size "
        f"parameter). {fortify_phrase}"
    )


def _access_fortify_phrase(build_flags: Optional[BuildFlagsContext]) -> str:
    """Same FORTIFY_SOURCE caveat shape as alloc_size — annotations
    unlock runtime checks when fortified intrinsics are active."""
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "FORTIFY_SOURCE status unknown; whether the declared access "
            "pattern triggers runtime bounds-checking depends on the "
            "compile-time _FORTIFY_SOURCE level."
        )
    level = build_flags.fortify_source_level
    if level is None:
        return (
            "FORTIFY_SOURCE not set in observed flags; access annotation "
            "is static-analyzer-only, no runtime bounds-check enforcement."
        )
    if level >= 2:
        return (
            f"FORTIFY_SOURCE={level} — runtime bounds-checking active "
            f"on the annotated parameters; caller overflows would be "
            f"caught."
        )
    if level == 1:
        return (
            "FORTIFY_SOURCE=1 — limited runtime bounds-checking active."
        )
    return (
        "_FORTIFY_SOURCE=0 — annotation is static-analyzer-only."
    )


def _enforcement_phrase(build_flags: Optional[BuildFlagsContext]) -> str:
    """Compose the compile-enforcement caveat from build flag context."""
    if build_flags is None or build_flags.extraction_confidence == "absent":
        return (
            "Compile-enforcement status unknown (build flags not in "
            "evidence); advisory only."
        )
    if build_flags.werror_unused_result is True:
        return (
            "Build flags include -Werror=unused-result — "
            "compile-enforced; callers that ignore the return "
            "would not compile."
        )
    if build_flags.werror_unused_result is False:
        return (
            "Build flags include -Wno-error=unused-result — "
            "warning suppressed; advisory only."
        )
    return (
        "Build flags observed but -Werror=unused-result not set; "
        "advisory unless -Werror is added."
    )


# =====================================================================
# Axis 6 — sanitizer build context
# =====================================================================


# Sanitizer names worth surfacing in evidence — both userspace
# (-fsanitize=X) and kernel-config-derived. Restricted to those that
# materially change exploitability reasoning for memory-corruption
# CWEs. We deliberately drop sanitizers that only affect undefined-
# behaviour (UBSAN) for non-memory CWEs, because the prose framing
# below is memory-specific.
_RELEVANT_SANITIZERS = frozenset({
    "address",       # -fsanitize=address (userspace ASan)
    "kasan",         # CONFIG_KASAN (kernel ASan)
    "kfence",        # CONFIG_KFENCE
    "hwaddress",     # -fsanitize=hwaddress (HW-tag ASan)
    "memory",        # -fsanitize=memory (MSan — uninit reads)
    "thread",        # -fsanitize=thread (TSan — races)
    "undefined",     # -fsanitize=undefined (UBSan — int overflows etc.)
    "ubsan",         # CONFIG_UBSAN
    "kcsan",         # CONFIG_KCSAN
    "kcov",          # CONFIG_KCOV (not a sanitizer per se but the
                     # fuzzer-coverage runtime that often pairs with KASAN)
})


def _render_sanitizers_line(
    build_flags: Optional[BuildFlagsContext],
    style: str,
) -> Optional[str]:
    """Render a single line summarising active sanitizers when any
    are present in ``build_flags``. Returns None when:
      * build_flags is None,
      * extraction_confidence == "absent",
      * sanitizers_enabled is empty,
      * no enabled sanitizer is in ``_RELEVANT_SANITIZERS``.

    The line is target-wide, not per-call-site — sanitizers are a
    build-wide property. Consumer prompt may dedup if multiple
    evidence blocks for the same target are rendered together; in
    practice each finding gets its own evidence block, so one line
    per finding is correct.
    """
    if build_flags is None:
        return None
    if build_flags.extraction_confidence == "absent":
        return None
    enabled = tuple(
        s for s in build_flags.sanitizers_enabled
        if s in _RELEVANT_SANITIZERS
    )
    if not enabled:
        return None

    listed = ", ".join(enabled)
    if style == "stage_d":
        prefix = "Build-flag context — active sanitizers"
    elif style == "exploit_plan":
        prefix = "Constraint — sanitizers active in build"
    else:
        prefix = "Variant hint — sanitizer build"

    # Memory-corruption-CWE prose. Surfaces both interpretations the
    # LLM should weigh: production-equivalent KASAN catches the bug
    # at panic-cost (DoS-only outcome), but if the binary under
    # analysis is the sanitizer build itself, the finding may not
    # reproduce in stripped production binaries.
    return (
        f"{prefix}: {listed}. If this is a production-equivalent build "
        f"with these sanitizers active, memory-corruption primitives "
        f"reaching runtime trigger a panic / abort — the bug is "
        f"DoS-only, not RCE. If this is a CI / fuzzer build only, "
        f"the production binary lacks these checks and the primitive "
        f"survives uninstrumented."
    )


# =====================================================================
# Helpers
# =====================================================================


def _truncate(lines: List[str], max_lines: Optional[int]) -> List[str]:
    """Cap line count for tight prompt budgets."""
    if max_lines is None or len(lines) <= max_lines:
        return lines
    return lines[:max_lines]


def derive_mitigations_found(
    result: SourceIntelResult,
    finding_function: Optional[str] = None,
    finding_file: Optional[str] = None,
    finding_line: Optional[int] = None,
) -> List[Mitigation]:
    """Return the structured `mitigations_found` list for a finding.

    Per design strict invariant: a positively-detected mitigation
    earns an entry; ABSENCE earns no entry (don't emit
    ``hardened: False`` because we may have missed signal).

    Walks every evidence axis on ``result`` that could meaningfully
    indicate hardening / verdict-suppression:

      * axis_2 abort  — abort-class call in finding's function
        (`abort_dominates` if grade=dominates, `abort_proximate`
        for same_function/same_path)
      * axis_4 priv   — privileged capable() in finding's function
        (`priv_dominates`)
      * axis_6 fortify — FORTIFY_SOURCE active + fortified call
        (`fortify_intercepted`)
      * axis_7 dead   — function dead per PR-4 + static
        (`dead_code`)
      * axis_8 valid  — downstream relational+early-exit guard
        (`downstream_validation`)
      * axis_3 paired — alloc paired with free in function
        (`paired_free` — informational for leak findings)

    Each entry includes location when known so Stage D LLM can
    cross-reference the source.
    """
    mitigations: List[Mitigation] = []

    # axis_2 abort — same function as finding
    for ab in result.aborts:
        if (finding_function and ab.enclosing_function
                and ab.enclosing_function != finding_function):
            continue
        if ab.grade == GRADE_DOMINATES:
            confidence = "high"
            name = "abort_dominates"
        elif ab.grade == GRADE_SAME_PATH:
            confidence = "medium"
            name = "abort_on_path"
        else:
            confidence = "low"
            name = "abort_proximate"
        mitigations.append(Mitigation(
            name=name, axis="axis_2", confidence=confidence,
            detail=f"{ab.macro} ({ab.grade})",
            location=ab.location,
        ))

    # axis_4 privilege — capable() in same function
    for cap in result.capabilities:
        if (finding_function and cap.enclosing_function
                and cap.enclosing_function != finding_function):
            continue
        mitigations.append(Mitigation(
            name="privilege_gate",
            axis="axis_4", confidence="medium",
            detail=f"{cap.cap_function} (grade={cap.grade})",
            location=cap.location,
        ))

    # axis_6 FORTIFY — surface only when level present
    if result.build_flags and result.build_flags.fortify_source_level:
        level = result.build_flags.fortify_source_level
        confidence = "high" if level >= 2 else "medium"
        mitigations.append(Mitigation(
            name="fortify_source",
            axis="axis_6", confidence=confidence,
            detail=f"_FORTIFY_SOURCE={level} ({result.build_flags.source})",
            location=None,
        ))

    # axis_3 paired-free — informational for cpp/memory-leak FPs
    for pf in result.paired_frees:
        if finding_function and pf.enclosing_function != finding_function:
            continue
        mitigations.append(Mitigation(
            name="paired_free",
            axis="axis_3", confidence="medium",
            detail=f"{pf.allocator} paired with {pf.free_fn}",
            location=pf.location,
        ))

    # axis_2 sub-class: warn-class is informational, not a real
    # mitigation; null-guards likewise (axis-3's `when !=` does the
    # verdict work). We don't emit these as mitigations to avoid
    # false-confidence in Stage D output.

    return mitigations


def aggregate_confidence(mitigations: List[Mitigation]) -> str:
    """Compute overall confidence per design strict invariant:
    "confidence capped at strongest individual signal; no
    multiplicative inflation".

    Multiple mediums do NOT combine into a high. Multiple highs
    don't combine into something stronger than high. The single
    strongest signal wins.

    Returns one of: "high" | "medium" | "low" | "none" (no
    evidence at all).
    """
    if not mitigations:
        return "none"
    ranks = {"high": 3, "medium": 2, "low": 1}
    best = max(ranks.get(m.confidence, 0) for m in mitigations)
    return next(
        (k for k, v in ranks.items() if v == best),
        "none",
    )
