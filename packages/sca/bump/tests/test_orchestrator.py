"""Tests for ``packages.sca.bump.orchestrator``.

End-to-end-ish: stub upstream / registry clients to avoid network,
exercise the candidate enumeration + verdict + apply paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from packages.sca.bump.orchestrator import (
    BumpCandidate, BumpReport, BumpResult,
    _VERDICT_BLOCK, _VERDICT_CLEAN, _VERDICT_REVIEW,
    render_report, run_bump,
)


# ---------------------------------------------------------------------------
# Stub HTTP — replies with operator-supplied JSON per URL.
# ---------------------------------------------------------------------------

class _StubResp:
    def __init__(self, body: dict, status=200):
        self._body = body
        self.status_code = status
        self.headers: Dict[str, str] = {}

    @property
    def content(self):
        import json
        return json.dumps(self._body).encode()


class _StubHttp:
    def __init__(self, responses: Dict[str, Any]):
        self._responses = responses

    def get_json(self, url: str, **kw):
        if url in self._responses:
            return self._responses[url]
        from core.http import HttpError
        raise HttpError(f"stub: no payload for {url}")

    def request(self, method, url, **kw):
        if url in self._responses:
            return _StubResp(self._responses[url])
        from core.http import HttpError
        raise HttpError(f"stub: no payload for {url}")


class _StubPyPI:
    def __init__(self, packages):
        self._p = packages

    def get_metadata(self, name):
        return self._p.get(name)


class _StubNpm:
    def __init__(self, packages):
        self._p = packages

    def get_metadata(self, name):
        return self._p.get(name)


# ---------------------------------------------------------------------------
# Discovery + candidate enumeration
# ---------------------------------------------------------------------------

def test_no_dockerfiles_returns_empty_report(tmp_path: Path) -> None:
    """Target with no Dockerfile → empty report (no error)."""
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    assert report.candidates == []
    assert report.results == []


def test_dockerfile_with_unknown_arg_skipped(tmp_path: Path) -> None:
    """ARG names not in the upstream-source map are silently
    skipped — operator can add via inline-comment override."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SOME_INTERNAL_VERSION=1.0\n"
    )
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    assert report.candidates == []
    assert report.results == []


def test_dockerfile_with_known_arg_at_latest_no_candidate(
    tmp_path: Path,
) -> None:
    """ARG already at upstream-latest → not a candidate. Avoids
    proposing identity bumps."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SEMGREP_VERSION=1.119.0\n"
    )
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    report = run_bump(tmp_path, http=http)
    assert report.candidates == []


def test_dockerfile_with_known_arg_below_latest_becomes_candidate(
    tmp_path: Path,
) -> None:
    """ARG below upstream-latest → candidate emitted; verdict
    computed."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SEMGREP_VERSION=1.50.0\n"
    )
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            # Published over 30 days ago — recent_publish silent
            "1.119.0": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now,
    )
    assert len(report.candidates) == 1
    c = report.candidates[0]
    assert c.arg_name == "SEMGREP_VERSION"
    assert c.current_version == "1.50.0"
    assert c.target_version == "1.119.0"
    # Verdict: Clean (no bump-tier signals fired — old enough).
    assert report.results[0].verdict == _VERDICT_CLEAN


def test_dockerfile_recent_publish_target_review_not_clean(
    tmp_path: Path,
) -> None:
    """Target published <30 days ago → recent_publish medium →
    Review (not Clean)."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SEMGREP_VERSION=1.50.0\n"
    )
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2026-05-09T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now,
    )
    assert report.results[0].verdict == _VERDICT_REVIEW
    # And the recent_publish finding is in the result for PR-comment
    # rendering / operator visibility.
    kinds = [f.kind for f in report.results[0].bump_supply_chain_findings]
    assert "recent_publish" in kinds


def test_upstream_lookup_failure_records_in_skipped(
    tmp_path: Path,
) -> None:
    """When the GitHub releases endpoint returns 404 (project
    doesn't cut releases), the ARG is recorded in ``skipped``
    so the operator sees the gap."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SEMGREP_VERSION=1.50.0\n"
    )
    http = _StubHttp({})    # everything 404s
    report = run_bump(tmp_path, http=http)
    assert report.candidates == []
    assert len(report.skipped) == 1
    arg, path, reason = report.skipped[0]
    assert arg == "SEMGREP_VERSION"
    assert "upstream lookup failed" in reason


# ---------------------------------------------------------------------------
# Apply path
# ---------------------------------------------------------------------------

def test_apply_writes_clean_bumps_in_place(tmp_path: Path) -> None:
    """``apply=True`` rewrites the Dockerfile when verdict is
    Clean."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("ARG SEMGREP_VERSION=1.50.0\n")
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now, apply=True,
    )
    # Verdict Clean + apply → rewrite applied.
    assert report.results[0].rewrite_result is not None
    assert report.results[0].rewrite_result.applied
    # File contents updated in place.
    assert "1.119.0" in dockerfile.read_text()
    assert "1.50.0" not in dockerfile.read_text()


def test_apply_does_not_write_review_bumps(tmp_path: Path) -> None:
    """``apply=True`` honours the suggest-only policy: Review /
    Block bumps do NOT get auto-written, even with --apply.
    Operator review required."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("ARG SEMGREP_VERSION=1.50.0\n")
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2026-05-09T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now, apply=True,
    )
    assert report.results[0].verdict == _VERDICT_REVIEW
    assert report.results[0].rewrite_result is None
    # File untouched.
    assert dockerfile.read_text() == "ARG SEMGREP_VERSION=1.50.0\n"


def test_apply_default_is_dry_run(tmp_path: Path) -> None:
    """Default ``apply=False`` → no writes even for Clean
    verdicts. The dry-run produces the verdict report; the
    operator decides whether to ``--apply``."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("ARG SEMGREP_VERSION=1.50.0\n")
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now,
    )
    assert report.results[0].verdict == _VERDICT_CLEAN
    assert report.results[0].rewrite_result is None
    assert dockerfile.read_text() == "ARG SEMGREP_VERSION=1.50.0\n"


# ---------------------------------------------------------------------------
# Render report
# ---------------------------------------------------------------------------

def test_render_report_shape_and_findings_in_table(tmp_path: Path) -> None:
    """The text report shows ARG / current / target / verdict
    per row, plus inline supply-chain findings for non-Clean
    verdicts (so operators see WHY)."""
    (tmp_path / "Dockerfile").write_text(
        "ARG SEMGREP_VERSION=1.50.0\n"
    )
    http = _StubHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2026-05-10T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now,
    )
    text = render_report(report)
    assert "SEMGREP_VERSION" in text
    assert "1.50.0" in text
    assert "1.119.0" in text
    assert "Review" in text
    # Inline finding annotation visible.
    assert "recent_publish" in text


def test_render_report_no_candidates_message(tmp_path: Path) -> None:
    """Friendly message when there are no candidates."""
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    text = render_report(report)
    assert "no bump candidates found" in text


# ---------------------------------------------------------------------------
# Cross-Dockerfile upstream-lookup deduplication
# ---------------------------------------------------------------------------

class _CountingHttp(_StubHttp):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.calls: List[str] = []

    def get_json(self, url: str, **kw):
        self.calls.append(url)
        return super().get_json(url, **kw)


# ---------------------------------------------------------------------------
# FROM image refs
# ---------------------------------------------------------------------------

def _tags_response(tags):
    import json
    return _StubResp({"name": "ignored", "tags": tags})


def test_from_image_with_clean_semver_tag_becomes_candidate(
    tmp_path: Path,
) -> None:
    """``FROM python:3.11`` → OCI tag lookup → bump candidate
    to highest stable tag."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11\n"
    )
    http = _StubHttp({
        "https://registry-1.docker.io/v2/library/python/tags/list?n=100":
            {"name": "library/python",
             "tags": ["3.11", "3.12", "3.13"]},
    })
    report = run_bump(tmp_path, http=http)
    from_cands = [c for c in report.candidates if c.kind == "from_image"]
    assert len(from_cands) == 1
    cand = from_cands[0]
    assert cand.locator == "docker.io/library/python"
    assert cand.current_version == "3.11"
    assert cand.target_version == "3.13"
    # No bump-tier signals available for OCI yet → Clean.
    matching_result = [r for r in report.results
                        if r.candidate is cand][0]
    assert matching_result.verdict == _VERDICT_CLEAN


def test_from_image_variant_tag_silently_skipped(tmp_path: Path) -> None:
    """``FROM python:3.12-bookworm`` — variant tag, not a clean
    semver. The walker skips silently (no bump-tier signal we
    can apply to a variant choice). Not in candidates, not in
    skipped."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-bookworm\n"
    )
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates
             if c.kind == "from_image"] == []
    # Not in skipped either — silent skip because we don't have
    # an upstream-latest path for variant tags.
    assert all(s[0] != "docker.io/library/python"
                for s in report.skipped)


def test_from_image_digest_pinned_silently_skipped(tmp_path: Path) -> None:
    """Digest-pinned FROM is immutable — not a bump target."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11@sha256:abc123\n"
    )
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates
             if c.kind == "from_image"] == []


def test_from_image_stage_reuse_skipped(tmp_path: Path) -> None:
    """Multi-stage builds: ``FROM build AS runtime`` (where
    ``build`` is a prior stage name, not an image) shouldn't be
    bump-attempted."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11 AS build\n"
        "RUN do-build\n"
        "FROM build AS runtime\n"
    )
    http = _StubHttp({
        "https://registry-1.docker.io/v2/library/python/tags/list?n=100":
            {"name": "library/python",
             "tags": ["3.11", "3.12"]},
    })
    report = run_bump(tmp_path, http=http)
    from_cands = [c for c in report.candidates if c.kind == "from_image"]
    assert len(from_cands) == 1     # python only, not the stage reuse
    assert from_cands[0].locator == "docker.io/library/python"


def test_from_image_already_at_latest_not_a_candidate(tmp_path: Path) -> None:
    """FROM at highest stable tag → not a bump target."""
    (tmp_path / "Dockerfile").write_text("FROM python:3.13\n")
    http = _StubHttp({
        "https://registry-1.docker.io/v2/library/python/tags/list?n=100":
            {"name": "library/python",
             "tags": ["3.11", "3.12", "3.13"]},
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates
             if c.kind == "from_image"] == []


def test_from_image_apply_writes_dockerfile(tmp_path: Path) -> None:
    """End-to-end with --apply: FROM gets rewritten in place."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.11\n")
    http = _StubHttp({
        "https://registry-1.docker.io/v2/library/python/tags/list?n=100":
            {"name": "library/python",
             "tags": ["3.11", "3.12", "3.13"]},
    })
    report = run_bump(tmp_path, http=http, apply=True)
    assert "FROM python:3.13" in dockerfile.read_text()


def test_mixed_arg_and_from_in_one_dockerfile(tmp_path: Path) -> None:
    """A devcontainer-shaped Dockerfile with both an ARG pin AND
    a FROM image — both surface as candidates."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.11\n"
        "ARG SEMGREP_VERSION=1.50.0\n"
    )
    http = _StubHttp({
        "https://registry-1.docker.io/v2/library/python/tags/list?n=100":
            {"name": "library/python", "tags": ["3.11", "3.12"]},
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    from datetime import datetime, timezone
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(tmp_path, http=http, pypi_client=pypi, now=now)
    by_kind = {c.kind for c in report.candidates}
    assert by_kind == {"arg", "from_image"}


# ---------------------------------------------------------------------------
# GHA uses refs (Phase 3.b)
# ---------------------------------------------------------------------------

def _workflow(tmp_path: Path, name: str, body: str) -> Path:
    wf = tmp_path / ".github" / "workflows" / name
    wf.parent.mkdir(parents=True, exist_ok=True)
    wf.write_text(body)
    return wf


def test_gha_tag_pinned_uses_becomes_candidate(tmp_path: Path) -> None:
    """Tag-pinned ``uses: foo/bar@v4`` with newer upstream
    release → bump candidate."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    gha_cands = [c for c in report.candidates if c.kind == "gha_uses"]
    assert len(gha_cands) == 1
    c = gha_cands[0]
    assert c.locator == "actions/checkout"
    assert c.current_version == "v4"
    assert c.target_version == "v5"


def test_gha_sha_pinned_uses_skipped(tmp_path: Path) -> None:
    """SHA-pinned ``uses: foo/bar@<40hex>`` — Phase 3.b skips
    silently (3.b.2 will handle SHA+comment with tag→SHA
    resolution)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@"
              "de0fac2e4500dabe0009e67214ff5f5447ce83dd  # was v6\n")
    http = _StubHttp({})    # no upstream fetched
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_branch_pinned_uses_skipped(tmp_path: Path) -> None:
    """Branch-pinned ``uses: foo/bar@main`` — out of scope for
    auto-bumper (would be a security upgrade, not a bump)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@main\n")
    http = _StubHttp({})
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_major_only_pin_no_same_major_bump(tmp_path: Path) -> None:
    """``uses: foo/bar@v4`` with upstream-latest ``v4.2.1`` —
    no candidate. Operator chose major-only pinning explicitly;
    proposing a same-major specific-version roll would be
    unwanted churn."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v4.2.1"},
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_major_only_pin_major_bump_is_a_candidate(tmp_path: Path) -> None:
    """``uses: foo/bar@v4`` with upstream-latest ``v5`` →
    candidate (cross-major bump is a real change)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    assert len([c for c in report.candidates
                 if c.kind == "gha_uses"]) == 1


def test_gha_sub_action_path_walker(tmp_path: Path) -> None:
    """``uses: github/codeql-action/init@v4`` — locator should
    be ``github/codeql-action`` (the repo, without the subpath)
    so the upstream lookup hits the right GitHub repo."""
    _workflow(tmp_path, "codeql.yml",
              "      - uses: github/codeql-action/init@v4\n"
              "      - uses: github/codeql-action/analyze@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/github/codeql-action/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    gha_cands = [c for c in report.candidates if c.kind == "gha_uses"]
    # Both ``init`` and ``analyze`` sub-actions surface the
    # same repo (same locator). Walker dedup via cache means we
    # only hit the upstream once, but each subpath line is its
    # own candidate.
    assert len(gha_cands) == 2
    assert all(c.locator == "github/codeql-action" for c in gha_cands)


def test_gha_already_at_latest_no_candidate(tmp_path: Path) -> None:
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@v5\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_apply_writes_workflow_file(tmp_path: Path) -> None:
    """End-to-end: ``--apply`` rewrites the workflow YAML."""
    wf = _workflow(tmp_path, "ci.yml",
                    "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    run_bump(tmp_path, http=http, apply=True)
    assert "uses: actions/checkout@v5" in wf.read_text()


# ---------------------------------------------------------------------------
# GHA SHA-pinned with ``# was vX`` comment (Phase 3.b.2)
# ---------------------------------------------------------------------------

def test_gha_sha_pinned_with_comment_becomes_candidate(tmp_path: Path) -> None:
    """Raptor's convention: SHA-pinned + ``# was vX`` comment.
    Walker detects the shape, looks up upstream-latest tag,
    resolves to target SHA, emits candidate with both."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@"
              "de0fac2e4500dabe0009e67214ff5f5447ce83dd  # was v6\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v7"},
        "https://api.github.com/repos/actions/checkout/git/refs/tags/v7":
            {"object": {"type": "commit",
                         "sha": "ffffffffffffffffffffffffffffffffffffffff"}},
    })
    report = run_bump(tmp_path, http=http)
    gha_cands = [c for c in report.candidates if c.kind == "gha_uses"]
    assert len(gha_cands) == 1
    c = gha_cands[0]
    assert c.locator == "actions/checkout"
    assert c.current_version == "v6"
    assert c.target_version == "v7"
    assert c.extra["old_sha"] == "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
    assert c.extra["new_sha"] == "f" * 40


def test_gha_sha_pinned_apply_writes_both_sha_and_comment(
    tmp_path: Path,
) -> None:
    """End-to-end: ``--apply`` on a SHA-pinned with ``# was vX``
    rewrites both the SHA and the comment tag."""
    wf = _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@"
              "0000000000000000000000000000000000000000  # was v6\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v7"},
        "https://api.github.com/repos/actions/checkout/git/refs/tags/v7":
            {"object": {"type": "commit", "sha": "1" * 40}},
    })
    run_bump(tmp_path, http=http, apply=True)
    text = wf.read_text()
    assert "@" + "1" * 40 in text
    assert "# was v7" in text
    assert "0000" not in text


def test_gha_sha_pinned_already_at_latest_no_candidate(tmp_path: Path) -> None:
    """SHA-pinned at latest tag → not a candidate (the bumper
    correctly handles the same-tag-but-different-SHA edge — only
    if upstream actually advanced)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@"
              "1111111111111111111111111111111111111111  # was v7\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v7"},
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_sha_pinned_same_major_pin_skipped(tmp_path: Path) -> None:
    """``# was v4`` and target is v4.x → same-major; skip
    (operator chose major-only pinning)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@"
              "0000000000000000000000000000000000000000  # was v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v4.2.1"},
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []


def test_gha_releases_latest_non_semver_falls_back_to_tags(
    tmp_path: Path,
) -> None:
    """``github/codeql-action``-shaped case: ``releases/latest``
    returns ``codeql-bundle-v2.25.4`` (a stable release but
    non-semver tag shape). The bumper can't substitute that for
    a ``v4`` pin; it should fall through to ``/tags`` and pick
    the highest stable-semver tag from there.

    Pre-fix the bumper proposed
    ``v4 → codeql-bundle-v2.25.4`` which would have produced an
    invalid pin. Live-output regression from raptor's actual
    workflow scan."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: github/codeql-action/init@v4\n")
    http = _StubHttp({
        # /releases/latest returns the non-semver bundle tag.
        "https://api.github.com/repos/github/codeql-action/releases/latest":
            {"tag_name": "codeql-bundle-v2.25.4"},
        # /tags has both bundle tags (skipped) AND stable-semver tags.
        "https://api.github.com/repos/github/codeql-action/tags?per_page=100":
            [
                {"name": "codeql-bundle-v2.25.4"},   # non-semver — skip
                {"name": "v5"},                       # stable — winner
                {"name": "v4.30.6"},
                {"name": "v4"},
            ],
    })
    report = run_bump(tmp_path, http=http)
    gha_cands = [c for c in report.candidates if c.kind == "gha_uses"]
    assert len(gha_cands) == 1
    # MUST be the stable-semver candidate, NOT codeql-bundle.
    assert gha_cands[0].target_version == "v5"
    assert "codeql-bundle" not in gha_cands[0].target_version


def test_gha_releases_latest_and_tags_both_non_semver_skipped(
    tmp_path: Path,
) -> None:
    """When NEITHER /releases/latest NOR /tags produces a
    stable-semver tag, the repo lands in ``skipped`` with a
    clear reason. Operator sees the gap rather than the bumper
    proposing a non-semver pin."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: weird/project@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/weird/project/releases/latest":
            {"tag_name": "release-2026-q1"},
        "https://api.github.com/repos/weird/project/tags?per_page=100":
            [{"name": "release-2026-q1"}, {"name": "rc-build-7"}],
    })
    report = run_bump(tmp_path, http=http)
    assert [c for c in report.candidates if c.kind == "gha_uses"] == []
    # Should be in skipped with explanatory reason.
    skipped_reasons = [s[2] for s in report.skipped
                        if s[0] == "weird/project"]
    assert len(skipped_reasons) == 1
    assert "non-semver" in skipped_reasons[0]


def test_gha_upstream_404_falls_back_to_tags(tmp_path: Path) -> None:
    """Some actions don't cut releases. Walker falls back to
    /tags (we already shipped ``latest_tag`` in Phase 2.a)."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: anthropics/claude-code@v2.0\n")
    # /releases/latest 404s; /tags returns a list.
    http = _StubHttp({
        "https://api.github.com/repos/anthropics/claude-code/tags?per_page=100":
            [{"name": "v2.1"}, {"name": "v2.0"}],
    })
    report = run_bump(tmp_path, http=http)
    gha_cands = [c for c in report.candidates if c.kind == "gha_uses"]
    assert len(gha_cands) == 1
    assert gha_cands[0].current_version == "v2.0"
    assert gha_cands[0].target_version == "v2.1"


# ---------------------------------------------------------------------------
# Render-side deduplication (Followup B)
# ---------------------------------------------------------------------------

def test_render_dedups_identical_candidates_across_files(
    tmp_path: Path,
) -> None:
    """When 3 workflow files all pin actions/checkout@v4, the
    rendered report shows ONE row with ``(3 files)`` — not three
    identical rows. The underlying ``results`` list still has
    three entries so --apply touches all three files.

    Pre-fix raptor's bump output showed 8 CODEQL_VERSION rows
    and 3 github/codeql-action rows; operators read it as
    duplicate noise."""
    for name in ("a.yml", "b.yml", "c.yml"):
        _workflow(tmp_path, name,
                  "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    # Underlying results: 3 (one per file).
    gha_results = [r for r in report.results
                    if r.candidate.kind == "gha_uses"]
    assert len(gha_results) == 3
    # Rendered: one row with "(3 files)" annotation.
    text = render_report(report)
    # Filter to candidate ROWS (kind=gha_uses appears in the
    # candidate rows but also in the header — exclude header
    # by looking for the locator field directly).
    rows = [l for l in text.splitlines() if "actions/checkout" in l]
    assert len(rows) == 1, (
        f"expected 1 deduped row; got {len(rows)}: {rows}"
    )
    # The result column carries the file count.
    assert "(3 files)" in rows[0]


def test_render_applied_count_in_dedup_row(tmp_path: Path) -> None:
    """When --apply runs, the dedup row shows ``applied (N
    files)`` rather than just ``applied``."""
    for name in ("a.yml", "b.yml"):
        _workflow(tmp_path, name,
                  "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http, apply=True)
    text = render_report(report)
    # All files applied → "applied (2 files)" suffix.
    assert "applied (2 files)" in text


def test_render_single_file_still_shows_filename_count(
    tmp_path: Path,
) -> None:
    """A non-duplicated row still renders cleanly without the
    file-count noise — single-file candidates were already fine
    pre-fix; preserve that."""
    _workflow(tmp_path, "ci.yml",
              "      - uses: actions/checkout@v4\n")
    http = _StubHttp({
        "https://api.github.com/repos/actions/checkout/releases/latest":
            {"tag_name": "v5"},
    })
    report = run_bump(tmp_path, http=http)
    text = render_report(report)
    # Single-file candidate: no "(N files)" suffix (renders empty
    # Result column).
    rows = [l for l in text.splitlines() if "actions/checkout" in l]
    assert len(rows) == 1
    assert "(1 file)" not in rows[0]
    assert "(2 files)" not in rows[0]


def test_upstream_lookup_dedups_across_dockerfiles(tmp_path: Path) -> None:
    """Two Dockerfiles both pinning SEMGREP_VERSION should hit
    the upstream-latest endpoint ONCE — the orchestrator caches
    per (kind, coordinate) within a single run."""
    (tmp_path / "Dockerfile").write_text("ARG SEMGREP_VERSION=1.50.0\n")
    (tmp_path / "Dockerfile.dev").write_text("ARG SEMGREP_VERSION=1.50.0\n")
    http = _CountingHttp({
        "https://api.github.com/repos/semgrep/semgrep/releases/latest":
            {"tag_name": "v1.119.0"},
    })
    pypi = _StubPyPI({
        "semgrep": {"releases": {
            "1.119.0": [{"upload_time_iso_8601": "2025-12-01T00:00:00Z"}],
        }},
    })
    now = datetime(2026, 5, 11, tzinfo=timezone.utc)
    report = run_bump(
        tmp_path, http=http, pypi_client=pypi, now=now,
    )
    assert len(report.candidates) == 2
    # ONE HTTP call to GitHub releases despite TWO Dockerfiles.
    gh_calls = [
        c for c in http.calls
        if "api.github.com" in c
    ]
    assert len(gh_calls) == 1
