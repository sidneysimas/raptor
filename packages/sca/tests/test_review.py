"""Tests for ``packages.sca.review`` (the ``raptor-sca check`` subcommand)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from packages.sca import review
from core.json import JsonCache
from packages.sca.osv import OSV_QUERY_BATCH_URL, OSV_VULN_URL_TEMPLATE


_LODASH_VULN_RECORD = {
    "id": "GHSA-jf85-cpcp-j695",
    "modified": "2024-01-01T00:00:00Z",
    "aliases": ["CVE-2019-10744"],
    "summary": "Prototype pollution in lodash",
    "details": "",
    "affected": [{
        "package": {"ecosystem": "npm", "name": "lodash"},
        "ranges": [{"type": "ECOSYSTEM",
                    "events": [{"introduced": "0"},
                               {"fixed": "4.17.12"}]}],
    }],
    "severity": [{"type": "CVSS_V3",
                  "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
    "references": [],
    "fixed_versions": ["4.17.12"],
}

_LOG4SHELL_RECORD = {
    "id": "GHSA-jfh8-c2jp-5v3q",
    "modified": "2024-01-01T00:00:00Z",
    "aliases": ["CVE-2021-44228"],
    "summary": "Log4Shell",
    "details": "",
    "affected": [{
        "package": {"ecosystem": "Maven",
                    "name": "org.apache.logging.log4j:log4j-core"},
        "ranges": [{"type": "ECOSYSTEM",
                    "events": [{"introduced": "2.0-beta9"},
                               {"fixed": "2.15.0"}]}],
    }],
    "severity": [{"type": "CVSS_V3",
                  "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
    "references": [],
}


class StubHttp:
    def __init__(self, posts: Dict[Any, Any] | None = None,
                 gets: Dict[str, Any] | None = None) -> None:
        self.posts: List[tuple] = []
        self.gets: List[str] = []
        self._post_response = posts or {"results": [{"vulns": []}]}
        self._get_responses = gets or {}

    def post_json(self, url, body, timeout=30, **kwargs):
        self.posts.append((url, body))
        return self._post_response

    def get_json(self, url, timeout=30, **kwargs):
        self.gets.append(url)
        if url in self._get_responses:
            return self._get_responses[url]
        if "cisa.gov" in url:
            return {"vulnerabilities": []}
        if "first.org" in url:
            return {"data": []}
        raise RuntimeError(f"unexpected GET {url}")

    def get_bytes(self, *a, **k):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def test_clean_dep_returns_zero(tmp_path: Path, capsys) -> None:
    """A safe version with no advisories and no typosquat → exit 0.

    Uses ``--no-transitive`` to skip the registry-metadata walk; the
    StubHttp doesn't model registry responses, and an unknown registry
    URL would otherwise trigger the seed-metadata-unverifiable warning.
    """
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(
        ["npm", "@types/node", "20.10.5",
         "--no-transitive",
         "--out", str(tmp_path / "r.md")],
        http=http, cache=cache,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "**Verdict:** Clean" in out
    assert "No advisories found" in out


def test_unknown_ecosystem_returns_2(tmp_path: Path, capsys) -> None:
    """Unrecognised ecosystem rejected before any OSV call."""
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(["Bogus", "requests", "0.1.0"], http=http, cache=cache)
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown ecosystem" in err


def test_lowercase_ecosystem_canonicalised(tmp_path: Path, capsys) -> None:
    """Lowercase ecosystem is canonicalised to the OSV-accepted form
    so the OSV query actually returns advisories.
    """
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(
        ["pypi", "requests", "2.31.0", "--no-transitive"],
        http=http, cache=cache,
    )
    # OSV would have been called with PyPI (canonical); StubHttp returns
    # no advisories, so verdict is Clean.
    assert rc == 0
    # Verify we sent PyPI (not pypi) to OSV.
    posts = http.posts
    assert any(
        any(q.get("package", {}).get("ecosystem") == "PyPI"
            for q in body.get("queries", []))
        for _url, body in posts
    )


def test_seed_metadata_unverifiable_escalates_to_review(
    tmp_path: Path, capsys,
) -> None:
    """When the registry can't confirm the package exists, escalate
    an otherwise-clean verdict to Review.
    """
    # StubHttp with no advisories AND no registry responses → seed
    # walk fails → seed_metadata_unverifiable=True.
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(
        ["npm", "nonexistent-package-xyz123", "1.0.0"],
        http=http, cache=cache,
    )
    # Verdict escalated from Clean to Review (exit 1).
    assert rc == 1
    out = capsys.readouterr().out
    assert "**Verdict:** Review" in out
    assert "could not confirm" in out


def test_existence_probe_runs_under_no_transitive(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    """The existence probe runs even when --no-transitive is set, so
    nonexistent packages are still escalated to Review.
    """
    http = StubHttp()
    cache = JsonCache(root=tmp_path)

    # Stub package_version_exists to return False (404), simulating a
    # nonexistent package without needing real network.
    from packages.sca import registry_metadata_walk
    monkeypatch.setattr(
        registry_metadata_walk, "package_version_exists",
        lambda *a, **kw: False,
    )
    rc = review.main(
        ["PyPI", "nonexistent-pkg-xyz", "1.0.0", "--no-transitive"],
        http=http, cache=cache,
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "**Verdict:** Review" in out
    assert "## Existence" in out
    assert "could not confirm" in out


def test_existence_probe_skipped_under_offline(
    tmp_path: Path, capsys, monkeypatch,
) -> None:
    """--offline skips the existence probe (no network available).
    Operators get Clean for nonexistent packages in offline mode;
    that's the documented trade-off.
    """
    http = StubHttp()
    cache = JsonCache(root=tmp_path)

    # The probe shouldn't be called under --offline; assert that.
    from packages.sca import registry_metadata_walk
    called = []
    def _record(*a, **kw):
        called.append(True)
        return False
    monkeypatch.setattr(
        registry_metadata_walk, "package_version_exists", _record,
    )
    rc = review.main(
        ["PyPI", "anything-x-y-z", "1.0.0", "--no-transitive", "--offline"],
        http=http, cache=cache,
    )
    assert rc == 0
    assert called == [], "probe should not run under --offline"


def test_kev_listed_dep_returns_block(tmp_path: Path, capsys) -> None:
    http = StubHttp(
        posts={"results": [{"vulns": [{"id": "GHSA-jfh8-c2jp-5v3q"}]}]},
        gets={
            OSV_VULN_URL_TEMPLATE.format("GHSA-jfh8-c2jp-5v3q"):
                _LOG4SHELL_RECORD,
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json":
                {"vulnerabilities": [{"cveID": "CVE-2021-44228"}]},
        },
    )
    cache = JsonCache(root=tmp_path)
    rc = review.main(
        ["Maven", "org.apache.logging.log4j:log4j-core", "2.14.1"],
        http=http, cache=cache,
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "**Verdict:** Block" in out
    assert "**KEV**" in out


def test_unfixable_critical_returns_block(tmp_path: Path, capsys) -> None:
    """A critical CVE without a fixed_versions entry blocks even
    without KEV listing."""
    record = dict(_LODASH_VULN_RECORD)
    record["affected"] = [{
        "package": {"ecosystem": "npm", "name": "lodash"},
        "ranges": [{"type": "ECOSYSTEM",
                    "events": [{"introduced": "0"}]}],   # no fixed event
    }]
    http = StubHttp(
        posts={"results": [{"vulns": [{"id": "GHSA-jf85-cpcp-j695"}]}]},
        gets={OSV_VULN_URL_TEMPLATE.format("GHSA-jf85-cpcp-j695"): record},
    )
    cache = JsonCache(root=tmp_path)
    rc = review.main(["npm", "lodash", "4.17.4"],
                     http=http, cache=cache)
    assert rc == 2
    assert "**Verdict:** Block" in capsys.readouterr().out


def test_typosquat_distance_one_blocks(tmp_path: Path, capsys) -> None:
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(["npm", "loadash", "1.0.0"],
                     http=http, cache=cache)
    assert rc == 2
    out = capsys.readouterr().out
    assert "Typosquat candidate" in out
    assert "**Verdict:** Block" in out


def test_typosquat_distance_two_returns_review(tmp_path: Path, capsys) -> None:
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(["npm", "lodaasch", "1.0.0"],
                     http=http, cache=cache)
    assert rc == 1
    assert "**Verdict:** Review" in capsys.readouterr().out


def test_advisory_with_fix_returns_review(tmp_path: Path, capsys) -> None:
    """A high-sev CVE with an upgrade path is a review, not a block."""
    http = StubHttp(
        posts={"results": [{"vulns": [{"id": "GHSA-jf85-cpcp-j695"}]}]},
        gets={OSV_VULN_URL_TEMPLATE.format("GHSA-jf85-cpcp-j695"):
              _LODASH_VULN_RECORD},
    )
    cache = JsonCache(root=tmp_path)
    rc = review.main(["npm", "lodash", "4.17.4"],
                     http=http, cache=cache)
    assert rc == 1
    out = capsys.readouterr().out
    assert "**Verdict:** Review" in out
    assert "Fix available: **4.17.12**" in out


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

def test_writes_report_to_out_when_supplied(tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "review.md"
    http = StubHttp()
    cache = JsonCache(root=tmp_path / "cache")
    review.main(["npm", "@types/node", "20.10.5", "--out", str(out_path)],
                http=http, cache=cache)
    assert out_path.exists()
    contents = out_path.read_text()
    assert "**Verdict:**" in contents
    # stdout still received the same body.
    assert capsys.readouterr().out == contents


def test_offline_mode_skips_network(tmp_path: Path, capsys) -> None:
    http = StubHttp()
    cache = JsonCache(root=tmp_path)
    rc = review.main(["npm", "@types/node", "20.10.5", "--offline"],
                     http=http, cache=cache)
    assert rc == 0
    assert http.posts == []
    assert http.gets == []


def test_purl_includes_ecosystem_lowercase(tmp_path: Path, capsys) -> None:
    """The header line shows a canonical purl so operators can paste
    it into other tools."""
    review.main(["PyPI", "django", "2.0.0"],
                http=StubHttp(), cache=JsonCache(root=tmp_path))
    out = capsys.readouterr().out
    assert "pkg:pypi/django@2.0.0" in out


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_missing_args_returns_2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        review.main(["npm"], http=StubHttp(), cache=JsonCache(root=tmp_path))
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Transitive surface (registry-metadata walk)
# ---------------------------------------------------------------------------

def test_transitive_walk_runs_by_default(tmp_path: Path, capsys) -> None:
    """Default: review walks one level of declared deps so the
    operator sees the full install surface, not just the named pkg."""
    http = StubHttp(gets={
        "https://pypi.org/pypi/django/2.0.0/json": {
            "info": {"requires_dist": ["pytz>=2017.2"]},
        },
        "https://pypi.org/pypi/pytz/2017.2/json": {
            "info": {"requires_dist": []},
        },
    })
    review.main(["PyPI", "django", "2.0.0"],
                 http=http, cache=JsonCache(root=tmp_path))
    out = capsys.readouterr().out
    assert "Transitive surface" in out
    assert "pytz" in out
    assert "1 declared dependency" in out


def test_transitive_walk_skipped_via_flag(tmp_path: Path, capsys) -> None:
    """--no-transitive disables the walk; no Transitive surface
    section appears."""
    review.main(["PyPI", "django", "2.0.0", "--no-transitive"],
                 http=StubHttp(), cache=JsonCache(root=tmp_path))
    out = capsys.readouterr().out
    assert "Transitive surface" not in out


def test_transitive_walk_skipped_when_offline(tmp_path: Path, capsys) -> None:
    """--offline implies no walk (the metadata fetch needs network)."""
    review.main(["PyPI", "django", "2.0.0", "--offline"],
                 http=StubHttp(), cache=JsonCache(root=tmp_path))
    out = capsys.readouterr().out
    assert "Transitive surface" not in out


def test_unsupported_ecosystem_emits_honest_section(
    tmp_path: Path, capsys,
) -> None:
    """Maven / RubyGems / etc. don't have a metadata walker yet —
    the section must say so explicitly so silence isn't mistaken
    for safety."""
    review.main(
        ["Maven", "org.apache.logging.log4j:log4j-core", "2.14.1"],
        http=StubHttp(), cache=JsonCache(root=tmp_path),
    )
    out = capsys.readouterr().out
    assert "Transitive surface" in out
    assert "not yet supported" in out


def test_kev_in_transitive_escalates_verdict_to_block(
    tmp_path: Path, capsys,
) -> None:
    """Block-class signal in a TRANSITIVE dep should still mean
    'don't install the named package' — the named package's clean
    bill is meaningless if installing it pulls in a KEV CVE."""

    transitive_advisory = {
        "id": "GHSA-fake-trans",
        "modified": "2024-01-01T00:00:00Z",
        "aliases": ["CVE-2024-FAKE-T"],
        "summary": "Hostile transitive",
        "details": "",
        "affected": [{
            "package": {"ecosystem": "PyPI", "name": "vulnerable-pkg"},
            "ranges": [{"type": "ECOSYSTEM",
                         "events": [{"introduced": "0"},
                                     {"fixed": "2.0"}]}],
        }],
        "severity": [{"type": "CVSS_V3",
                      "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
        "references": [],
    }

    class _SeqHttp(StubHttp):
        def __init__(self):
            super().__init__()
            self._batch_count = 0

        def post_json(self, url, body, timeout=30, **kwargs):
            self.posts.append((url, body))
            self._batch_count += 1
            # First OSV batch = direct dep (clean).
            # Second OSV batch = transitives (one hit).
            if self._batch_count == 1:
                return {"results": [{"vulns": []}]}
            return {"results": [{"vulns": [{"id": "GHSA-fake-trans"}]}]}

        def get_json(self, url, timeout=30, **kwargs):
            self.gets.append(url)
            if "pypi.org/pypi/safe-pkg/1.0/json" in url:
                return {"info": {
                    "requires_dist": ["vulnerable-pkg==1.0"],
                }}
            if "pypi.org/pypi/vulnerable-pkg/1.0/json" in url:
                return {"info": {"requires_dist": []}}
            if "GHSA-fake-trans" in url:
                return transitive_advisory
            if "cisa.gov" in url:
                return {"vulnerabilities": [{"cveID": "CVE-2024-FAKE-T"}]}
            if "first.org" in url:
                return {"data": []}
            raise RuntimeError(f"unexpected GET {url}")

    rc = review.main(
        ["PyPI", "safe-pkg", "1.0"],
        http=_SeqHttp(), cache=JsonCache(root=tmp_path),
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "**Verdict:** Block" in out
    assert "Transitive surface" in out
    assert "vulnerable-pkg" in out
    assert "KEV" in out
