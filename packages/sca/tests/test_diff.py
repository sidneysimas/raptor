"""Tests for ``packages.sca.diff``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from packages.sca import diff


def _vuln_row(
    *,
    eco: str = "npm",
    name: str = "lodash",
    version: str = "4.17.4",
    advisory_id: str = "GHSA-jf85",
    aliases: List[str] | None = None,
    severity: str = "critical",
    in_kev: bool = False,
    epss: float | None = None,
    suppressed: bool = False,
    reason: str | None = None,
) -> Dict[str, Any]:
    return {
        "id": f"sca:vuln:{eco}:{name}:{version}:{advisory_id}",
        "vuln_type": "sca:vulnerable_dependency",
        "severity": severity,
        "suppressed": suppressed,
        "suppression_reason": reason,
        "sca": {
            "ecosystem": eco, "name": name, "version": version,
            "advisory": {"id": advisory_id, "aliases": aliases or []},
            "in_kev": in_kev,
            "epss": epss,
        },
    }


def _hygiene_row(kind: str = "loose_pin", eco: str = "npm",
                 name: str = "lodash", version: str = "4.17.4",
                 severity: str = "low",
                 suppressed: bool = False) -> Dict[str, Any]:
    return {
        "id": f"sca:hygiene:{kind}:{eco}:{name}",
        "vuln_type": f"sca:hygiene:{kind}",
        "severity": severity,
        "suppressed": suppressed,
        "sca": {"ecosystem": eco, "name": name, "version": version,
                 "kind": kind},
    }


def _write(tmp_path: Path, name: str, rows: List[Dict[str, Any]]) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# compute_delta
# ---------------------------------------------------------------------------

def test_new_finding_in_b_only() -> None:
    a = []
    b = [_vuln_row()]
    d = diff.compute_delta(a, b)
    assert len(d.new) == 1
    assert d.resolved == []


def test_resolved_finding_in_a_only() -> None:
    a = [_vuln_row()]
    b = []
    d = diff.compute_delta(a, b)
    assert len(d.resolved) == 1
    assert d.new == []


def test_unchanged_findings_drop_from_diff() -> None:
    """Same finding in both → not in new, not in resolved."""
    a = [_vuln_row()]
    b = [_vuln_row()]
    d = diff.compute_delta(a, b)
    assert d.new == [] and d.resolved == []


def test_canonical_key_uses_cve_alias_for_dedup() -> None:
    """GHSA-X (CVE-Y) in A vs PYSEC-X (CVE-Y) in B = same finding."""
    a = [_vuln_row(advisory_id="GHSA-x", aliases=["CVE-2023-X"])]
    b = [_vuln_row(advisory_id="PYSEC-x", aliases=["CVE-2023-X"])]
    d = diff.compute_delta(a, b)
    assert d.new == [] and d.resolved == []


def test_suppressed_findings_excluded_by_default() -> None:
    """A finding suppressed in B looks like 'resolved' to default mode
    (because we don't see it as active), even though it still exists."""
    a = [_vuln_row()]
    b = [_vuln_row(suppressed=True, reason="ack")]
    d = diff.compute_delta(a, b)
    # New/resolved respect the visibility filter:
    assert d.resolved == [] and d.new == []
    # The suppression-state diff sees it as a state change, separate stream:
    assert len(d.suppression_added) == 1


def test_suppression_lifted_detected() -> None:
    a = [_vuln_row(suppressed=True, reason="ack")]
    b = [_vuln_row()]
    d = diff.compute_delta(a, b)
    assert len(d.suppression_lifted) == 1


def test_include_suppressed_treats_them_as_visible() -> None:
    a = [_vuln_row()]
    b = [_vuln_row(suppressed=True, reason="ack"),
         _vuln_row(name="other-lib", advisory_id="GHSA-other")]
    d = diff.compute_delta(a, b, include_suppressed=True)
    # Now the suppressed row counts as present, so the only "new" is
    # other-lib:
    new_names = [r["sca"]["name"] for r in d.new]
    assert new_names == ["other-lib"]
    # And it's *also* a suppression change:
    assert len(d.suppression_added) == 1


def test_hygiene_findings_keyed_separately(tmp_path: Path) -> None:
    a = [_hygiene_row(kind="loose_pin")]
    b = [_hygiene_row(kind="lockfile_drift")]
    d = diff.compute_delta(a, b)
    assert len(d.new) == 1
    assert len(d.resolved) == 1


def test_findings_without_canonical_key_skipped() -> None:
    """Rows from other tools (no advisory id, not hygiene/supply_chain)
    are silently dropped — diff is SCA-only."""
    a = [{"vuln_type": "scan:other", "severity": "high"}]
    b = []
    d = diff.compute_delta(a, b)
    assert d.resolved == []


# ---------------------------------------------------------------------------
# CLI / argparse
# ---------------------------------------------------------------------------

def test_main_writes_markdown_by_default(tmp_path: Path, capsys) -> None:
    a = _write(tmp_path, "a.json", [])
    b = _write(tmp_path, "b.json", [_vuln_row()])
    rc = diff.main([str(a), str(b)])
    assert rc == 1   # B introduces a critical → above default --severity high
    out = capsys.readouterr().out
    assert "# raptor-sca diff" in out
    assert "## New findings" in out
    assert "lodash" in out


def test_main_emits_json_with_flag(tmp_path: Path, capsys) -> None:
    a = _write(tmp_path, "a.json", [])
    b = _write(tmp_path, "b.json", [_vuln_row()])
    diff.main([str(a), str(b), "--json"])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["summary"]["new"] == 1
    assert parsed["summary"]["resolved"] == 0


def test_main_writes_to_out_path(tmp_path: Path, capsys) -> None:
    a = _write(tmp_path, "a.json", [_vuln_row()])
    b = _write(tmp_path, "b.json", [])
    out = tmp_path / "delta.md"
    diff.main([str(a), str(b), "--out", str(out)])
    assert out.exists()
    body = out.read_text()
    assert "## Resolved findings" in body
    # stdout matches the file body.
    assert capsys.readouterr().out.startswith(body)


def test_exit_code_zero_when_only_resolutions(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.json", [_vuln_row()])
    b = _write(tmp_path, "b.json", [])
    rc = diff.main([str(a), str(b)])
    assert rc == 0


def test_exit_code_zero_when_new_below_severity_threshold(
    tmp_path: Path,
) -> None:
    a = _write(tmp_path, "a.json", [])
    b = _write(tmp_path, "b.json", [_vuln_row(severity="medium")])
    rc = diff.main([str(a), str(b), "--fail-on-severity", "high"])
    assert rc == 0


def test_exit_code_two_for_missing_file(tmp_path: Path) -> None:
    rc = diff.main([str(tmp_path / "nope.json"),
                    str(tmp_path / "also-nope.json")])
    assert rc == 2


def test_exit_code_two_for_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    good = _write(tmp_path, "ok.json", [])
    assert diff.main([str(bad), str(good)]) == 2


def test_exit_code_two_for_non_list_top_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"results": []}), encoding="utf-8")
    good = _write(tmp_path, "ok.json", [])
    assert diff.main([str(bad), str(good)]) == 2


def test_no_changes_renders_explanatory_message(
    tmp_path: Path, capsys,
) -> None:
    rows = [_vuln_row()]
    a = _write(tmp_path, "a.json", rows)
    b = _write(tmp_path, "b.json", rows)
    diff.main([str(a), str(b)])
    out = capsys.readouterr().out
    assert "No changes." in out
