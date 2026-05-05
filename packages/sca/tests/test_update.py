"""Tests for ``packages.sca.update`` (the ``raptor-sca fix --cve-only`` subcommand)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from packages.sca import update


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _vuln_row(
    *,
    ecosystem: str,
    name: str,
    version: str,
    fixed_version: str | None,
    manifest: Path,
    advisory_id: str = "GHSA-x",
    pin_style: str = "exact",
    aliases: List[str] | None = None,
) -> dict:
    return {
        "id": f"sca:vuln:{ecosystem}:{name}:{version}:{advisory_id}",
        "vuln_type": "sca:vulnerable_dependency",
        "tool": "sca",
        "file": str(manifest),
        "function": name,
        "line": 0,
        "severity": "high",
        "description": f"{name}@{version} test",
        "sca": {
            "ecosystem": ecosystem,
            "name": name,
            "version": version,
            "purl": f"pkg:{ecosystem.lower()}/{name}@{version}",
            "pin_style": pin_style,
            "fixed_version": fixed_version,
            "advisory": {
                "id": advisory_id,
                "aliases": aliases or [],
                "summary": "test",
                "fixed_versions": [fixed_version] if fixed_version else [],
                "references": [],
                "severity": None,
            },
            "all_advisories": [],
            "in_kev": False,
            "epss": None,
            "reachability": {"verdict": "imported",
                             "confidence": {"level": "high",
                                            "numeric": 0.95, "reason": "t"},
                             "evidence": []},
            "cvss_score": 7.5,
            "cvss_vector": None,
            "version_match_confidence": {"level": "high", "numeric": 0.95,
                                         "reason": "t"},
            "parser_confidence": {"level": "high", "numeric": 0.95,
                                  "reason": "t"},
            "exposure_factor": 0.0,
            "transitive_depth": 0,
            "related_findings": [],
        },
    }


def _findings_file(tmp_path: Path, rows: list) -> Path:
    p = tmp_path / "findings.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# pom.xml rewriter
# ---------------------------------------------------------------------------

def test_pom_xml_rewrite_bumps_version(tmp_path: Path) -> None:
    pom = tmp_path / "pom.xml"
    pom.write_text("""\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <dependencies>
    <dependency>
      <groupId>org.apache.logging.log4j</groupId>
      <artifactId>log4j-core</artifactId>
      <version>2.14.1</version>
    </dependency>
  </dependencies>
</project>
""", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="Maven",
        name="org.apache.logging.log4j:log4j-core",
        version="2.14.1", fixed_version="2.17.1",
        manifest=pom,
    )])
    out = tmp_path / "out"
    rc = update.main([
        "--findings", str(findings), "--out", str(out), "--allow-major",
    ])
    assert rc == 0
    rewritten = (out / "proposed" / pom.resolve().relative_to(Path.cwd().resolve()) if False else None)
    # Find the proposed file by walking the tree (path layout depends on cwd).
    found = list((out / "proposed").rglob("pom.xml"))
    assert len(found) == 1
    body = found[0].read_text()
    assert "<version>2.17.1</version>" in body
    assert "<version>2.14.1</version>" not in body
    changes = json.loads((out / "changes.json").read_text())
    assert changes[0]["new_version"] == "2.17.1"


def test_pom_xml_with_property_reference_skipped(tmp_path: Path) -> None:
    pom = tmp_path / "pom.xml"
    pom.write_text("""\
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <properties><log4j.version>2.14.1</log4j.version></properties>
  <dependencies><dependency>
    <groupId>org.apache.logging.log4j</groupId>
    <artifactId>log4j-core</artifactId>
    <version>${log4j.version}</version>
  </dependency></dependencies>
</project>
""", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="Maven",
        name="org.apache.logging.log4j:log4j-core",
        version="2.14.1", fixed_version="2.17.1",
        manifest=pom,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out), "--allow-major"])
    changes = json.loads((out / "changes.json").read_text())
    assert changes[0]["skipped_reason"] is not None
    assert "property reference" in changes[0]["skipped_reason"]


# ---------------------------------------------------------------------------
# package.json rewriter
# ---------------------------------------------------------------------------

def test_package_json_caret_preserved(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "name": "demo",
        "dependencies": {"lodash": "^4.17.4"},
    }, indent=2), encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="npm", name="lodash",
        version="4.17.4", fixed_version="4.17.21",
        manifest=pkg, pin_style="caret",
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("package.json"))[0]
    obj = json.loads(proposed.read_text())
    assert obj["dependencies"]["lodash"] == "^4.17.21"


def test_package_json_exact_pin_replaced(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "name": "demo",
        "dependencies": {"lodash": "4.17.4"},
    }, indent=2), encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="npm", name="lodash",
        version="4.17.4", fixed_version="4.17.21",
        manifest=pkg,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("package.json"))[0]
    assert json.loads(proposed.read_text())["dependencies"]["lodash"] == "4.17.21"


def test_package_json_git_url_skipped(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "dependencies": {"lodash": "git+https://github.com/lodash/lodash.git#v4.17.4"},
    }), encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="npm", name="lodash",
        version="4.17.4", fixed_version="4.17.21",
        manifest=pkg,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    changes = json.loads((out / "changes.json").read_text())
    assert changes[0]["skipped_reason"] is not None


# ---------------------------------------------------------------------------
# requirements.txt rewriter
# ---------------------------------------------------------------------------

def test_requirements_txt_rewrite(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text("# pinned\ndjango==4.2.7\nrequests>=2.31.0\n", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="PyPI", name="django",
        version="4.2.7", fixed_version="4.2.10",
        manifest=req,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("requirements.txt"))[0]
    body = proposed.read_text()
    assert "django==4.2.10" in body
    # Untouched line preserved.
    assert "requests>=2.31.0" in body
    assert "# pinned" in body


def test_requirements_txt_pep503_normalisation(tmp_path: Path) -> None:
    """``Foo_Bar.Baz`` in the manifest should match the PEP 503 form
    ``foo-bar-baz`` carried in findings."""
    req = tmp_path / "requirements.txt"
    req.write_text("Foo_Bar.Baz==1.0.0\n", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="PyPI", name="foo-bar-baz",
        version="1.0.0", fixed_version="1.0.1",
        manifest=req,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("requirements.txt"))[0]
    assert "==1.0.1" in proposed.read_text()


# ---------------------------------------------------------------------------
# pyproject.toml rewriter
# ---------------------------------------------------------------------------

def test_pyproject_toml_pep621_rewrite(tmp_path: Path) -> None:
    py = tmp_path / "pyproject.toml"
    py.write_text("""\
[project]
dependencies = [
  "django==4.2.7",
  "requests~=2.31.0",
]
""", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="PyPI", name="django",
        version="4.2.7", fixed_version="4.2.10",
        manifest=py,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("pyproject.toml"))[0]
    body = proposed.read_text()
    assert '"django==4.2.10"' in body
    assert '"requests~=2.31.0"' in body


def test_pyproject_toml_poetry_caret_preserved(tmp_path: Path) -> None:
    py = tmp_path / "pyproject.toml"
    py.write_text("""\
[tool.poetry.dependencies]
python = "^3.10"
django = "^4.2.7"
""", encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="PyPI", name="django",
        version="4.2.7", fixed_version="4.2.10",
        manifest=py,
    )])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("pyproject.toml"))[0]
    body = proposed.read_text()
    assert 'django = "^4.2.10"' in body
    assert 'python = "^3.10"' in body


# ---------------------------------------------------------------------------
# Mode flags
# ---------------------------------------------------------------------------

def test_fix_filter_restricts_to_listed_advisories(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "dependencies": {"a": "1.0.0", "b": "2.0.0"},
    }), encoding="utf-8")
    findings = _findings_file(tmp_path, [
        _vuln_row(ecosystem="npm", name="a", version="1.0.0",
                  fixed_version="1.5.0", manifest=pkg,
                  advisory_id="GHSA-keep"),
        _vuln_row(ecosystem="npm", name="b", version="2.0.0",
                  fixed_version="2.5.0", manifest=pkg,
                  advisory_id="GHSA-skip"),
    ])
    out = tmp_path / "out"
    update.main([
        "--findings", str(findings),
        "--out", str(out),
        "--fix", "GHSA-keep",
    ])
    changes = json.loads((out / "changes.json").read_text())
    names = {c["name"] for c in changes}
    assert names == {"a"}


def test_minimal_picks_max_fix_across_findings(tmp_path: Path) -> None:
    """Two CVEs against the same dep with fixes 1.5 and 1.10 — the
    proposed bump must be 1.10."""
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"dependencies": {"x": "1.0.0"}}),
                   encoding="utf-8")
    findings = _findings_file(tmp_path, [
        _vuln_row(ecosystem="npm", name="x", version="1.0.0",
                  fixed_version="1.5.0", manifest=pkg,
                  advisory_id="GHSA-1"),
        _vuln_row(ecosystem="npm", name="x", version="1.0.0",
                  fixed_version="1.10.0", manifest=pkg,
                  advisory_id="GHSA-2"),
    ])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out)])
    proposed = list((out / "proposed").rglob("package.json"))[0]
    assert json.loads(proposed.read_text())["dependencies"]["x"] == "1.10.0"


def test_allow_major_gates_cross_major_upgrade(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({"dependencies": {"x": "1.0.0"}}),
                   encoding="utf-8")
    findings = _findings_file(tmp_path, [_vuln_row(
        ecosystem="npm", name="x", version="1.0.0",
        fixed_version="2.0.0", manifest=pkg,
    )])
    out_no_major = tmp_path / "out_no_major"
    rc = update.main([
        "--findings", str(findings), "--out", str(out_no_major),
    ])
    # No proposed file because the only fix crosses a major boundary
    # and --allow-major wasn't supplied.
    assert rc == 0
    assert not (out_no_major / "proposed").exists()

    out_allow = tmp_path / "out_allow"
    update.main([
        "--findings", str(findings), "--out", str(out_allow),
        "--allow-major",
    ])
    proposed = list((out_allow / "proposed").rglob("package.json"))[0]
    assert json.loads(proposed.read_text())["dependencies"]["x"] == "2.0.0"


def test_pin_only_skips_loose_pins(tmp_path: Path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text(json.dumps({
        "dependencies": {"x": "^1.0.0", "y": "1.0.0"},
    }), encoding="utf-8")
    findings = _findings_file(tmp_path, [
        _vuln_row(ecosystem="npm", name="x", version="1.0.0",
                  fixed_version="1.5.0", manifest=pkg, pin_style="caret"),
        _vuln_row(ecosystem="npm", name="y", version="1.0.0",
                  fixed_version="1.5.0", manifest=pkg, pin_style="exact"),
    ])
    out = tmp_path / "out"
    update.main(["--findings", str(findings), "--out", str(out),
                 "--pin-only"])
    changes = {c["name"]: c for c in json.loads(
        (out / "changes.json").read_text(),
    )}
    assert changes["y"]["skipped_reason"] is None
    assert changes["x"]["skipped_reason"] is not None
    assert "pin-only" in changes["x"]["skipped_reason"]


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_neither_findings_nor_target_returns_2(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        update.main(["--out", str(tmp_path / "out")])
    assert exc.value.code == 2


def test_findings_missing_returns_2(tmp_path: Path) -> None:
    rc = update.main(["--findings", str(tmp_path / "nope.json"),
                      "--out", str(tmp_path / "out")])
    assert rc == 2


def test_offline_and_allow_cascade_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """``--offline --allow-cascade`` is operator confusion: the cascade
    resolver shells out to npm/pip/go which all need network. Reject
    at argparse time so the operator sees a clear message instead of
    a confusing resolver-can't-reach-registry failure deep in the run.
    """
    findings = tmp_path / "findings.json"
    findings.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        update.main([
            "--findings", str(findings),
            "--out", str(tmp_path / "out"),
            "--offline", "--allow-cascade",
        ])
    assert exc.value.code == 2     # argparse error exit
    err = capsys.readouterr().err
    assert "mutually exclusive" in err
    assert "--offline" in err and "--allow-cascade" in err
