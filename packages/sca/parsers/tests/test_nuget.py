"""Tests for the NuGet parser (.csproj + packages.config + packages.lock.json)."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.sca.models import PinStyle
from packages.sca.parsers.nuget import (
    parse_lockfile,
    parse_msbuild_project,
    parse_packages_config,
)


def _write(tmp_path: Path, body: str, name: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# .csproj — modern SDK-style
# ---------------------------------------------------------------------------

def test_csproj_attribute_form(tmp_path: Path) -> None:
    body = """\
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
    <PackageReference Include="Serilog" Version="3.1.0" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    by_name = {d.name: d for d in parse_msbuild_project(p)}
    assert "Newtonsoft.Json" in by_name
    assert by_name["Newtonsoft.Json"].version == "13.0.1"


def test_csproj_child_element_version(tmp_path: Path) -> None:
    """Some projects use ``<Version>X</Version>`` as a child."""
    body = """\
<Project>
  <ItemGroup>
    <PackageReference Include="Foo">
      <Version>1.2.3</Version>
    </PackageReference>
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert len(deps) == 1
    assert deps[0].version == "1.2.3"


def test_csproj_namespaced(tmp_path: Path) -> None:
    """Legacy projects with the MSBuild XML namespace."""
    body = """\
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <PackageReference Include="OldPkg" Version="1.0.0" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert len(deps) == 1
    assert deps[0].name == "OldPkg"


def test_csproj_private_assets_marks_build(tmp_path: Path) -> None:
    body = """\
<Project>
  <ItemGroup>
    <PackageReference Include="Analyzer.Pkg" Version="1.0.0"
                      PrivateAssets="all" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert deps[0].scope == "build"


def test_csproj_exact_pin_brackets(tmp_path: Path) -> None:
    body = """\
<Project>
  <ItemGroup>
    <PackageReference Include="Foo" Version="[1.2.3]" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert deps[0].pin_style is PinStyle.EXACT
    assert deps[0].version == "1.2.3"


def test_csproj_range_brackets(tmp_path: Path) -> None:
    body = """\
<Project>
  <ItemGroup>
    <PackageReference Include="Foo" Version="[1.0,2.0)" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert deps[0].pin_style is PinStyle.RANGE


def test_csproj_minimum_form(tmp_path: Path) -> None:
    """``Version="1.2.3"`` (no brackets) is NuGet's "≥1.2.3" — RANGE."""
    body = """\
<Project>
  <ItemGroup>
    <PackageReference Include="Foo" Version="1.2.3" />
  </ItemGroup>
</Project>
"""
    p = _write(tmp_path, body, "App.csproj")
    deps = parse_msbuild_project(p)
    assert deps[0].pin_style is PinStyle.RANGE
    assert deps[0].version == "1.2.3"


def test_csproj_invalid_xml_returns_empty(tmp_path: Path) -> None:
    p = _write(tmp_path, "<Project><ItemGroup></Project>", "App.csproj")
    assert parse_msbuild_project(p) == []


@pytest.mark.parametrize("spec", [
    "(1.2.3)",   # exclusive on both sides + single value = empty interval
    "[1.2.3)",   # inclusive lower, exclusive upper, single value = empty
    "(1.2.3]",   # exclusive lower, inclusive upper, single value = empty
])
def test_csproj_pathological_brackets_classified_unknown(
    tmp_path: Path, spec: str,
) -> None:
    """Regression for the 2026-05-21 lint-sweep find: only ``[V]``
    (both inclusive, single value) is a valid EXACT pin per the
    NuGet version-range spec. The pathological one-value forms
    ``(V)`` / ``[V)`` / ``(V]`` describe empty intervals and
    must NOT be classified as EXACT — otherwise the harden
    planner would treat a malformed manifest entry as a
    concrete pinned version and propagate the bad version
    downstream. Pre-fix: ``_classify_version_spec`` extracted
    both bracket tokens but ignored them, returning EXACT
    whenever the regex matched a single value. Post-fix:
    bracket tokens are checked; non-``[..]`` single-value forms
    return ``PinStyle.UNKNOWN``."""
    from packages.sca.parsers.nuget import _classify_version_spec

    style, version = _classify_version_spec(spec)
    assert style is PinStyle.UNKNOWN, (
        f"pathological spec {spec!r} classified as {style!r} "
        f"(should be UNKNOWN); version={version!r}"
    )
    # No concrete version pinned — the planner gets nothing to
    # mistake for a real version.
    assert version is None, (
        f"pathological spec {spec!r} returned version={version!r} "
        f"(should be None to prevent downstream propagation)"
    )


def test_csproj_canonical_single_value_still_exact(
    tmp_path: Path,
) -> None:
    """Companion to the pathological-bracket test: ``[1.2.3]``
    (the ONLY valid single-value form per spec) must still be
    classified as EXACT. Pins the regression-fix didn't
    over-correct into rejecting valid input."""
    from packages.sca.parsers.nuget import _classify_version_spec

    style, version = _classify_version_spec("[1.2.3]")
    assert style is PinStyle.EXACT
    assert version == "1.2.3"


# ---------------------------------------------------------------------------
# packages.config — legacy
# ---------------------------------------------------------------------------

def test_packages_config_basic(tmp_path: Path) -> None:
    body = """\
<?xml version="1.0" encoding="utf-8"?>
<packages>
  <package id="Newtonsoft.Json" version="13.0.1" />
  <package id="Serilog" version="3.1.0" />
</packages>
"""
    p = _write(tmp_path, body, "packages.config")
    deps = parse_packages_config(p)
    assert {d.name for d in deps} == {"Newtonsoft.Json", "Serilog"}


# ---------------------------------------------------------------------------
# packages.lock.json — lockfile
# ---------------------------------------------------------------------------

def test_lockfile_direct_vs_transitive(tmp_path: Path) -> None:
    body = """\
{
  "version": 1,
  "dependencies": {
    "net8.0": {
      "Newtonsoft.Json": {
        "type": "Direct",
        "requested": "[13.0.1, )",
        "resolved": "13.0.1"
      },
      "Microsoft.Foo": {
        "type": "Transitive",
        "resolved": "5.0.0"
      }
    }
  }
}
"""
    p = _write(tmp_path, body, "packages.lock.json")
    deps = parse_lockfile(p)
    by_name = {d.name: d for d in deps}
    assert by_name["Newtonsoft.Json"].direct is True
    assert by_name["Microsoft.Foo"].direct is False
    assert by_name["Newtonsoft.Json"].pin_style is PinStyle.EXACT


def test_lockfile_dedup_across_targets(tmp_path: Path) -> None:
    """Same dep present in multiple target frameworks — emit once."""
    body = """\
{
  "dependencies": {
    "net8.0": {"Foo": {"type": "Direct", "resolved": "1.0.0"}},
    "net6.0": {"Foo": {"type": "Direct", "resolved": "1.0.0"}}
  }
}
"""
    p = _write(tmp_path, body, "packages.lock.json")
    deps = parse_lockfile(p)
    assert len(deps) == 1


# ---------------------------------------------------------------------------
# Discovery → parser dispatch
# ---------------------------------------------------------------------------

def test_dispatch_csproj_via_suffix(tmp_path: Path) -> None:
    from packages.sca.discovery import find_manifests
    from packages.sca.parsers import parse_manifest as dispatch
    repo = tmp_path / "dotnet-proj"
    repo.mkdir()
    (repo / "App.csproj").write_text(
        '<Project><ItemGroup>'
        '<PackageReference Include="Foo" Version="1.0.0" />'
        '</ItemGroup></Project>',
        encoding="utf-8",
    )
    manifests = find_manifests(repo)
    csproj = next(m for m in manifests if m.path.suffix == ".csproj")
    assert csproj.ecosystem == "NuGet"
    deps = dispatch(csproj)
    assert deps and deps[0].name == "Foo"


def test_dispatch_lockfile_via_filename(tmp_path: Path) -> None:
    from packages.sca.discovery import find_manifests
    from packages.sca.parsers import parse_manifest as dispatch
    repo = tmp_path / "dotnet-proj"
    repo.mkdir()
    (repo / "packages.lock.json").write_text(
        '{"dependencies":{"net8":{"Foo":{"type":"Direct","resolved":"1.0"}}}}',
        encoding="utf-8",
    )
    manifests = find_manifests(repo)
    lock = next(m for m in manifests if m.path.name == "packages.lock.json")
    assert lock.ecosystem == "NuGet"
    deps = dispatch(lock)
    assert deps and deps[0].name == "Foo"
