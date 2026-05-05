"""Regression tests for discovery exclusion rules."""

from __future__ import annotations

from pathlib import Path

from packages.sca.discovery import EXCLUDED_DIR_NAMES, find_manifests


def test_top_level_packages_dir_not_excluded(tmp_path: Path) -> None:
    """`packages/` is a legitimate monorepo layout (raptor, rush, lerna).

    A previous version of the exclude list dropped it silently, hiding
    real manifests. Guard against the regression.
    """
    repo = tmp_path / "proj"
    (repo / "packages" / "web").mkdir(parents=True)
    (repo / "packages" / "web" / "requirements.txt").write_text(
        "django==4.2.7\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("requests==2.31.0\n",
                                            encoding="utf-8")

    manifests = find_manifests(repo)
    paths = {str(m.path.relative_to(repo)) for m in manifests}
    assert "packages/web/requirements.txt" in paths
    assert "requirements.txt" in paths


def test_node_modules_still_excluded(tmp_path: Path) -> None:
    """``node_modules`` must stay excluded — it's vendored deps."""
    repo = tmp_path / "proj"
    (repo / "node_modules" / "lodash").mkdir(parents=True)
    (repo / "node_modules" / "lodash" / "package.json").write_text(
        '{"name":"lodash","version":"4.17.21"}\n', encoding="utf-8")
    (repo / "package.json").write_text(
        '{"name":"app","dependencies":{"lodash":"^4"}}\n', encoding="utf-8")

    manifests = find_manifests(repo)
    paths = {str(m.path.relative_to(repo)) for m in manifests}
    assert "package.json" in paths
    assert not any("node_modules" in p for p in paths)


def test_packages_not_in_excludes() -> None:
    """Belt-and-braces: bare 'packages' must not be in EXCLUDED_DIR_NAMES."""
    assert "packages" not in EXCLUDED_DIR_NAMES


def test_claude_dir_excluded() -> None:
    """.claude/ contains agent worktrees — must not be scanned."""
    assert ".claude" in EXCLUDED_DIR_NAMES


def test_claude_worktrees_skipped(tmp_path: Path) -> None:
    """Manifests inside .claude/worktrees/ must not be discovered."""
    (tmp_path / ".claude" / "worktrees" / "agent-abc123").mkdir(parents=True)
    (tmp_path / ".claude" / "worktrees" / "agent-abc123" / "requirements.txt").write_text(
        "requests>=2.31.0\n"
    )
    (tmp_path / "requirements.txt").write_text("flask>=2.3.0\n")
    manifests = find_manifests(tmp_path)
    paths = [str(m.path) for m in manifests]
    assert any("flask" in Path(p).read_text() for p in paths)
    assert not any(".claude" in p for p in paths)
