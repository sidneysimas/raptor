#!/usr/bin/env python3
"""Subprocess-compatible SCA entry point + cross-tool launch helpers.

When invoked as a script::

    python3 packages/sca/agent.py --repo /path/to/target --out /path/to/out

Runs the full SCA analyse pipeline and writes findings.json + SARIF +
report.md into ``--out``, then prints a one-line JSON summary to stdout
so the caller can parse it.

When imported as a module, exposes two helpers used by
``raptor_agentic.py`` and other RAPTOR-side callers that want to launch
SCA as a sandboxed subprocess rather than in-process:

  - :func:`_find_sca_agent` — discover the SCA agent entry point.
    Returns the resolved path to this file (or to an external override
    set via ``RAPTOR_SCA_AGENT`` env). Pre-merge, this used to bridge
    to a separate ``raptor-sca`` repo; post-merge SCA lives in-tree.
  - :func:`run_sca_subprocess` — launch the agent under
    ``core.sandbox.run`` with egress restricted to
    :data:`packages.sca.SCA_ALLOWED_HOSTS`.

When ``--sandbox`` is passed to the script form, the analysis runs
inside a sandbox context with egress proxy enabled — the
``EgressClient`` default in ``packages.sca.__init__`` already routes
HTTP through the proxy, and the resolver subprocesses already sandbox
themselves, so the outer context adds Landlock FS confinement for the
manifest-parsing phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parents[2]  # raptor-sca repo root
sys.path.insert(0, str(_REPO))

from packages.sca import SCA_ALLOWED_HOSTS  # noqa: E402
from packages.sca.api import analyse  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-tool launch helpers
# ---------------------------------------------------------------------------

def _find_sca_agent() -> Optional[Path]:
    """Discover the SCA agent entry point.

    Post-merge, SCA lives in-tree — this file IS the agent. So the
    default answer is ``Path(__file__).resolve()``. The
    ``RAPTOR_SCA_AGENT`` env var still allows pointing at an external
    agent (e.g. a vendored or pinned version) for CI / custom layouts.
    Returns ``None`` only when the override path is set but invalid.
    """
    env_path = os.environ.get("RAPTOR_SCA_AGENT")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p
        logger.warning("RAPTOR_SCA_AGENT=%s does not exist — ignoring",
                       env_path)
        # Fall through: env override missing is intentional → return
        # None per the bridge contract test expectations.
        return None
    return Path(__file__).resolve()


def run_sca_subprocess(
    agent_path: Path,
    target: Path,
    output_dir: Path,
    *,
    sandbox_args: Sequence[str] = (),
    env: Optional[dict] = None,
    timeout: int = 600,
) -> tuple:
    """Run the SCA agent as a sandboxed subprocess.

    Uses :func:`core.sandbox.run` with ``use_egress_proxy=True`` so the
    child's outbound HTTPS is funnelled through the in-process proxy
    with :data:`packages.sca.SCA_ALLOWED_HOSTS` as the hostname
    allowlist. Landlock confines writes to ``output_dir``.

    Returns ``(returncode, stdout, stderr)``.
    """
    from core.config import RaptorConfig
    from core.sandbox import run as sandbox_run

    cmd: list = [
        sys.executable, str(agent_path),
        "--repo", str(target),
        "--out", str(output_dir),
        *sandbox_args,
    ]

    result = sandbox_run(
        cmd,
        use_egress_proxy=True,
        proxy_hosts=_compose_proxy_hosts(target),
        caller_label="sca-agent",
        target=str(target),
        output=str(output_dir),
        env=env or RaptorConfig.get_safe_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _compose_proxy_hosts(target: Path) -> list:
    """Build the sandbox proxy_hosts allowlist for a SCA run.

    Always includes :data:`SCA_ALLOWED_HOSTS` (the static set of
    OSV / KEV / EPSS / registry-metadata hosts). When the target
    contains Dockerfiles, also adds the registry hosts for every
    FROM image — without these the B9 base-image scanner's
    manifest / blob requests fail at the proxy with a confusing
    network-unreachable error.

    Order: static set first (deterministic), then the dynamic
    Dockerfile-derived hosts. Deduplicated by the union.
    """
    hosts = list(SCA_ALLOWED_HOSTS)
    seen = set(hosts)
    try:
        from .dockerfile_from import image_source_registry_hosts
        for h in image_source_registry_hosts(target):
            if h not in seen:
                hosts.append(h)
                seen.add(h)
    except Exception:                               # noqa: BLE001
        # Best effort: a malformed target shouldn't prevent the
        # sandbox from launching with the static allowlist. The
        # logged exception is sufficient diagnostic.
        logger.warning(
            "sca.agent: failed to derive image-source registry hosts "
            "for sandbox allowlist", exc_info=True,
        )
    return hosts


# ---------------------------------------------------------------------------
# Subprocess entry point — when invoked as `python3 packages/sca/agent.py …`
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RAPTOR SCA agent")
    ap.add_argument("--repo", required=True, help="Target project root")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sarif-dirs", nargs="*",
                    help="Sibling SARIF directories for cross-tool linking")
    ap.add_argument("--sandbox", choices=["full", "network-only", "none"],
                    default=None,
                    help="Sandbox profile (default: use egress proxy only)")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="Disable all sandbox isolation")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--audit-verbose", action="store_true")
    args = ap.parse_args(argv)

    sarif_dirs = [Path(p) for p in args.sarif_dirs] if args.sarif_dirs else None
    target = Path(args.repo).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_sandbox = args.sandbox is not None and not args.no_sandbox

    if use_sandbox:
        result = _run_sandboxed(
            target=target,
            output_dir=output_dir,
            offline=args.offline,
            no_cache=args.no_cache,
            sarif_dirs=sarif_dirs,
            profile=args.sandbox,
        )
    else:
        result = analyse(
            target=target,
            output_dir=output_dir,
            offline=args.offline,
            no_cache=args.no_cache,
            sarif_dirs=sarif_dirs,
        )

    print(json.dumps(result))
    return 0 if result.get("status") == "ok" else 1


def _run_sandboxed(*, target, output_dir, offline, no_cache, sarif_dirs, profile):
    """Run analyse() inside a sandbox context."""
    try:
        from core.sandbox.context import sandbox
    except ImportError:
        logger.warning("sca.agent: sandbox not available, running unsandboxed")
        return analyse(
            target=target, output_dir=output_dir,
            offline=offline, no_cache=no_cache, sarif_dirs=sarif_dirs,
        )

    with sandbox(
        target=str(target),
        output=str(output_dir),
        profile=profile,
        use_egress_proxy=True,
        proxy_hosts=_compose_proxy_hosts(target),
        caller_label="sca-agent",
    ):
        return analyse(
            target=target, output_dir=output_dir,
            offline=offline, no_cache=no_cache, sarif_dirs=sarif_dirs,
        )


if __name__ == "__main__":
    sys.exit(main())
