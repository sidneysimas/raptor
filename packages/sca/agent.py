#!/usr/bin/env python3
"""Subprocess-compatible SCA entry point.

Designed to be called by ``raptor_agentic.py`` the same way it calls
``packages/llm_analysis/agent.py`` or ``packages/codeql/agent.py``::

    python3 packages/sca/agent.py --repo /path/to/target --out /path/to/out

Writes findings.json + SARIF + report.md into ``--out``, then prints
a one-line JSON summary to stdout so the caller can parse it.

When ``--sandbox`` is passed, the analysis runs inside a sandbox
context with egress proxy enabled — the ``EgressClient`` default in
``packages.sca.__init__`` already routes HTTP through the proxy, and
the resolver subprocesses already sandbox themselves, so the outer
context adds Landlock FS confinement for the manifest-parsing phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]  # raptor-sca repo root
sys.path.insert(0, str(_REPO))

from packages.sca import SCA_ALLOWED_HOSTS  # noqa: E402
from packages.sca.api import analyse  # noqa: E402

logger = logging.getLogger(__name__)


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
        proxy_hosts=list(SCA_ALLOWED_HOSTS),
        caller_label="sca-agent",
    ):
        return analyse(
            target=target, output_dir=output_dir,
            offline=offline, no_cache=no_cache, sarif_dirs=sarif_dirs,
        )


if __name__ == "__main__":
    sys.exit(main())
