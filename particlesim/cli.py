"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import particlesim
from particlesim.core.config import SCENARIOS, config_from_dict, load_config
from particlesim.theories import list_theories


def _cmd_theories(args: argparse.Namespace) -> int:
    for tid, cls in sorted(list_theories().items()):
        print(
            f"{tid:24s} tier {cls.tier}  D={cls.dimension}  {cls.formulation:14s} {cls.provenance}"
        )
    return 0


def _dispatch(config, out: str | None) -> int:
    if out:
        config.output.dir = out
    if config.scenario == "warp.analyze":
        from particlesim.scenarios.warp.analyze import run

        result = run(config)
    elif config.scenario == "cosmo.linear":
        from particlesim.scenarios.cosmo.linear import run

        result = run(config)
        result.save(config.output.dir)
    else:
        print(f"scenario {config.scenario} has no runner yet", file=sys.stderr)
        return 2
    print(json.dumps(result.report, indent=2, default=str))
    print(f"outputs written to {config.output.dir}", file=sys.stderr)
    # Every run gets a dashboard (issue #83), whatever formats it asked for.
    from particlesim.viz.dashboard import write_run_dashboard

    print(f"dashboard: {write_run_dashboard(config.output.dir)}", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return _dispatch(load_config(args.config), args.out)


def _cmd_rerun(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    config = config_from_dict(manifest["config"])
    return _dispatch(config, args.out)


def _cmd_check_limits(args: argparse.Namespace) -> int:
    from particlesim.theories.limits import check_all

    reports = check_all(check_action=not args.no_action)
    failures = 0
    for tid, r in sorted(reports.items()):
        if r.passed:
            status = "ok"
        elif r.checked:
            status = "FAIL"
            failures += 1
        else:
            status = "skipped"
        print(f"{tid:24s} {status}")
        for reason in r.reasons:
            print(f"    {reason}")
    return 1 if failures else 0


def _cmd_hypothesis(args: argparse.Namespace) -> int:
    from particlesim.scenarios.singularity.harness import evaluate
    from particlesim.theories import get_theory

    theory = get_theory(args.theory)
    card = evaluate(theory)
    if args.report:
        from particlesim.scenarios.singularity.harness import write_report

        path = write_report(card, args.report, theory)
        print(f"report written to {path}", file=sys.stderr)
    if args.json:
        print(json.dumps(card.summary(), indent=2, default=str))
    else:
        print(card.render())
    # Exit non-zero when a hypothesis is contradicted, so the harness can gate
    # a pipeline rather than only inform a reader.
    return 0 if card.passed else 1


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from particlesim.viz.dashboard import (
        read_outcomes,
        render_benchmark_dashboard,
        write_run_dashboard,
    )

    for target in map(Path, args.paths):
        if target.is_dir():
            path = write_run_dashboard(target)
        elif target.suffix == ".json":
            outcomes, meta = read_outcomes(target)
            docs = Path(args.docs)
            path = render_benchmark_dashboard(
                outcomes,
                target.with_suffix(".html"),
                documentation=docs if docs.is_file() else None,
                meta=meta,
            )
        else:
            print(f"{target}: neither a run directory nor benchmark results", file=sys.stderr)
            return 2
        print(path)
    return 0


def _cmd_scenarios(args: argparse.Namespace) -> int:
    for name in sorted(SCENARIOS):
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="particlesim", description=particlesim.__doc__)
    p.add_argument("--version", action="version", version=f"particlesim {particlesim.__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    th = sub.add_parser("theories", help="list discoverable theory plugins")
    th.set_defaults(func=_cmd_theories)

    sc = sub.add_parser("scenarios", help="list scenario types")
    sc.set_defaults(func=_cmd_scenarios)

    cl = sub.add_parser("check-limits", help="verify every plugin's declared GR limit")
    cl.add_argument("--no-action", action="store_true", help="skip the slower Lagrangian check")
    cl.set_defaults(func=_cmd_check_limits)

    hy = sub.add_parser("hypothesis", help="score a theory plugin against the singularity battery")
    hy.add_argument("theory", help="theory id, for example lqg.lqc")
    hy.add_argument("--json", action="store_true", help="machine-readable report card")
    hy.add_argument("--report", help="also write a self-contained HTML report to this path")
    hy.set_defaults(func=_cmd_hypothesis)

    rn = sub.add_parser("run", help="run a scenario from a YAML config")
    rn.add_argument("config")
    rn.add_argument("--out", help="override the output directory")
    rn.set_defaults(func=_cmd_run)

    db = sub.add_parser(
        "dashboard",
        help="write the HTML dashboard for run directories or benchmark results (.json)",
    )
    db.add_argument("paths", nargs="+", help="run directories, or --dashboard's .json results")
    db.add_argument(
        "--docs", default="docs/benchmarks.md", help="benchmark documentation to join with"
    )
    db.set_defaults(func=_cmd_dashboard)

    rr = sub.add_parser("rerun", help="re-run a scenario from a run manifest")
    rr.add_argument("manifest")
    rr.add_argument("--out", help="override the output directory")
    rr.set_defaults(func=_cmd_rerun)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
