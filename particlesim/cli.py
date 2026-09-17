"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys

import particlesim
from particlesim.core.config import SCENARIOS, load_config
from particlesim.theories import list_theories


def _cmd_theories(args: argparse.Namespace) -> int:
    for tid, cls in sorted(list_theories().items()):
        print(
            f"{tid:24s} tier {cls.tier}  D={cls.dimension}  {cls.formulation:14s} {cls.provenance}"
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.out:
        config.output.dir = args.out
    if config.scenario == "warp.analyze":
        from particlesim.scenarios.warp.analyze import run

        result = run(config)
        print(json.dumps(result.report, indent=2, default=str))
        print(f"outputs written to {config.output.dir}", file=sys.stderr)
        return 0
    print(f"scenario {config.scenario} has no runner yet", file=sys.stderr)
    return 2


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

    rn = sub.add_parser("run", help="run a scenario from a YAML config")
    rn.add_argument("config")
    rn.add_argument("--out", help="override the output directory")
    rn.set_defaults(func=_cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
