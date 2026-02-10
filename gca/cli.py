"""
GCA v0 — CLI for quick inspection of a GCA store

Optional but recommended (plan Layer 2.1).

Usage::

    python -m gca.cli --store ./gca_store top-strategies
    python -m gca.cli --store ./gca_store programs-for-strategy <strategy_id>
    python -m gca.cli --store ./gca_store timeline --run run_001
    python -m gca.cli --store ./gca_store info
"""

from __future__ import annotations

import argparse
import json
import sys

from gca.query import load_query
from gca.persistence import Persistence


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gca",
        description="GCA v0 — inspect strategy archives",
    )
    parser.add_argument(
        "--store",
        default="./gca_store",
        help="Path to the GCA store directory (default: ./gca_store)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- top-strategies ----------------------------------------------------
    p_top = sub.add_parser("top-strategies", help="List strategies by usage count")
    p_top.add_argument("-n", "--limit", type=int, default=20, help="Max results")

    # -- programs-for-strategy ---------------------------------------------
    p_progs = sub.add_parser("programs-for-strategy", help="Programs exhibiting a strategy")
    p_progs.add_argument("strategy_id", help="Strategy ID (or prefix)")

    # -- timeline ----------------------------------------------------------
    p_tl = sub.add_parser("timeline", help="Strategy counts per iteration")
    p_tl.add_argument("--run", required=True, help="Run ID")

    # -- info --------------------------------------------------------------
    sub.add_parser("info", help="Show store metadata")

    # -- strategies --------------------------------------------------------
    p_list = sub.add_parser("strategies", help="List all strategies")
    p_list.add_argument("-n", "--limit", type=int, default=50, help="Max results")

    args = parser.parse_args(argv)
    query = load_query(args.store)

    if args.command == "top-strategies":
        _cmd_top_strategies(query, args.limit)

    elif args.command == "programs-for-strategy":
        _cmd_programs_for_strategy(query, args.strategy_id)

    elif args.command == "timeline":
        _cmd_timeline(query, args.run)

    elif args.command == "info":
        _cmd_info(args.store)

    elif args.command == "strategies":
        _cmd_list_strategies(query, args.limit)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_top_strategies(query, limit: int) -> None:
    results = query.top_strategies(limit)
    if not results:
        print("No strategies found.")
        return
    print(f"{'Count':>6}  {'ID (prefix)':14}  Description")
    print("-" * 70)
    for strat, count in results:
        print(f"{count:>6}  {strat.strategy_id[:12]:14}  {strat.description[:60]}")


def _cmd_programs_for_strategy(query, sid_prefix: str) -> None:
    # Try exact match first, then prefix
    strat = query.get_strategy(sid_prefix)
    if strat is None:
        # Prefix search
        for s in query.list_strategies(limit=500):
            if s.strategy_id.startswith(sid_prefix):
                strat = s
                break
    if strat is None:
        print(f"Strategy not found: {sid_prefix}")
        return

    print(f"Strategy: {strat.description}")
    print(f"ID:       {strat.strategy_id}")
    print()

    program_ids = query.programs_with_strategy(strat.strategy_id)
    if not program_ids:
        print("No programs found.")
        return
    for pid in program_ids:
        print(f"  - {pid}")


def _cmd_timeline(query, run_id: str) -> None:
    tl = query.timeline(run_id)
    if not tl:
        print(f"No data for run: {run_id}")
        return
    for iteration, counts in tl.items():
        total = sum(counts.values())
        unique = len(counts)
        print(f"  iter {iteration:>5}:  {total} occurrences, {unique} unique strategies")


def _cmd_info(store_path: str) -> None:
    persistence = Persistence(store_path)
    meta = persistence.load_metadata()
    print(json.dumps(meta, indent=2))

    n_events = len(persistence.load_program_events())
    n_occ = len(persistence.load_occurrence_events())
    n_strat = len(persistence.load_strategies())
    print()
    print(f"Program events:      {n_events}")
    print(f"Strategy occurrences:{n_occ}")
    print(f"Strategies (deduped):{n_strat}")


def _cmd_list_strategies(query, limit: int) -> None:
    strats = query.list_strategies(limit)
    if not strats:
        print("No strategies found.")
        return
    for s in strats:
        print(f"  {s.strategy_id[:12]}  {s.description[:70]}")


if __name__ == "__main__":
    main()
